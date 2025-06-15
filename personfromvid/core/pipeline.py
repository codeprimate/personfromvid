"""Main processing pipeline orchestrator.

This module implements the core ProcessingPipeline class that coordinates
the entire video processing workflow with state management and resumption.
"""

import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

import numpy as np

from ..data import (
    Config,
    PipelineState,
    VideoMetadata,
    FrameData,
    SourceInfo,
    ImageProperties,
    StepProgress,
    ProcessingContext,
)
from ..utils.logging import get_logger, get_formatter
from ..utils.exceptions import (
    PersonFromVidError,
    VideoProcessingError,
    StateManagementError,
    InterruptionError,
    VideoFileError,
)
from ..data.constants import (
    get_pipeline_steps,
    get_pipeline_step_names,
    get_total_pipeline_steps,
)
from .steps import (
    InitializationStep,
    FrameExtractionStep,
    FaceDetectionStep,
    PoseAnalysisStep,
    CloseupDetectionStep,
    QualityAssessmentStep,
    FrameSelectionStep,
    OutputGenerationStep,
    PipelineStep,
)


@dataclass
class ProcessingResult:
    """Result of pipeline processing."""

    success: bool
    total_frames_extracted: int = 0
    faces_found: int = 0
    poses_found: Dict[str, int] = None
    head_angles_found: Dict[str, int] = None
    output_files: List[str] = None
    processing_time_seconds: float = 0.0
    error_message: Optional[str] = None

    def __post_init__(self):
        """Initialize default values."""
        if self.poses_found is None:
            self.poses_found = {}
        if self.head_angles_found is None:
            self.head_angles_found = {}
        if self.output_files is None:
            self.output_files = []


@dataclass
class PipelineStatus:
    """Current status of pipeline processing."""

    current_step: str
    total_steps: int
    completed_steps: int
    overall_progress: float
    step_progress: Dict[str, float]
    is_completed: bool
    is_running: bool
    can_resume: bool
    error_message: Optional[str] = None


class ProcessingPipeline:
    """Main processing pipeline orchestrator.

    Coordinates the entire video processing workflow with state management,
    resumption, and interruption handling.
    """

    def __init__(self, context: ProcessingContext, formatter: Optional[Any] = None):
        """Initialize processing pipeline.

        Args:
            context: ProcessingContext with unified pipeline data
            formatter: Optional consolidated formatter for output
        """
        self.context = context
        self.video_path = context.video_path
        self.config = context.config
        self.temp_manager = context.temp_manager

        self.logger = get_logger("pipeline")

        # Use provided formatter or create default one
        self.formatter = formatter
        if not self.formatter and self.config.logging.enable_structured_output:
            self.formatter = get_formatter()

        # State management
        self.state: Optional[PipelineState] = None
        self.state_manager = None  # Will be initialized in process()

        # Processing components (will be initialized as needed)
        self.video_processor = None

        # Interruption handling
        self._interrupted = False
        self._original_sigint_handler = None
        self._original_sigterm_handler = None

        # Processing tracking
        self._start_time: Optional[float] = None
        self._step_start_time: Optional[float] = None

        # Step definitions
        self._steps: List[PipelineStep] = []

        # Validate inputs
        self._validate_inputs()

    def is_interrupted(self) -> bool:
        """Check if the pipeline has been interrupted."""
        return self._interrupted

    def get_step_start_time(self) -> Optional[float]:
        """Get the start time of the current step."""
        return self._step_start_time

    def _validate_inputs(self) -> None:
        """Validate pipeline inputs."""
        if not self.video_path.exists():
            raise VideoFileError(f"Video file not found: {self.video_path}")

        if not self.video_path.is_file():
            raise VideoFileError(f"Path is not a file: {self.video_path}")

        # Basic file size check
        file_size = self.video_path.stat().st_size
        if file_size == 0:
            raise VideoFileError(f"Video file is empty: {self.video_path}")

        self.logger.info(f"Pipeline initialized for video: {self.video_path}")
        self.logger.info(f"Video file size: {file_size / (1024*1024):.1f} MB")

    def process(self) -> ProcessingResult:
        """Execute the complete processing pipeline.

        Returns:
            ProcessingResult with success status and metrics
        """
        self._start_time = time.time()

        if not self.formatter or not hasattr(self.formatter, "start_processing"):
            self.logger.info("Starting video processing pipeline")

        try:
            self._setup_interruption_handling()
            self._initialize_state_management()

            self._initialize_steps()

            if self.state and self.state.can_resume():
                self.logger.info("Resuming from previous processing state")
                result = self._resume_processing()
            else:
                self.logger.info("Starting new processing from beginning")
                result = self._start_new_processing()

            # Handle temp cleanup after successful processing
            if result.success and not self.config.storage.keep_temp:
                if self.config.storage.cleanup_temp_on_success:
                    self.logger.info(
                        "Cleaning up temporary files after successful processing"
                    )
                    self.temp_manager.cleanup_temp_files()
                else:
                    self.logger.info("Temporary files kept (cleanup disabled)")

            return result

        except InterruptionError:
            self.logger.warning("Processing interrupted by user")
            self._save_current_state()

            # Handle temp cleanup after interruption
            if (
                not self.config.storage.keep_temp
                and self.config.storage.cleanup_temp_on_failure
            ):
                self.logger.info("Cleaning up temporary files after interruption")
                self.temp_manager.cleanup_temp_files()

            return ProcessingResult(
                success=False,
                error_message="Processing interrupted by user",
                processing_time_seconds=self._get_elapsed_time(),
            )
        except Exception as e:
            error_msg = f"Pipeline processing failed: {e}"
            self.logger.error(error_msg, exc_info=True)
            if self.formatter:
                self.formatter.print_error(str(e))
            self._save_current_state()

            # Handle temp cleanup after failure
            if (
                not self.config.storage.keep_temp
                and self.config.storage.cleanup_temp_on_failure
            ):
                self.logger.info("Cleaning up temporary files after failure")
                self.temp_manager.cleanup_temp_files()

            return ProcessingResult(
                success=False,
                error_message=str(e),
                processing_time_seconds=self._get_elapsed_time(),
            )
        finally:
            self._restore_signal_handlers()

    def resume(self) -> ProcessingResult:
        """Resume processing from saved state.

        Returns:
            ProcessingResult with success status and metrics
        """
        self.logger.info("Resuming processing from saved state")

        self._initialize_state_management()

        if not self.state or not self.state.can_resume():
            raise StateManagementError("No resumable state found")

        return self._resume_processing()

    def get_status(self) -> PipelineStatus:
        """Get current pipeline status.

        Returns:
            PipelineStatus with current processing information
        """
        if not self.state:
            return PipelineStatus(
                current_step="not_started",
                total_steps=get_total_pipeline_steps(),
                completed_steps=0,
                overall_progress=0.0,
                step_progress={},
                is_completed=False,
                is_running=False,
                can_resume=False,
            )

        total_steps = get_total_pipeline_steps()
        completed_steps = len(self.state.completed_steps)
        overall_progress = (completed_steps / total_steps) * 100.0

        step_progress = {
            name: progress.progress_percentage
            for name, progress in self.state.step_progress.items()
        }

        return PipelineStatus(
            current_step=self.state.current_step,
            total_steps=total_steps,
            completed_steps=completed_steps,
            overall_progress=overall_progress,
            step_progress=step_progress,
            is_completed=self.state.is_fully_completed(),
            is_running=not self._interrupted,
            can_resume=self.state.can_resume(),
        )

    def interrupt_gracefully(self) -> None:
        """Request graceful interruption of processing."""
        self.logger.info("Graceful interruption requested")
        self._interrupted = True

    def _initialize_state_management(self) -> None:
        """Initialize state management and load existing state if available."""
        from .state_manager import StateManager
        from .video_processor import VideoProcessor

        # Initialize components using ProcessingContext
        self.state_manager = StateManager(context=self.context)
        self.video_processor = VideoProcessor(str(self.video_path))

        # Handle force restart if requested - delete state before loading
        if self.config.processing.force_restart:
            self.logger.info(
                "Force restart requested - deleting existing state to start fresh"
            )
            self.state_manager.delete_state()

        existing_state = self.state_manager.load_state()

        if existing_state:
            self.state = existing_state
            self.logger.info(
                f"Loaded existing pipeline state, currently at step: {self.state.current_step}"
            )
        else:
            self.logger.info("No resumable state found, starting new processing...")
            self._create_initial_state()

    def _initialize_steps(self) -> None:
        """Initialize the pipeline step handlers."""
        step_classes = [
            InitializationStep,
            FrameExtractionStep,
            FaceDetectionStep,
            PoseAnalysisStep,
            CloseupDetectionStep,
            QualityAssessmentStep,
            FrameSelectionStep,
            OutputGenerationStep,
        ]
        self._steps = [step_class(self) for step_class in step_classes]

    def _create_initial_state(self) -> None:
        """Create initial pipeline state."""
        metadata = self._extract_video_metadata()
        video_hash = self._calculate_video_hash()

        self.state = PipelineState(
            video_file=str(self.video_path),
            video_hash=video_hash,
            video_metadata=metadata,
            model_versions={},
            created_at=datetime.now(),
            last_updated=datetime.now(),
        )
        self.logger.info(f"Created initial state for video: {self.video_path.name}")

    def _start_new_processing(self) -> ProcessingResult:
        """Start new processing from beginning."""
        return self._execute_pipeline_steps()

    def _resume_processing(self) -> ProcessingResult:
        """Resume processing from saved state."""
        resume_point = self.state.get_resume_point()

        if not resume_point:
            self.logger.info("All steps completed, nothing to resume")
            return self._create_success_result()

        self.logger.info(f"Resuming from step: {resume_point}")
        return self._execute_pipeline_steps(start_from=resume_point)

    def _execute_pipeline_steps(
        self, start_from: Optional[str] = None
    ) -> ProcessingResult:
        """Execute all pipeline steps in sequence."""
        step_map = {step.step_name: step for step in self._steps}
        steps_to_execute = [
            (name, step_map[name])
            for name in get_pipeline_step_names()
            if name in step_map
        ]

        start_index = 0
        if start_from:
            try:
                start_index = get_pipeline_step_names().index(start_from)
            except ValueError:
                self.logger.error(f"Cannot resume from unknown step: {start_from}")
                raise StateManagementError(f"Resume step '{start_from}' not found.")

        for i, (step_name, step_instance) in enumerate(
            steps_to_execute[start_index:], start_index + 1
        ):
            if self._interrupted:
                raise InterruptionError("Processing interrupted")

            if self.state.is_step_completed(step_name):
                self.logger.debug(f"Step '{step_name}' already completed, skipping")
                continue

            if self.formatter:
                self.formatter.start_step(
                    i, step_name, step_name.replace("_", " ").title()
                )
            else:
                self.logger.info(
                    f"STEP {i}/{get_total_pipeline_steps()}: {step_name.replace('_', ' ').upper()}"
                )

            self._step_start_time = time.time()

            try:
                step_instance.execute()
                self.state.complete_step(step_name)
                self._save_current_state()

                step_results = self._extract_step_results(step_name)

                if self.formatter:
                    self.formatter.complete_step(step_name, results=step_results)
                else:
                    self.logger.info(
                        f"✅ Step completed in {time.time() - self._step_start_time:.1f}s"
                    )

            except Exception as e:
                if self.formatter:
                    self.formatter.print_error(
                        f"Step '{step_name}' failed: {e}", step_name
                    )
                else:
                    self.logger.error(f"❌ Step '{step_name}' failed: {e}")

                self.state.fail_step(step_name, str(e))
                self._save_current_state()
                raise

        result = self._create_success_result()

        if self.formatter:
            output_path = str(self.config.storage.temp_directory or "auto-generated")
            self.formatter.print_completion_summary(result, output_path)

        return result

    def _extract_step_results(self, step_name: str) -> Dict[str, Any]:
        """Extract step results for consolidated display."""
        step_progress = self.state.get_step_progress(step_name)
        if not (step_progress and hasattr(step_progress, "step_data")):
            return {"summary": f"{step_name.replace('_', ' ').title()} completed"}

        try:
            return step_progress.step_data.get("step_results", {})
        except Exception as e:
            self.logger.debug(f"Could not extract results for {step_name}: {e}")
            return {}

    def _find_frame_by_id(self, frame_id: str) -> Optional[FrameData]:
        """Find frame by ID in the centralized frames list."""
        for frame in self.state.frames:
            if frame.frame_id == frame_id:
                return frame
        return None

    def _extract_video_metadata(self) -> VideoMetadata:
        """Extract metadata from video file."""
        self.logger.info(f"Extracting metadata from: {self.video_path}")
        metadata = self.video_processor.extract_metadata()
        self.logger.info(
            f"Video metadata extracted: {metadata.width}x{metadata.height}, {metadata.duration:.1f}s, {metadata.fps:.1f}fps"
        )
        return metadata

    def _calculate_video_hash(self) -> str:
        """Calculate SHA256 hash of video file for integrity checking."""
        return self.video_processor.calculate_hash()

    def _save_current_state(self) -> None:
        """Save current pipeline state."""
        if self.state and self.state_manager:
            try:
                self.state_manager.save_state(self.state)
                self.logger.debug("Pipeline state saved successfully")
            except Exception as e:
                self.logger.error(f"Failed to save pipeline state: {e}")

    def _create_success_result(self) -> ProcessingResult:
        """Create success result from current state."""
        self.logger.info("=" * 60)
        self.logger.info("PROCESSING COMPLETE ✅")

        total_frames = self.state.get_total_frames_extracted()
        faces_found = self.state.get_faces_found()
        poses_found = self.state.get_poses_found()
        head_angles_found = self.state.get_head_angles_found()
        processing_time = self._get_elapsed_time()

        self.logger.info(f"📊 RESULTS SUMMARY:")
        self.logger.info(f"   • Total frames extracted: {total_frames}")
        self.logger.info(f"   • Faces detected: {faces_found}")

        if poses_found:
            total_body_poses = sum(poses_found.values())
            self.logger.info(f"   • Body poses found: {total_body_poses}")

        if head_angles_found:
            total_head_poses = sum(head_angles_found.values())
            self.logger.info(f"   • Head poses found: {total_head_poses}")

        self.logger.info(f"   • Processing time: {processing_time:.1f}s")
        self.logger.info("")

        if poses_found:
            self.logger.info(f"🏆 TOP BODY POSES:")
            sorted_poses = sorted(poses_found.items(), key=lambda x: x[1], reverse=True)
            for pose, count in sorted_poses[:3]:
                self.logger.info(f"   • {pose}: {count} instances")

        if head_angles_found:
            self.logger.info(f"🎯 TOP HEAD POSES:")
            sorted_head_poses = sorted(
                head_angles_found.items(), key=lambda x: x[1], reverse=True
            )
            for pose, count in sorted_head_poses[:3]:
                self.logger.info(f"   • {pose}: {count} instances")

        self.logger.info("=" * 60)

        return ProcessingResult(
            success=True,
            total_frames_extracted=total_frames,
            faces_found=faces_found,
            poses_found=poses_found,
            head_angles_found=head_angles_found,
            output_files=self.state.processing_stats.get("output_files", []),
            processing_time_seconds=processing_time,
        )

    def _get_elapsed_time(self) -> float:
        """Get elapsed processing time."""
        if self._start_time is None:
            return 0.0
        return time.time() - self._start_time

    def _setup_interruption_handling(self) -> None:
        """Set up signal handlers for graceful interruption."""
        self._original_sigint_handler = signal.signal(
            signal.SIGINT, self._signal_handler
        )
        self._original_sigterm_handler = signal.signal(
            signal.SIGTERM, self._signal_handler
        )
        self.logger.debug("Interruption handling configured")

    def _signal_handler(self, signum: int, frame) -> None:
        """Handle interruption signals."""
        signal_name = "SIGINT" if signum == signal.SIGINT else "SIGTERM"
        self.logger.info(f"Received {signal_name}, initiating graceful shutdown...")
        self._interrupted = True

    def _restore_signal_handlers(self) -> None:
        """Restore original signal handlers."""
        if self._original_sigint_handler is not None:
            signal.signal(signal.SIGINT, self._original_sigint_handler)
        if self._original_sigterm_handler is not None:
            signal.signal(signal.SIGTERM, self._original_sigterm_handler)
        self.logger.debug("Signal handlers restored")

"""Unit tests for the ProcessingPipeline class."""

import signal
import subprocess
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from personfromvid.core.pipeline import (
    PipelineStatus,
    ProcessingPipeline,
    ProcessingResult,
)
from personfromvid.data import Config, ProcessingContext, VideoMetadata
from personfromvid.data.constants import get_total_pipeline_steps
from personfromvid.utils.exceptions import (
    InterruptionError,
    StateManagementError,
    VideoFileError,
)


@pytest.fixture
def sample_config():
    """Create sample configuration for testing."""
    return Config()

@pytest.fixture
def processing_context(temp_video_file, sample_config):
    """Create ProcessingContext for testing."""
    output_dir = Path(temp_video_file).parent / 'test_output'

    with patch('personfromvid.core.temp_manager.TempManager'):
        # Configure any necessary mock behavior on mock_temp_manager here

        context = ProcessingContext(
            video_path=Path(temp_video_file),
            video_base_name=Path(temp_video_file).stem,
            config=sample_config,
            output_directory=output_dir
        )

        yield context

        # No cleanup needed for the mock, but if the context created dirs, clean them.
        if output_dir.exists():
            import shutil
            shutil.rmtree(output_dir, ignore_errors=True)

@pytest.fixture
def temp_video_file():
    """Create temporary video file for testing."""
    temp_path = None
    try:
        # Create a minimal valid video file using ffmpeg
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            temp_path = f.name

        # Generate a simple 1-second black video
        result = subprocess.run([
            'ffmpeg', '-f', 'lavfi', '-i', 'color=black:size=320x240:duration=1:rate=1',
            '-y', temp_path
        ], capture_output=True, text=True)

        if result.returncode != 0:
            # Fallback: create a file with fake binary content that's large enough
            with open(temp_path, 'wb') as f:
                # Write a fake video header that won't crash ffprobe immediately
                f.write(b'\x00\x00\x00\x20ftypmp41' + b'\x00' * 1000)

        yield temp_path

    finally:
        # Cleanup
        if temp_path:
            Path(temp_path).unlink(missing_ok=True)

class TestProcessingPipeline:
    """Test suite for ProcessingPipeline class."""

    def test_pipeline_initialization(self, processing_context):
        """Test pipeline initialization with valid inputs."""
        pipeline = ProcessingPipeline(context=processing_context)

        assert pipeline.video_path == processing_context.video_path
        assert pipeline.config == processing_context.config
        assert pipeline.context == processing_context
        assert pipeline.temp_manager == processing_context.temp_manager
        assert pipeline._interrupted is False
        assert pipeline.state is None
        assert pipeline.state_manager is None

    def test_pipeline_initialization_invalid_video(self, sample_config):
        """Test pipeline initialization with invalid video file."""
        # ProcessingContext now validates file existence, so we expect FileNotFoundError
        with pytest.raises(FileNotFoundError, match="Video file does not exist"):
            with patch('personfromvid.core.temp_manager.TempManager'):
                ProcessingContext(
                    video_path=Path("/nonexistent/video.mp4"),
                    video_base_name="video",
                    config=sample_config,
                    output_directory=Path("/tmp/output")
                )

    def test_validate_inputs_empty_file(self, sample_config):
        """Test validation of empty video file."""
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            empty_file = f.name

        try:
            with patch('personfromvid.core.temp_manager.TempManager'):
                context = ProcessingContext(
                    video_path=Path(empty_file),
                    video_base_name=Path(empty_file).stem,
                    config=sample_config,
                    output_directory=Path(empty_file).parent / 'output'
                )
            with pytest.raises(VideoFileError, match="Video file is empty"):
                ProcessingPipeline(context=context)
        finally:
            Path(empty_file).unlink(missing_ok=True)

    def test_interrupt_gracefully(self, processing_context):
        """Test graceful interruption."""
        pipeline = ProcessingPipeline(context=processing_context)

        assert pipeline._interrupted is False
        pipeline.interrupt_gracefully()
        assert pipeline._interrupted is True

    def test_get_status_no_state(self, processing_context):
        """Test getting status when no state exists."""
        pipeline = ProcessingPipeline(context=processing_context)

        status = pipeline.get_status()

        assert isinstance(status, PipelineStatus)
        assert status.current_step == "not_started"
        assert status.total_steps == get_total_pipeline_steps()
        assert status.completed_steps == 0
        assert status.overall_progress == 0.0
        assert status.step_progress == {}
        assert status.is_completed is False
        assert status.is_running is False
        assert status.can_resume is False

    def test_get_status_with_state(self, processing_context):
        """Test getting status when state exists."""
        # Create mock state
        mock_state = Mock()
        mock_state.current_step = "face_detection"
        mock_state.completed_steps = ["initialization", "frame_extraction"]
        mock_state.step_progress = {"face_detection": Mock(progress_percentage=50.0)}
        mock_state.is_fully_completed.return_value = False
        mock_state.can_resume.return_value = True

        pipeline = ProcessingPipeline(context=processing_context)
        pipeline.state = mock_state

        status = pipeline.get_status()

        assert status.current_step == "face_detection"
        assert status.total_steps == get_total_pipeline_steps()
        assert status.completed_steps == 2
        assert status.overall_progress == pytest.approx(25.0, rel=1e-2)
        assert status.step_progress == {"face_detection": 50.0}
        assert status.is_completed is False
        assert status.is_running is True
        assert status.can_resume is True

    @patch('personfromvid.core.state_manager.StateManager')
    @patch('personfromvid.core.video_processor.VideoProcessor')
    def test_initialize_state_management_existing_state(self, mock_video_processor_class, mock_state_manager_class, processing_context):
        """Test state management initialization with existing state."""
        mock_state_manager = Mock()
        mock_existing_state = Mock()
        mock_state_manager.load_state.return_value = mock_existing_state
        mock_state_manager_class.return_value = mock_state_manager

        pipeline = ProcessingPipeline(context=processing_context)
        pipeline._initialize_state_management()

        assert pipeline.state_manager == mock_state_manager
        assert pipeline.state == mock_existing_state
        mock_state_manager.load_state.assert_called_once()

    @patch('personfromvid.core.state_manager.StateManager')
    @patch('personfromvid.core.video_processor.VideoProcessor')
    def test_initialize_state_management_no_existing_state(self, mock_video_processor_class, mock_state_manager_class, processing_context):
        """Test state management initialization without existing state."""
        # Mock video processor to avoid ffprobe issues
        mock_video_processor = Mock()
        mock_metadata = VideoMetadata(
            duration=60.0, fps=30.0, width=1920, height=1080,
            codec="h264", total_frames=1800, file_size_bytes=50000000, format="mp4"
        )
        mock_video_processor.extract_metadata.return_value = mock_metadata
        mock_video_processor.calculate_hash.return_value = "test_hash"
        mock_video_processor_class.return_value = mock_video_processor

        mock_state_manager = Mock()
        mock_state_manager.load_state.return_value = None
        mock_state_manager_class.return_value = mock_state_manager

        pipeline = ProcessingPipeline(context=processing_context)
        pipeline._initialize_state_management()

        assert pipeline.state_manager == mock_state_manager
        # State should be created when none exists
        mock_state_manager.load_state.assert_called_once()

    def test_create_initial_state(self, processing_context):
        """Test creation of initial pipeline state."""
        pipeline = ProcessingPipeline(context=processing_context)

        with patch.object(pipeline, '_extract_video_metadata') as mock_metadata, \
             patch.object(pipeline, '_calculate_video_hash') as mock_hash:

            mock_metadata.return_value = VideoMetadata(
                duration=120.0, fps=30.0, width=1920, height=1080,
                codec="h264", total_frames=3600, file_size_bytes=1000000, format="mp4"
            )
            mock_hash.return_value = "test_hash_123"

            pipeline._create_initial_state()

            assert pipeline.state is not None
            assert pipeline.state.video_file == str(processing_context.video_path)
            assert pipeline.state.video_hash == "test_hash_123"
            assert pipeline.state.video_metadata.duration == 120.0
            mock_metadata.assert_called_once()
            mock_hash.assert_called_once()

    @patch('personfromvid.core.video_processor.VideoProcessor')
    def test_extract_video_metadata(self, mock_video_processor_class, processing_context):
        """Test video metadata extraction."""
        # Set up mock video processor
        mock_video_processor = Mock()
        mock_metadata = VideoMetadata(
            duration=60.0,
            fps=30.0,
            width=1920,
            height=1080,
            codec="h264",
            total_frames=1800,
            file_size_bytes=50000000,
            format="mp4"
        )
        mock_video_processor.extract_metadata.return_value = mock_metadata
        mock_video_processor_class.return_value = mock_video_processor

        pipeline = ProcessingPipeline(context=processing_context)
        # Initialize video processor manually for testing
        pipeline.video_processor = mock_video_processor

        metadata = pipeline._extract_video_metadata()

        assert isinstance(metadata, VideoMetadata)
        assert metadata.duration == 60.0
        assert metadata.fps == 30.0
        assert metadata.width == 1920
        assert metadata.height == 1080
        assert metadata.codec == "h264"

    @patch('personfromvid.core.video_processor.VideoProcessor')
    def test_calculate_video_hash(self, mock_video_processor_class, processing_context):
        """Test video hash calculation."""
        # Set up mock video processor
        mock_video_processor = Mock()
        mock_hash = "abc123def456789abcdef0123456789abcdef0123456789abcdef0123456789a"
        mock_video_processor.calculate_hash.return_value = mock_hash
        mock_video_processor_class.return_value = mock_video_processor

        pipeline = ProcessingPipeline(context=processing_context)
        # Initialize video processor manually for testing
        pipeline.video_processor = mock_video_processor

        hash_value = pipeline._calculate_video_hash()

        assert isinstance(hash_value, str)
        assert len(hash_value) == 64  # SHA256 hex digest length
        assert hash_value == mock_hash

        # Test consistency - same file should produce same hash
        hash_value2 = pipeline._calculate_video_hash()
        assert hash_value == hash_value2

    @patch('personfromvid.core.state_manager.StateManager')
    def test_save_current_state(self, mock_state_manager_class, processing_context):
        """Test saving current state."""
        mock_state_manager = Mock()
        mock_state_manager_class.return_value = mock_state_manager

        pipeline = ProcessingPipeline(context=processing_context)
        pipeline.state_manager = mock_state_manager
        pipeline.state = Mock()

        pipeline._save_current_state()

        mock_state_manager.save_state.assert_called_once_with(pipeline.state)

    def test_save_current_state_no_state_manager(self, processing_context):
        """Test saving state when no state manager exists."""
        pipeline = ProcessingPipeline(context=processing_context)
        pipeline.state = Mock()
        pipeline.state_manager = None

        # Should not raise exception
        pipeline._save_current_state()

    @patch('personfromvid.core.state_manager.StateManager')
    def test_process_new_video(self, mock_state_manager_class, processing_context):
        """Test processing new video from start."""
        mock_state_manager = Mock()
        mock_state_manager.load_state.return_value = None
        mock_state_manager_class.return_value = mock_state_manager

        pipeline = ProcessingPipeline(context=processing_context)

        with patch.object(pipeline, '_setup_interruption_handling'), \
             patch.object(pipeline, '_create_initial_state') as mock_create_state, \
             patch.object(pipeline, '_execute_pipeline_steps') as mock_execute:

            mock_result = ProcessingResult(success=True)
            mock_execute.return_value = mock_result

            result = pipeline.process()

            mock_create_state.assert_called_once()
            mock_execute.assert_called_once_with()
            assert result == mock_result

    @patch('personfromvid.core.state_manager.StateManager')
    def test_process_resume_existing(self, mock_state_manager_class, processing_context):
        """Test processing with resume from existing state."""
        mock_state = Mock()
        mock_state.can_resume.return_value = True
        mock_state.get_resume_point.return_value = "face_detection"

        mock_state_manager = Mock()
        mock_state_manager.load_state.return_value = mock_state
        mock_state_manager_class.return_value = mock_state_manager

        pipeline = ProcessingPipeline(context=processing_context)

        with patch.object(pipeline, '_setup_interruption_handling'), \
             patch.object(pipeline, '_execute_pipeline_steps') as mock_execute:

            mock_result = ProcessingResult(success=True)
            mock_execute.return_value = mock_result

            result = pipeline.process()

            mock_execute.assert_called_once_with(start_from="face_detection")
            assert result == mock_result

    @patch('personfromvid.core.state_manager.StateManager')
    def test_process_interruption(self, mock_state_manager_class, processing_context):
        """Test processing with interruption."""
        mock_state_manager = Mock()
        mock_state_manager.load_state.return_value = None
        mock_state_manager_class.return_value = mock_state_manager

        pipeline = ProcessingPipeline(context=processing_context)

        with patch.object(pipeline, '_setup_interruption_handling'), \
             patch.object(pipeline, '_create_initial_state'), \
             patch.object(pipeline, '_execute_pipeline_steps') as mock_execute, \
             patch.object(pipeline, '_save_current_state'):

            mock_execute.side_effect = InterruptionError("Processing interrupted")

            result = pipeline.process()

            assert result.success is False
            assert "interrupted" in result.error_message.lower()

    @patch('personfromvid.core.state_manager.StateManager')
    def test_resume_method(self, mock_state_manager_class, processing_context):
        """Test the resume method."""
        mock_state = Mock()
        mock_state.can_resume.return_value = True
        mock_state.get_resume_point.return_value = "pose_analysis"

        mock_state_manager = Mock()
        mock_state_manager.load_state.return_value = mock_state
        mock_state_manager_class.return_value = mock_state_manager

        pipeline = ProcessingPipeline(context=processing_context)

        with patch.object(pipeline, '_execute_pipeline_steps') as mock_execute:
            mock_result = ProcessingResult(success=True)
            mock_execute.return_value = mock_result

            result = pipeline.resume()

            mock_execute.assert_called_once_with(start_from="pose_analysis")
            assert result == mock_result

    @patch('personfromvid.core.state_manager.StateManager')
    @patch('personfromvid.core.video_processor.VideoProcessor')
    def test_resume_no_resumable_state(self, mock_video_processor_class, mock_state_manager_class, processing_context):
        """Test resume when no resumable state exists."""
        # Mock video processor to avoid ffprobe issues
        mock_video_processor = Mock()
        mock_metadata = VideoMetadata(
            duration=60.0, fps=30.0, width=1920, height=1080,
            codec="h264", total_frames=1800, file_size_bytes=50000000, format="mp4"
        )
        mock_video_processor.extract_metadata.return_value = mock_metadata
        mock_video_processor.calculate_hash.return_value = "test_hash"
        mock_video_processor_class.return_value = mock_video_processor

        mock_state_manager = Mock()
        mock_state_manager.load_state.return_value = None
        mock_state_manager_class.return_value = mock_state_manager

        pipeline = ProcessingPipeline(context=processing_context)

        with pytest.raises(StateManagementError, match="No resumable state found"):
            pipeline.resume()

    def test_step_initialization(self, processing_context):
        """Test initialization step."""
        pipeline = ProcessingPipeline(context=processing_context)
        pipeline.state = Mock()

        # Mock the step initialization instead of calling pipeline methods directly
        with patch('personfromvid.core.steps.InitializationStep') as mock_init_step_class:
            mock_init_step = Mock()
            mock_init_step.step_name = "initialization"
            mock_init_step_class.return_value = mock_init_step

            pipeline._initialize_steps()

            # Verify step is created and can be executed
            assert len(pipeline._steps) > 0
            step_names = [step.step_name for step in pipeline._steps]
            assert "initialization" in step_names

    def test_step_placeholders(self, processing_context):
        """Test step class instantiation."""
        pipeline = ProcessingPipeline(context=processing_context)
        pipeline.state = Mock()

        # Test that steps can be initialized without errors
        with patch('personfromvid.core.pipeline.InitializationStep') as mock_init, \
             patch('personfromvid.core.pipeline.FrameExtractionStep') as mock_frame, \
             patch('personfromvid.core.pipeline.FaceDetectionStep') as mock_face, \
             patch('personfromvid.core.pipeline.PoseAnalysisStep') as mock_pose, \
             patch('personfromvid.core.pipeline.CloseupDetectionStep') as mock_closeup, \
             patch('personfromvid.core.pipeline.QualityAssessmentStep') as mock_quality, \
             patch('personfromvid.core.pipeline.FrameSelectionStep') as mock_selection, \
             patch('personfromvid.core.pipeline.OutputGenerationStep') as mock_output:

            # Mock each step class to return a mock instance
            for mock_step_class in [mock_init, mock_frame, mock_face, mock_pose,
                                  mock_closeup, mock_quality, mock_selection, mock_output]:
                mock_step = Mock()
                mock_step.step_name = "test_step"
                mock_step_class.return_value = mock_step

            pipeline._initialize_steps()

            # Verify all step classes were instantiated
            mock_init.assert_called_once_with(pipeline)
            mock_frame.assert_called_once_with(pipeline)
            mock_face.assert_called_once_with(pipeline)
            mock_pose.assert_called_once_with(pipeline)
            mock_closeup.assert_called_once_with(pipeline)
            mock_quality.assert_called_once_with(pipeline)
            mock_selection.assert_called_once_with(pipeline)
            mock_output.assert_called_once_with(pipeline)

    def test_execute_pipeline_steps_full_run(self, processing_context):
        """Test executing all pipeline steps."""
        pipeline = ProcessingPipeline(context=processing_context)
        pipeline.state = Mock()
        pipeline.state.is_step_completed.return_value = False
        pipeline.state.complete_step = Mock()
        pipeline.state.get_resume_point.return_value = None
        pipeline.state.frames = []  # Mock frames list

        # Mock video processor
        pipeline.video_processor = Mock()
        pipeline.video_processor.validate_format = Mock()
        pipeline.video_processor.get_video_info_summary.return_value = {'resolution': '320x240', 'duration_seconds': 1.0, 'fps': 1.0}

        # Mock the state to return empty lists to avoid len() issues
        def mock_get_step_progress(step_name):
            mock_step = Mock()
            mock_step.get_data.return_value = []  # Return empty list instead of Mock
            mock_step.set_data = Mock()
            mock_step.step_data = {'step_results': {}}
            return mock_step

        pipeline.state.get_step_progress.side_effect = mock_get_step_progress

        # Mock all step classes at the import level in pipeline module
        with patch('personfromvid.core.pipeline.InitializationStep') as mock_init, \
             patch('personfromvid.core.pipeline.FrameExtractionStep') as mock_frame, \
             patch('personfromvid.core.pipeline.FaceDetectionStep') as mock_face, \
             patch('personfromvid.core.pipeline.PoseAnalysisStep') as mock_pose, \
             patch('personfromvid.core.pipeline.CloseupDetectionStep') as mock_closeup, \
             patch('personfromvid.core.pipeline.QualityAssessmentStep') as mock_quality, \
             patch('personfromvid.core.pipeline.FrameSelectionStep') as mock_selection, \
             patch('personfromvid.core.pipeline.OutputGenerationStep') as mock_output, \
             patch.object(pipeline, '_save_current_state'), \
             patch.object(pipeline, '_create_success_result') as mock_result:

            # Set up mock step instances
            mock_steps = []
            step_names = ["initialization", "frame_extraction", "face_detection", "pose_analysis",
                         "closeup_detection", "quality_assessment", "frame_selection", "output_generation"]

            for mock_step_class, step_name in zip([mock_init, mock_frame, mock_face, mock_pose,
                                                 mock_closeup, mock_quality, mock_selection, mock_output], step_names, strict=False):
                mock_step = Mock()
                mock_step.step_name = step_name
                mock_step.execute = Mock()
                mock_step_class.return_value = mock_step
                mock_steps.append(mock_step)

            mock_result.return_value = ProcessingResult(success=True)

            pipeline._initialize_steps()
            result = pipeline._execute_pipeline_steps()

            # All steps should be executed
            for mock_step in mock_steps:
                mock_step.execute.assert_called_once()

            assert result.success is True

    def test_execute_pipeline_steps_with_resume_point(self, processing_context):
        """Test executing pipeline steps with resume point."""
        pipeline = ProcessingPipeline(context=processing_context)
        pipeline.state = Mock()
        pipeline.state.is_step_completed.return_value = False
        pipeline.state.complete_step = Mock()
        pipeline.state.frames = []  # Mock frames list

        # Mock video processor
        pipeline.video_processor = Mock()

        # Mock the state to return empty lists to avoid len() issues
        def mock_get_step_progress(step_name):
            mock_step = Mock()
            mock_step.get_data.return_value = []  # Return empty list instead of Mock
            mock_step.set_data = Mock()
            mock_step.step_data = {'step_results': {}}
            return mock_step

        pipeline.state.get_step_progress.side_effect = mock_get_step_progress

        # Mock all step classes at the import level in pipeline module
        with patch('personfromvid.core.pipeline.InitializationStep') as mock_init, \
             patch('personfromvid.core.pipeline.FrameExtractionStep') as mock_frame, \
             patch('personfromvid.core.pipeline.FaceDetectionStep') as mock_face, \
             patch('personfromvid.core.pipeline.PoseAnalysisStep') as mock_pose, \
             patch('personfromvid.core.pipeline.CloseupDetectionStep') as mock_closeup, \
             patch('personfromvid.core.pipeline.QualityAssessmentStep') as mock_quality, \
             patch('personfromvid.core.pipeline.FrameSelectionStep') as mock_selection, \
             patch('personfromvid.core.pipeline.OutputGenerationStep') as mock_output, \
             patch.object(pipeline, '_save_current_state'), \
             patch.object(pipeline, '_create_success_result') as mock_result:

            # Set up mock step instances
            mock_steps = []
            step_names = ["initialization", "frame_extraction", "face_detection", "pose_analysis",
                         "closeup_detection", "quality_assessment", "frame_selection", "output_generation"]

            for mock_step_class, step_name in zip([mock_init, mock_frame, mock_face, mock_pose,
                                                 mock_closeup, mock_quality, mock_selection, mock_output], step_names, strict=False):
                mock_step = Mock()
                mock_step.step_name = step_name
                mock_step.execute = Mock()
                mock_step_class.return_value = mock_step
                mock_steps.append(mock_step)

            mock_result.return_value = ProcessingResult(success=True)

            pipeline._initialize_steps()
            result = pipeline._execute_pipeline_steps(start_from="face_detection")

            # Only steps from face_detection onwards should be executed
            mock_steps[0].execute.assert_not_called()  # initialization
            mock_steps[1].execute.assert_not_called()  # frame_extraction
            mock_steps[2].execute.assert_called_once()  # face_detection
            mock_steps[3].execute.assert_called_once()  # pose_analysis
            mock_steps[4].execute.assert_called_once()  # closeup_detection
            mock_steps[5].execute.assert_called_once()  # quality_assessment
            mock_steps[6].execute.assert_called_once()  # frame_selection
            mock_steps[7].execute.assert_called_once()  # output_generation

            assert result.success is True

    def test_execute_pipeline_steps_skip_completed(self, processing_context):
        """Test executing pipeline steps when some are already completed."""
        pipeline = ProcessingPipeline(context=processing_context)
        pipeline.state = Mock()
        pipeline.state.frames = []  # Mock frames list

        def mock_is_completed(step):
            return step in ["initialization", "frame_extraction"]

        pipeline.state.is_step_completed.side_effect = mock_is_completed
        pipeline.state.complete_step = Mock()

        # Mock video processor
        pipeline.video_processor = Mock()

        # Mock the state to return empty lists to avoid len() issues
        def mock_get_step_progress(step_name):
            mock_step = Mock()
            mock_step.get_data.return_value = []  # Return empty list instead of Mock
            mock_step.set_data = Mock()
            mock_step.step_data = {'step_results': {}}
            return mock_step

        pipeline.state.get_step_progress.side_effect = mock_get_step_progress

        # Mock all step classes at the import level in pipeline module
        with patch('personfromvid.core.pipeline.InitializationStep') as mock_init, \
             patch('personfromvid.core.pipeline.FrameExtractionStep') as mock_frame, \
             patch('personfromvid.core.pipeline.FaceDetectionStep') as mock_face, \
             patch('personfromvid.core.pipeline.PoseAnalysisStep') as mock_pose, \
             patch('personfromvid.core.pipeline.CloseupDetectionStep') as mock_closeup, \
             patch('personfromvid.core.pipeline.QualityAssessmentStep') as mock_quality, \
             patch('personfromvid.core.pipeline.FrameSelectionStep') as mock_selection, \
             patch('personfromvid.core.pipeline.OutputGenerationStep') as mock_output, \
             patch.object(pipeline, '_save_current_state'), \
             patch.object(pipeline, '_create_success_result') as mock_result:

            # Set up mock step instances
            mock_steps = []
            step_names = ["initialization", "frame_extraction", "face_detection", "pose_analysis",
                         "closeup_detection", "quality_assessment", "frame_selection", "output_generation"]

            for mock_step_class, step_name in zip([mock_init, mock_frame, mock_face, mock_pose,
                                                 mock_closeup, mock_quality, mock_selection, mock_output], step_names, strict=False):
                mock_step = Mock()
                mock_step.step_name = step_name
                mock_step.execute = Mock()
                mock_step_class.return_value = mock_step
                mock_steps.append(mock_step)

            mock_result.return_value = ProcessingResult(success=True)

            pipeline._initialize_steps()
            result = pipeline._execute_pipeline_steps()

            # Only non-completed steps should be executed
            mock_steps[0].execute.assert_not_called()  # initialization (completed)
            mock_steps[1].execute.assert_not_called()  # frame_extraction (completed)

            # Remaining steps should be executed
            for i in range(2, len(mock_steps)):
                mock_steps[i].execute.assert_called_once()

            assert result.success is True

    def test_execute_pipeline_steps_with_interruption(self, processing_context):
        """Test executing pipeline steps with interruption."""
        pipeline = ProcessingPipeline(context=processing_context)
        pipeline.state = Mock()
        pipeline.state.is_step_completed.return_value = False
        pipeline._interrupted = True  # Simulate interruption

        with patch('personfromvid.core.steps.InitializationStep') as mock_init:
            mock_step = Mock()
            mock_step.step_name = "initialization"
            mock_init.return_value = mock_step

            pipeline._initialize_steps()

            with pytest.raises(InterruptionError):
                pipeline._execute_pipeline_steps()

    def test_create_success_result(self, processing_context):
        """Test creating success result."""
        pipeline = ProcessingPipeline(context=processing_context)
        pipeline._start_time = time.time() - 10.0  # 10 seconds ago

        mock_state = Mock()
        mock_state.get_total_frames_extracted.return_value = 100
        mock_state.get_faces_found.return_value = 25
        mock_state.get_poses_found.return_value = {"standing": 10, "sitting": 8}
        mock_state.get_head_angles_found.return_value = {"front": 15, "profile_left": 5}
        pipeline.state = mock_state

        result = pipeline._create_success_result()

        assert result.success is True
        assert result.total_frames_extracted == 100
        assert result.faces_found == 25
        assert result.poses_found == {"standing": 10, "sitting": 8}
        assert result.head_angles_found == {"front": 15, "profile_left": 5}
        assert result.processing_time_seconds >= 9.0  # Should be close to 10 seconds

    def test_signal_handlers(self, processing_context):
        """Test signal handler setup and restoration."""
        pipeline = ProcessingPipeline(context=processing_context)

        original_sigint = signal.signal(signal.SIGINT, signal.SIG_DFL)
        original_sigterm = signal.signal(signal.SIGTERM, signal.SIG_DFL)

        try:
            # Setup interruption handling
            pipeline._setup_interruption_handling()

            # Handlers should be changed
            current_sigint = signal.signal(signal.SIGINT, signal.SIG_DFL)
            current_sigterm = signal.signal(signal.SIGTERM, signal.SIG_DFL)

            assert current_sigint != original_sigint
            assert current_sigterm != original_sigterm

            # Store the handlers we just retrieved
            signal.signal(signal.SIGINT, current_sigint)
            signal.signal(signal.SIGTERM, current_sigterm)

            # Test signal handler
            assert pipeline._interrupted is False
            pipeline._signal_handler(signal.SIGINT, None)
            assert pipeline._interrupted is True

            # Restore handlers
            pipeline._restore_signal_handlers()

        finally:
            # Ensure we restore original handlers
            signal.signal(signal.SIGINT, original_sigint)
            signal.signal(signal.SIGTERM, original_sigterm)

    def test_get_elapsed_time(self, processing_context):
        """Test elapsed time calculation."""
        pipeline = ProcessingPipeline(context=processing_context)

        # No start time
        assert pipeline._get_elapsed_time() == 0.0

        # With start time
        start = time.time()
        pipeline._start_time = start
        time.sleep(0.1)  # Small delay

        elapsed = pipeline._get_elapsed_time()
        assert elapsed >= 0.1
        assert elapsed < 1.0  # Should be much less than a second

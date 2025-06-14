"""Pipeline state data models.

This module defines data classes for tracking the complete state of pipeline processing
and enabling resumption from interruptions.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

from .constants import get_pipeline_step_names, get_total_pipeline_steps
from .frame_data import FrameData


class PathEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles Path objects and other complex types."""

    def default(self, obj):
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)


@dataclass
class VideoMetadata:
    """Video file metadata."""

    duration: float  # Video duration in seconds
    fps: float  # Frames per second
    width: int  # Video width in pixels
    height: int  # Video height in pixels
    codec: str  # Video codec (e.g., "h264")
    total_frames: int  # Total frame count
    file_size_bytes: int  # Video file size
    format: str  # Container format (e.g., "mp4")

    def __post_init__(self):
        """Validate video metadata."""
        if self.duration <= 0:
            raise ValueError("duration must be positive")
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be positive")
        if self.total_frames <= 0:
            raise ValueError("total_frames must be positive")
        if self.file_size_bytes < 0:
            raise ValueError("file_size_bytes cannot be negative")

    @property
    def aspect_ratio(self) -> float:
        """Calculate aspect ratio."""
        return self.width / self.height

    @property
    def resolution_string(self) -> str:
        """Get resolution as string."""
        return f"{self.width}x{self.height}"


@dataclass
class StepProgress:
    """Progress information for a pipeline step."""

    total_items: int = 0
    processed_count: int = 0
    completed: bool = False
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Step-specific data
    step_data: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate step progress."""
        if self.total_items < 0:
            raise ValueError("total_items cannot be negative")
        if self.processed_count < 0:
            raise ValueError("processed_count cannot be negative")
        if self.processed_count > self.total_items:
            raise ValueError("processed_count cannot exceed total_items")

    @property
    def progress_percentage(self) -> float:
        """Calculate progress percentage."""
        if self.total_items == 0:
            return 100.0 if self.completed else 0.0
        return (self.processed_count / self.total_items) * 100.0

    @property
    def remaining_items(self) -> int:
        """Get number of remaining items."""
        return max(0, self.total_items - self.processed_count)

    def start(self, total_items: int) -> None:
        """Start step processing."""
        self.total_items = total_items
        self.processed_count = 0
        self.completed = False
        self.started_at = datetime.now()
        self.completed_at = None

    def update_progress(self, processed_count: int) -> None:
        """Update processing progress."""
        self.processed_count = min(processed_count, self.total_items)

    def complete(self) -> None:
        """Mark step as completed."""
        self.processed_count = self.total_items
        self.completed = True
        self.completed_at = datetime.now()

    def get_data(self, key: str, default: Any = None) -> Any:
        """Get step-specific data."""
        return self.step_data.get(key, default)

    def set_data(self, key: str, value: Any) -> None:
        """Set step-specific data."""
        self.step_data[key] = value


@dataclass
class PipelineState:
    """Complete pipeline processing state."""

    video_file: str  # Path to video file
    video_hash: str  # SHA256 hash of video file
    video_metadata: VideoMetadata  # Video file metadata
    model_versions: Dict[str, str]  # Model names and versions used

    # Timeline tracking
    created_at: datetime
    last_updated: datetime

    # Processing state
    current_step: str = "initialization"
    completed_steps: List[str] = field(default_factory=list)
    failed_steps: List[str] = field(default_factory=list)

    # Step progress tracking
    step_progress: Dict[str, StepProgress] = field(default_factory=dict)

    # Processing statistics
    processing_stats: Dict[str, Any] = field(default_factory=dict)

    # Configuration snapshot
    config_snapshot: Dict[str, Any] = field(default_factory=dict)

    # Frame data - centralized storage for all frame information
    frames: List[FrameData] = field(default_factory=list, repr=False)

    def __post_init__(self):
        """Initialize default step progress tracking."""
        if not self.step_progress:
            self._initialize_step_progress()

    def _initialize_step_progress(self) -> None:
        """Initialize progress tracking for all pipeline steps."""
        steps = get_pipeline_step_names()

        for step in steps:
            if step not in self.step_progress:
                self.step_progress[step] = StepProgress()

    # Step management
    def start_step(self, step_name: str, total_items: int = 0) -> None:
        """Start a processing step."""
        if step_name not in self.step_progress:
            self.step_progress[step_name] = StepProgress()

        self.step_progress[step_name].start(total_items)
        self.current_step = step_name
        self.last_updated = datetime.now()

    def update_step_progress(self, step_name: str, processed_count: int) -> None:
        """Update step progress."""
        if step_name not in self.step_progress:
            self.step_progress[step_name] = StepProgress()

        self.step_progress[step_name].update_progress(processed_count)
        self.last_updated = datetime.now()

    def complete_step(self, step_name: str) -> None:
        """Mark step as completed."""
        if step_name not in self.step_progress:
            self.step_progress[step_name] = StepProgress()

        self.step_progress[step_name].complete()

        if step_name not in self.completed_steps:
            self.completed_steps.append(step_name)

        # Remove from failed steps if it was there
        if step_name in self.failed_steps:
            self.failed_steps.remove(step_name)

        self.last_updated = datetime.now()

    def fail_step(self, step_name: str, error_message: str = "") -> None:
        """Mark step as failed."""
        if step_name not in self.failed_steps:
            self.failed_steps.append(step_name)

        # Add error to step data
        if step_name in self.step_progress:
            self.step_progress[step_name].set_data("error_message", error_message)

        self.last_updated = datetime.now()

    def is_step_completed(self, step_name: str) -> bool:
        """Check if step is completed."""
        return (
            step_name in self.completed_steps
            and step_name in self.step_progress
            and self.step_progress[step_name].completed
        )

    def is_step_failed(self, step_name: str) -> bool:
        """Check if step has failed."""
        return step_name in self.failed_steps

    def get_step_progress(self, step_name: str) -> Optional[StepProgress]:
        """Get progress for specific step."""
        return self.step_progress.get(step_name)

    # Resume logic
    def can_resume(self) -> bool:
        """Check if processing can be resumed."""
        return len(self.completed_steps) > 0 and not self.is_fully_completed()

    def get_next_step(self) -> Optional[str]:
        """Get the next step to process."""
        step_order = get_pipeline_step_names()

        for step in step_order:
            if not self.is_step_completed(step):
                return step

        return None

    def get_resume_point(self) -> Optional[str]:
        """Get the step to resume from."""
        # If current step is not completed, resume from it
        if not self.is_step_completed(self.current_step):
            return self.current_step

        # Otherwise get next step
        return self.get_next_step()

    def is_fully_completed(self) -> bool:
        """Check if all processing is completed."""
        required_steps = get_pipeline_step_names()

        return all(self.is_step_completed(step) for step in required_steps)

    # Statistics
    def get_total_frames_extracted(self) -> int:
        """Get total number of frames extracted."""
        return len(self.frames)

    def get_faces_found(self) -> int:
        """Get total number of faces found."""
        return sum(len(frame.face_detections) for frame in self.frames)

    def get_poses_found(self) -> Dict[str, int]:
        """Get poses found by category."""
        poses_by_category = {}
        for frame in self.frames:
            for pose_classification in frame.get_pose_classifications():
                poses_by_category[pose_classification] = (
                    poses_by_category.get(pose_classification, 0) + 1
                )
        return poses_by_category

    def get_head_angles_found(self) -> Dict[str, int]:
        """Get head angles found by category."""
        head_angles_by_category = {}
        for frame in self.frames:
            for head_direction in frame.get_head_directions():
                head_angles_by_category[head_direction] = (
                    head_angles_by_category.get(head_direction, 0) + 1
                )
        return head_angles_by_category

    def get_frame_selections(self) -> Dict[str, Any]:
        """Get frame selection results."""
        if "frame_selection" in self.step_progress:
            return self.step_progress["frame_selection"].get_data(
                "frame_selections", {}
            )
        return {}

    # Frame data access methods
    def add_frame(self, frame: FrameData) -> None:
        """Add a frame to the centralized frames list."""
        self.frames.append(frame)

    def get_frames_with_faces(self) -> List[FrameData]:
        """Get all frames that have face detections."""
        return [frame for frame in self.frames if frame.has_faces()]

    def get_frames_with_poses(self) -> List[FrameData]:
        """Get all frames that have pose detections."""
        return [frame for frame in self.frames if frame.has_poses()]

    def get_frames_with_head_poses(self) -> List[FrameData]:
        """Get all frames that have head pose results."""
        return [frame for frame in self.frames if frame.has_head_poses()]

    def get_usable_frames(self) -> List[FrameData]:
        """Get all frames that are marked as usable."""
        return [frame for frame in self.frames if frame.is_usable()]

    def get_selected_frames(self) -> List[FrameData]:
        """Get all frames that are selected for output."""
        return [frame for frame in self.frames if frame.is_selected()]

    def clear_frames(self) -> None:
        """Clear all frames (useful for testing or restarting)."""
        self.frames.clear()

    # Serialization
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "video_file": self.video_file,
            "video_hash": self.video_hash,
            "video_metadata": {
                "duration": self.video_metadata.duration,
                "fps": self.video_metadata.fps,
                "width": self.video_metadata.width,
                "height": self.video_metadata.height,
                "codec": self.video_metadata.codec,
                "total_frames": self.video_metadata.total_frames,
                "file_size_bytes": self.video_metadata.file_size_bytes,
                "format": self.video_metadata.format,
            },
            "model_versions": self.model_versions,
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "current_step": self.current_step,
            "completed_steps": self.completed_steps,
            "failed_steps": self.failed_steps,
            "step_progress": {
                step_name: {
                    "total_items": progress.total_items,
                    "processed_count": progress.processed_count,
                    "completed": progress.completed,
                    "started_at": (
                        progress.started_at.isoformat() if progress.started_at else None
                    ),
                    "completed_at": (
                        progress.completed_at.isoformat()
                        if progress.completed_at
                        else None
                    ),
                    "step_data": progress.step_data,
                }
                for step_name, progress in self.step_progress.items()
            },
            "processing_stats": self.processing_stats,
            "config_snapshot": self.config_snapshot,
            "frame_selections": self.get_frame_selections(),
            # Serialize frames to dictionaries for JSON storage
            "frames": [frame.to_dict() for frame in self.frames],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineState":
        """Create from dictionary (JSON deserialization)."""
        video_metadata = VideoMetadata(**data["video_metadata"])

        # Parse timestamps
        created_at = datetime.fromisoformat(data["created_at"])
        last_updated = datetime.fromisoformat(data["last_updated"])

        # Create step progress objects
        step_progress = {}
        for step_name, progress_data in data.get("step_progress", {}).items():
            progress = StepProgress(
                total_items=progress_data["total_items"],
                processed_count=progress_data["processed_count"],
                completed=progress_data["completed"],
                step_data=progress_data.get("step_data", {}),
            )

            if progress_data.get("started_at"):
                progress.started_at = datetime.fromisoformat(
                    progress_data["started_at"]
                )
            if progress_data.get("completed_at"):
                progress.completed_at = datetime.fromisoformat(
                    progress_data["completed_at"]
                )

            step_progress[step_name] = progress

        # Deserialize frames from dictionaries
        frames = []
        for frame_dict in data.get("frames", []):
            try:
                frame = FrameData.from_dict(frame_dict)
                frames.append(frame)
            except Exception as e:
                # Log warning but continue - don't fail entire state load for one bad frame
                import logging

                logging.getLogger("pipeline_state").warning(
                    f"Failed to deserialize frame: {e}"
                )

        return cls(
            video_file=data["video_file"],
            video_hash=data["video_hash"],
            video_metadata=video_metadata,
            model_versions=data["model_versions"],
            created_at=created_at,
            last_updated=last_updated,
            current_step=data.get("current_step", "initialization"),
            completed_steps=data.get("completed_steps", []),
            failed_steps=data.get("failed_steps", []),
            step_progress=step_progress,
            processing_stats=data.get("processing_stats", {}),
            config_snapshot=data.get("config_snapshot", {}),
            frames=frames,
        )

    def save_to_file(self, file_path: Path) -> None:
        """Save state to JSON file."""
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False, cls=PathEncoder)

    @classmethod
    def load_from_file(cls, file_path: Path) -> "PipelineState":
        """Load state from JSON file."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


@dataclass
class ProcessingResult:
    """Result of pipeline processing."""

    success: bool
    video_file: str
    processing_time_seconds: float = 0.0
    error_message: Optional[str] = None
    output_files: List[str] = field(default_factory=list)
    statistics: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate processing result."""
        if not self.success and not self.error_message:
            raise ValueError("Failed result must have error_message")


@dataclass
class PipelineStatus:
    """Current status of pipeline processing."""

    video_file: str
    current_step: Optional[str] = None
    progress_percentage: float = 0.0
    completed_steps: List[str] = field(default_factory=list)
    total_steps: int = field(
        default_factory=get_total_pipeline_steps
    )  # Total number of processing steps
    processing_rate: Optional[float] = None  # items/second
    estimated_time_remaining: Optional[float] = None  # seconds
    error_message: Optional[str] = None

    @property
    def is_completed(self) -> bool:
        """Check if processing is fully completed."""
        return len(self.completed_steps) >= self.total_steps

    @property
    def is_error(self) -> bool:
        """Check if there's an error state."""
        return self.error_message is not None

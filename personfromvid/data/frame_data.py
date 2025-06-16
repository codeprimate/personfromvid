"""Frame metadata structures.

This module defines data classes for storing comprehensive frame metadata
throughout the processing pipeline.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from .detection_results import (
    FaceDetection,
    PoseDetection,
    HeadPoseResult,
    QualityMetrics,
    CloseupDetection,
    ProcessingTimings,
)


@dataclass
class SourceInfo:
    """Source information for frame extraction."""

    video_timestamp: float  # Timestamp in video (seconds)
    extraction_method: str  # "i_frame", "temporal_sampling", "manual"
    original_frame_number: int  # Frame number in original video
    video_fps: float  # Original video FPS

    def __post_init__(self):
        """Validate source info."""
        if self.video_timestamp < 0:
            raise ValueError("video_timestamp cannot be negative")
        if self.original_frame_number < 0:
            raise ValueError("original_frame_number cannot be negative")
        if self.video_fps <= 0:
            raise ValueError("video_fps must be positive")


@dataclass
class ImageProperties:
    """Image properties and metadata."""

    width: int
    height: int
    channels: int
    file_size_bytes: int
    format: str  # "JPEG", "PNG", etc.
    color_space: str = "RGB"

    def __post_init__(self):
        """Validate image properties."""
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be positive")
        if self.channels not in [1, 3, 4]:
            raise ValueError("channels must be 1, 3, or 4")
        if self.file_size_bytes < 0:
            raise ValueError("file_size_bytes cannot be negative")

    @property
    def aspect_ratio(self) -> float:
        """Calculate aspect ratio."""
        return self.width / self.height

    @property
    def total_pixels(self) -> int:
        """Calculate total pixel count."""
        return self.width * self.height


@dataclass
class SelectionInfo:
    """Information about frame selection for output."""
    
    selected_for_poses: List[str] = field(default_factory=list)
    selected_for_head_angles: List[str] = field(default_factory=list)
    final_output: bool = False
    output_files: List[str] = field(default_factory=list)
    crop_regions: Dict[str, Tuple[int, int, int, int]] = field(default_factory=dict)
    selection_rank: Optional[int] = None
    quality_rank: Optional[int] = None
    category_scores: Dict[str, float] = field(default_factory=dict)
    category_score_breakdowns: Dict[str, Dict[str, float]] = field(default_factory=dict)
    category_ranks: Dict[str, int] = field(default_factory=dict)
    primary_selection_category: Optional[str] = None
    selection_competition: Dict[str, str] = field(default_factory=dict)
    final_selection_score: Optional[float] = None
    rejection_reason: Optional[str] = None


@dataclass
class ProcessingStepInfo:
    """Information about a single processing step."""

    timestamp: datetime
    model_version: Optional[str] = None
    processing_time_ms: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

    def add_warning(self, warning: str) -> None:
        """Add a warning message."""
        self.warnings.append(warning)


@dataclass
class FrameData:
    """Comprehensive frame data containing all processing information."""

    frame_id: str
    file_path: Path
    source_info: SourceInfo
    image_properties: ImageProperties

    # Optional cached image data (loaded on demand)
    _image: Optional[np.ndarray] = field(default=None, init=False, repr=False)

    # Processing results (populated during pipeline)
    face_detections: List[FaceDetection] = field(default_factory=list)
    pose_detections: List[PoseDetection] = field(default_factory=list)
    head_poses: List[HeadPoseResult] = field(default_factory=list)
    quality_metrics: Optional[QualityMetrics] = None
    closeup_detections: List[CloseupDetection] = field(default_factory=list)

    # Selection and output information
    selections: SelectionInfo = field(default_factory=SelectionInfo)

    # Processing metadata
    processing_steps: Dict[str, ProcessingStepInfo] = field(default_factory=dict)
    processing_timings: ProcessingTimings = field(default_factory=ProcessingTimings)

    # Debug and analysis
    debug_info: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate frame data."""
        if not self.frame_id:
            raise ValueError("frame_id cannot be empty")
        if not self.file_path:
            raise ValueError("file_path cannot be empty")

    @property
    def image(self) -> Optional[np.ndarray]:
        """Get the image data, loading it from file if not already cached.

        Returns:
            BGR image array (OpenCV format) or None if loading fails
        """
        if self._image is None:
            self._load_image()
        return self._image

    def _load_image(self) -> None:
        """Load image from file path and cache it."""
        try:
            import cv2

            if self.file_path.exists():
                self._image = cv2.imread(str(self.file_path))
                if self._image is None:
                    raise ValueError(f"Failed to load image from {self.file_path}")
            else:
                raise FileNotFoundError(f"Image file not found: {self.file_path}")
        except Exception as e:
            import logging

            logger = logging.getLogger(__name__)
            logger.error(f"Failed to load image for frame {self.frame_id}: {e}")
            self._image = None

    def unload_image(self) -> None:
        """Unload cached image to free memory."""
        self._image = None

    def start_processing_step(
        self, step_name: str, model_version: Optional[str] = None
    ) -> None:
        """Start tracking a processing step."""
        self.processing_steps[step_name] = ProcessingStepInfo(
            timestamp=datetime.now(), model_version=model_version
        )

    def complete_processing_step(
        self, step_name: str, processing_time_ms: Optional[float] = None
    ) -> None:
        """Complete a processing step."""
        if step_name in self.processing_steps:
            step_info = self.processing_steps[step_name]
            step_info.processing_time_ms = processing_time_ms
            step_info.success = True

            # Update processing timings
            if processing_time_ms is not None:
                self.processing_timings.add_timing(step_name, processing_time_ms)

    def fail_processing_step(self, step_name: str, error_message: str) -> None:
        """Mark a processing step as failed."""
        if step_name in self.processing_steps:
            step_info = self.processing_steps[step_name]
            step_info.success = False
            step_info.error_message = error_message

    def add_step_warning(self, step_name: str, warning: str) -> None:
        """Add warning to a processing step."""
        if step_name in self.processing_steps:
            self.processing_steps[step_name].add_warning(warning)

    def has_faces(self) -> bool:
        """Check if frame has any face detections."""
        return len(self.face_detections) > 0

    def has_poses(self) -> bool:
        """Check if frame has any pose detections."""
        return len(self.pose_detections) > 0

    def has_head_poses(self) -> bool:
        """Check if frame has any head pose results."""
        return len(self.head_poses) > 0

    def has_closeup_detections(self) -> bool:
        """Check if frame has any closeup detections."""
        return len(self.closeup_detections) > 0

    def get_best_face(self) -> Optional[FaceDetection]:
        """Get face detection with highest confidence."""
        if not self.face_detections:
            return None
        return max(self.face_detections, key=lambda f: f.confidence)

    def get_best_pose(self) -> Optional[PoseDetection]:
        """Get pose detection with highest confidence."""
        if not self.pose_detections:
            return None
        return max(self.pose_detections, key=lambda p: p.confidence)

    def get_best_head_pose(self) -> Optional[HeadPoseResult]:
        """Get head pose result with highest confidence."""
        if not self.head_poses:
            return None
        return max(self.head_poses, key=lambda h: h.confidence)

    def get_best_closeup(self) -> Optional[CloseupDetection]:
        """Get closeup detection with highest confidence."""
        if not self.closeup_detections:
            return None
        return max(self.closeup_detections, key=lambda c: c.confidence)

    def get_pose_classifications(self) -> List[str]:
        """Get all unique pose classifications."""
        classifications = []
        for pose in self.pose_detections:
            for classification, _ in pose.pose_classifications:
                if classification not in classifications:
                    classifications.append(classification)
        return classifications

    def get_head_directions(self) -> List[str]:
        """Get all unique head directions."""
        directions = []
        for head_pose in self.head_poses:
            if head_pose.direction and head_pose.direction not in directions:
                directions.append(head_pose.direction)
        return directions

    def has_pose_classification(self, classification: str) -> bool:
        """Check if frame has specific pose classification."""
        return classification in self.get_pose_classifications()

    def has_head_direction(self, direction: str) -> bool:
        """Check if frame has specific head direction."""
        return direction in self.get_head_directions()

    def get_shot_types(self) -> List[str]:
        """Get all unique shot types from closeup detections."""
        shot_types = []
        for closeup in self.closeup_detections:
            if closeup.shot_type and closeup.shot_type not in shot_types:
                shot_types.append(closeup.shot_type)
        return shot_types

    def has_shot_type(self, shot_type: str) -> bool:
        """Check if frame has specific shot type."""
        return shot_type in self.get_shot_types()

    def is_closeup_shot(self) -> bool:
        """Check if frame is classified as any type of closeup shot."""
        for closeup in self.closeup_detections:
            if closeup.is_closeup:
                return True
        return False

    def is_high_quality(self) -> bool:
        """Check if frame meets high quality standards."""
        return self.quality_metrics is not None and self.quality_metrics.is_high_quality

    def is_usable(self) -> bool:
        """Check if frame is usable for output."""
        return self.has_faces() and (
            self.quality_metrics is None or self.quality_metrics.usable
        )

    def is_selected(self) -> bool:
        """Check if frame is selected for any output."""
        return self.selections.final_output

    def is_selected_for_pose(self, pose: str) -> bool:
        """Check if frame is selected for specific pose category."""
        return pose in self.selections.selected_for_poses

    def is_selected_for_head_angle(self, angle: str) -> bool:
        """Check if frame is selected for specific head angle category."""
        return angle in self.selections.selected_for_head_angles

    def to_dict(self) -> Dict[str, Any]:
        """Convert frame data to dictionary for JSON serialization."""
        return {
            "frame_id": self.frame_id,
            "file_path": str(self.file_path),
            "source_info": {
                "video_timestamp": self.source_info.video_timestamp,
                "extraction_method": self.source_info.extraction_method,
                "original_frame_number": self.source_info.original_frame_number,
                "video_fps": self.source_info.video_fps,
            },
            "image_properties": {
                "width": self.image_properties.width,
                "height": self.image_properties.height,
                "channels": self.image_properties.channels,
                "file_size_bytes": self.image_properties.file_size_bytes,
                "format": self.image_properties.format,
                "color_space": self.image_properties.color_space,
            },
            "face_detections": [
                {
                    "bbox": face.bbox,
                    "confidence": face.confidence,
                    "landmarks": face.landmarks,
                    "area": face.area,
                }
                for face in self.face_detections
            ],
            "pose_detections": [
                {
                    "bbox": pose.bbox,
                    "confidence": pose.confidence,
                    "keypoints": pose.keypoints,
                    "pose_classifications": pose.pose_classifications,
                }
                for pose in self.pose_detections
            ],
            "head_poses": [
                {
                    "yaw": head.yaw,
                    "pitch": head.pitch,
                    "roll": head.roll,
                    "confidence": head.confidence,
                    "direction": head.direction,
                    "face_id": head.face_id,
                }
                for head in self.head_poses
            ],
            "quality_metrics": (
                {
                    "laplacian_variance": self.quality_metrics.laplacian_variance,
                    "sobel_variance": self.quality_metrics.sobel_variance,
                    "brightness_score": self.quality_metrics.brightness_score,
                    "contrast_score": self.quality_metrics.contrast_score,
                    "overall_quality": self.quality_metrics.overall_quality,
                    "quality_issues": self.quality_metrics.quality_issues,
                    "usable": self.quality_metrics.usable,
                }
                if self.quality_metrics
                else None
            ),
            "closeup_detections": [
                {
                    "is_closeup": closeup.is_closeup,
                    "shot_type": closeup.shot_type,
                    "confidence": closeup.confidence,
                    "face_area_ratio": closeup.face_area_ratio,
                    "inter_ocular_distance": closeup.inter_ocular_distance,
                    "estimated_distance": closeup.estimated_distance,
                    "shoulder_width_ratio": closeup.shoulder_width_ratio,
                }
                for closeup in self.closeup_detections
            ],
            "selections": {
                "selected_for_poses": self.selections.selected_for_poses,
                "selected_for_head_angles": self.selections.selected_for_head_angles,
                "final_output": self.selections.final_output,
                "output_files": self.selections.output_files,
                "crop_regions": self.selections.crop_regions,
                "selection_rank": self.selections.selection_rank,
                "quality_rank": self.selections.quality_rank,
                "category_scores": self.selections.category_scores,
                "category_score_breakdowns": self.selections.category_score_breakdowns,
                "category_ranks": self.selections.category_ranks,
                "primary_selection_category": self.selections.primary_selection_category,
                "selection_competition": self.selections.selection_competition,
                "final_selection_score": self.selections.final_selection_score,
                "rejection_reason": self.selections.rejection_reason,
            },
            "processing_timings": {
                "face_detection_ms": self.processing_timings.face_detection_ms,
                "pose_estimation_ms": self.processing_timings.pose_estimation_ms,
                "head_pose_estimation_ms": self.processing_timings.head_pose_estimation_ms,
                "quality_assessment_ms": self.processing_timings.quality_assessment_ms,
                "closeup_detection_ms": self.processing_timings.closeup_detection_ms,
                "total_processing_ms": self.processing_timings.total_processing_ms,
            },
            "debug_info": self.debug_info,
        }

    @classmethod
    def from_dict(cls, frame_dict: Dict[str, Any]) -> "FrameData":
        """Convert frame dictionary back to FrameData object.

        Args:
            frame_dict: Dictionary representation of frame data

        Returns:
            FrameData object reconstructed from dictionary
        """
        from .detection_results import (
            FaceDetection,
            PoseDetection,
            HeadPoseResult,
            QualityMetrics,
            CloseupDetection,
            ProcessingTimings,
        )

        # Reconstruct source info
        source_info_dict = frame_dict.get("source_info", {})
        source_info = SourceInfo(
            video_timestamp=source_info_dict.get("video_timestamp", 0.0),
            extraction_method=source_info_dict.get("extraction_method", "unknown"),
            original_frame_number=source_info_dict.get("original_frame_number", 0),
            video_fps=source_info_dict.get("video_fps", 30.0),
        )

        # Reconstruct image properties
        image_props_dict = frame_dict.get("image_properties", {})
        image_properties = ImageProperties(
            width=image_props_dict.get("width", 0),
            height=image_props_dict.get("height", 0),
            channels=image_props_dict.get("channels", 3),
            file_size_bytes=image_props_dict.get("file_size_bytes", 0),
            format=image_props_dict.get("format", "JPEG"),
            color_space=image_props_dict.get("color_space", "RGB"),
        )

        # Reconstruct face detections
        face_detections = []
        for face_dict in frame_dict.get("face_detections", []):
            face_detection = FaceDetection(
                bbox=tuple(face_dict.get("bbox", [0, 0, 0, 0])),
                confidence=face_dict.get("confidence", 0.0),
                landmarks=face_dict.get("landmarks"),
            )
            face_detections.append(face_detection)

        # Reconstruct pose detections
        pose_detections = []
        for pose_dict in frame_dict.get("pose_detections", []):
            pose_detection = PoseDetection(
                bbox=tuple(pose_dict.get("bbox", [0, 0, 0, 0])),
                confidence=pose_dict.get("confidence", 0.0),
                keypoints=pose_dict.get("keypoints", {}),
                pose_classifications=[
                    tuple(pc) for pc in pose_dict.get("pose_classifications", [])
                ],
            )
            pose_detections.append(pose_detection)

        # Reconstruct head poses
        head_poses = []
        for head_dict in frame_dict.get("head_poses", []):
            head_pose = HeadPoseResult(
                yaw=head_dict.get("yaw", 0.0),
                pitch=head_dict.get("pitch", 0.0),
                roll=head_dict.get("roll", 0.0),
                confidence=head_dict.get("confidence", 0.0),
                direction=head_dict.get("direction", "unknown"),
                face_id=head_dict.get("face_id", 0),
            )
            head_poses.append(head_pose)

        # Reconstruct quality metrics
        quality_metrics = None
        quality_dict = frame_dict.get("quality_metrics")
        if quality_dict:
            quality_metrics = QualityMetrics(
                laplacian_variance=quality_dict.get("laplacian_variance", 0.0),
                sobel_variance=quality_dict.get("sobel_variance", 0.0),
                brightness_score=quality_dict.get("brightness_score", 0.0),
                contrast_score=quality_dict.get("contrast_score", 0.0),
                overall_quality=quality_dict.get("overall_quality", 0.0),
                quality_issues=quality_dict.get("quality_issues", []),
                usable=quality_dict.get("usable", True),
            )

        # Reconstruct closeup detections
        closeup_detections = []
        for closeup_dict in frame_dict.get("closeup_detections", []):
            closeup_detection = CloseupDetection(
                is_closeup=closeup_dict.get("is_closeup", False),
                shot_type=closeup_dict.get("shot_type", "unknown"),
                confidence=closeup_dict.get("confidence", 0.0),
                face_area_ratio=closeup_dict.get("face_area_ratio", 0.0),
                inter_ocular_distance=closeup_dict.get("inter_ocular_distance"),
                estimated_distance=closeup_dict.get("estimated_distance"),
                shoulder_width_ratio=closeup_dict.get("shoulder_width_ratio"),
            )
            closeup_detections.append(closeup_detection)

        # Reconstruct processing timings
        timings_dict = frame_dict.get("processing_timings", {})
        processing_timings = ProcessingTimings(
            face_detection_ms=timings_dict.get("face_detection_ms"),
            pose_estimation_ms=timings_dict.get("pose_estimation_ms"),
            head_pose_estimation_ms=timings_dict.get("head_pose_estimation_ms"),
            quality_assessment_ms=timings_dict.get("quality_assessment_ms"),
            closeup_detection_ms=timings_dict.get("closeup_detection_ms"),
            total_processing_ms=timings_dict.get("total_processing_ms"),
        )

        # Reconstruct selections - NEW FORMAT REQUIRED (no backward compatibility)
        selections_dict = frame_dict["selections"]  # Will raise KeyError if missing
        selections = SelectionInfo(
            selected_for_poses=selections_dict.get("selected_for_poses", []),
            selected_for_head_angles=selections_dict.get("selected_for_head_angles", []),
            final_output=selections_dict.get("final_output", False),
            output_files=selections_dict.get("output_files", []),
            crop_regions=selections_dict.get("crop_regions", {}),
            selection_rank=selections_dict.get("selection_rank"),
            quality_rank=selections_dict.get("quality_rank"),
            category_scores=selections_dict.get("category_scores", {}),
            category_score_breakdowns=selections_dict.get("category_score_breakdowns", {}),
            category_ranks=selections_dict.get("category_ranks", {}),
            primary_selection_category=selections_dict.get("primary_selection_category"),
            selection_competition=selections_dict.get("selection_competition", {}),
            final_selection_score=selections_dict.get("final_selection_score"),
            rejection_reason=selections_dict.get("rejection_reason"),
        )

        # Create FrameData object
        frame_data = cls(
            frame_id=frame_dict.get("frame_id", ""),
            file_path=Path(frame_dict.get("file_path", "")),
            source_info=source_info,
            image_properties=image_properties,
            face_detections=face_detections,
            pose_detections=pose_detections,
            head_poses=head_poses,
            quality_metrics=quality_metrics,
            closeup_detections=closeup_detections,
            selections=selections,
            processing_timings=processing_timings,
        )
        
        # Set debug_info separately since it's not a constructor parameter
        frame_data.debug_info = frame_dict.get("debug_info", {})

        return frame_data

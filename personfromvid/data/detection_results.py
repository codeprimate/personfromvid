"""Data structures for AI model outputs.

This module defines data classes for storing detection results from face detection,
pose estimation, and head pose estimation models.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .constants import QualityMethod


@dataclass
class FaceDetection:
    """Face detection result containing bounding box and landmarks."""

    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    landmarks: Optional[List[Tuple[float, float]]] = None  # List of (x, y) coordinates

    def __post_init__(self):
        """Validate bbox format and confidence range."""
        if len(self.bbox) != 4:
            raise ValueError("bbox must contain exactly 4 values (x1, y1, x2, y2)")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("confidence must be between 0.0 and 1.0")

    @property
    def width(self) -> int:
        """Get width of face bounding box."""
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        """Get height of face bounding box."""
        return self.bbox[3] - self.bbox[1]

    @property
    def area(self) -> int:
        """Get area of face bounding box."""
        return self.width * self.height

    @property
    def center(self) -> Tuple[float, float]:
        """Get center point of face bounding box."""
        return ((self.bbox[0] + self.bbox[2]) / 2, (self.bbox[1] + self.bbox[3]) / 2)


@dataclass
class PoseDetection:
    """Body pose detection result containing keypoints and classification."""

    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    keypoints: Dict[
        str, Tuple[float, float, float]
    ]  # {keypoint_name: (x, y, confidence)}
    pose_classifications: List[Tuple[str, float]] = field(default_factory=list)

    def __post_init__(self):
        """Validate detection data."""
        if len(self.bbox) != 4:
            raise ValueError("bbox must contain exactly 4 values (x1, y1, x2, y2)")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("confidence must be between 0.0 and 1.0")
        for _, confidence in self.pose_classifications:
            if not (0.0 <= confidence <= 1.0):
                raise ValueError("pose_confidence must be between 0.0 and 1.0")

    def get_keypoint(self, name: str) -> Optional[Tuple[float, float, float]]:
        """Get specific keypoint by name."""
        return self.keypoints.get(name)

    def has_keypoint(self, name: str) -> bool:
        """Check if keypoint exists and has sufficient confidence."""
        keypoint = self.keypoints.get(name)
        return keypoint is not None and keypoint[2] > 0.5

    @property
    def valid_keypoints(self) -> Dict[str, Tuple[float, float, float]]:
        """Get keypoints with confidence > 0.5."""
        return {name: kp for name, kp in self.keypoints.items() if kp[2] > 0.5}

    @property
    def center(self) -> Tuple[float, float]:
        """Get center point of body bounding box."""
        return ((self.bbox[0] + self.bbox[2]) / 2, (self.bbox[1] + self.bbox[3]) / 2)


@dataclass
class HeadPoseResult:
    """Head pose estimation result containing orientation angles."""

    yaw: float  # Rotation around vertical axis
    pitch: float  # Rotation around horizontal axis (nodding)
    roll: float  # Rotation around depth axis (tilting)
    confidence: float
    face_id: int = 0  # Index of face in frame
    direction: Optional[str] = None  # Classified cardinal direction
    direction_confidence: Optional[
        float
    ] = None  # Confidence of direction classification

    def __post_init__(self):
        """Validate angle ranges and confidence."""
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("confidence must be between 0.0 and 1.0")
        if self.direction_confidence is not None and not (
            0.0 <= self.direction_confidence <= 1.0
        ):
            raise ValueError("direction_confidence must be between 0.0 and 1.0")

        # Angles can be outside typical ranges, but warn about extreme values
        if abs(self.yaw) > 180 or abs(self.pitch) > 90 or abs(self.roll) > 180:
            import warnings

            warnings.warn(
                f"Extreme angle values detected: yaw={self.yaw}, pitch={self.pitch}, roll={self.roll}",
                stacklevel=2,
            )

    @property
    def is_valid_orientation(self) -> bool:
        """Check if head orientation is valid (not excessively tilted)."""
        return abs(self.roll) <= 30.0  # Max acceptable tilt

    @property
    def angles_dict(self) -> Dict[str, float]:
        """Get angles as dictionary."""
        return {"yaw": self.yaw, "pitch": self.pitch, "roll": self.roll}


@dataclass
class CloseupDetection:
    """Close-up shot detection result with detailed analysis."""

    is_closeup: bool
    shot_type: str  # "extreme_closeup", "closeup", "medium_closeup", "medium_shot", "wide_shot"
    confidence: float
    face_area_ratio: float  # Face area / total frame area
    inter_ocular_distance: Optional[float] = None  # Distance between eyes in pixels
    estimated_distance: Optional[str] = None  # "very_close", "close", "medium", "far"
    shoulder_width_ratio: Optional[float] = None  # Shoulder width / frame width

    def __post_init__(self):
        """Validate detection results and initialize lists."""
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("confidence must be between 0.0 and 1.0")
        if not (0.0 <= self.face_area_ratio <= 1.0):
            raise ValueError("face_area_ratio must be between 0.0 and 1.0")

        # Validate shot type
        valid_shot_types = [
            "extreme_closeup",
            "closeup",
            "medium_closeup",
            "medium_shot",
            "wide_shot",
            "unknown",
        ]
        if self.shot_type not in valid_shot_types:
            raise ValueError(f"shot_type must be one of {valid_shot_types}")

    @property
    def is_close_shot(self) -> bool:
        """Check if this is any kind of close shot (extreme/close/medium closeup)."""
        return self.shot_type in ["extreme_closeup", "closeup", "medium_closeup"]

    @property
    def quality_factors(self) -> Dict[str, float]:
        """Get quality factors as dictionary."""
        return {
            "face_area_ratio": self.face_area_ratio,
            "inter_ocular_distance": self.inter_ocular_distance or 0.0,
            "shoulder_width_ratio": self.shoulder_width_ratio or 0.0,
        }


@dataclass
class QualityMetrics:
    """Image quality assessment metrics."""

    laplacian_variance: float
    sobel_variance: float
    brightness_score: float
    contrast_score: float
    overall_quality: float
    method: QualityMethod = QualityMethod.DIRECT
    quality_issues: List[str] = None
    usable: bool = True

    def __post_init__(self):
        """Initialize quality issues list if None."""
        if self.quality_issues is None:
            self.quality_issues = []

    @property
    def is_high_quality(self) -> bool:
        """Check if image meets high quality thresholds."""
        return (
            self.overall_quality >= 0.7
            and self.laplacian_variance >= 100.0
            and self.usable
        )

    @property
    def has_issues(self) -> bool:
        """Check if image has quality issues."""
        return len(self.quality_issues) > 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "laplacian_variance": self.laplacian_variance,
            "sobel_variance": self.sobel_variance,
            "brightness_score": self.brightness_score,
            "contrast_score": self.contrast_score,
            "overall_quality": self.overall_quality,
            "method": self.method.value,
            "quality_issues": self.quality_issues,
            "usable": self.usable,
        }

    @classmethod
    def from_dict(cls, quality_dict: Dict[str, Any]) -> "QualityMetrics":
        """Create QualityMetrics from dictionary with backward compatibility."""
        # Handle method field with backward compatibility
        method_value = quality_dict.get("method", "direct")
        try:
            method = QualityMethod(method_value)
        except ValueError:
            # Default to DIRECT for unknown method values
            method = QualityMethod.DIRECT

        return cls(
            laplacian_variance=quality_dict.get("laplacian_variance", 0.0),
            sobel_variance=quality_dict.get("sobel_variance", 0.0),
            brightness_score=quality_dict.get("brightness_score", 0.0),
            contrast_score=quality_dict.get("contrast_score", 0.0),
            overall_quality=quality_dict.get("overall_quality", 0.0),
            method=method,
            quality_issues=quality_dict.get("quality_issues", []),
            usable=quality_dict.get("usable", True),
        )


@dataclass
class ProcessingTimings:
    """Processing time tracking for performance analysis."""

    face_detection_ms: Optional[float] = None
    pose_estimation_ms: Optional[float] = None
    head_pose_estimation_ms: Optional[float] = None
    quality_assessment_ms: Optional[float] = None
    closeup_detection_ms: Optional[float] = None
    total_processing_ms: Optional[float] = None

    def add_timing(self, step: str, duration_ms: float) -> None:
        """Add timing for a processing step."""
        if step == "face_detection":
            self.face_detection_ms = duration_ms
        elif step == "pose_estimation":
            self.pose_estimation_ms = duration_ms
        elif step == "head_pose_estimation":
            self.head_pose_estimation_ms = duration_ms
        elif step == "quality_assessment":
            self.quality_assessment_ms = duration_ms
        elif step == "closeup_detection":
            self.closeup_detection_ms = duration_ms

        # Update total
        self._update_total()

    def _update_total(self) -> None:
        """Update total processing time."""
        timings = [
            self.face_detection_ms,
            self.pose_estimation_ms,
            self.head_pose_estimation_ms,
            self.quality_assessment_ms,
            self.closeup_detection_ms,
        ]
        valid_timings = [t for t in timings if t is not None]
        if valid_timings:
            self.total_processing_ms = sum(valid_timings)

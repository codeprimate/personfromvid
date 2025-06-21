"""Person representation for multi-person frame analysis.

This module defines data structures for representing individual people
detected in frames, including associated face/body detections and
composite quality metrics.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union

from .detection_results import FaceDetection, HeadPoseResult, PoseDetection


@dataclass
class PersonQuality:
    """Composite quality assessment for a person in a frame."""

    face_quality: float
    body_quality: float
    overall_quality: float = field(init=False)

    def __post_init__(self):
        """Calculate overall quality and validate ranges."""
        # Validate quality score ranges
        if not (0.0 <= self.face_quality <= 1.0):
            raise ValueError("face_quality must be between 0.0 and 1.0")
        if not (0.0 <= self.body_quality <= 1.0):
            raise ValueError("body_quality must be between 0.0 and 1.0")

        # Calculate weighted overall quality: 70% face + 30% body
        self.overall_quality = 0.7 * self.face_quality + 0.3 * self.body_quality

    @property
    def is_high_quality(self) -> bool:
        """Check if person meets high quality thresholds."""
        return self.overall_quality >= 0.7

    @property
    def is_usable(self) -> bool:
        """Check if person quality is usable for output."""
        return self.overall_quality >= 0.3

    @property
    def quality_factors(self) -> Dict[str, float]:
        """Get quality factors as dictionary."""
        return {
            "face_quality": self.face_quality,
            "body_quality": self.body_quality,
            "overall_quality": self.overall_quality,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "face_quality": self.face_quality,
            "body_quality": self.body_quality,
            "overall_quality": self.overall_quality,
        }

    @classmethod
    def from_dict(cls, quality_dict: Dict[str, Any]) -> "PersonQuality":
        """Create PersonQuality from dictionary."""
        return cls(
            face_quality=quality_dict.get("face_quality", 0.0),
            body_quality=quality_dict.get("body_quality", 0.0),
        )


class FaceUnknown(FaceDetection):
    """Singleton sentinel object representing unknown/missing face detection."""

    _instance = None

    def __new__(cls):
        """Ensure only one instance exists."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize with default unknown values."""
        if not hasattr(self, "_initialized"):
            super().__init__(bbox=(0, 0, 0, 0), confidence=0.0, landmarks=None)
            self._initialized = True

    def __post_init__(self):
        """Override validation for sentinel object."""
        pass  # Skip validation for sentinel


class BodyUnknown(PoseDetection):
    """Singleton sentinel object representing unknown/missing body detection."""

    _instance = None

    def __new__(cls):
        """Ensure only one instance exists."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize with default unknown values."""
        if not hasattr(self, "_initialized"):
            super().__init__(
                bbox=(0, 0, 0, 0), confidence=0.0, keypoints={}, pose_classifications=[]
            )
            self._initialized = True

    def __post_init__(self):
        """Override validation for sentinel object."""
        pass  # Skip validation for sentinel


@dataclass
class Person:
    """Individual person detected in a frame with associated detections."""

    person_id: int
    face: Union[FaceDetection, FaceUnknown]
    body: Union[PoseDetection, BodyUnknown]
    head_pose: Optional[HeadPoseResult]
    quality: PersonQuality

    def __post_init__(self):
        """Validate person data."""
        if self.person_id < 0:
            raise ValueError("person_id must be non-negative")

        # Ensure at least one primary detection exists
        has_face = not isinstance(self.face, FaceUnknown)
        has_body = not isinstance(self.body, BodyUnknown)

        if not (has_face or has_body):
            raise ValueError("Person must have at least one detection (face or body)")

    @property
    def has_face(self) -> bool:
        """Check if person has a valid face detection."""
        return not isinstance(self.face, FaceUnknown)

    @property
    def has_body(self) -> bool:
        """Check if person has a valid body detection."""
        return not isinstance(self.body, BodyUnknown)

    @property
    def has_head_pose(self) -> bool:
        """Check if person has head pose information."""
        return self.head_pose is not None

    @property
    def center(self) -> Tuple[float, float]:
        """Get center point for spatial matching (body preferred, face fallback)."""
        if self.has_body:
            return self.body.center
        elif self.has_face:
            return self.face.center
        else:
            return (0.0, 0.0)

    @property
    def is_high_quality(self) -> bool:
        """Check if person meets high quality standards."""
        return self.quality.is_high_quality

    @property
    def is_usable(self) -> bool:
        """Check if person is usable for output."""
        return self.quality.is_usable and (self.has_face or self.has_body)

    def to_dict(self) -> Dict[str, Any]:
        """Convert Person to dictionary for JSON serialization."""
        # Handle face serialization
        if isinstance(self.face, FaceUnknown):
            face_dict = {"type": "FaceUnknown"}
        else:
            face_dict = {
                "type": "FaceDetection",
                "bbox": self.face.bbox,
                "confidence": self.face.confidence,
                "landmarks": self.face.landmarks,
            }

        # Handle body serialization
        if isinstance(self.body, BodyUnknown):
            body_dict = {"type": "BodyUnknown"}
        else:
            body_dict = {
                "type": "PoseDetection",
                "bbox": self.body.bbox,
                "confidence": self.body.confidence,
                "keypoints": self.body.keypoints,
                "pose_classifications": self.body.pose_classifications,
            }

        # Handle head pose serialization
        head_pose_dict = None
        if self.head_pose:
            head_pose_dict = {
                "yaw": self.head_pose.yaw,
                "pitch": self.head_pose.pitch,
                "roll": self.head_pose.roll,
                "confidence": self.head_pose.confidence,
                "face_id": self.head_pose.face_id,
                "direction": self.head_pose.direction,
                "direction_confidence": self.head_pose.direction_confidence,
            }

        return {
            "person_id": self.person_id,
            "face": face_dict,
            "body": body_dict,
            "head_pose": head_pose_dict,
            "quality": self.quality.to_dict(),
        }

    @classmethod
    def from_dict(cls, person_dict: Dict[str, Any]) -> "Person":
        """Create Person from dictionary representation."""
        # Reconstruct face detection
        face_dict = person_dict.get("face", {})
        if face_dict.get("type") == "FaceUnknown":
            face = FaceUnknown()
        else:
            face = FaceDetection(
                bbox=tuple(face_dict.get("bbox", [0, 0, 0, 0])),
                confidence=face_dict.get("confidence", 0.0),
                landmarks=face_dict.get("landmarks"),
            )

        # Reconstruct body detection
        body_dict = person_dict.get("body", {})
        if body_dict.get("type") == "BodyUnknown":
            body = BodyUnknown()
        else:
            body = PoseDetection(
                bbox=tuple(body_dict.get("bbox", [0, 0, 0, 0])),
                confidence=body_dict.get("confidence", 0.0),
                keypoints=body_dict.get("keypoints", {}),
                pose_classifications=[
                    tuple(pc) for pc in body_dict.get("pose_classifications", [])
                ],
            )

        # Reconstruct head pose
        head_pose = None
        head_pose_dict = person_dict.get("head_pose")
        if head_pose_dict:
            head_pose = HeadPoseResult(
                yaw=head_pose_dict.get("yaw", 0.0),
                pitch=head_pose_dict.get("pitch", 0.0),
                roll=head_pose_dict.get("roll", 0.0),
                confidence=head_pose_dict.get("confidence", 0.0),
                face_id=head_pose_dict.get("face_id", 0),
                direction=head_pose_dict.get("direction"),
                direction_confidence=head_pose_dict.get("direction_confidence"),
            )

        # Reconstruct quality
        quality_dict = person_dict.get("quality", {})
        quality = PersonQuality.from_dict(quality_dict)

        return cls(
            person_id=person_dict.get("person_id", 0),
            face=face,
            body=body,
            head_pose=head_pose,
            quality=quality,
        )

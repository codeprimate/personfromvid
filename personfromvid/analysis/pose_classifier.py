"""Body pose classification logic.

This module implements geometric analysis of pose keypoints to classify
body poses into standing, sitting, squatting, and closeup categories.
"""

import logging
import math
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from ..data.detection_results import PoseDetection
from ..data.frame_data import FrameData
from ..utils.exceptions import PoseClassificationError

logger = logging.getLogger(__name__)

# Angle thresholds for pose classification (in degrees)
STANDING_HIP_KNEE_MIN = 160.0  # Hip-knee angle for standing
SITTING_HIP_KNEE_MIN = 80.0  # Hip-knee angle range for sitting
SITTING_HIP_KNEE_MAX = 120.0
SQUATTING_HIP_KNEE_MAX = 90.0  # Hip-knee angle for squatting
TORSO_VERTICAL_THRESHOLD = 30.0  # Max deviation from vertical for torso

# Closeup detection thresholds
CLOSEUP_FACE_RATIO_THRESHOLD = 0.15  # Face area vs total frame area
CLOSEUP_SHOULDER_WIDTH_THRESHOLD = 0.35  # Shoulder width vs frame width

# Confidence thresholds for reliable keypoints
MIN_KEYPOINT_CONFIDENCE = 0.5
MIN_POSE_CONFIDENCE = 0.3

# Required keypoints for classification
REQUIRED_KEYPOINTS_STANDING = [
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]
REQUIRED_KEYPOINTS_SITTING = ["left_hip", "right_hip", "left_knee", "right_knee"]
REQUIRED_KEYPOINTS_SQUATTING = [
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]
REQUIRED_KEYPOINTS_CLOSEUP = ["left_shoulder", "right_shoulder", "nose"]


class PoseClassifier:
    """Classifies body poses from keypoint data using geometric analysis.

    This class provides pose classification capabilities using geometric
    relationships between body landmarks to determine pose categories.

    Supported pose categories:
    - standing: Upright posture with extended legs
    - sitting: Seated posture with bent knees
    - squatting: Crouched posture with deeply bent knees
    - closeup: Close-up shots (face/upper body only)

    Examples:
        Basic usage:
        >>> classifier = PoseClassifier()
        >>> classifier.classify_poses_in_frame(frame_data)

        Custom thresholds:
        >>> classifier.set_angle_thresholds(standing_min=165.0, sitting_max=115.0)
    """

    def __init__(self):
        """Initialize pose classifier with default thresholds."""
        self.standing_hip_knee_min = STANDING_HIP_KNEE_MIN
        self.sitting_hip_knee_min = SITTING_HIP_KNEE_MIN
        self.sitting_hip_knee_max = SITTING_HIP_KNEE_MAX
        self.squatting_hip_knee_max = SQUATTING_HIP_KNEE_MAX
        self.torso_vertical_threshold = TORSO_VERTICAL_THRESHOLD
        self.closeup_face_ratio_threshold = CLOSEUP_FACE_RATIO_THRESHOLD
        self.closeup_shoulder_width_threshold = CLOSEUP_SHOULDER_WIDTH_THRESHOLD
        self.min_keypoint_confidence = MIN_KEYPOINT_CONFIDENCE
        self.min_pose_confidence = MIN_POSE_CONFIDENCE

        logger.info("Initialized PoseClassifier with default thresholds")

    def classify_poses_in_frame(self, frame: FrameData) -> None:
        """Classifies all pose detections in a given frame and updates them in place.

        Args:
            frame: The FrameData object containing pose_detections and image_properties.
        """
        image_shape = (frame.image_properties.height, frame.image_properties.width)
        for pose_detection in frame.pose_detections:
            try:
                classifications = self._classify_single_pose(
                    pose_detection, image_shape
                )
                pose_detection.pose_classifications = classifications
            except PoseClassificationError:
                # Assign empty list for failed classifications
                pose_detection.pose_classifications = []

    def _classify_single_pose(
        self, pose_detection: PoseDetection, image_shape: Tuple[int, int]
    ) -> List[Tuple[str, float]]:
        """Classify a single pose detection, allowing for multiple classifications.

        Args:
            pose_detection: Pose detection with keypoints and bounding box
            image_shape: Image dimensions (height, width) for closeup detection

        Returns:
            A list of (pose_classification, confidence_score) tuples.

        Raises:
            PoseClassificationError: If classification fails due to an unexpected error.
        """
        classifications = []
        try:
            keypoints = pose_detection.keypoints
            bbox = pose_detection.bbox

            # A frame can be both a closeup and another pose, so check separately.
            if self._is_closeup(keypoints, bbox, image_shape):
                classifications.append(
                    (
                        "closeup",
                        self._calculate_closeup_confidence(
                            keypoints, bbox, image_shape
                        ),
                    )
                )

            if self._is_standing(keypoints):
                classifications.append(
                    ("standing", self._calculate_standing_confidence(keypoints))
                )

            if self._is_sitting(keypoints):
                classifications.append(
                    ("sitting", self._calculate_sitting_confidence(keypoints))
                )

            if self._is_squatting(keypoints):
                classifications.append(
                    ("squatting", self._calculate_squatting_confidence(keypoints))
                )

            return classifications

        except Exception as e:
            logger.error(f"Failed to classify pose: {e}", exc_info=True)
            raise PoseClassificationError(f"Failed to classify pose: {str(e)}") from e

    def _is_closeup(
        self,
        keypoints: Dict[str, Tuple[float, float, float]],
        bbox: Tuple[int, int, int, int],
        image_shape: Tuple[int, int],
    ) -> bool:
        """Check if pose represents a closeup shot.

        Args:
            keypoints: Pose keypoints
            bbox: Bounding box coordinates
            image_shape: Image dimensions (height, width)

        Returns:
            True if this is a closeup shot
        """
        height, width = image_shape
        x1, y1, x2, y2 = bbox

        # Check if required keypoints are available
        if not self._has_required_keypoints(keypoints, REQUIRED_KEYPOINTS_CLOSEUP):
            return False

        # Method 1: Check shoulder width ratio
        left_shoulder = keypoints.get("left_shoulder")
        right_shoulder = keypoints.get("right_shoulder")

        if (
            left_shoulder
            and right_shoulder
            and left_shoulder[2] >= self.min_keypoint_confidence
            and right_shoulder[2] >= self.min_keypoint_confidence
        ):

            shoulder_width = abs(right_shoulder[0] - left_shoulder[0])
            shoulder_width_ratio = shoulder_width / width

            if shoulder_width_ratio >= self.closeup_shoulder_width_threshold:
                return True

        # Method 2: Check face/head area ratio
        nose = keypoints.get("nose")
        if nose and nose[2] >= self.min_keypoint_confidence:
            # Estimate face area based on bounding box
            bbox_area = (x2 - x1) * (y2 - y1)
            total_area = height * width
            bbox_ratio = bbox_area / total_area

            if bbox_ratio >= self.closeup_face_ratio_threshold:
                return True

        # Method 3: Check if lower body keypoints are missing
        lower_body_keypoints = [
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
            "left_ankle",
            "right_ankle",
        ]
        lower_body_count = sum(
            1
            for kp_name in lower_body_keypoints
            if kp_name in keypoints
            and keypoints[kp_name][2] >= self.min_keypoint_confidence
        )

        if lower_body_count < 3:  # Less than half of lower body keypoints visible
            return True

        return False

    def _is_standing(self, keypoints: Dict[str, Tuple[float, float, float]]) -> bool:
        """Check if pose represents standing posture.

        Args:
            keypoints: Pose keypoints

        Returns:
            True if pose is standing
        """
        # Check for minimum required keypoints (hips and knees are essential)
        essential_keypoints = ["left_hip", "right_hip", "left_knee", "right_knee"]
        if not self._has_required_keypoints(keypoints, essential_keypoints):
            return False

        # Try to calculate hip-knee angles (ankles may not be visible)
        left_hip_knee_angle = self._calculate_hip_knee_angle_flexible(keypoints, "left")
        right_hip_knee_angle = self._calculate_hip_knee_angle_flexible(
            keypoints, "right"
        )

        # Check if legs appear extended (standing)
        left_standing = (
            left_hip_knee_angle is not None
            and left_hip_knee_angle >= self.standing_hip_knee_min
        )
        right_standing = (
            right_hip_knee_angle is not None
            and right_hip_knee_angle >= self.standing_hip_knee_min
        )

        # At least one leg should be standing
        if left_standing or right_standing:
            # Check torso alignment (should be relatively vertical)
            torso_angle = self._calculate_torso_angle(keypoints)
            if (
                torso_angle is not None
                and abs(torso_angle) <= self.torso_vertical_threshold
            ):
                return True

        # Fallback: If ankles aren't visible but hips and knees suggest standing posture
        if left_hip_knee_angle is None and right_hip_knee_angle is None:
            # Check if hips are above knees (typical standing posture)
            if self._hips_above_knees(keypoints):
                torso_angle = self._calculate_torso_angle(keypoints)
                if (
                    torso_angle is not None
                    and abs(torso_angle) <= self.torso_vertical_threshold
                ):
                    return True

        return False

    def _is_sitting(self, keypoints: Dict[str, Tuple[float, float, float]]) -> bool:
        """Check if pose represents sitting posture.

        Args:
            keypoints: Pose keypoints

        Returns:
            True if pose is sitting
        """
        if not self._has_required_keypoints(keypoints, REQUIRED_KEYPOINTS_SITTING):
            return False

        # Calculate hip-knee angles for both legs
        left_hip_knee_angle = self._calculate_hip_knee_angle(keypoints, "left")
        right_hip_knee_angle = self._calculate_hip_knee_angle(keypoints, "right")

        # Check if both legs are in sitting position
        left_sitting = (
            left_hip_knee_angle is not None
            and self.sitting_hip_knee_min
            <= left_hip_knee_angle
            <= self.sitting_hip_knee_max
        )
        right_sitting = (
            right_hip_knee_angle is not None
            and self.sitting_hip_knee_min
            <= right_hip_knee_angle
            <= self.sitting_hip_knee_max
        )

        # At least one leg should be in sitting position
        return left_sitting or right_sitting

    def _is_squatting(self, keypoints: Dict[str, Tuple[float, float, float]]) -> bool:
        """Check if pose represents squatting posture.

        Args:
            keypoints: Pose keypoints

        Returns:
            True if pose is squatting
        """
        if not self._has_required_keypoints(keypoints, REQUIRED_KEYPOINTS_SQUATTING):
            return False

        # Calculate hip-knee angles for both legs
        left_hip_knee_angle = self._calculate_hip_knee_angle(keypoints, "left")
        right_hip_knee_angle = self._calculate_hip_knee_angle(keypoints, "right")

        # Check if both legs are deeply bent (squatting)
        left_squatting = (
            left_hip_knee_angle is not None
            and left_hip_knee_angle <= self.squatting_hip_knee_max
        )
        right_squatting = (
            right_hip_knee_angle is not None
            and right_hip_knee_angle <= self.squatting_hip_knee_max
        )

        # At least one leg should be squatting
        if left_squatting or right_squatting:
            # Check torso alignment (should be relatively vertical for squatting)
            torso_angle = self._calculate_torso_angle(keypoints)
            if (
                torso_angle is not None
                and abs(torso_angle) <= self.torso_vertical_threshold
            ):
                return True

        return False

    def _has_required_keypoints(
        self, keypoints: Dict[str, Tuple[float, float, float]], required: List[str]
    ) -> bool:
        """Check if required keypoints are available with sufficient confidence.

        Args:
            keypoints: Keypoint dictionary
            required: List of required keypoint names

        Returns:
            True if all required keypoints are available
        """
        for kp_name in required:
            if (
                kp_name not in keypoints
                or keypoints[kp_name][2] < self.min_keypoint_confidence
            ):
                return False
        return True

    def _calculate_hip_knee_angle(
        self, keypoints: Dict[str, Tuple[float, float, float]], side: str
    ) -> Optional[float]:
        """Calculate hip-knee angle for specified side.

        Args:
            keypoints: Pose keypoints
            side: 'left' or 'right'

        Returns:
            Hip-knee angle in degrees, or None if cannot calculate
        """
        hip_key = f"{side}_hip"
        knee_key = f"{side}_knee"
        ankle_key = f"{side}_ankle"

        if not all(key in keypoints for key in [hip_key, knee_key, ankle_key]):
            return None

        hip = keypoints[hip_key]
        knee = keypoints[knee_key]
        ankle = keypoints[ankle_key]

        # Check confidence
        if any(point[2] < self.min_keypoint_confidence for point in [hip, knee, ankle]):
            return None

        # Calculate vectors: hip->knee and knee->ankle
        vec1 = (knee[0] - hip[0], knee[1] - hip[1])
        vec2 = (ankle[0] - knee[0], ankle[1] - knee[1])

        # Calculate angle between vectors
        vector_angle = self._calculate_angle_between_vectors(vec1, vec2)

        # Convert to joint angle (exterior angle at knee)
        # Joint angle = 180° - vector_angle
        joint_angle = 180.0 - vector_angle

        return joint_angle

    def _calculate_hip_knee_angle_flexible(
        self, keypoints: Dict[str, Tuple[float, float, float]], side: str
    ) -> Optional[float]:
        """Calculate hip-knee angle with fallback when ankle is not visible.

        Args:
            keypoints: Pose keypoints
            side: 'left' or 'right'

        Returns:
            Hip-knee angle in degrees, or None if cannot calculate
        """
        hip_key = f"{side}_hip"
        knee_key = f"{side}_knee"
        ankle_key = f"{side}_ankle"

        if not all(key in keypoints for key in [hip_key, knee_key]):
            return None

        hip = keypoints[hip_key]
        knee = keypoints[knee_key]

        # Check confidence for hip and knee
        if any(point[2] < self.min_keypoint_confidence for point in [hip, knee]):
            return None

        # Try to use ankle if available
        if ankle_key in keypoints:
            ankle = keypoints[ankle_key]
            if ankle[2] >= self.min_keypoint_confidence:
                # Use normal calculation with ankle
                vec1 = (knee[0] - hip[0], knee[1] - hip[1])
                vec2 = (ankle[0] - knee[0], ankle[1] - knee[1])
                vector_angle = self._calculate_angle_between_vectors(vec1, vec2)
                return 180.0 - vector_angle

        # Fallback: Estimate angle based on hip-knee vector relative to vertical
        # If knee is directly below hip, assume standing (180°)
        # If knee is significantly forward/back, assume bent
        hip_knee_vec = (knee[0] - hip[0], knee[1] - hip[1])
        vertical_vec = (0, 1)  # Downward vertical

        angle_from_vertical = self._calculate_angle_between_vectors(
            hip_knee_vec, vertical_vec
        )

        # Convert to approximate joint angle
        # If leg is nearly vertical (small angle), assume extended (standing)
        # If leg is angled significantly, assume bent
        if angle_from_vertical <= 20:  # Nearly vertical
            return 170.0  # Assume standing
        elif angle_from_vertical <= 45:  # Moderately angled
            return 150.0  # Slightly bent
        else:  # Highly angled
            return 120.0  # More bent

    def _hips_above_knees(
        self, keypoints: Dict[str, Tuple[float, float, float]]
    ) -> bool:
        """Check if hips are positioned above knees (typical standing posture).

        Args:
            keypoints: Pose keypoints

        Returns:
            True if hips are above knees
        """
        left_hip = keypoints.get("left_hip")
        right_hip = keypoints.get("right_hip")
        left_knee = keypoints.get("left_knee")
        right_knee = keypoints.get("right_knee")

        # Check if we have at least one hip-knee pair
        valid_pairs = []

        if (
            left_hip
            and left_knee
            and left_hip[2] >= self.min_keypoint_confidence
            and left_knee[2] >= self.min_keypoint_confidence
        ):
            valid_pairs.append((left_hip[1], left_knee[1]))  # y-coordinates

        if (
            right_hip
            and right_knee
            and right_hip[2] >= self.min_keypoint_confidence
            and right_knee[2] >= self.min_keypoint_confidence
        ):
            valid_pairs.append((right_hip[1], right_knee[1]))  # y-coordinates

        if not valid_pairs:
            return False

        # Check if hips are above knees (smaller y-coordinate = higher in image)
        for hip_y, knee_y in valid_pairs:
            if hip_y >= knee_y:  # Hip is below or at same level as knee
                return False

        return True

    def _calculate_torso_angle(
        self, keypoints: Dict[str, Tuple[float, float, float]]
    ) -> Optional[float]:
        """Calculate torso angle relative to vertical.

        Args:
            keypoints: Pose keypoints

        Returns:
            Torso angle in degrees from vertical, or None if cannot calculate
        """
        # Use shoulder midpoint to hip midpoint as torso vector
        left_shoulder = keypoints.get("left_shoulder")
        right_shoulder = keypoints.get("right_shoulder")
        left_hip = keypoints.get("left_hip")
        right_hip = keypoints.get("right_hip")

        if not all(
            kp and kp[2] >= self.min_keypoint_confidence
            for kp in [left_shoulder, right_shoulder, left_hip, right_hip]
        ):
            return None

        # Calculate midpoints
        shoulder_mid = (
            (left_shoulder[0] + right_shoulder[0]) / 2,
            (left_shoulder[1] + right_shoulder[1]) / 2,
        )
        hip_mid = ((left_hip[0] + right_hip[0]) / 2, (left_hip[1] + right_hip[1]) / 2)

        # Calculate torso vector (from hip to shoulder)
        torso_vec = (shoulder_mid[0] - hip_mid[0], shoulder_mid[1] - hip_mid[1])

        # Calculate angle from vertical (0, -1) vector
        vertical_vec = (0, -1)
        angle = self._calculate_angle_between_vectors(torso_vec, vertical_vec)

        return angle

    def _calculate_angle_between_vectors(
        self, vec1: Tuple[float, float], vec2: Tuple[float, float]
    ) -> float:
        """Calculate angle between two 2D vectors.

        Args:
            vec1: First vector (x, y)
            vec2: Second vector (x, y)

        Returns:
            Angle in degrees
        """
        # Calculate dot product
        dot_product = vec1[0] * vec2[0] + vec1[1] * vec2[1]

        # Calculate magnitudes
        mag1 = math.sqrt(vec1[0] ** 2 + vec1[1] ** 2)
        mag2 = math.sqrt(vec2[0] ** 2 + vec2[1] ** 2)

        if mag1 == 0 or mag2 == 0:
            return 0.0

        # Calculate angle
        cos_angle = dot_product / (mag1 * mag2)
        cos_angle = max(-1.0, min(1.0, cos_angle))  # Clamp to valid range

        angle_rad = math.acos(cos_angle)
        angle_deg = math.degrees(angle_rad)

        return angle_deg

    def _calculate_closeup_confidence(
        self,
        keypoints: Dict[str, Tuple[float, float, float]],
        bbox: Tuple[int, int, int, int],
        image_shape: Tuple[int, int],
    ) -> float:
        """Calculate confidence score for closeup classification.

        Args:
            keypoints: Pose keypoints
            bbox: Bounding box coordinates
            image_shape: Image dimensions

        Returns:
            Confidence score (0.0 to 1.0)
        """
        confidence_factors = []

        # Factor 1: Shoulder width ratio
        left_shoulder = keypoints.get("left_shoulder")
        right_shoulder = keypoints.get("right_shoulder")

        if (
            left_shoulder
            and right_shoulder
            and left_shoulder[2] >= self.min_keypoint_confidence
            and right_shoulder[2] >= self.min_keypoint_confidence
        ):

            shoulder_width = abs(right_shoulder[0] - left_shoulder[0])
            shoulder_width_ratio = shoulder_width / image_shape[1]
            shoulder_confidence = min(
                1.0, shoulder_width_ratio / self.closeup_shoulder_width_threshold
            )
            confidence_factors.append(shoulder_confidence)

        # Factor 2: Bounding box area ratio
        bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        total_area = image_shape[0] * image_shape[1]
        bbox_ratio = bbox_area / total_area
        bbox_confidence = min(1.0, bbox_ratio / self.closeup_face_ratio_threshold)
        confidence_factors.append(bbox_confidence)

        # Factor 3: Missing lower body keypoints
        lower_body_keypoints = [
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
            "left_ankle",
            "right_ankle",
        ]
        missing_count = sum(
            1
            for kp_name in lower_body_keypoints
            if kp_name not in keypoints
            or keypoints[kp_name][2] < self.min_keypoint_confidence
        )
        missing_confidence = missing_count / len(lower_body_keypoints)
        confidence_factors.append(missing_confidence)

        return max(0.3, np.mean(confidence_factors))

    def _calculate_standing_confidence(
        self, keypoints: Dict[str, Tuple[float, float, float]]
    ) -> float:
        """Calculate confidence score for standing classification."""
        confidence_factors = []

        # Factor 1: Hip-knee angles
        for side in ["left", "right"]:
            angle = self._calculate_hip_knee_angle(keypoints, side)
            if angle is not None:
                if angle >= self.standing_hip_knee_min:
                    angle_confidence = min(
                        1.0, (angle - self.standing_hip_knee_min) / 20.0 + 0.7
                    )
                    confidence_factors.append(angle_confidence)

        # Factor 2: Torso alignment
        torso_angle = self._calculate_torso_angle(keypoints)
        if torso_angle is not None:
            torso_confidence = max(0.3, 1.0 - abs(torso_angle) / 45.0)
            confidence_factors.append(torso_confidence)

        return max(0.3, np.mean(confidence_factors)) if confidence_factors else 0.3

    def _calculate_sitting_confidence(
        self, keypoints: Dict[str, Tuple[float, float, float]]
    ) -> float:
        """Calculate confidence score for sitting classification."""
        confidence_factors = []

        # Factor 1: Hip-knee angles in sitting range
        for side in ["left", "right"]:
            angle = self._calculate_hip_knee_angle(keypoints, side)
            if angle is not None:
                if self.sitting_hip_knee_min <= angle <= self.sitting_hip_knee_max:
                    # Higher confidence for angles closer to 90 degrees
                    optimal_angle = 90.0
                    angle_confidence = max(0.5, 1.0 - abs(angle - optimal_angle) / 30.0)
                    confidence_factors.append(angle_confidence)

        return max(0.4, np.mean(confidence_factors)) if confidence_factors else 0.4

    def _calculate_squatting_confidence(
        self, keypoints: Dict[str, Tuple[float, float, float]]
    ) -> float:
        """Calculate confidence score for squatting classification."""
        confidence_factors = []

        # Factor 1: Hip-knee angles in squatting range
        for side in ["left", "right"]:
            angle = self._calculate_hip_knee_angle(keypoints, side)
            if angle is not None:
                if angle <= self.squatting_hip_knee_max:
                    # Higher confidence for smaller angles
                    angle_confidence = max(
                        0.5, 1.0 - angle / self.squatting_hip_knee_max
                    )
                    confidence_factors.append(angle_confidence)

        # Factor 2: Torso alignment
        torso_angle = self._calculate_torso_angle(keypoints)
        if torso_angle is not None:
            torso_confidence = max(0.3, 1.0 - abs(torso_angle) / 45.0)
            confidence_factors.append(torso_confidence)

        return max(0.4, np.mean(confidence_factors)) if confidence_factors else 0.4

    def set_angle_thresholds(
        self,
        standing_min: Optional[float] = None,
        sitting_min: Optional[float] = None,
        sitting_max: Optional[float] = None,
        squatting_max: Optional[float] = None,
    ) -> None:
        """Set custom angle thresholds for pose classification.

        Args:
            standing_min: Minimum hip-knee angle for standing
            sitting_min: Minimum hip-knee angle for sitting
            sitting_max: Maximum hip-knee angle for sitting
            squatting_max: Maximum hip-knee angle for squatting
        """
        if standing_min is not None:
            self.standing_hip_knee_min = standing_min
        if sitting_min is not None:
            self.sitting_hip_knee_min = sitting_min
        if sitting_max is not None:
            self.sitting_hip_knee_max = sitting_max
        if squatting_max is not None:
            self.squatting_hip_knee_max = squatting_max

        logger.info(
            f"Updated angle thresholds: standing_min={self.standing_hip_knee_min}, "
            f"sitting_range=[{self.sitting_hip_knee_min}, {self.sitting_hip_knee_max}], "
            f"squatting_max={self.squatting_hip_knee_max}"
        )

    def get_classification_info(self) -> Dict[str, Any]:
        """Get information about the current classification settings.

        Returns:
            Dictionary containing thresholds and configuration
        """
        return {
            "angle_thresholds": {
                "standing_hip_knee_min": self.standing_hip_knee_min,
                "sitting_hip_knee_min": self.sitting_hip_knee_min,
                "sitting_hip_knee_max": self.sitting_hip_knee_max,
                "squatting_hip_knee_max": self.squatting_hip_knee_max,
                "torso_vertical_threshold": self.torso_vertical_threshold,
            },
            "closeup_thresholds": {
                "face_ratio_threshold": self.closeup_face_ratio_threshold,
                "shoulder_width_threshold": self.closeup_shoulder_width_threshold,
            },
            "confidence_thresholds": {
                "min_keypoint_confidence": self.min_keypoint_confidence,
                "min_pose_confidence": self.min_pose_confidence,
            },
        }

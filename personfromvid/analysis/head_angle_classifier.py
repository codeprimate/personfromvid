"""Head angle classification logic.

This module implements head pose angle classification into 9 cardinal directions
based on yaw, pitch, and roll angles from head pose estimation models.
"""

import logging
from typing import Dict, List, Tuple, Any, Optional
import warnings

from ..data.detection_results import HeadPoseResult
from ..data.frame_data import FrameData
from ..utils.exceptions import HeadAngleClassificationError

logger = logging.getLogger(__name__)

# Default angle thresholds for direction classification (in degrees)
DEFAULT_YAW_THRESHOLD = 22.5  # ±22.5° for front/looking directions
DEFAULT_PITCH_THRESHOLD = 22.5  # ±22.5° for up/down directions
DEFAULT_PROFILE_YAW_THRESHOLD = 67.5  # ±67.5° for profile directions
DEFAULT_MAX_ROLL = 30.0  # Max acceptable roll angle

# 9 cardinal directions mapping
CARDINAL_DIRECTIONS = [
    "front",
    "looking_left",
    "looking_right",
    "profile_left",
    "profile_right",
    "looking_up",
    "looking_down",
    "looking_up_left",
    "looking_up_right",
]

# Direction confidence weights
CONFIDENCE_WEIGHTS = {
    "front": 1.0,
    "looking_left": 0.9,
    "looking_right": 0.9,
    "profile_left": 0.8,
    "profile_right": 0.8,
    "looking_up": 0.85,
    "looking_down": 0.85,
    "looking_up_left": 0.75,
    "looking_up_right": 0.75,
}


class HeadAngleClassifier:
    """Classifies head pose angles into 9 cardinal directions.

    This class provides head angle classification capabilities using yaw, pitch,
    and roll angles from head pose estimation to determine head orientation.

    Supported directions (9 cardinal directions):
    - front: Looking straight ahead
    - looking_left/right: Slight turn left/right
    - profile_left/right: Full profile view
    - looking_up/down: Vertical head movement
    - looking_up_left/up_right: Diagonal combinations

    Examples:
        Basic usage:
        >>> classifier = HeadAngleClassifier()
        >>> direction = classifier.classify_head_angle(yaw=15.0, pitch=-5.0, roll=2.0)

        Batch classification:
        >>> directions = classifier.classify_head_poses(head_pose_results)

        Custom thresholds:
        >>> classifier.set_angle_thresholds(yaw=25.0, pitch=25.0)
    """

    def __init__(self):
        """Initialize head angle classifier with default thresholds."""
        self.yaw_threshold = DEFAULT_YAW_THRESHOLD
        self.pitch_threshold = DEFAULT_PITCH_THRESHOLD
        self.profile_yaw_threshold = DEFAULT_PROFILE_YAW_THRESHOLD
        self.max_roll = DEFAULT_MAX_ROLL

        logger.info("Initialized HeadAngleClassifier with default thresholds")

    def classify_head_poses_in_frame(self, frame: FrameData) -> None:
        """Classifies all head pose detections in a given frame and updates them in place.

        Args:
            frame: The FrameData object containing head_poses.
        """
        for head_pose_result in frame.head_poses:
            direction, confidence = self._classify_single_head_pose(head_pose_result)
            head_pose_result.direction = direction
            head_pose_result.direction_confidence = confidence

    def classify_head_angle(self, yaw: float, pitch: float, roll: float) -> str:
        """Classify head pose angles into cardinal directions.

        Args:
            yaw: Yaw angle in degrees
            pitch: Pitch angle in degrees  
            roll: Roll angle in degrees

        Returns:
            Direction string from the 9 cardinal directions
        """
        try:
            # Validate input angles
            self._validate_angles(yaw, pitch, roll)

            # Determine direction based on yaw and pitch
            direction = self._get_direction_from_angles(yaw, pitch)
            
            return direction

        except Exception as e:
            logger.error(f"Failed to classify head angles yaw={yaw}, pitch={pitch}, roll={roll}: {e}")
            # Return default classification for failed angles
            return "front"

    def _classify_single_head_pose(
        self, head_pose_result: HeadPoseResult
    ) -> Tuple[str, float]:
        """Classify a single head pose result with confidence scoring.

        This is the core classification logic for a single detection.

        Args:
            head_pose_result: Head pose estimation result containing yaw, pitch, roll.

        Returns:
            Tuple of (direction, adjusted_confidence)
        """
        try:
            # Validate input angles
            self._validate_angles(
                head_pose_result.yaw, head_pose_result.pitch, head_pose_result.roll
            )

            # Check if roll is within acceptable range
            if not self.is_valid_orientation(head_pose_result.roll):
                # Still classify but could be marked as low quality
                logger.debug(
                    f"High roll angle detected: {head_pose_result.roll}°, classification may be less reliable"
                )

            # Determine direction based on yaw and pitch
            direction = self._get_direction_from_angles(
                head_pose_result.yaw, head_pose_result.pitch
            )

            # Calculate adjusted confidence based on direction and pose quality
            adjusted_confidence = self._calculate_direction_confidence(
                direction,
                head_pose_result.yaw,
                head_pose_result.pitch,
                head_pose_result.roll,
                head_pose_result.confidence,
            )

            return direction, adjusted_confidence

        except Exception as e:
            logger.error(
                f"Failed to classify head pose for face_id={head_pose_result.face_id}: {e}"
            )
            # Return default classification for failed poses
            return "front", 0.1

    def _get_direction_from_angles(self, yaw: float, pitch: float) -> str:
        """Determine cardinal direction from yaw and pitch angles."""
        if abs(yaw) <= self.yaw_threshold:
            # Frontal directions
            if abs(pitch) <= self.pitch_threshold:
                return "front"
            elif pitch > self.pitch_threshold:
                return "looking_up"
            else:  # pitch < -self.pitch_threshold
                return "looking_down"

        elif abs(yaw) >= self.profile_yaw_threshold:
            # Profile directions
            if yaw > 0:
                return "profile_left"
            else:
                return "profile_right"

        else:
            # Intermediate directions
            if yaw > 0:  # Looking left
                if abs(pitch) <= self.pitch_threshold:
                    return "looking_left"
                elif pitch > self.pitch_threshold:
                    return "looking_up_left"
                else:  # Inverted from original logic to be correct
                    return "looking_down_left"
            else:  # Looking right
                if abs(pitch) <= self.pitch_threshold:
                    return "looking_right"
                elif pitch > self.pitch_threshold:
                    return "looking_up_right"
                else:  # Inverted from original logic to be correct
                    return "looking_down_right"

    def is_valid_orientation(self, roll: float) -> bool:
        """Check if head orientation is valid (not excessively tilted).

        Args:
            roll: Roll angle in degrees

        Returns:
            True if orientation is valid
        """
        return abs(roll) <= self.max_roll

    def get_angle_ranges(self) -> Dict[str, Dict[str, Tuple[float, float]]]:
        """Get angle ranges for each direction classification.

        Returns:
            Dictionary mapping directions to their angle ranges
        """
        return {
            "front": {
                "yaw": (-self.yaw_threshold, self.yaw_threshold),
                "pitch": (-self.pitch_threshold, self.pitch_threshold),
            },
            "looking_left": {
                "yaw": (self.yaw_threshold, self.profile_yaw_threshold),
                "pitch": (-self.pitch_threshold, self.pitch_threshold),
            },
            "looking_right": {
                "yaw": (-self.profile_yaw_threshold, -self.yaw_threshold),
                "pitch": (-self.pitch_threshold, self.pitch_threshold),
            },
            "profile_left": {
                "yaw": (self.profile_yaw_threshold, 180.0),
                "pitch": (-90.0, 90.0),
            },
            "profile_right": {
                "yaw": (-180.0, -self.profile_yaw_threshold),
                "pitch": (-90.0, 90.0),
            },
            "looking_up": {
                "yaw": (-self.yaw_threshold, self.yaw_threshold),
                "pitch": (self.pitch_threshold, 90.0),
            },
            "looking_down": {
                "yaw": (-self.yaw_threshold, self.yaw_threshold),
                "pitch": (-90.0, -self.pitch_threshold),
            },
            "looking_up_left": {
                "yaw": (self.yaw_threshold, self.profile_yaw_threshold),
                "pitch": (self.pitch_threshold, 90.0),
            },
            "looking_up_right": {
                "yaw": (-self.profile_yaw_threshold, -self.yaw_threshold),
                "pitch": (self.pitch_threshold, 90.0),
            },
        }

    def _validate_angles(self, yaw: float, pitch: float, roll: float) -> None:
        """Validate angle ranges and warn about extreme values.

        Args:
            yaw: Yaw angle in degrees
            pitch: Pitch angle in degrees
            roll: Roll angle in degrees

        Raises:
            HeadAngleClassificationError: If angles are invalid
        """
        # Check for NaN or infinite values
        if not all(
            isinstance(angle, (int, float)) and not (angle != angle)
            for angle in [yaw, pitch, roll]
        ):
            raise HeadAngleClassificationError(
                f"Invalid angle values: yaw={yaw}, pitch={pitch}, roll={roll}"
            )

        # Warn about extreme values but don't fail
        if abs(yaw) > 180 or abs(pitch) > 90 or abs(roll) > 180:
            warnings.warn(
                f"Extreme angle values detected: yaw={yaw}, pitch={pitch}, roll={roll}"
            )

    def _calculate_direction_confidence(
        self,
        direction: str,
        yaw: float,
        pitch: float,
        roll: float,
        base_confidence: float,
    ) -> float:
        """Calculate adjusted confidence for direction classification.

        Args:
            direction: Classified direction
            yaw: Yaw angle in degrees
            pitch: Pitch angle in degrees
            roll: Roll angle in degrees
            base_confidence: Original head pose confidence

        Returns:
            Adjusted confidence score (0.0 to 1.0)
        """
        # Start with base confidence
        confidence = base_confidence

        # Apply direction-specific weight
        direction_weight = CONFIDENCE_WEIGHTS.get(direction, 0.5)
        confidence *= direction_weight

        # Penalize high roll angles
        if abs(roll) > self.max_roll:
            roll_penalty = (
                1.0 - (abs(roll) - self.max_roll) / 60.0
            )  # Reduce confidence for high roll
            confidence *= max(0.1, roll_penalty)

        # Boost confidence for angles close to direction centers
        angle_confidence = self._calculate_angle_centeredness(direction, yaw, pitch)
        confidence = (confidence + angle_confidence) / 2.0

        return max(0.1, min(1.0, confidence))

    def _calculate_angle_centeredness(
        self, direction: str, yaw: float, pitch: float
    ) -> float:
        """Calculate how close angles are to the center of their direction range.

        Args:
            direction: Direction classification
            yaw: Yaw angle in degrees
            pitch: Pitch angle in degrees

        Returns:
            Centeredness score (0.0 to 1.0)
        """
        # Define ideal center angles for each direction
        direction_centers = {
            "front": (0.0, 0.0),
            "looking_left": (45.0, 0.0),
            "looking_right": (-45.0, 0.0),
            "profile_left": (90.0, 0.0),
            "profile_right": (-90.0, 0.0),
            "looking_up": (0.0, 30.0),
            "looking_down": (0.0, -30.0),
            "looking_up_left": (45.0, 30.0),
            "looking_up_right": (-45.0, 30.0),
        }

        if direction not in direction_centers:
            return 0.5

        center_yaw, center_pitch = direction_centers[direction]

        # Calculate distance from center
        yaw_distance = abs(yaw - center_yaw)
        pitch_distance = abs(pitch - center_pitch)

        # Convert to confidence score (closer to center = higher confidence)
        yaw_conf = max(0.0, 1.0 - yaw_distance / 90.0)
        pitch_conf = max(0.0, 1.0 - pitch_distance / 45.0)

        return (yaw_conf + pitch_conf) / 2.0

    def set_angle_thresholds(
        self,
        yaw: Optional[float] = None,
        pitch: Optional[float] = None,
        profile_yaw: Optional[float] = None,
        max_roll: Optional[float] = None,
    ) -> None:
        """Set custom angle thresholds for direction classification.

        Args:
            yaw: Yaw threshold for front/looking directions
            pitch: Pitch threshold for up/down directions
            profile_yaw: Yaw threshold for profile directions
            max_roll: Maximum acceptable roll angle
        """
        if yaw is not None:
            self.yaw_threshold = yaw
        if pitch is not None:
            self.pitch_threshold = pitch
        if profile_yaw is not None:
            self.profile_yaw_threshold = profile_yaw
        if max_roll is not None:
            self.max_roll = max_roll

        logger.info(
            f"Updated angle thresholds: yaw={self.yaw_threshold}°, "
            f"pitch={self.pitch_threshold}°, profile_yaw={self.profile_yaw_threshold}°, "
            f"max_roll={self.max_roll}°"
        )

    def get_classification_info(self) -> Dict[str, Any]:
        """Get information about the current classification settings.

        Returns:
            Dictionary containing thresholds and configuration
        """
        return {
            "angle_thresholds": {
                "yaw_threshold": self.yaw_threshold,
                "pitch_threshold": self.pitch_threshold,
                "profile_yaw_threshold": self.profile_yaw_threshold,
                "max_roll": self.max_roll,
            },
            "supported_directions": CARDINAL_DIRECTIONS.copy(),
            "confidence_weights": CONFIDENCE_WEIGHTS.copy(),
            "angle_ranges": self.get_angle_ranges(),
        }

    def validate_direction(self, direction: str) -> bool:
        """Validate if a direction string is supported.

        Args:
            direction: Direction string to validate

        Returns:
            True if direction is supported
        """
        return direction in CARDINAL_DIRECTIONS

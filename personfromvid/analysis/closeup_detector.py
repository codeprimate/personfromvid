"""Close-up shot detection and frame composition analysis.

This module provides comprehensive closeup detection capabilities including:
- Shot type classification (extreme closeup, closeup, medium closeup, etc.)
- Distance estimation using facial landmarks and geometry
- Frame composition assessment using rule of thirds and positioning
- Face size ratio analysis for shot classification
"""

import logging
import math
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from ..data.detection_results import FaceDetection, PoseDetection, CloseupDetection
from ..data.frame_data import FrameData, ImageProperties
from ..utils.exceptions import AnalysisError

logger = logging.getLogger(__name__)

# Shot classification thresholds (face area ratio relative to frame)
EXTREME_CLOSEUP_THRESHOLD = 0.25  # Face takes up >25% of frame
CLOSEUP_THRESHOLD = 0.15  # Face takes up >15% of frame
MEDIUM_CLOSEUP_THRESHOLD = 0.08  # Face takes up >8% of frame
MEDIUM_SHOT_THRESHOLD = 0.03  # Face takes up >3% of frame

# Distance estimation thresholds (inter-ocular distance in pixels)
VERY_CLOSE_IOD_THRESHOLD = 80  # >80 pixels between eyes
CLOSE_IOD_THRESHOLD = 50  # >50 pixels between eyes
MEDIUM_IOD_THRESHOLD = 25  # >25 pixels between eyes

# Composition assessment constants
RULE_OF_THIRDS_TOLERANCE = 0.1  # ±10% tolerance for rule of thirds
IDEAL_FACE_HEIGHT_RATIO = (
    0.4  # Face should be ~40% of frame height for good composition
)
SHOULDER_WIDTH_CLOSEUP_THRESHOLD = 0.35  # Shoulder width ratio for closeup detection

# Confidence thresholds
MIN_FACE_CONFIDENCE = 0.3
MIN_LANDMARK_CONFIDENCE = 0.5


class CloseupDetectionError(AnalysisError):
    """Raised when closeup detection fails."""

    pass


class CloseupDetector:
    """Comprehensive closeup detection and frame composition analysis.

    This class provides advanced closeup detection capabilities including:
    - Multi-criteria shot classification
    - Distance estimation using facial geometry
    - Frame composition assessment
    - Portrait suitability scoring

    Examples:
        Basic usage with FrameData:
        >>> detector = CloseupDetector()
        >>> detector.detect_closeups_in_frame(frame_data)

        Batch processing:
        >>> detector.process_frame_batch(frames_with_faces)
    """

    def __init__(
        self,
        extreme_closeup_threshold: float = EXTREME_CLOSEUP_THRESHOLD,
        closeup_threshold: float = CLOSEUP_THRESHOLD,
        medium_closeup_threshold: float = MEDIUM_CLOSEUP_THRESHOLD,
        medium_shot_threshold: float = MEDIUM_SHOT_THRESHOLD,
    ):
        """Initialize closeup detector with configurable thresholds.

        Args:
            extreme_closeup_threshold: Face area ratio for extreme closeup
            closeup_threshold: Face area ratio for closeup
            medium_closeup_threshold: Face area ratio for medium closeup
            medium_shot_threshold: Face area ratio for medium shot
        """
        self.extreme_closeup_threshold = extreme_closeup_threshold
        self.closeup_threshold = closeup_threshold
        self.medium_closeup_threshold = medium_closeup_threshold
        self.medium_shot_threshold = medium_shot_threshold

        logger.info(
            f"Initialized CloseupDetector with thresholds: "
            f"extreme={extreme_closeup_threshold}, closeup={closeup_threshold}, "
            f"medium_closeup={medium_closeup_threshold}, medium_shot={medium_shot_threshold}"
        )

    def detect_closeups_in_frame(self, frame: FrameData) -> None:
        """Detect closeup characteristics for all faces in a frame and update the frame in place.

        This is the primary method that operates on FrameData objects and follows the
        standardized pattern of using FrameData as the unit of work.

        Args:
            frame: FrameData object containing face detections and image properties
        """
        if not frame.face_detections:
            return

        image_properties = frame.image_properties

        # Process each face detection
        for face_idx, face_detection in enumerate(frame.face_detections):
            try:
                # Check if we have corresponding pose data for enhanced detection
                pose_detection = None
                if frame.pose_detections and len(frame.pose_detections) > face_idx:
                    pose_detection = frame.pose_detections[face_idx]

                # Perform closeup detection
                if pose_detection:
                    closeup_result = self._detect_closeup_with_pose_data(
                        face_detection, pose_detection, image_properties
                    )
                else:
                    closeup_result = self._detect_closeup_from_face(
                        face_detection, image_properties
                    )

                frame.closeup_detections.append(closeup_result)

            except Exception as e:
                logger.error(
                    f"Failed to detect closeup for face {face_idx} in frame {frame.frame_id}: {e}"
                )
                # Continue processing other faces

    def _detect_closeup_from_face(
        self, face_detection: FaceDetection, image_properties: ImageProperties
    ) -> CloseupDetection:
        """Detect closeup shot characteristics from face detection using data models.

        Args:
            face_detection: Face detection result with bbox and landmarks
            image_properties: Image properties containing dimensions and metadata

        Returns:
            CloseupDetection with comprehensive analysis results

        Raises:
            CloseupDetectionError: If detection fails
        """
        try:
            frame_area = image_properties.total_pixels

            # Calculate face area ratio
            face_area = face_detection.area
            face_area_ratio = face_area / frame_area

            # Classify shot type based on face area ratio
            shot_type = self._classify_shot_type(face_area_ratio)

            # Calculate inter-ocular distance if landmarks available
            inter_ocular_distance = None
            estimated_distance = None
            if face_detection.landmarks and len(face_detection.landmarks) >= 5:
                inter_ocular_distance = self._calculate_inter_ocular_distance(
                    face_detection.landmarks
                )
                estimated_distance = self._estimate_distance(inter_ocular_distance)

            # Assess frame composition
            composition_score, composition_notes, face_position = (
                self._assess_composition_with_properties(
                    face_detection, image_properties
                )
            )

            # Determine if this is a closeup
            is_closeup = shot_type in ["extreme_closeup", "closeup", "medium_closeup"]

            # Calculate confidence based on multiple factors
            confidence = self._calculate_detection_confidence(
                face_detection,
                face_area_ratio,
                inter_ocular_distance,
                composition_score,
            )

            return CloseupDetection(
                is_closeup=is_closeup,
                shot_type=shot_type,
                confidence=confidence,
                face_area_ratio=face_area_ratio,
                inter_ocular_distance=inter_ocular_distance,
                estimated_distance=estimated_distance,
                composition_score=composition_score,
                composition_notes=composition_notes,
                face_position=face_position,
            )

        except Exception as e:
            raise CloseupDetectionError(f"Failed to detect closeup: {str(e)}") from e

    def _detect_closeup_with_pose_data(
        self,
        face_detection: FaceDetection,
        pose_detection: PoseDetection,
        image_properties: ImageProperties,
    ) -> CloseupDetection:
        """Enhanced closeup detection using both face and pose data models.

        Args:
            face_detection: Face detection result
            pose_detection: Pose detection with keypoints
            image_properties: Image properties containing dimensions

        Returns:
            CloseupDetection with enhanced analysis including shoulder width
        """
        # Start with basic face-based detection
        result = self._detect_closeup_from_face(face_detection, image_properties)

        # Add shoulder width analysis from pose data
        shoulder_width_ratio = self._calculate_shoulder_width_ratio_from_pose(
            pose_detection, image_properties
        )
        if shoulder_width_ratio is not None:
            result.shoulder_width_ratio = shoulder_width_ratio

            # Update shot type if shoulder analysis suggests different classification
            if shoulder_width_ratio >= SHOULDER_WIDTH_CLOSEUP_THRESHOLD:
                if result.shot_type in ["medium_shot", "wide_shot"]:
                    result.shot_type = "medium_closeup"
                    result.is_closeup = True

                    # Update confidence with shoulder information
                    result.confidence = min(1.0, result.confidence + 0.1)

        return result

    def _assess_composition_with_properties(
        self, face_detection: FaceDetection, image_properties: ImageProperties
    ) -> Tuple[float, List[str], Tuple[str, str]]:
        """Assess frame composition quality using ImageProperties data model.

        Args:
            face_detection: Face detection result
            image_properties: Image properties data model

        Returns:
            Tuple of (composition_score, notes, face_position)
        """
        height = image_properties.height
        width = image_properties.width
        x1, y1, x2, y2 = face_detection.bbox
        face_center_x = (x1 + x2) / 2
        face_center_y = (y1 + y2) / 2
        face_height = y2 - y1

        composition_score = 0.0
        notes = []

        # Rule of thirds assessment
        third_width = width / 3
        third_height = height / 3

        # Check horizontal positioning
        horizontal_pos = "center"
        if face_center_x < third_width:
            horizontal_pos = "left"
        elif face_center_x > 2 * third_width:
            horizontal_pos = "right"

        # Check vertical positioning
        vertical_pos = "center"
        if face_center_y < third_height:
            vertical_pos = "upper"
        elif face_center_y > 2 * third_height:
            vertical_pos = "lower"

        face_position = (horizontal_pos, vertical_pos)

        # Score rule of thirds positioning
        # Prefer center or slightly off-center for portraits
        if horizontal_pos == "center":
            composition_score += 0.3
            notes.append("good_horizontal_centering")
        else:
            composition_score += 0.2
            notes.append("rule_of_thirds_horizontal")

        if vertical_pos in ["center", "upper"]:
            composition_score += 0.3
            notes.append("good_vertical_positioning")
        else:
            composition_score += 0.1
            notes.append("face_too_low")

        # Face size relative to frame assessment
        face_height_ratio = face_height / height
        if 0.3 <= face_height_ratio <= 0.5:
            composition_score += 0.3
            notes.append("ideal_face_size")
        elif 0.2 <= face_height_ratio <= 0.6:
            composition_score += 0.2
            notes.append("acceptable_face_size")
        else:
            composition_score += 0.1
            if face_height_ratio < 0.2:
                notes.append("face_too_small")
            else:
                notes.append("face_too_large")

        # Headroom assessment (space above face)
        headroom_ratio = y1 / height
        if 0.1 <= headroom_ratio <= 0.2:
            composition_score += 0.1
            notes.append("good_headroom")
        elif headroom_ratio < 0.05:
            notes.append("insufficient_headroom")
        else:
            notes.append("excessive_headroom")

        return min(1.0, composition_score), notes, face_position

    def _calculate_shoulder_width_ratio_from_pose(
        self, pose_detection: PoseDetection, image_properties: ImageProperties
    ) -> Optional[float]:
        """Calculate shoulder width ratio from pose detection data model.

        Args:
            pose_detection: Pose detection with keypoints
            image_properties: Image properties containing dimensions

        Returns:
            Shoulder width ratio or None if keypoints unavailable
        """
        pose_keypoints = pose_detection.keypoints
        left_shoulder = pose_keypoints.get("left_shoulder")
        right_shoulder = pose_keypoints.get("right_shoulder")

        if (
            left_shoulder
            and right_shoulder
            and left_shoulder[2] >= MIN_LANDMARK_CONFIDENCE
            and right_shoulder[2] >= MIN_LANDMARK_CONFIDENCE
        ):

            shoulder_width = abs(right_shoulder[0] - left_shoulder[0])
            return shoulder_width / image_properties.width

        return None

    def _classify_shot_type(self, face_area_ratio: float) -> str:
        """Classify shot type based on face area ratio.

        Args:
            face_area_ratio: Ratio of face area to total frame area

        Returns:
            Shot type classification string
        """
        if face_area_ratio >= self.extreme_closeup_threshold:
            return "extreme_closeup"
        elif face_area_ratio >= self.closeup_threshold:
            return "closeup"
        elif face_area_ratio >= self.medium_closeup_threshold:
            return "medium_closeup"
        elif face_area_ratio >= self.medium_shot_threshold:
            return "medium_shot"
        else:
            return "wide_shot"

    def _calculate_inter_ocular_distance(
        self, landmarks: List[Tuple[float, float]]
    ) -> float:
        """Calculate distance between eyes using facial landmarks.

        Args:
            landmarks: List of facial landmark points (typically 5 points)
                      Expected format: [left_eye, right_eye, nose, left_mouth, right_mouth]

        Returns:
            Distance between eyes in pixels
        """
        if len(landmarks) < 2:
            return 0.0

        # Assuming first two landmarks are left and right eyes
        left_eye = landmarks[0]
        right_eye = landmarks[1]

        # Calculate Euclidean distance
        distance = math.sqrt(
            (right_eye[0] - left_eye[0]) ** 2 + (right_eye[1] - left_eye[1]) ** 2
        )
        return distance

    def _estimate_distance(self, inter_ocular_distance: float) -> str:
        """Estimate relative distance based on inter-ocular distance.

        Args:
            inter_ocular_distance: Distance between eyes in pixels

        Returns:
            Distance category string
        """
        if inter_ocular_distance >= VERY_CLOSE_IOD_THRESHOLD:
            return "very_close"
        elif inter_ocular_distance >= CLOSE_IOD_THRESHOLD:
            return "close"
        elif inter_ocular_distance >= MEDIUM_IOD_THRESHOLD:
            return "medium"
        else:
            return "far"

    def _calculate_detection_confidence(
        self,
        face_detection: FaceDetection,
        face_area_ratio: float,
        inter_ocular_distance: Optional[float],
        composition_score: float,
    ) -> float:
        """Calculate overall detection confidence based on multiple factors.

        Args:
            face_detection: Face detection result
            face_area_ratio: Face area ratio
            inter_ocular_distance: Inter-ocular distance
            composition_score: Composition quality score

        Returns:
            Overall confidence score (0.0 to 1.0)
        """
        confidence_factors = []

        # Factor 1: Face detection confidence
        confidence_factors.append(face_detection.confidence)

        # Factor 2: Face area ratio consistency with classification
        area_confidence = min(1.0, face_area_ratio / self.medium_closeup_threshold)
        confidence_factors.append(area_confidence)

        # Factor 3: Landmark quality (if available)
        if face_detection.landmarks and inter_ocular_distance:
            landmark_confidence = min(1.0, inter_ocular_distance / CLOSE_IOD_THRESHOLD)
            confidence_factors.append(landmark_confidence)

        # Factor 4: Composition quality
        confidence_factors.append(composition_score)

        return max(0.3, np.mean(confidence_factors))

    def process_frame_batch(
        self,
        frames_with_faces: List["FrameData"],
        progress_callback: Optional[callable] = None,
    ) -> None:
        """Process a batch of frames with closeup detection.

        Args:
            frames_with_faces: List of FrameData objects with face detections
            progress_callback: Optional callback for progress updates
        """
        if not frames_with_faces:
            return

        total_frames = len(frames_with_faces)

        logger.info(f"Starting closeup detection on {total_frames} frames")

        for i, frame_data in enumerate(frames_with_faces):
            try:
                # Use the new standardized method
                self.detect_closeups_in_frame(frame_data)

            except Exception as e:
                frame_id = getattr(frame_data, "frame_id", f"frame_{i}")
                logger.error(f"Closeup detection failed for frame {frame_id}: {e}")
                # Continue processing other frames

            # Update progress
            if progress_callback:
                progress_callback(i + 1)

        logger.info(f"Closeup detection completed: {total_frames} frames processed")

    def get_detection_info(self) -> Dict[str, Any]:
        """Get information about the current detection settings.

        Returns:
            Dictionary containing thresholds and configuration
        """
        return {
            "shot_thresholds": {
                "extreme_closeup_threshold": self.extreme_closeup_threshold,
                "closeup_threshold": self.closeup_threshold,
                "medium_closeup_threshold": self.medium_closeup_threshold,
                "medium_shot_threshold": self.medium_shot_threshold,
            },
            "distance_thresholds": {
                "very_close_iod": VERY_CLOSE_IOD_THRESHOLD,
                "close_iod": CLOSE_IOD_THRESHOLD,
                "medium_iod": MEDIUM_IOD_THRESHOLD,
            },
            "composition_constants": {
                "rule_of_thirds_tolerance": RULE_OF_THIRDS_TOLERANCE,
                "ideal_face_height_ratio": IDEAL_FACE_HEIGHT_RATIO,
                "shoulder_width_closeup_threshold": SHOULDER_WIDTH_CLOSEUP_THRESHOLD,
            },
        }

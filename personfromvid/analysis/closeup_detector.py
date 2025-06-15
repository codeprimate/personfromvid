"""Close-up shot detection and distance estimation.

This module provides comprehensive closeup detection capabilities including:
- Shot type classification (extreme closeup, closeup, medium closeup, etc.)
- Distance estimation using facial landmarks and geometry
- Face size ratio analysis for shot classification
"""

import logging
import math
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import time

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

SHOULDER_WIDTH_CLOSEUP_THRESHOLD = 0.35  # Shoulder width ratio for closeup detection

# Confidence thresholds
MIN_FACE_CONFIDENCE = 0.3
MIN_LANDMARK_CONFIDENCE = 0.5


class CloseupDetectionError(AnalysisError):
    """Raised when closeup detection fails."""

    pass


class CloseupDetector:
    """Comprehensive closeup detection and distance estimation.

    This class provides advanced closeup detection capabilities including:
    - Multi-criteria shot classification
    - Distance estimation using facial geometry
    - Portrait suitability analysis

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

            # Determine if this is a closeup
            is_closeup = shot_type in ["extreme_closeup", "closeup", "medium_closeup"]

            # Calculate confidence based on multiple factors
            confidence = self._calculate_detection_confidence(
                face_detection,
                face_area_ratio,
                inter_ocular_distance,
            )

            return CloseupDetection(
                is_closeup=is_closeup,
                shot_type=shot_type,
                confidence=confidence,
                face_area_ratio=face_area_ratio,
                inter_ocular_distance=inter_ocular_distance,
                estimated_distance=estimated_distance,
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
    ) -> float:
        """Calculate overall detection confidence based on multiple factors.

        Args:
            face_detection: Face detection result
            face_area_ratio: Face area ratio
            inter_ocular_distance: Inter-ocular distance

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

        return max(0.3, np.mean(confidence_factors))

    def process_frame_batch(
        self,
        frames_with_faces: List["FrameData"],
        progress_callback: Optional[callable] = None,
        interruption_check: Optional[callable] = None,
    ) -> None:
        """Process a batch of frames with closeup detection.

        Args:
            frames_with_faces: List of FrameData objects with face detections
            progress_callback: Optional callback for progress updates
            interruption_check: Optional callback to check for interruption
        """
        if not frames_with_faces:
            return

        total_frames = len(frames_with_faces)
        start_time = time.time()

        logger.info(f"Starting closeup detection on {total_frames} frames")

        for i, frame_data in enumerate(frames_with_faces):
            # Check for interruption at regular intervals
            if interruption_check and i % 10 == 0:
                interruption_check()
                
            try:
                # Use the new standardized method
                self.detect_closeups_in_frame(frame_data)

            except Exception as e:
                frame_id = getattr(frame_data, "frame_id", f"frame_{i}")
                logger.error(f"Closeup detection failed for frame {frame_id}: {e}")
                # Continue processing other frames

            # Update progress with rate calculation
            if progress_callback:
                processed_count = i + 1
                # Calculate current processing rate
                elapsed = time.time() - start_time
                current_rate = processed_count / elapsed if elapsed > 0 else 0
                # Check if progress_callback accepts rate parameter
                try:
                    progress_callback(processed_count, rate=current_rate)
                except TypeError:
                    # Fallback to single argument for backwards compatibility
                    progress_callback(processed_count)

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
            "detection_constants": {
                "shoulder_width_closeup_threshold": SHOULDER_WIDTH_CLOSEUP_THRESHOLD,
            },
        }

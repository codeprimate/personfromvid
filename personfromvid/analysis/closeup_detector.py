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

from ..data.detection_results import FaceDetection, CloseupDetection
from ..utils.exceptions import AnalysisError

logger = logging.getLogger(__name__)

# Shot classification thresholds (face area ratio relative to frame)
EXTREME_CLOSEUP_THRESHOLD = 0.25  # Face takes up >25% of frame
CLOSEUP_THRESHOLD = 0.15          # Face takes up >15% of frame
MEDIUM_CLOSEUP_THRESHOLD = 0.08   # Face takes up >8% of frame
MEDIUM_SHOT_THRESHOLD = 0.03      # Face takes up >3% of frame

# Distance estimation thresholds (inter-ocular distance in pixels)
VERY_CLOSE_IOD_THRESHOLD = 80     # >80 pixels between eyes
CLOSE_IOD_THRESHOLD = 50          # >50 pixels between eyes
MEDIUM_IOD_THRESHOLD = 25         # >25 pixels between eyes

# Composition assessment constants
RULE_OF_THIRDS_TOLERANCE = 0.1    # ±10% tolerance for rule of thirds
IDEAL_FACE_HEIGHT_RATIO = 0.4     # Face should be ~40% of frame height for good composition
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
        Basic usage:
        >>> detector = CloseupDetector()
        >>> result = detector.detect_closeup(face_detection, image_shape)
        
        With pose keypoints for enhanced analysis:
        >>> result = detector.detect_closeup_with_pose(face_detection, pose_keypoints, image_shape)
        
        Batch processing:
        >>> results = detector.process_frame_batch(frames_with_faces)
    """
    
    def __init__(self, 
                 extreme_closeup_threshold: float = EXTREME_CLOSEUP_THRESHOLD,
                 closeup_threshold: float = CLOSEUP_THRESHOLD,
                 medium_closeup_threshold: float = MEDIUM_CLOSEUP_THRESHOLD,
                 medium_shot_threshold: float = MEDIUM_SHOT_THRESHOLD):
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
        
        logger.info(f"Initialized CloseupDetector with thresholds: "
                   f"extreme={extreme_closeup_threshold}, closeup={closeup_threshold}, "
                   f"medium_closeup={medium_closeup_threshold}, medium_shot={medium_shot_threshold}")
    
    def detect_closeup(self, face_detection: FaceDetection, 
                      image_shape: Tuple[int, int]) -> CloseupDetection:
        """Detect closeup shot characteristics from face detection.
        
        Args:
            face_detection: Face detection result with bbox and landmarks
            image_shape: Image dimensions (height, width)
            
        Returns:
            CloseupDetection with comprehensive analysis results
            
        Raises:
            CloseupDetectionError: If detection fails
        """
        try:
            height, width = image_shape
            frame_area = height * width
            
            # Calculate face area ratio
            face_area = face_detection.area
            face_area_ratio = face_area / frame_area
            
            # Classify shot type based on face area ratio
            shot_type = self._classify_shot_type(face_area_ratio)
            
            # Calculate inter-ocular distance if landmarks available
            inter_ocular_distance = None
            estimated_distance = None
            if face_detection.landmarks and len(face_detection.landmarks) >= 5:
                inter_ocular_distance = self._calculate_inter_ocular_distance(face_detection.landmarks)
                estimated_distance = self._estimate_distance(inter_ocular_distance)
            
            # Assess frame composition
            composition_score, composition_notes, face_position = self._assess_composition(
                face_detection, image_shape
            )
            
            # Determine if this is a closeup
            is_closeup = shot_type in ["extreme_closeup", "closeup", "medium_closeup"]
            
            # Calculate confidence based on multiple factors
            confidence = self._calculate_detection_confidence(
                face_detection, face_area_ratio, inter_ocular_distance, composition_score
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
                face_position=face_position
            )
            
        except Exception as e:
            raise CloseupDetectionError(f"Failed to detect closeup: {str(e)}") from e
    
    def detect_closeup_with_pose(self, face_detection: FaceDetection,
                                pose_keypoints: Dict[str, Tuple[float, float, float]],
                                image_shape: Tuple[int, int]) -> CloseupDetection:
        """Enhanced closeup detection using both face and pose information.
        
        Args:
            face_detection: Face detection result
            pose_keypoints: Pose keypoints for additional context
            image_shape: Image dimensions (height, width)
            
        Returns:
            CloseupDetection with enhanced analysis including shoulder width
        """
        # Start with basic face-based detection
        result = self.detect_closeup(face_detection, image_shape)
        
        # Add shoulder width analysis if available
        shoulder_width_ratio = self._calculate_shoulder_width_ratio(pose_keypoints, image_shape)
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
    
    def _calculate_inter_ocular_distance(self, landmarks: List[Tuple[float, float]]) -> float:
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
        distance = math.sqrt((right_eye[0] - left_eye[0])**2 + (right_eye[1] - left_eye[1])**2)
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
    
    def _assess_composition(self, face_detection: FaceDetection, 
                          image_shape: Tuple[int, int]) -> Tuple[float, List[str], Tuple[str, str]]:
        """Assess frame composition quality for portrait photography.
        
        Args:
            face_detection: Face detection result
            image_shape: Image dimensions (height, width)
            
        Returns:
            Tuple of (composition_score, notes, face_position)
        """
        height, width = image_shape
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
    
    def _calculate_shoulder_width_ratio(self, pose_keypoints: Dict[str, Tuple[float, float, float]],
                                      image_shape: Tuple[int, int]) -> Optional[float]:
        """Calculate shoulder width ratio from pose keypoints.
        
        Args:
            pose_keypoints: Pose keypoints dictionary
            image_shape: Image dimensions (height, width)
            
        Returns:
            Shoulder width ratio or None if keypoints unavailable
        """
        left_shoulder = pose_keypoints.get('left_shoulder')
        right_shoulder = pose_keypoints.get('right_shoulder')
        
        if (left_shoulder and right_shoulder and
            left_shoulder[2] >= MIN_LANDMARK_CONFIDENCE and
            right_shoulder[2] >= MIN_LANDMARK_CONFIDENCE):
            
            shoulder_width = abs(right_shoulder[0] - left_shoulder[0])
            return shoulder_width / image_shape[1]
        
        return None
    
    def _calculate_detection_confidence(self, face_detection: FaceDetection,
                                      face_area_ratio: float,
                                      inter_ocular_distance: Optional[float],
                                      composition_score: float) -> float:
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
    
    def process_frame_batch(self, frames_with_faces: List['FrameData'], 
                           progress_callback: Optional[callable] = None) -> None:
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
                # Get image dimensions from FrameData object
                image_shape = (frame_data.image_properties.height, 
                             frame_data.image_properties.width)
                
                # Get face and pose detections from FrameData object
                face_detections = frame_data.face_detections
                pose_detections = frame_data.pose_detections
                
                closeup_results = []
                
                # Process each face detection
                for face_idx, face_detection in enumerate(face_detections):
                    # Check if we have pose data for enhanced detection
                    if pose_detections and len(pose_detections) > face_idx:
                        pose_keypoints = pose_detections[face_idx].keypoints
                        closeup_result = self.detect_closeup_with_pose(
                            face_detection, pose_keypoints, image_shape
                        )
                    else:
                        closeup_result = self.detect_closeup(face_detection, image_shape)
                    
                    closeup_results.append(closeup_result)
                
                # Add closeup results to FrameData object
                frame_data.closeup_detections.extend(closeup_results)
                
            except Exception as e:
                frame_id = getattr(frame_data, 'frame_id', f'frame_{i}')
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
            'shot_thresholds': {
                'extreme_closeup_threshold': self.extreme_closeup_threshold,
                'closeup_threshold': self.closeup_threshold,
                'medium_closeup_threshold': self.medium_closeup_threshold,
                'medium_shot_threshold': self.medium_shot_threshold
            },
            'distance_thresholds': {
                'very_close_iod': VERY_CLOSE_IOD_THRESHOLD,
                'close_iod': CLOSE_IOD_THRESHOLD,
                'medium_iod': MEDIUM_IOD_THRESHOLD
            },
            'composition_constants': {
                'rule_of_thirds_tolerance': RULE_OF_THIRDS_TOLERANCE,
                'ideal_face_height_ratio': IDEAL_FACE_HEIGHT_RATIO,
                'shoulder_width_closeup_threshold': SHOULDER_WIDTH_CLOSEUP_THRESHOLD
            }
        } 
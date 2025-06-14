"""Unit tests for closeup detection functionality."""

import pytest
from unittest.mock import Mock, patch
import numpy as np

from personfromvid.analysis.closeup_detector import (
    CloseupDetector, 
    CloseupDetectionError,
    EXTREME_CLOSEUP_THRESHOLD,
    CLOSEUP_THRESHOLD,
    MEDIUM_CLOSEUP_THRESHOLD,
    MEDIUM_SHOT_THRESHOLD
)
from personfromvid.data.detection_results import FaceDetection, CloseupDetection


class TestCloseupDetector:
    """Test suite for CloseupDetector class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.detector = CloseupDetector()
        self.image_shape = (1080, 1920)  # height, width
    
    def create_face_detection(self, bbox, confidence=0.9, landmarks=None):
        """Helper to create FaceDetection object."""
        return FaceDetection(
            bbox=bbox,
            confidence=confidence,
            landmarks=landmarks
        )
    
    def test_initialization(self):
        """Test CloseupDetector initialization."""
        detector = CloseupDetector(
            extreme_closeup_threshold=0.3,
            closeup_threshold=0.2,
            medium_closeup_threshold=0.1,
            medium_shot_threshold=0.05
        )
        
        assert detector.extreme_closeup_threshold == 0.3
        assert detector.closeup_threshold == 0.2
        assert detector.medium_closeup_threshold == 0.1
        assert detector.medium_shot_threshold == 0.05
    
    def test_shot_type_classification_extreme_closeup(self):
        """Test classification of extreme closeup shots."""
        # Large face taking up 30% of frame
        face_area = int(self.image_shape[0] * self.image_shape[1] * 0.30)
        bbox_size = int(np.sqrt(face_area))
        bbox = (500, 300, 500 + bbox_size, 300 + bbox_size)
        
        face_detection = self.create_face_detection(bbox)
        result = self.detector.detect_closeup(face_detection, self.image_shape)
        
        assert result.shot_type == "extreme_closeup"
        assert result.is_closeup is True
        assert result.confidence > 0.5
        assert result.face_area_ratio >= EXTREME_CLOSEUP_THRESHOLD
    
    def test_shot_type_classification_closeup(self):
        """Test classification of closeup shots."""
        # Face taking up 18% of frame
        face_area = int(self.image_shape[0] * self.image_shape[1] * 0.18)
        bbox_size = int(np.sqrt(face_area))
        bbox = (500, 300, 500 + bbox_size, 300 + bbox_size)
        
        face_detection = self.create_face_detection(bbox)
        result = self.detector.detect_closeup(face_detection, self.image_shape)
        
        assert result.shot_type == "closeup"
        assert result.is_closeup is True
        assert CLOSEUP_THRESHOLD <= result.face_area_ratio < EXTREME_CLOSEUP_THRESHOLD
    
    def test_shot_type_classification_medium_closeup(self):
        """Test classification of medium closeup shots."""
        # Face taking up 10% of frame
        face_area = int(self.image_shape[0] * self.image_shape[1] * 0.10)
        bbox_size = int(np.sqrt(face_area))
        bbox = (500, 300, 500 + bbox_size, 300 + bbox_size)
        
        face_detection = self.create_face_detection(bbox)
        result = self.detector.detect_closeup(face_detection, self.image_shape)
        
        assert result.shot_type == "medium_closeup"
        assert result.is_closeup is True
        assert MEDIUM_CLOSEUP_THRESHOLD <= result.face_area_ratio < CLOSEUP_THRESHOLD
    
    def test_shot_type_classification_medium_shot(self):
        """Test classification of medium shots."""
        # Face taking up 5% of frame
        face_area = int(self.image_shape[0] * self.image_shape[1] * 0.05)
        bbox_size = int(np.sqrt(face_area))
        bbox = (500, 300, 500 + bbox_size, 300 + bbox_size)
        
        face_detection = self.create_face_detection(bbox)
        result = self.detector.detect_closeup(face_detection, self.image_shape)
        
        assert result.shot_type == "medium_shot"
        assert result.is_closeup is False
        assert MEDIUM_SHOT_THRESHOLD <= result.face_area_ratio < MEDIUM_CLOSEUP_THRESHOLD
    
    def test_shot_type_classification_wide_shot(self):
        """Test classification of wide shots."""
        # Small face taking up 1% of frame
        face_area = int(self.image_shape[0] * self.image_shape[1] * 0.01)
        bbox_size = int(np.sqrt(face_area))
        bbox = (500, 300, 500 + bbox_size, 300 + bbox_size)
        
        face_detection = self.create_face_detection(bbox)
        result = self.detector.detect_closeup(face_detection, self.image_shape)
        
        assert result.shot_type == "wide_shot"
        assert result.is_closeup is False
        assert result.face_area_ratio < MEDIUM_SHOT_THRESHOLD
    
    def test_inter_ocular_distance_calculation(self):
        """Test inter-ocular distance calculation."""
        # Create landmarks with known eye positions
        landmarks = [
            (100.0, 200.0),  # left eye
            (180.0, 200.0),  # right eye (80 pixels apart)
            (140.0, 220.0),  # nose
            (120.0, 250.0),  # left mouth
            (160.0, 250.0)   # right mouth
        ]
        
        bbox = (50, 150, 250, 300)
        face_detection = self.create_face_detection(bbox, landmarks=landmarks)
        result = self.detector.detect_closeup(face_detection, self.image_shape)
        
        assert result.inter_ocular_distance == 80.0
        assert result.estimated_distance == "very_close"  # 80 > VERY_CLOSE_IOD_THRESHOLD
    
    def test_distance_estimation_categories(self):
        """Test distance estimation categories."""
        # Test different inter-ocular distances
        test_cases = [
            (90.0, "very_close"),
            (60.0, "close"),
            (35.0, "medium"),
            (15.0, "far")
        ]
        
        for iod, expected_distance in test_cases:
            distance = self.detector._estimate_distance(iod)
            assert distance == expected_distance, f"IOD {iod} should be {expected_distance}"
    
    def test_composition_assessment_centered_face(self):
        """Test composition assessment for centered face."""
        # Centered face with good proportions
        bbox = (800, 300, 1120, 620)  # 320x320 face centered horizontally
        face_detection = self.create_face_detection(bbox)
        result = self.detector.detect_closeup(face_detection, self.image_shape)
        
        assert result.face_position == ("center", "center")
        assert result.composition_score > 0.5
        assert "good_horizontal_centering" in result.composition_notes
    
    def test_composition_assessment_rule_of_thirds(self):
        """Test composition assessment for rule of thirds positioning."""
        width_third = self.image_shape[1] // 3
        
        # Face in left third
        bbox = (width_third // 2, 300, width_third // 2 + 200, 500)
        face_detection = self.create_face_detection(bbox)
        result = self.detector.detect_closeup(face_detection, self.image_shape)
        
        assert result.face_position[0] == "left"
        assert "rule_of_thirds_horizontal" in result.composition_notes
    
    def test_composition_assessment_face_size(self):
        """Test composition assessment for different face sizes."""
        # Ideal face size (40% of frame height)
        face_height = int(self.image_shape[0] * 0.4)
        bbox = (800, 300, 1000, 300 + face_height)
        face_detection = self.create_face_detection(bbox)
        result = self.detector.detect_closeup(face_detection, self.image_shape)
        
        assert "ideal_face_size" in result.composition_notes
        assert result.composition_score > 0.6
    
    def test_detect_closeup_with_pose_keypoints(self):
        """Test enhanced closeup detection with pose keypoints."""
        bbox = (500, 300, 700, 500)
        face_detection = self.create_face_detection(bbox)
        
        # Pose keypoints with wide shoulders indicating closeup
        pose_keypoints = {
            'left_shoulder': (400.0, 450.0, 0.9),
            'right_shoulder': (1000.0, 450.0, 0.9),  # Wide shoulders
            'nose': (600.0, 400.0, 0.8)
        }
        
        result = self.detector.detect_closeup_with_pose(
            face_detection, pose_keypoints, self.image_shape
        )
        
        assert result.shoulder_width_ratio is not None
        assert result.shoulder_width_ratio > 0.3
    
    def test_batch_processing(self):
        """Test batch processing of frames."""
        from personfromvid.data.frame_data import FrameData
        from personfromvid.data.detection_results import FaceDetection, PoseDetection
        from personfromvid.data.frame_data import SourceInfo, ImageProperties
        from pathlib import Path
        
        # Create mock FrameData objects
        frames_with_faces = [
            FrameData(
                frame_id='frame_001',
                file_path=Path('/path/to/frame1.jpg'),
                source_info=SourceInfo(
                    video_timestamp=1.0, 
                    extraction_method='test',
                    original_frame_number=30,
                    video_fps=30.0
                ),
                image_properties=ImageProperties(
                    width=self.image_shape[1], 
                    height=self.image_shape[0], 
                    channels=3,
                    file_size_bytes=1024000,
                    format='jpg'
                ),
                face_detections=[
                    FaceDetection(
                        bbox=(500, 300, 800, 600),
                        confidence=0.9,
                        landmarks=None
                    )
                ],
                pose_detections=[]
            ),
            FrameData(
                frame_id='frame_002',
                file_path=Path('/path/to/frame2.jpg'),
                source_info=SourceInfo(
                    video_timestamp=2.0, 
                    extraction_method='test',
                    original_frame_number=60,
                    video_fps=30.0
                ),
                image_properties=ImageProperties(
                    width=self.image_shape[1], 
                    height=self.image_shape[0], 
                    channels=3,
                    file_size_bytes=1024000,
                    format='jpg'
                ),
                face_detections=[
                    FaceDetection(
                        bbox=(100, 100, 200, 200),
                        confidence=0.8,
                        landmarks=[
                            (120.0, 140.0), (160.0, 140.0),
                            (140.0, 160.0), (130.0, 180.0), (150.0, 180.0)
                        ]
                    )
                ],
                pose_detections=[]
            )
        ]
        
        # The method now modifies frames in-place and returns None
        result = self.detector.process_frame_batch(frames_with_faces)
        
        # Verify the method returns None (in-place modification)
        assert result is None
        
        # Verify frame data was updated in-place
        assert len(frames_with_faces) == 2
        
        # First frame should have closeup detection results
        assert len(frames_with_faces[0].closeup_detections) == 1
        closeup_result = frames_with_faces[0].closeup_detections[0]
        assert hasattr(closeup_result, 'shot_type')
        assert hasattr(closeup_result, 'is_closeup')
        assert hasattr(closeup_result, 'confidence')
        
        # Second frame should also have results
        assert len(frames_with_faces[1].closeup_detections) == 1
    
    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        # Test with very small face (edge case)
        invalid_face = Mock()
        invalid_face.area = 0
        invalid_face.bbox = (0, 0, 0, 0)
        invalid_face.landmarks = None
        invalid_face.confidence = 0.9
        
        # The detector should handle this gracefully without raising an exception
        result = self.detector.detect_closeup(invalid_face, self.image_shape)
        assert result.shot_type == "wide_shot"
        assert result.face_area_ratio == 0.0
        assert result.is_closeup is False
    
    def test_confidence_calculation_factors(self):
        """Test confidence calculation considers multiple factors."""
        # High quality face detection with landmarks
        landmarks = [
            (100.0, 200.0), (180.0, 200.0), (140.0, 220.0),
            (120.0, 250.0), (160.0, 250.0)
        ]
        bbox = (500, 300, 900, 700)  # Large face for closeup
        face_detection = self.create_face_detection(bbox, confidence=0.95, landmarks=landmarks)
        
        result = self.detector.detect_closeup(face_detection, self.image_shape)
        
        # Should have high confidence due to:
        # - High face detection confidence
        # - Large face area ratio
        # - Available landmarks
        # - Good composition
        assert result.confidence > 0.7
    
    def test_get_detection_info(self):
        """Test getting detection configuration information."""
        info = self.detector.get_detection_info()
        
        assert 'shot_thresholds' in info
        assert 'distance_thresholds' in info
        assert 'composition_constants' in info
        
        # Check shot thresholds
        shot_thresholds = info['shot_thresholds']
        assert shot_thresholds['extreme_closeup_threshold'] == EXTREME_CLOSEUP_THRESHOLD
        assert shot_thresholds['closeup_threshold'] == CLOSEUP_THRESHOLD
        
        # Check distance thresholds
        distance_thresholds = info['distance_thresholds']
        assert 'very_close_iod' in distance_thresholds
        assert 'close_iod' in distance_thresholds
    
    def test_empty_frame_batch(self):
        """Test processing empty frame batch."""
        result = self.detector.process_frame_batch([])
        # Method returns None for empty input (no frames to modify)
        assert result is None
    
    def test_frame_with_no_faces(self):
        """Test processing frame with no face detections."""
        from personfromvid.data.frame_data import FrameData
        from personfromvid.data.frame_data import SourceInfo, ImageProperties
        from pathlib import Path
        
        frames_with_faces = [
            FrameData(
                frame_id='frame_001',
                file_path=Path('/path/to/frame1.jpg'),
                source_info=SourceInfo(
                    video_timestamp=1.0, 
                    extraction_method='test',
                    original_frame_number=30,
                    video_fps=30.0
                ),
                image_properties=ImageProperties(
                    width=self.image_shape[1], 
                    height=self.image_shape[0], 
                    channels=3,
                    file_size_bytes=1024000,
                    format='jpg'
                ),
                face_detections=[],  # No faces
                pose_detections=[]
            )
        ]
        
        result = self.detector.process_frame_batch(frames_with_faces)
        
        # Method returns None (in-place modification)
        assert result is None
        
        # Frame should have no closeup detections since there are no faces
        assert len(frames_with_faces) == 1
        assert len(frames_with_faces[0].closeup_detections) == 0
    
    def test_batch_processing_edge_cases(self):
        """Test batch processing with edge cases that are actually possible."""
        from personfromvid.data.frame_data import FrameData
        from personfromvid.data.detection_results import FaceDetection
        from personfromvid.data.frame_data import SourceInfo, ImageProperties
        from pathlib import Path
        
        # Create a FrameData with a very small face (edge case that's actually possible)
        frames_with_faces = [
            FrameData(
                frame_id='frame_001',
                file_path=Path('/path/to/frame1.jpg'),
                source_info=SourceInfo(
                    video_timestamp=1.0, 
                    extraction_method='test',
                    original_frame_number=30,
                    video_fps=30.0
                ),
                image_properties=ImageProperties(
                    width=self.image_shape[1], 
                    height=self.image_shape[0], 
                    channels=3,
                    file_size_bytes=1024000,
                    format='jpg'
                ),
                face_detections=[
                    FaceDetection(
                        bbox=(0, 0, 1, 1),  # Extremely small face (1x1 pixel)
                        confidence=0.9,
                        landmarks=None
                    )
                ],
                pose_detections=[]
            )
        ]
        
        result = self.detector.process_frame_batch(frames_with_faces)
        
        # Method returns None (in-place modification)
        assert result is None
        
        # Should still process the frame, even with tiny face
        assert len(frames_with_faces) == 1
        assert len(frames_with_faces[0].closeup_detections) == 1
        
        # Tiny face should be classified as wide shot
        closeup_result = frames_with_faces[0].closeup_detections[0]
        assert closeup_result.shot_type == "wide_shot"
        assert closeup_result.is_closeup is False
    
    def test_shoulder_width_ratio_calculation(self):
        """Test shoulder width ratio calculation from pose keypoints."""
        pose_keypoints = {
            'left_shoulder': (400.0, 450.0, 0.9),
            'right_shoulder': (800.0, 450.0, 0.9),
            'nose': (600.0, 400.0, 0.8)
        }
        
        ratio = self.detector._calculate_shoulder_width_ratio(pose_keypoints, self.image_shape)
        
        expected_ratio = 400.0 / self.image_shape[1]  # shoulder width / frame width
        assert abs(ratio - expected_ratio) < 0.001
    
    def test_shoulder_width_ratio_insufficient_confidence(self):
        """Test shoulder width calculation with low confidence keypoints."""
        pose_keypoints = {
            'left_shoulder': (400.0, 450.0, 0.3),  # Low confidence
            'right_shoulder': (800.0, 450.0, 0.9),
            'nose': (600.0, 400.0, 0.8)
        }
        
        ratio = self.detector._calculate_shoulder_width_ratio(pose_keypoints, self.image_shape)
        assert ratio is None
    
    def test_composition_headroom_assessment(self):
        """Test headroom assessment in composition scoring."""
        # Face with good headroom (15% from top)
        y_offset = int(self.image_shape[0] * 0.15)
        bbox = (800, y_offset, 1000, y_offset + 200)
        face_detection = self.create_face_detection(bbox)
        result = self.detector.detect_closeup(face_detection, self.image_shape)
        
        assert "good_headroom" in result.composition_notes
        
        # Face with insufficient headroom (too close to top)
        bbox = (800, 10, 1000, 210)
        face_detection = self.create_face_detection(bbox)
        result = self.detector.detect_closeup(face_detection, self.image_shape)
        
        assert "insufficient_headroom" in result.composition_notes 
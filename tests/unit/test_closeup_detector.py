"""Unit tests for closeup detection functionality."""

from pathlib import Path
from unittest.mock import Mock

import numpy as np

from personfromvid.analysis.closeup_detector import (
    CLOSEUP_THRESHOLD,
    EXTREME_CLOSEUP_THRESHOLD,
    MEDIUM_CLOSEUP_THRESHOLD,
    MEDIUM_SHOT_THRESHOLD,
    CloseupDetector,
)
from personfromvid.data.detection_results import FaceDetection, PoseDetection
from personfromvid.data.frame_data import FrameData, ImageProperties, SourceInfo


class TestCloseupDetector:
    """Test suite for CloseupDetector class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.detector = CloseupDetector()
        self.image_shape = (1080, 1920)  # height, width

        # Create sample FrameData components
        self.source_info = SourceInfo(
            video_timestamp=1.0,
            extraction_method="test",
            original_frame_number=1,
            video_fps=30.0
        )

        self.image_properties = ImageProperties(
            width=1920,
            height=1080,
            channels=3,
            file_size_bytes=1000000,
            format="JPG"
        )

    def create_face_detection(self, bbox, confidence=0.9, landmarks=None):
        """Helper to create FaceDetection object."""
        return FaceDetection(
            bbox=bbox,
            confidence=confidence,
            landmarks=landmarks
        )

    def create_frame_data(self, face_detections=None, pose_detections=None):
        """Helper to create FrameData object with detections."""
        return FrameData(
            frame_id="test_frame",
            file_path=Path("/tmp/test.jpg"),
            source_info=self.source_info,
            image_properties=self.image_properties,
            face_detections=face_detections or [],
            pose_detections=pose_detections or []
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
        frame_data = self.create_frame_data([face_detection])

        self.detector.detect_closeups_in_frame(frame_data)

        assert len(frame_data.closeup_detections) == 1
        result = frame_data.closeup_detections[0]
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
        frame_data = self.create_frame_data([face_detection])

        self.detector.detect_closeups_in_frame(frame_data)

        result = frame_data.closeup_detections[0]
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
        frame_data = self.create_frame_data([face_detection])

        self.detector.detect_closeups_in_frame(frame_data)

        result = frame_data.closeup_detections[0]
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
        frame_data = self.create_frame_data([face_detection])

        self.detector.detect_closeups_in_frame(frame_data)

        result = frame_data.closeup_detections[0]
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
        frame_data = self.create_frame_data([face_detection])

        self.detector.detect_closeups_in_frame(frame_data)

        result = frame_data.closeup_detections[0]
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
        frame_data = self.create_frame_data([face_detection])

        self.detector.detect_closeups_in_frame(frame_data)

        result = frame_data.closeup_detections[0]
        assert result.inter_ocular_distance == 80.0
        assert result.estimated_distance == "very_close"  # 80 > VERY_CLOSE_IOD_THRESHOLD

    def test_distance_estimation_categories(self):
        """Test distance estimation based on inter-ocular distance."""
        # Test various inter-ocular distances
        test_cases = [
            (100, "very_close"),
            (60, "close"),
            (30, "medium"),
            (10, "far")
        ]

        for iod, expected_distance in test_cases:
            distance = self.detector._estimate_distance(iod)
            assert distance == expected_distance

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

        pose_detection = PoseDetection(
            bbox=(400, 300, 1000, 600),
            confidence=0.9,
            keypoints=pose_keypoints
        )

        frame_data = self.create_frame_data([face_detection], [pose_detection])

        self.detector.detect_closeups_in_frame(frame_data)

        result = frame_data.closeup_detections[0]
        assert result.shoulder_width_ratio is not None
        assert result.shoulder_width_ratio > 0.3  # Wide shoulders

    def test_batch_processing(self):
        """Test batch processing of frames."""
        from pathlib import Path

        from personfromvid.data.detection_results import FaceDetection
        from personfromvid.data.frame_data import FrameData, ImageProperties, SourceInfo

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

        frame_data = self.create_frame_data([invalid_face])

        # The detector should handle this gracefully without raising an exception
        self.detector.detect_closeups_in_frame(frame_data)

        assert len(frame_data.closeup_detections) == 1
        result = frame_data.closeup_detections[0]
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
        frame_data = self.create_frame_data([face_detection])

        self.detector.detect_closeups_in_frame(frame_data)

        result = frame_data.closeup_detections[0]
        # Should have high confidence due to:
        # - High face detection confidence
        # - Large face area ratio
        # - Available landmarks
        assert result.confidence > 0.7

    def test_get_detection_info(self):
        """Test getting detection configuration information."""
        info = self.detector.get_detection_info()

        assert 'shot_thresholds' in info
        assert 'distance_thresholds' in info
        assert 'detection_constants' in info

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
        assert result is None

    def test_frame_with_no_faces(self):
        """Test processing frame with no face detections."""
        frame_data = self.create_frame_data([])  # No face detections

        self.detector.detect_closeups_in_frame(frame_data)

        # Should not add any closeup detections
        assert len(frame_data.closeup_detections) == 0

    def test_multiple_faces_in_frame(self):
        """Test processing frame with multiple face detections."""
        # Create two different face detections
        face1 = self.create_face_detection((100, 100, 300, 300))  # Small face
        face2 = self.create_face_detection((400, 200, 1000, 800))  # Large face for closeup

        frame_data = self.create_frame_data([face1, face2])

        self.detector.detect_closeups_in_frame(frame_data)

        # Should have two closeup detection results
        assert len(frame_data.closeup_detections) == 2

        # First face should be wide shot or medium shot
        result1 = frame_data.closeup_detections[0]
        assert result1.shot_type in ["wide_shot", "medium_shot"]

        # Second face should be closeup (large face area)
        result2 = frame_data.closeup_detections[1]
        assert result2.is_closeup is True

    def test_batch_processing_edge_cases(self):
        """Test batch processing with edge cases."""
        # Frame with face but no pose data
        frame1 = self.create_frame_data([
            self.create_face_detection((400, 300, 600, 500))
        ])

        # Frame with both face and pose data
        pose_keypoints = {
            'left_shoulder': (300.0, 400.0, 0.9),
            'right_shoulder': (700.0, 400.0, 0.9),
            'nose': (500.0, 350.0, 0.8)
        }
        pose_detection = PoseDetection(
            bbox=(300, 300, 700, 600),
            confidence=0.9,
            keypoints=pose_keypoints
        )
        frame2 = self.create_frame_data([
            self.create_face_detection((400, 300, 600, 500))
        ], [pose_detection])

        # Frame with no faces
        frame3 = self.create_frame_data([])

        frames = [frame1, frame2, frame3]

        # Process batch
        self.detector.process_frame_batch(frames)

        # Verify results
        assert len(frame1.closeup_detections) == 1
        assert len(frame2.closeup_detections) == 1
        assert len(frame3.closeup_detections) == 0

        # Frame with pose data should have shoulder width ratio
        assert frame2.closeup_detections[0].shoulder_width_ratio is not None

        # Frame without pose data should not have shoulder width ratio
        assert frame1.closeup_detections[0].shoulder_width_ratio is None


# Head angle classifier tests

"""Unit tests for head angle classifier module."""

import warnings
from pathlib import Path
from unittest.mock import Mock

from personfromvid.analysis.head_angle_classifier import HeadAngleClassifier
from personfromvid.data.detection_results import HeadPoseResult
from personfromvid.data.frame_data import FrameData, ImageProperties, SourceInfo


class TestHeadAngleClassifier:
    """Tests for HeadAngleClassifier class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.classifier = HeadAngleClassifier()

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

    def create_head_pose_result(self, yaw, pitch, roll, confidence=0.9):
        """Helper to create a HeadPoseResult object."""
        return HeadPoseResult(
            yaw=yaw,
            pitch=pitch,
            roll=roll,
            confidence=confidence
        )

    def create_frame_data(self, head_pose_results):
        """Helper to create FrameData object with head pose results."""
        return FrameData(
            frame_id="test_frame",
            file_path=Path("/tmp/test.jpg"),
            source_info=self.source_info,
            image_properties=self.image_properties,
            head_poses=head_pose_results
        )

    def test_init_default_thresholds(self):
        """Test HeadAngleClassifier initialization with default thresholds."""
        assert self.classifier.yaw_threshold == 22.5
        assert self.classifier.pitch_threshold == 22.5
        assert self.classifier.profile_yaw_threshold == 67.5
        assert self.classifier.max_roll == 30.0

    def test_classify_front_direction(self):
        """Test classification of front direction."""
        head_pose = self.create_head_pose_result(yaw=0.0, pitch=0.0, roll=0.0)
        frame_data = self.create_frame_data([head_pose])
        self.classifier.classify_head_poses_in_frame(frame_data)
        assert frame_data.head_poses[0].direction == "front"

        # Test within front threshold
        head_pose = self.create_head_pose_result(yaw=15.0, pitch=-10.0, roll=5.0)
        frame_data = self.create_frame_data([head_pose])
        self.classifier.classify_head_poses_in_frame(frame_data)
        assert frame_data.head_poses[0].direction == "front"

    def test_classify_looking_left_right(self):
        """Test classification of looking left/right directions."""
        # Looking left
        head_pose = self.create_head_pose_result(yaw=45.0, pitch=0.0, roll=0.0)
        frame_data = self.create_frame_data([head_pose])
        self.classifier.classify_head_poses_in_frame(frame_data)
        assert frame_data.head_poses[0].direction == "looking_left"

        # Looking right
        head_pose = self.create_head_pose_result(yaw=-45.0, pitch=0.0, roll=0.0)
        frame_data = self.create_frame_data([head_pose])
        self.classifier.classify_head_poses_in_frame(frame_data)
        assert frame_data.head_poses[0].direction == "looking_right"

    def test_classify_profile_directions(self):
        """Test classification of profile directions."""
        # Profile left
        head_pose = self.create_head_pose_result(yaw=90.0, pitch=0.0, roll=0.0)
        frame_data = self.create_frame_data([head_pose])
        self.classifier.classify_head_poses_in_frame(frame_data)
        assert frame_data.head_poses[0].direction == "profile_left"

        # Profile right
        head_pose = self.create_head_pose_result(yaw=-90.0, pitch=0.0, roll=0.0)
        frame_data = self.create_frame_data([head_pose])
        self.classifier.classify_head_poses_in_frame(frame_data)
        assert frame_data.head_poses[0].direction == "profile_right"

    def test_classify_looking_up_down(self):
        """Test classification of looking up/down directions."""
        # Looking up
        head_pose = self.create_head_pose_result(yaw=0.0, pitch=30.0, roll=0.0)
        frame_data = self.create_frame_data([head_pose])
        self.classifier.classify_head_poses_in_frame(frame_data)
        assert frame_data.head_poses[0].direction == "looking_up"

        # Looking down
        head_pose = self.create_head_pose_result(yaw=0.0, pitch=-30.0, roll=0.0)
        frame_data = self.create_frame_data([head_pose])
        self.classifier.classify_head_poses_in_frame(frame_data)
        assert frame_data.head_poses[0].direction == "looking_down"

    def test_classify_diagonal_directions(self):
        """Test classification of diagonal directions."""
        # Looking up left
        head_pose = self.create_head_pose_result(yaw=45.0, pitch=30.0, roll=0.0)
        frame_data = self.create_frame_data([head_pose])
        self.classifier.classify_head_poses_in_frame(frame_data)
        assert frame_data.head_poses[0].direction == "looking_up_left"

        # Looking up right
        head_pose = self.create_head_pose_result(yaw=-45.0, pitch=30.0, roll=0.0)
        frame_data = self.create_frame_data([head_pose])
        self.classifier.classify_head_poses_in_frame(frame_data)
        assert frame_data.head_poses[0].direction == "looking_up_right"

        # Looking down left
        head_pose = self.create_head_pose_result(yaw=45.0, pitch=-30.0, roll=0.0)
        frame_data = self.create_frame_data([head_pose])
        self.classifier.classify_head_poses_in_frame(frame_data)
        assert frame_data.head_poses[0].direction == "looking_down_left"

        # Looking down right
        head_pose = self.create_head_pose_result(yaw=-45.0, pitch=-30.0, roll=0.0)
        frame_data = self.create_frame_data([head_pose])
        self.classifier.classify_head_poses_in_frame(frame_data)
        assert frame_data.head_poses[0].direction == "looking_down_right"

    def test_classify_head_pose_with_confidence(self):
        """Test classification with confidence scoring."""
        head_pose = self.create_head_pose_result(yaw=10.0, pitch=5.0, roll=2.0, confidence=0.9)
        frame_data = self.create_frame_data([head_pose])
        self.classifier.classify_head_poses_in_frame(frame_data)

        result = frame_data.head_poses[0]
        assert result.direction == "front"
        assert 0.1 <= result.direction_confidence <= 1.0

    def test_classify_head_poses_batch(self):
        """Test batch classification of head poses."""
        head_poses = [
            self.create_head_pose_result(yaw=0.0, pitch=0.0, roll=0.0, confidence=0.9),
            self.create_head_pose_result(yaw=45.0, pitch=0.0, roll=0.0, confidence=0.8),
            self.create_head_pose_result(yaw=-90.0, pitch=0.0, roll=0.0, confidence=0.85),
        ]

        frame_data = self.create_frame_data(head_poses)
        self.classifier.classify_head_poses_in_frame(frame_data)

        results = frame_data.head_poses
        assert len(results) == 3
        assert results[0].direction == "front"
        assert results[1].direction == "looking_left"
        assert results[2].direction == "profile_right"
        assert all(0.1 <= r.direction_confidence <= 1.0 for r in results)

    def test_is_valid_orientation(self):
        """Test valid orientation checking."""
        # Valid roll angles
        assert self.classifier.is_valid_orientation(0.0) is True
        assert self.classifier.is_valid_orientation(15.0) is True
        assert self.classifier.is_valid_orientation(-25.0) is True

        # Invalid roll angles
        assert self.classifier.is_valid_orientation(45.0) is False
        assert self.classifier.is_valid_orientation(-60.0) is False

    def test_get_angle_ranges(self):
        """Test getting angle ranges for each direction."""
        ranges = self.classifier.get_angle_ranges()

        assert "front" in ranges
        assert "looking_left" in ranges
        assert "profile_right" in ranges

        # Check front range
        front_range = ranges["front"]
        assert "yaw" in front_range
        assert "pitch" in front_range
        assert front_range["yaw"] == (-22.5, 22.5)
        assert front_range["pitch"] == (-22.5, 22.5)

    def test_set_angle_thresholds(self):
        """Test setting custom angle thresholds."""
        self.classifier.set_angle_thresholds(
            yaw=25.0,
            pitch=20.0,
            profile_yaw=70.0,
            max_roll=25.0
        )

        assert self.classifier.yaw_threshold == 25.0
        assert self.classifier.pitch_threshold == 20.0
        assert self.classifier.profile_yaw_threshold == 70.0
        assert self.classifier.max_roll == 25.0

    def test_get_classification_info(self):
        """Test getting classification configuration info."""
        info = self.classifier.get_classification_info()

        assert 'angle_thresholds' in info
        assert 'supported_directions' in info
        assert 'confidence_weights' in info
        assert 'angle_ranges' in info

        assert len(info['supported_directions']) == 9
        assert 'front' in info['supported_directions']
        assert 'profile_left' in info['supported_directions']

    def test_validate_direction(self):
        """Test direction validation."""
        # Valid directions
        assert self.classifier.validate_direction("front") is True
        assert self.classifier.validate_direction("looking_left") is True
        assert self.classifier.validate_direction("profile_right") is True

        # Invalid directions
        assert self.classifier.validate_direction("invalid") is False
        assert self.classifier.validate_direction("backward") is False

    def test_extreme_angle_values(self):
        """Test handling of extreme angle values."""
        # Test with extreme but valid angles
        head_pose = self.create_head_pose_result(yaw=179.0, pitch=89.0, roll=179.0)
        frame_data = self.create_frame_data([head_pose])
        self.classifier.classify_head_poses_in_frame(frame_data)
        assert frame_data.head_poses[0].direction == "profile_left"

        # Test with angles that should trigger warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            head_pose = self.create_head_pose_result(yaw=200.0, pitch=100.0, roll=200.0)
            # The warning is raised in HeadPoseResult.__post_init__
            frame_data = self.create_frame_data([head_pose])
            self.classifier.classify_head_poses_in_frame(frame_data)
            assert len(w) > 0
            assert "Extreme angle values" in str(w[0].message)

    def test_edge_case_thresholds(self):
        """Test edge cases at threshold boundaries."""
        # Test exact threshold values
        head_pose = self.create_head_pose_result(yaw=22.5, pitch=0.0, roll=0.0)
        frame_data = self.create_frame_data([head_pose])
        self.classifier.classify_head_poses_in_frame(frame_data)
        assert frame_data.head_poses[0].direction == "front"

        head_pose = self.create_head_pose_result(yaw=22.6, pitch=0.0, roll=0.0)
        frame_data = self.create_frame_data([head_pose])
        self.classifier.classify_head_poses_in_frame(frame_data)
        assert frame_data.head_poses[0].direction == "looking_left"

        head_pose = self.create_head_pose_result(yaw=67.4, pitch=0.0, roll=0.0)
        frame_data = self.create_frame_data([head_pose])
        self.classifier.classify_head_poses_in_frame(frame_data)
        assert frame_data.head_poses[0].direction == "looking_left"

        head_pose = self.create_head_pose_result(yaw=67.6, pitch=0.0, roll=0.0)
        frame_data = self.create_frame_data([head_pose])
        self.classifier.classify_head_poses_in_frame(frame_data)
        assert frame_data.head_poses[0].direction == "profile_left"

    def test_invalid_angles(self):
        """Test handling of invalid angle values."""
        # Test NaN values
        head_pose = self.create_head_pose_result(yaw=float('nan'), pitch=0.0, roll=0.0)
        frame_data = self.create_frame_data([head_pose])
        # Should be handled gracefully and assigned a default direction
        self.classifier.classify_head_poses_in_frame(frame_data)
        result = frame_data.head_poses[0]
        assert result.direction == "front"
        assert result.direction_confidence == 0.1

        # Test infinite values
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            head_pose = self.create_head_pose_result(yaw=float('inf'), pitch=0.0, roll=0.0)
            frame_data = self.create_frame_data([head_pose])
            self.classifier.classify_head_poses_in_frame(frame_data)
            result = frame_data.head_poses[0]
            assert len(w) > 0
            assert "Extreme angle values" in str(w[0].message)
            assert result.direction in ["profile_left", "profile_right"]

    def test_confidence_calculation(self):
        """Test confidence calculation for different directions."""
        # Test high-confidence front pose
        head_pose = self.create_head_pose_result(yaw=0.0, pitch=0.0, roll=0.0, confidence=0.95)
        frame_data = self.create_frame_data([head_pose])
        self.classifier.classify_head_poses_in_frame(frame_data)
        result = frame_data.head_poses[0]
        assert result.direction == "front"
        assert result.direction_confidence > 0.7  # Should be high confidence

        # Test lower confidence for extreme angles
        head_pose = self.create_head_pose_result(yaw=80.0, pitch=40.0, roll=25.0, confidence=0.6)
        frame_data = self.create_frame_data([head_pose])
        self.classifier.classify_head_poses_in_frame(frame_data)
        result = frame_data.head_poses[0]
        assert result.direction in ["profile_left", "profile_right"]
        assert result.direction_confidence < 0.8  # Should be lower confidence

    def test_roll_penalty(self):
        """Test roll angle penalty in confidence calculation."""
        # Test normal roll angle
        head_pose_normal = self.create_head_pose_result(yaw=0.0, pitch=0.0, roll=5.0, confidence=0.9)
        frame_data_normal = self.create_frame_data([head_pose_normal])
        self.classifier.classify_head_poses_in_frame(frame_data_normal)

        # Test high roll angle
        head_pose_tilted = self.create_head_pose_result(yaw=0.0, pitch=0.0, roll=45.0, confidence=0.9)
        frame_data_tilted = self.create_frame_data([head_pose_tilted])
        self.classifier.classify_head_poses_in_frame(frame_data_tilted)

        result_normal = frame_data_normal.head_poses[0]
        result_tilted = frame_data_tilted.head_poses[0]

        # Tilted head should have lower confidence
        assert result_tilted.direction == result_normal.direction
        assert result_tilted.direction_confidence < result_normal.direction_confidence

    def test_error_recovery_in_batch(self):
        """Test error recovery in batch processing."""
        # Mix valid and invalid head pose results
        head_poses = [
            self.create_head_pose_result(yaw=0.0, pitch=0.0, roll=0.0, confidence=0.9),
            Mock(),  # Invalid head pose
        ]

        # Mock the invalid pose to raise an exception
        head_poses[1].yaw = float('nan')
        head_poses[1].pitch = 0.0
        head_poses[1].roll = 0.0
        head_poses[1].confidence = 0.8

        frame_data = self.create_frame_data(head_poses)
        self.classifier.classify_head_poses_in_frame(frame_data)

        results = frame_data.head_poses
        assert len(results) == 2
        assert results[0].direction == "front"
        assert results[1].direction == "front"  # Default for failed pose
        assert results[1].direction_confidence == 0.1  # Low confidence for failed pose


class TestHeadAngleClassifierIntegration:
    """Integration tests for HeadAngleClassifier."""

    def test_realistic_head_poses(self):
        """Test classification of realistic head pose scenarios."""
        classifier = HeadAngleClassifier()

        # Realistic conversation poses
        test_cases = [
            # (yaw, pitch, roll, expected_direction)
            (2.5, -3.1, 1.2, "front"),
            (35.2, -8.4, 5.6, "looking_left"),
            (-28.7, 12.3, -2.1, "looking_right"),
            (85.6, -15.2, 8.9, "profile_left"),
            (-92.1, 5.4, -12.3, "profile_right"),
            (12.4, 28.7, 3.2, "looking_up"),
            (-8.9, -32.1, -1.8, "looking_down"),
            (42.3, 25.6, 6.7, "looking_up_left"),
            (-38.4, 35.2, -4.5, "looking_up_right"),
        ]

        for yaw, pitch, roll, expected in test_cases:
            head_pose = HeadPoseResult(yaw=yaw, pitch=pitch, roll=roll, confidence=0.9)
            frame_data = FrameData(
                frame_id="test",
                file_path=Path("test.jpg"),
                source_info=SourceInfo(video_timestamp=0, extraction_method="test", original_frame_number=0, video_fps=30),
                image_properties=ImageProperties(width=1920, height=1080, channels=3, file_size_bytes=0, format="jpg"),
                head_poses=[head_pose]
            )
            classifier.classify_head_poses_in_frame(frame_data)
            assert frame_data.head_poses[0].direction == expected

    def test_borderline_cases(self):
        """Test classification of borderline angle cases."""
        classifier = HeadAngleClassifier()

        # Test angles very close to thresholds
        # Should be front (just within threshold)
        head_pose = HeadPoseResult(yaw=22.4, pitch=22.4, roll=0.0, confidence=0.9)
        frame_data = FrameData(
            frame_id="test",
            file_path=Path("test.jpg"),
            source_info=SourceInfo(video_timestamp=0, extraction_method="test", original_frame_number=0, video_fps=30),
            image_properties=ImageProperties(width=1920, height=1080, channels=3, file_size_bytes=0, format="jpg"),
            head_poses=[head_pose]
        )
        classifier.classify_head_poses_in_frame(frame_data)
        assert frame_data.head_poses[0].direction == "front"

        # Should be looking left (just over threshold)
        head_pose = HeadPoseResult(yaw=22.6, pitch=0.0, roll=0.0, confidence=0.9)
        frame_data = FrameData(
            frame_id="test",
            file_path=Path("test.jpg"),
            source_info=SourceInfo(video_timestamp=0, extraction_method="test", original_frame_number=0, video_fps=30),
            image_properties=ImageProperties(width=1920, height=1080, channels=3, file_size_bytes=0, format="jpg"),
            head_poses=[head_pose]
        )
        classifier.classify_head_poses_in_frame(frame_data)
        assert frame_data.head_poses[0].direction == "looking_left"

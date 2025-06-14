# Head angle classifier tests

"""Unit tests for head angle classifier module."""

import pytest
import numpy as np
from unittest.mock import Mock
import warnings

from personfromvid.analysis.head_angle_classifier import HeadAngleClassifier
from personfromvid.data.detection_results import HeadPoseResult
from personfromvid.utils.exceptions import HeadAngleClassificationError


class TestHeadAngleClassifier:
    """Tests for HeadAngleClassifier class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.classifier = HeadAngleClassifier()
    
    def test_init_default_thresholds(self):
        """Test HeadAngleClassifier initialization with default thresholds."""
        classifier = HeadAngleClassifier()
        
        assert classifier.yaw_threshold == 22.5
        assert classifier.pitch_threshold == 22.5
        assert classifier.profile_yaw_threshold == 67.5
        assert classifier.max_roll == 30.0
    
    def test_classify_front_direction(self):
        """Test classification of front direction."""
        direction = self.classifier.classify_head_angle(yaw=0.0, pitch=0.0, roll=0.0)
        assert direction == "front"
        
        # Test within front threshold
        direction = self.classifier.classify_head_angle(yaw=15.0, pitch=-10.0, roll=5.0)
        assert direction == "front"
    
    def test_classify_looking_left_right(self):
        """Test classification of looking left/right directions."""
        # Looking left
        direction = self.classifier.classify_head_angle(yaw=45.0, pitch=0.0, roll=0.0)
        assert direction == "looking_left"
        
        # Looking right
        direction = self.classifier.classify_head_angle(yaw=-45.0, pitch=0.0, roll=0.0)
        assert direction == "looking_right"
    
    def test_classify_profile_directions(self):
        """Test classification of profile directions."""
        # Profile left
        direction = self.classifier.classify_head_angle(yaw=90.0, pitch=0.0, roll=0.0)
        assert direction == "profile_left"
        
        # Profile right
        direction = self.classifier.classify_head_angle(yaw=-90.0, pitch=0.0, roll=0.0)
        assert direction == "profile_right"
    
    def test_classify_looking_up_down(self):
        """Test classification of looking up/down directions."""
        # Looking up
        direction = self.classifier.classify_head_angle(yaw=0.0, pitch=30.0, roll=0.0)
        assert direction == "looking_up"
        
        # Looking down
        direction = self.classifier.classify_head_angle(yaw=0.0, pitch=-30.0, roll=0.0)
        assert direction == "looking_down"
    
    def test_classify_diagonal_directions(self):
        """Test classification of diagonal directions."""
        # Looking up left
        direction = self.classifier.classify_head_angle(yaw=45.0, pitch=30.0, roll=0.0)
        assert direction == "looking_up_left"
        
        # Looking up right
        direction = self.classifier.classify_head_angle(yaw=-45.0, pitch=30.0, roll=0.0)
        assert direction == "looking_up_right"
        
        # Looking down left
        direction = self.classifier.classify_head_angle(yaw=45.0, pitch=-30.0, roll=0.0)
        assert direction == "looking_down_left"
        
        # Looking down right
        direction = self.classifier.classify_head_angle(yaw=-45.0, pitch=-30.0, roll=0.0)
        assert direction == "looking_down_right"
    
    def test_classify_head_pose_with_confidence(self):
        """Test classification with confidence scoring."""
        head_pose = HeadPoseResult(
            yaw=10.0,
            pitch=5.0,
            roll=2.0,
            confidence=0.9,
            direction="front"  # This will be overridden by classifier
        )
        
        direction, confidence = self.classifier.classify_head_pose(head_pose)
        
        assert direction == "front"
        assert 0.1 <= confidence <= 1.0
    
    def test_classify_head_poses_batch(self):
        """Test batch classification of head poses."""
        head_poses = [
            HeadPoseResult(yaw=0.0, pitch=0.0, roll=0.0, confidence=0.9, direction="front"),
            HeadPoseResult(yaw=45.0, pitch=0.0, roll=0.0, confidence=0.8, direction="looking_left"),
            HeadPoseResult(yaw=-90.0, pitch=0.0, roll=0.0, confidence=0.85, direction="profile_right"),
        ]
        
        results = self.classifier.classify_head_poses(head_poses)
        
        assert len(results) == 3
        assert results[0][0] == "front"
        assert results[1][0] == "looking_left"
        assert results[2][0] == "profile_right"
        assert all(0.1 <= conf <= 1.0 for _, conf in results)
    
    def test_is_valid_orientation(self):
        """Test valid orientation checking."""
        # Valid roll angles
        assert self.classifier.is_valid_orientation(0.0) == True
        assert self.classifier.is_valid_orientation(15.0) == True
        assert self.classifier.is_valid_orientation(-25.0) == True
        
        # Invalid roll angles
        assert self.classifier.is_valid_orientation(45.0) == False
        assert self.classifier.is_valid_orientation(-60.0) == False
    
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
        assert self.classifier.validate_direction("front") == True
        assert self.classifier.validate_direction("looking_left") == True
        assert self.classifier.validate_direction("profile_right") == True
        
        # Invalid directions
        assert self.classifier.validate_direction("invalid") == False
        assert self.classifier.validate_direction("backward") == False
    
    def test_extreme_angle_values(self):
        """Test handling of extreme angle values."""
        # Test with extreme but valid angles
        direction = self.classifier.classify_head_angle(yaw=179.0, pitch=89.0, roll=179.0)
        assert direction == "profile_left"
        
        # Test with angles that should trigger warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            direction = self.classifier.classify_head_angle(yaw=200.0, pitch=100.0, roll=200.0)
            assert len(w) > 0
            assert "Extreme angle values" in str(w[0].message)
    
    def test_edge_case_thresholds(self):
        """Test edge cases at threshold boundaries."""
        # Test exact threshold values
        direction = self.classifier.classify_head_angle(yaw=22.5, pitch=0.0, roll=0.0)
        assert direction == "front"
        
        direction = self.classifier.classify_head_angle(yaw=22.6, pitch=0.0, roll=0.0)
        assert direction == "looking_left"
        
        direction = self.classifier.classify_head_angle(yaw=67.4, pitch=0.0, roll=0.0)
        assert direction == "looking_left"
        
        direction = self.classifier.classify_head_angle(yaw=67.6, pitch=0.0, roll=0.0)
        assert direction == "profile_left"
    
    def test_invalid_angles(self):
        """Test handling of invalid angle values."""
        # Test NaN values
        with pytest.raises(HeadAngleClassificationError):
            self.classifier.classify_head_angle(yaw=float('nan'), pitch=0.0, roll=0.0)
        
        # Test infinite values (should just issue warnings, not raise exceptions)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            direction = self.classifier.classify_head_angle(yaw=float('inf'), pitch=0.0, roll=0.0)
            assert len(w) == 1
            assert "Extreme angle values" in str(w[0].message)
            assert direction in ["profile_left", "profile_right"]  # Infinite yaw should be profile
    
    def test_confidence_calculation(self):
        """Test confidence calculation for different directions."""
        # Test high-confidence front pose
        head_pose = HeadPoseResult(yaw=0.0, pitch=0.0, roll=0.0, confidence=0.95, direction="front")
        direction, confidence = self.classifier.classify_head_pose(head_pose)
        assert confidence > 0.7  # Should be high confidence
        
        # Test lower confidence for extreme angles
        head_pose = HeadPoseResult(yaw=80.0, pitch=40.0, roll=25.0, confidence=0.6, direction="profile_left")
        direction, confidence = self.classifier.classify_head_pose(head_pose)
        assert confidence < 0.8  # Should be lower confidence
    
    def test_roll_penalty(self):
        """Test roll angle penalty in confidence calculation."""
        # Test normal roll angle
        head_pose_normal = HeadPoseResult(yaw=0.0, pitch=0.0, roll=5.0, confidence=0.9, direction="front")
        _, confidence_normal = self.classifier.classify_head_pose(head_pose_normal)
        
        # Test high roll angle
        head_pose_tilted = HeadPoseResult(yaw=0.0, pitch=0.0, roll=45.0, confidence=0.9, direction="front")
        _, confidence_tilted = self.classifier.classify_head_pose(head_pose_tilted)
        
        # Tilted head should have lower confidence
        assert confidence_tilted < confidence_normal
    
    def test_error_recovery_in_batch(self):
        """Test error recovery in batch processing."""
        # Mix valid and invalid head pose results
        head_poses = [
            HeadPoseResult(yaw=0.0, pitch=0.0, roll=0.0, confidence=0.9, direction="front"),
            Mock(),  # Invalid head pose
        ]
        
        # Mock the invalid pose to raise an exception
        head_poses[1].yaw = float('nan')
        head_poses[1].pitch = 0.0
        head_poses[1].roll = 0.0
        head_poses[1].confidence = 0.8
        
        results = self.classifier.classify_head_poses(head_poses)
        
        # Should return results for all poses, with defaults for failed ones
        assert len(results) == 2
        assert results[0][0] == "front"
        assert results[1][0] == "front"  # Default for failed pose
        assert results[1][1] == 0.1  # Low confidence for failed pose


class TestHeadAngleClassifierIntegration:
    """Integration tests for HeadAngleClassifier with realistic scenarios."""
    
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
            direction = classifier.classify_head_angle(yaw, pitch, roll)
            assert direction == expected, f"Failed for angles ({yaw}, {pitch}, {roll}): expected {expected}, got {direction}"
    
    def test_borderline_cases(self):
        """Test classification of borderline angle cases."""
        classifier = HeadAngleClassifier()
        
        # Test angles very close to thresholds
        # Should be front (just within threshold)
        direction = classifier.classify_head_angle(yaw=22.4, pitch=22.4, roll=0.0)
        assert direction == "front"
        
        # Should be looking_left (just outside threshold)
        direction = classifier.classify_head_angle(yaw=22.6, pitch=0.0, roll=0.0)
        assert direction == "looking_left"

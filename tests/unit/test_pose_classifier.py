# Pose classifier tests

"""Unit tests for pose classifier module."""

import pytest
import numpy as np
from unittest.mock import Mock

from personfromvid.analysis.pose_classifier import PoseClassifier
from personfromvid.data.detection_results import PoseDetection
from personfromvid.utils.exceptions import PoseClassificationError


class TestPoseClassifier:
    """Tests for PoseClassifier class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.classifier = PoseClassifier()
        self.image_shape = (1080, 1920)  # height, width
    
    def test_init_default_thresholds(self):
        """Test PoseClassifier initialization with default thresholds."""
        classifier = PoseClassifier()
        
        assert classifier.standing_hip_knee_min == 160.0
        assert classifier.sitting_hip_knee_min == 80.0
        assert classifier.sitting_hip_knee_max == 120.0
        assert classifier.squatting_hip_knee_max == 90.0
        assert classifier.closeup_face_ratio_threshold == 0.15
        assert classifier.min_keypoint_confidence == 0.5
    
    def test_classify_standing_pose(self):
        """Test classification of standing pose."""
        # Create mock pose detection with standing keypoints (straight legs)
        keypoints = {
            'left_hip': (500.0, 600.0, 0.9),
            'right_hip': (600.0, 600.0, 0.9),
            'left_knee': (500.0, 800.0, 0.9),      # Straight down from hip
            'right_knee': (600.0, 800.0, 0.9),     # Straight down from hip
            'left_ankle': (500.0, 1000.0, 0.9),    # Straight down from knee
            'right_ankle': (600.0, 1000.0, 0.9),   # Straight down from knee
            'left_shoulder': (480.0, 400.0, 0.9),
            'right_shoulder': (620.0, 400.0, 0.9),
        }
        
        pose_detection = PoseDetection(
            bbox=(400, 300, 700, 1050),
            confidence=0.85,
            keypoints=keypoints
        )
        
        classifications = self.classifier.classify_pose(pose_detection, self.image_shape)
        
        assert isinstance(classifications, list)
        assert len(classifications) >= 1
        
        # Check that 'standing' is one of the classifications
        found_standing = any(c[0] == "standing" for c in classifications)
        assert found_standing, "Standing classification not found"
        
        # Check confidence of standing classification
        standing_confidence = next((c[1] for c in classifications if c[0] == "standing"), None)
        assert standing_confidence is not None
        assert 0.3 <= standing_confidence <= 1.0
    
    def test_classify_sitting_pose(self):
        """Test classification of sitting pose."""
        # Create mock pose detection with sitting keypoints (bent knees ~90° joint angle)
        keypoints = {
            'left_hip': (500.0, 600.0, 0.9),
            'right_hip': (600.0, 600.0, 0.9),
            'left_knee': (500.0, 700.0, 0.9),   # Knee directly below hip
            'right_knee': (600.0, 700.0, 0.9),  # Knee directly below hip
            'left_ankle': (400.0, 700.0, 0.9),  # Ankle to the left of knee (90° joint angle)
            'right_ankle': (700.0, 700.0, 0.9), # Ankle to the right of knee (90° joint angle)
            'left_shoulder': (480.0, 400.0, 0.9),
            'right_shoulder': (620.0, 400.0, 0.9),
        }
        
        pose_detection = PoseDetection(
            bbox=(400, 300, 700, 750),
            confidence=0.85,
            keypoints=keypoints
        )
        
        classifications = self.classifier.classify_pose(pose_detection, self.image_shape)
        
        assert isinstance(classifications, list)
        # May also be classified as squatting, so check for sitting
        found_sitting = any(c[0] == "sitting" for c in classifications)
        assert found_sitting, "Sitting classification not found"
        
        sitting_confidence = next((c[1] for c in classifications if c[0] == "sitting"), None)
        assert sitting_confidence is not None
        assert 0.4 <= sitting_confidence <= 1.0
    
    def test_classify_closeup_by_face_ratio(self):
        """Test closeup classification based on face area ratio."""
        # Large bounding box suggesting closeup shot
        keypoints = {
            'left_shoulder': (200.0, 400.0, 0.9),
            'right_shoulder': (800.0, 400.0, 0.9),  # Wide shoulders = closeup
            'nose': (500.0, 300.0, 0.9),
        }
        
        pose_detection = PoseDetection(
            bbox=(100, 200, 900, 800),  # Large bbox
            confidence=0.85,
            keypoints=keypoints
        )
        
        classifications = self.classifier.classify_pose(pose_detection, self.image_shape)
        
        assert isinstance(classifications, list)
        found_closeup = any(c[0] == "closeup" for c in classifications)
        assert found_closeup, "Closeup classification not found"
        
        closeup_confidence = next((c[1] for c in classifications if c[0] == "closeup"), None)
        assert closeup_confidence is not None
        assert 0.3 <= closeup_confidence <= 1.0
    
    def test_classify_multiple_poses(self):
        """Test classification of a pose that meets multiple criteria (e.g., standing and closeup)."""
        keypoints = {
            'left_hip': (500.0, 600.0, 0.9),
            'right_hip': (600.0, 600.0, 0.9),
            'left_knee': (500.0, 800.0, 0.9),
            'right_knee': (600.0, 800.0, 0.9),
            'left_ankle': (500.0, 1000.0, 0.9),
            'right_ankle': (600.0, 1000.0, 0.9),
            'left_shoulder': (200.0, 400.0, 0.9),
            'right_shoulder': (800.0, 400.0, 0.9),  # Wide shoulders for closeup
            'nose': (500.0, 300.0, 0.9),
        }
        
        pose_detection = PoseDetection(
            bbox=(100, 200, 900, 1050),  # Large bbox for closeup
            confidence=0.9,
            keypoints=keypoints
        )
        
        classifications = self.classifier.classify_pose(pose_detection, self.image_shape)
        
        assert len(classifications) >= 2
        
        class_names = [c[0] for c in classifications]
        assert "standing" in class_names
        assert "closeup" in class_names
    
    def test_hip_knee_angle_calculation(self):
        """Test hip-knee angle calculation."""
        # Test perfect standing pose (straight leg = 180 degree joint angle)
        keypoints = {
            'left_hip': (500.0, 600.0, 0.9),
            'left_knee': (500.0, 800.0, 0.9),    # Straight down from hip
            'left_ankle': (500.0, 1000.0, 0.9),  # Straight down from knee
        }
        
        angle = self.classifier._calculate_hip_knee_angle(keypoints, 'left')
        
        # Should be close to 180 degrees (straight leg = extended joint)
        assert 170 <= angle <= 180
    
    def test_set_angle_thresholds(self):
        """Test setting custom angle thresholds."""
        self.classifier.set_angle_thresholds(
            standing_min=165.0,
            sitting_max=115.0,
            squatting_max=85.0
        )
        
        assert self.classifier.standing_hip_knee_min == 165.0
        assert self.classifier.sitting_hip_knee_max == 115.0
        assert self.classifier.squatting_hip_knee_max == 85.0

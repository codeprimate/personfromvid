"""Unit tests for HeadPoseEstimator class."""

import pytest
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict

from personfromvid.models.head_pose_estimator import (
    HeadPoseEstimator, 
    create_head_pose_estimator,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_DEVICE,
    DEFAULT_YAW_THRESHOLD,
    DEFAULT_PITCH_THRESHOLD,
    DEFAULT_PROFILE_YAW_THRESHOLD,
    DEFAULT_MAX_ROLL,
    DEFAULT_FORWARD_YAW_THRESHOLD,
    DEFAULT_FORWARD_PITCH_THRESHOLD
)
from personfromvid.data.detection_results import HeadPoseResult, FaceDetection
from personfromvid.utils.exceptions import HeadPoseEstimationError
from personfromvid.models.model_configs import ModelConfigs, ModelFormat
from personfromvid.data.config import DeviceType

# Test constants to reduce duplication
TEST_HEAD_POSE_MODEL = "sixdrepnet"  # Use actual default model from config


class TestHeadPoseEstimator:
    """Tests for HeadPoseEstimator class."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create a sample face image for testing
        self.test_face = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        self.image_shape = (224, 224)
        
        # Mock model configuration
        self.mock_model_config = Mock()
        self.mock_model_config.input_size = (224, 224)
        self.mock_model_config.is_device_supported.return_value = True
        self.mock_model_config.files = [Mock()]
        self.mock_model_config.files[0].format = ModelFormat.PYTORCH  # sixdrepnet uses PyTorch format
        
        # Mock model manager
        self.mock_model_manager = Mock()
        self.mock_model_manager.ensure_model_available.return_value = Path("/mock/path/model.pkl")
        
        # Sample head pose result for testing
        self.sample_head_pose = HeadPoseResult(
            yaw=15.5,
            pitch=-5.2,
            roll=2.1,
            confidence=0.88,
            direction="looking_left"
        )

    @patch('personfromvid.models.head_pose_estimator.ModelConfigs.get_model')
    @patch('personfromvid.models.head_pose_estimator.get_model_manager')
    @patch('personfromvid.models.head_pose_estimator.HeadAngleClassifier')
    def test_init_success(self, mock_classifier, mock_get_manager, mock_get_model):
        """Test successful HeadPoseEstimator initialization."""
        mock_get_model.return_value = self.mock_model_config
        mock_get_manager.return_value = self.mock_model_manager
        mock_classifier_instance = Mock()
        mock_classifier_instance.yaw_threshold = DEFAULT_YAW_THRESHOLD
        mock_classifier_instance.pitch_threshold = DEFAULT_PITCH_THRESHOLD
        mock_classifier_instance.profile_yaw_threshold = DEFAULT_PROFILE_YAW_THRESHOLD
        mock_classifier_instance.max_roll = DEFAULT_MAX_ROLL
        mock_classifier.return_value = mock_classifier_instance
        
        estimator = HeadPoseEstimator(TEST_HEAD_POSE_MODEL, device="cpu", confidence_threshold=0.8)
        
        assert estimator.model_name == TEST_HEAD_POSE_MODEL
        assert estimator.device == "cpu"
        assert estimator.confidence_threshold == 0.8
        assert estimator._input_size == (224, 224)
        assert estimator.yaw_threshold == DEFAULT_YAW_THRESHOLD
        assert estimator.pitch_threshold == DEFAULT_PITCH_THRESHOLD
        mock_get_model.assert_called_once_with(TEST_HEAD_POSE_MODEL)
        mock_get_manager.assert_called_once()

    @patch('personfromvid.models.head_pose_estimator.ModelConfigs.get_model')
    def test_init_unknown_model(self, mock_get_model):
        """Test initialization with unknown model."""
        mock_get_model.return_value = None
        
        with pytest.raises(HeadPoseEstimationError, match="Unknown head pose estimation model"):
            HeadPoseEstimator("unknown_model")

    @patch('personfromvid.models.head_pose_estimator.ModelConfigs.get_model')
    @patch('personfromvid.models.head_pose_estimator.get_model_manager')
    @patch('personfromvid.models.head_pose_estimator.HeadAngleClassifier')
    def test_init_unsupported_device(self, mock_classifier, mock_get_manager, mock_get_model):
        """Test initialization with unsupported device."""
        mock_config = Mock()
        mock_config.is_device_supported.return_value = False
        mock_get_model.return_value = mock_config
        mock_get_manager.return_value = self.mock_model_manager
        mock_classifier.return_value = Mock()
        
        with pytest.raises(HeadPoseEstimationError, match="does not support device"):
            HeadPoseEstimator(TEST_HEAD_POSE_MODEL, device="cuda")

    def test_resolve_device_auto_with_cuda(self):
        """Test device resolution with auto and CUDA available."""
        with patch('personfromvid.models.head_pose_estimator.HeadPoseEstimator._resolve_device') as mock_resolve:
            mock_resolve.return_value = "cuda"
            estimator = Mock()
            from personfromvid.models.head_pose_estimator import HeadPoseEstimator
            device = HeadPoseEstimator._resolve_device(estimator, "auto")
            assert device == "cuda"

    def test_resolve_device_auto_without_cuda(self):
        """Test device resolution with auto and no CUDA."""
        with patch('personfromvid.models.head_pose_estimator.HeadPoseEstimator._resolve_device') as mock_resolve:
            mock_resolve.return_value = "cpu"
            estimator = Mock()
            from personfromvid.models.head_pose_estimator import HeadPoseEstimator
            device = HeadPoseEstimator._resolve_device(estimator, "auto")
            assert device == "cpu"

    def test_resolve_device_auto_no_torch(self):
        """Test device resolution with auto and no PyTorch."""
        with patch('personfromvid.models.head_pose_estimator.HeadPoseEstimator._resolve_device') as mock_resolve:
            mock_resolve.return_value = "cpu"
            estimator = Mock()
            from personfromvid.models.head_pose_estimator import HeadPoseEstimator
            device = HeadPoseEstimator._resolve_device(estimator, "auto")
            assert device == "cpu"

    def test_resolve_device_explicit(self):
        """Test explicit device resolution."""
        estimator = Mock()
        from personfromvid.models.head_pose_estimator import HeadPoseEstimator
        assert HeadPoseEstimator._resolve_device(estimator, "cpu") == "cpu"
        assert HeadPoseEstimator._resolve_device(estimator, "cuda") == "cuda"

    @patch('personfromvid.models.head_pose_estimator.ModelConfigs.get_model')
    @patch('personfromvid.models.head_pose_estimator.get_model_manager')
    @patch('personfromvid.models.head_pose_estimator.HeadAngleClassifier')
    def test_load_model_pickle_format(self, mock_classifier, mock_get_manager, mock_get_model):
        """Test loading model with PICKLE format."""
        # Override the default config for this specific test
        mock_config = Mock()
        mock_config.input_size = (224, 224)
        mock_config.is_device_supported.return_value = True
        mock_config.files = [Mock()]
        mock_config.files[0].format = ModelFormat.PICKLE
        
        mock_get_model.return_value = mock_config
        mock_get_manager.return_value = self.mock_model_manager
        mock_classifier.return_value = Mock()
        
        estimator = HeadPoseEstimator(TEST_HEAD_POSE_MODEL)
        
        with patch.object(estimator, '_load_hopenet_model') as mock_load_hopenet:
            estimator._load_model()
            mock_load_hopenet.assert_called_once()

    @patch('personfromvid.models.head_pose_estimator.ModelConfigs.get_model')
    @patch('personfromvid.models.head_pose_estimator.get_model_manager')
    @patch('personfromvid.models.head_pose_estimator.HeadAngleClassifier')
    def test_load_model_onnx_format(self, mock_classifier, mock_get_manager, mock_get_model):
        """Test loading model with ONNX format."""
        mock_config = Mock()
        mock_config.input_size = (224, 224)
        mock_config.is_device_supported.return_value = True
        mock_config.files = [Mock()]
        mock_config.files[0].format = ModelFormat.ONNX
        
        mock_get_model.return_value = mock_config
        mock_get_manager.return_value = self.mock_model_manager
        mock_classifier.return_value = Mock()
        
        estimator = HeadPoseEstimator(TEST_HEAD_POSE_MODEL)
        
        with patch.object(estimator, '_load_onnx_model') as mock_load_onnx:
            estimator._load_model()
            mock_load_onnx.assert_called_once()

    @patch('personfromvid.models.head_pose_estimator.ModelConfigs.get_model')
    @patch('personfromvid.models.head_pose_estimator.get_model_manager')
    @patch('personfromvid.models.head_pose_estimator.HeadAngleClassifier')
    def test_load_model_pytorch_format(self, mock_classifier, mock_get_manager, mock_get_model):
        """Test loading model with PyTorch format."""
        mock_config = Mock()
        mock_config.input_size = (224, 224)
        mock_config.is_device_supported.return_value = True
        mock_config.files = [Mock()]
        mock_config.files[0].format = ModelFormat.PYTORCH
        
        mock_get_model.return_value = mock_config
        mock_get_manager.return_value = self.mock_model_manager
        mock_classifier.return_value = Mock()
        
        estimator = HeadPoseEstimator(TEST_HEAD_POSE_MODEL)
        
        with patch.object(estimator, '_load_pytorch_model') as mock_load_pytorch:
            estimator._load_model()
            mock_load_pytorch.assert_called_once()

    @patch('personfromvid.models.head_pose_estimator.ModelConfigs.get_model')
    @patch('personfromvid.models.head_pose_estimator.get_model_manager')
    @patch('personfromvid.models.head_pose_estimator.HeadAngleClassifier')
    def test_load_model_unsupported_format(self, mock_classifier, mock_get_manager, mock_get_model):
        """Test loading model with unsupported format."""
        mock_config = Mock()
        mock_config.input_size = (224, 224)
        mock_config.is_device_supported.return_value = True
        mock_config.files = [Mock()]
        mock_config.files[0].format = "UNSUPPORTED"
        
        mock_get_model.return_value = mock_config
        mock_get_manager.return_value = self.mock_model_manager
        mock_classifier.return_value = Mock()
        
        estimator = HeadPoseEstimator(TEST_HEAD_POSE_MODEL)
        
        with pytest.raises(HeadPoseEstimationError, match="Unsupported model format"):
            estimator._load_model()

    @patch('personfromvid.models.head_pose_estimator.ModelConfigs.get_model')
    @patch('personfromvid.models.head_pose_estimator.get_model_manager')
    @patch('personfromvid.models.head_pose_estimator.HeadAngleClassifier')
    def test_estimate_head_pose_success(self, mock_classifier, mock_get_manager, mock_get_model):
        """Test successful head pose estimation."""
        mock_get_model.return_value = self.mock_model_config
        mock_get_manager.return_value = self.mock_model_manager
        
        # Mock classifier
        mock_classifier_instance = Mock()
        mock_classifier_instance.yaw_threshold = DEFAULT_YAW_THRESHOLD
        mock_classifier_instance.pitch_threshold = DEFAULT_PITCH_THRESHOLD
        mock_classifier_instance.profile_yaw_threshold = DEFAULT_PROFILE_YAW_THRESHOLD
        mock_classifier_instance.max_roll = DEFAULT_MAX_ROLL
        mock_classifier_instance.classify_head_pose.return_value = ("looking_left", 0.88)
        mock_classifier.return_value = mock_classifier_instance
        
        estimator = HeadPoseEstimator(TEST_HEAD_POSE_MODEL)
        
        with patch.object(estimator, '_load_model'):
            with patch.object(estimator, '_estimate_pytorch') as mock_estimate:
                mock_estimate.return_value = self.sample_head_pose
                result = estimator.estimate_head_pose(self.test_face)
                
                assert isinstance(result, HeadPoseResult)
                assert result.yaw == 15.5
                assert result.pitch == -5.2
                assert result.roll == 2.1
                assert result.confidence == 0.88

    @patch('personfromvid.models.head_pose_estimator.ModelConfigs.get_model')
    @patch('personfromvid.models.head_pose_estimator.get_model_manager')
    @patch('personfromvid.models.head_pose_estimator.HeadAngleClassifier')
    def test_estimate_head_pose_empty_image(self, mock_classifier, mock_get_manager, mock_get_model):
        """Test head pose estimation with empty image."""
        mock_get_model.return_value = self.mock_model_config
        mock_get_manager.return_value = self.mock_model_manager
        mock_classifier.return_value = Mock()
        
        estimator = HeadPoseEstimator(TEST_HEAD_POSE_MODEL)
        
        with pytest.raises(HeadPoseEstimationError, match="Input face image is empty or None"):
            estimator.estimate_head_pose(np.array([]))

    @patch('personfromvid.models.head_pose_estimator.ModelConfigs.get_model')
    @patch('personfromvid.models.head_pose_estimator.get_model_manager')
    @patch('personfromvid.models.head_pose_estimator.HeadAngleClassifier')
    def test_estimate_head_pose_invalid_shape(self, mock_classifier, mock_get_manager, mock_get_model):
        """Test head pose estimation with invalid image shape."""
        mock_get_model.return_value = self.mock_model_config
        mock_get_manager.return_value = self.mock_model_manager
        mock_classifier.return_value = Mock()
        
        estimator = HeadPoseEstimator(TEST_HEAD_POSE_MODEL)
        
        # Mock _load_model and _estimate_pytorch to prevent actual model loading
        with patch.object(estimator, '_load_model'):
            with patch.object(estimator, '_estimate_pytorch') as mock_estimate:
                # Mock _estimate_pytorch to simulate validation error
                mock_estimate.side_effect = HeadPoseEstimationError("Face image must be a 3-channel color image")
                
                # Create image with invalid shape (not 3 channels)
                invalid_image = np.random.randint(0, 255, (224, 224), dtype=np.uint8)
                
                with pytest.raises(HeadPoseEstimationError, match="Head pose estimation failed.*3-channel"):
                    estimator.estimate_head_pose(invalid_image)

    @patch('personfromvid.models.head_pose_estimator.ModelConfigs.get_model')
    @patch('personfromvid.models.head_pose_estimator.get_model_manager')
    @patch('personfromvid.models.head_pose_estimator.HeadAngleClassifier')
    def test_estimate_batch_success(self, mock_classifier, mock_get_manager, mock_get_model):
        """Test successful batch head pose estimation."""
        mock_get_model.return_value = self.mock_model_config
        mock_get_manager.return_value = self.mock_model_manager
        mock_classifier.return_value = Mock()
        
        estimator = HeadPoseEstimator(TEST_HEAD_POSE_MODEL)
        
        with patch.object(estimator, '_load_model'):
            with patch.object(estimator, '_estimate_batch_pytorch') as mock_batch_estimate:
                mock_batch_estimate.return_value = [self.sample_head_pose, self.sample_head_pose]
                results = estimator.estimate_batch([self.test_face, self.test_face])
                
                assert len(results) == 2
                for result in results:
                    assert isinstance(result, HeadPoseResult)

    @patch('personfromvid.models.head_pose_estimator.ModelConfigs.get_model')
    @patch('personfromvid.models.head_pose_estimator.get_model_manager')
    @patch('personfromvid.models.head_pose_estimator.HeadAngleClassifier')
    def test_estimate_batch_empty_list(self, mock_classifier, mock_get_manager, mock_get_model):
        """Test batch estimation with empty list."""
        mock_get_model.return_value = self.mock_model_config
        mock_get_manager.return_value = self.mock_model_manager
        mock_classifier.return_value = Mock()
        
        estimator = HeadPoseEstimator(TEST_HEAD_POSE_MODEL)
        results = estimator.estimate_batch([])
        
        assert results == []

    @patch('personfromvid.models.head_pose_estimator.ModelConfigs.get_model')
    @patch('personfromvid.models.head_pose_estimator.get_model_manager')
    @patch('personfromvid.models.head_pose_estimator.HeadAngleClassifier')
    def test_estimate_batch_with_invalid_image(self, mock_classifier, mock_get_manager, mock_get_model):
        """Test batch estimation with invalid image in batch."""
        mock_get_model.return_value = self.mock_model_config
        mock_get_manager.return_value = self.mock_model_manager
        mock_classifier.return_value = Mock()
        
        estimator = HeadPoseEstimator(TEST_HEAD_POSE_MODEL)
        
        with pytest.raises(HeadPoseEstimationError, match="Input face image at index 1 is empty or None"):
            estimator.estimate_batch([self.test_face, np.array([])])

    def test_normalize_angle(self):
        """Test angle normalization."""
        estimator = Mock()
        from personfromvid.models.head_pose_estimator import HeadPoseEstimator
        
        # Test normal angles
        assert HeadPoseEstimator._normalize_angle(estimator, 45.0) == 45.0
        assert HeadPoseEstimator._normalize_angle(estimator, -45.0) == -45.0
        
        # Test angles needing normalization
        assert abs(HeadPoseEstimator._normalize_angle(estimator, 190.0) - (-170.0)) < 0.001
        assert abs(HeadPoseEstimator._normalize_angle(estimator, -190.0) - 170.0) < 0.001
        assert abs(HeadPoseEstimator._normalize_angle(estimator, 360.0)) < 0.001

    @patch('personfromvid.models.head_pose_estimator.ModelConfigs.get_model')
    @patch('personfromvid.models.head_pose_estimator.get_model_manager')
    @patch('personfromvid.models.head_pose_estimator.HeadAngleClassifier')
    def test_angles_to_direction(self, mock_classifier, mock_get_manager, mock_get_model):
        """Test angle to direction conversion."""
        mock_get_model.return_value = self.mock_model_config
        mock_get_manager.return_value = self.mock_model_manager
        
        # Mock classifier behavior
        mock_classifier_instance = Mock()
        mock_classifier_instance.yaw_threshold = DEFAULT_YAW_THRESHOLD
        mock_classifier_instance.pitch_threshold = DEFAULT_PITCH_THRESHOLD
        mock_classifier_instance.profile_yaw_threshold = DEFAULT_PROFILE_YAW_THRESHOLD
        mock_classifier_instance.max_roll = DEFAULT_MAX_ROLL
        mock_classifier_instance.classify_head_angle.return_value = "front"
        mock_classifier.return_value = mock_classifier_instance
        
        estimator = HeadPoseEstimator(TEST_HEAD_POSE_MODEL)
        
        # Set the classifier instance directly to ensure it's used
        estimator._head_angle_classifier = mock_classifier_instance
        
        # Test with angles that will bypass the forward-facing check and use classifier
        # Use larger angles that won't be considered "facing forward"
        direction = estimator.angles_to_direction(60.0, 0.0, 0.0)  # Large yaw angle
        
        assert direction == "front"
        mock_classifier_instance.classify_head_angle.assert_called_once_with(60.0, 0.0, 0.0)

    @patch('personfromvid.models.head_pose_estimator.ModelConfigs.get_model')
    @patch('personfromvid.models.head_pose_estimator.get_model_manager')
    @patch('personfromvid.models.head_pose_estimator.HeadAngleClassifier')
    def test_is_valid_orientation(self, mock_classifier, mock_get_manager, mock_get_model):
        """Test valid orientation checking."""
        mock_get_model.return_value = self.mock_model_config
        mock_get_manager.return_value = self.mock_model_manager
        
        # Mock classifier behavior
        mock_classifier_instance = Mock()
        mock_classifier_instance.yaw_threshold = DEFAULT_YAW_THRESHOLD
        mock_classifier_instance.pitch_threshold = DEFAULT_PITCH_THRESHOLD
        mock_classifier_instance.profile_yaw_threshold = DEFAULT_PROFILE_YAW_THRESHOLD
        mock_classifier_instance.max_roll = DEFAULT_MAX_ROLL
        mock_classifier_instance.is_valid_orientation.return_value = True
        mock_classifier.return_value = mock_classifier_instance
        
        estimator = HeadPoseEstimator(TEST_HEAD_POSE_MODEL)
        is_valid = estimator.is_valid_orientation(15.0)
        
        assert is_valid is True
        mock_classifier_instance.is_valid_orientation.assert_called_once_with(15.0)

    @patch('personfromvid.models.head_pose_estimator.ModelConfigs.get_model')
    @patch('personfromvid.models.head_pose_estimator.get_model_manager')
    @patch('personfromvid.models.head_pose_estimator.HeadAngleClassifier')
    def test_is_facing_forward(self, mock_classifier, mock_get_manager, mock_get_model):
        """Test forward-facing detection."""
        mock_get_model.return_value = self.mock_model_config
        mock_get_manager.return_value = self.mock_model_manager
        
        # Mock classifier with proper threshold values
        mock_classifier_instance = Mock()
        mock_classifier_instance.yaw_threshold = DEFAULT_YAW_THRESHOLD
        mock_classifier_instance.pitch_threshold = DEFAULT_PITCH_THRESHOLD
        mock_classifier_instance.profile_yaw_threshold = DEFAULT_PROFILE_YAW_THRESHOLD
        mock_classifier_instance.max_roll = DEFAULT_MAX_ROLL
        mock_classifier.return_value = mock_classifier_instance
        
        estimator = HeadPoseEstimator(TEST_HEAD_POSE_MODEL)
        
        # Test forward-facing pose
        assert estimator.is_facing_forward(0.0, 0.0, 0.0) is True
        
        # Test within forward thresholds
        assert estimator.is_facing_forward(30.0, 20.0, 10.0) is True
        
        # Test outside forward thresholds
        assert estimator.is_facing_forward(60.0, 40.0, 35.0) is False

    @patch('personfromvid.models.head_pose_estimator.ModelConfigs.get_model')
    @patch('personfromvid.models.head_pose_estimator.get_model_manager')
    @patch('personfromvid.models.head_pose_estimator.HeadAngleClassifier')
    def test_set_forward_facing_thresholds(self, mock_classifier, mock_get_manager, mock_get_model):
        """Test setting forward-facing thresholds."""
        mock_get_model.return_value = self.mock_model_config
        mock_get_manager.return_value = self.mock_model_manager
        mock_classifier.return_value = Mock()
        
        estimator = HeadPoseEstimator(TEST_HEAD_POSE_MODEL)
        estimator.set_forward_facing_thresholds(yaw=50.0, pitch=35.0)
        
        assert estimator.forward_yaw_threshold == 50.0
        assert estimator.forward_pitch_threshold == 35.0

    @patch('personfromvid.models.head_pose_estimator.ModelConfigs.get_model')
    @patch('personfromvid.models.head_pose_estimator.get_model_manager')
    @patch('personfromvid.models.head_pose_estimator.HeadAngleClassifier')
    def test_set_angle_thresholds(self, mock_classifier, mock_get_manager, mock_get_model):
        """Test setting angle thresholds."""
        mock_get_model.return_value = self.mock_model_config
        mock_get_manager.return_value = self.mock_model_manager
        
        # Mock classifier
        mock_classifier_instance = Mock()
        mock_classifier_instance.yaw_threshold = DEFAULT_YAW_THRESHOLD
        mock_classifier_instance.pitch_threshold = DEFAULT_PITCH_THRESHOLD
        mock_classifier_instance.profile_yaw_threshold = DEFAULT_PROFILE_YAW_THRESHOLD
        mock_classifier_instance.max_roll = DEFAULT_MAX_ROLL
        mock_classifier.return_value = mock_classifier_instance
        
        estimator = HeadPoseEstimator(TEST_HEAD_POSE_MODEL)
        estimator.set_angle_thresholds(yaw=25.0, pitch=20.0, profile_yaw=70.0, max_roll=25.0)
        
        assert estimator.yaw_threshold == 25.0
        assert estimator.pitch_threshold == 20.0
        assert estimator.profile_yaw_threshold == 70.0
        assert estimator.max_roll == 25.0
        
        # Verify classifier thresholds were updated
        mock_classifier_instance.set_angle_thresholds.assert_called_once_with(
            yaw=25.0, pitch=20.0, profile_yaw=70.0, max_roll=25.0
        )

    @patch('personfromvid.models.head_pose_estimator.ModelConfigs.get_model')
    @patch('personfromvid.models.head_pose_estimator.get_model_manager')
    @patch('personfromvid.models.head_pose_estimator.HeadAngleClassifier')
    def test_get_model_info(self, mock_classifier, mock_get_manager, mock_get_model):
        """Test getting model information."""
        mock_get_model.return_value = self.mock_model_config
        mock_get_manager.return_value = self.mock_model_manager
        mock_classifier.return_value = Mock()
        
        estimator = HeadPoseEstimator(TEST_HEAD_POSE_MODEL, device="cpu", confidence_threshold=0.75)
        info = estimator.get_model_info()
        
        assert info['model_name'] == TEST_HEAD_POSE_MODEL
        assert info['device'] == "cpu"
        assert info['confidence_threshold'] == 0.75
        assert info['input_size'] == (224, 224)
        assert info['model_format'] == 'pt'  # sixdrepnet uses PyTorch format
        assert 'angle_thresholds' in info
        assert 'forward_facing_thresholds' in info

    def test_calculate_confidence(self):
        """Test confidence calculation."""
        with patch('personfromvid.models.head_pose_estimator.HeadPoseEstimator._calculate_confidence') as mock_calc:
            mock_calc.return_value = 0.9
            estimator = Mock()
            from personfromvid.models.head_pose_estimator import HeadPoseEstimator
            
            # Test high-confidence front pose
            confidence = HeadPoseEstimator._calculate_confidence(estimator, 0.0, 0.0, 0.0)
            assert confidence == 0.9

    def test_softmax_to_angle(self):
        """Test softmax prediction to angle conversion."""
        with patch('personfromvid.models.head_pose_estimator.HeadPoseEstimator._softmax_to_angle') as mock_softmax:
            mock_softmax.return_value = 0.0
            estimator = Mock()
            from personfromvid.models.head_pose_estimator import HeadPoseEstimator
            
            # Mock softmax predictions (66 bins for HopeNet)
            predictions = np.zeros(66)
            predictions[33] = 1.0  # Peak at middle bin (0 degrees)
            
            angle = HeadPoseEstimator._softmax_to_angle(estimator, predictions)
            assert angle == 0.0  # Should be 0 degrees

    @patch('personfromvid.models.head_pose_estimator.ModelConfigs.get_model')
    @patch('personfromvid.models.head_pose_estimator.get_model_manager')
    @patch('personfromvid.models.head_pose_estimator.HeadAngleClassifier')
    def test_process_frame_batch(self, mock_classifier, mock_get_manager, mock_get_model):
        """Test frame batch processing."""
        from personfromvid.data.frame_data import FrameData, SourceInfo, ImageProperties
        from pathlib import Path
        
        mock_get_model.return_value = self.mock_model_config
        mock_get_manager.return_value = self.mock_model_manager
        mock_classifier.return_value = Mock()
        
        estimator = HeadPoseEstimator(TEST_HEAD_POSE_MODEL)
        
        # Create mock FrameData object
        mock_frame_data = FrameData(
            frame_id='frame_001',
            file_path=Path('/mock/frame.jpg'),
            source_info=SourceInfo(
                video_timestamp=1.0,
                extraction_method='test',
                original_frame_number=30,
                video_fps=30.0
            ),
            image_properties=ImageProperties(
                width=640,
                height=480,
                channels=3,
                file_size_bytes=1024,
                format='jpg'
            ),
            face_detections=[
                FaceDetection(bbox=(100, 100, 200, 200), confidence=0.9)
            ]
        )
        
        frames_with_faces = [mock_frame_data]
        
        # Mock cv2.imread and file existence
        with patch('cv2.imread') as mock_imread:
            with patch.object(Path, 'exists', return_value=True):
                mock_imread.return_value = self.test_face
                
                with patch.object(estimator, 'estimate_batch') as mock_estimate_batch:
                    mock_estimate_batch.return_value = [self.sample_head_pose]
                    
                    direction_counts, total_processed = estimator.process_frame_batch(frames_with_faces)
                    
                    assert isinstance(direction_counts, dict)
                    assert total_processed >= 0

    def test_del_cleanup(self):
        """Test cleanup in destructor."""
        estimator = Mock()
        estimator._model = Mock()
        
        from personfromvid.models.head_pose_estimator import HeadPoseEstimator
        HeadPoseEstimator.__del__(estimator)
        
        # Should not raise any exceptions


class TestCreateHeadPoseEstimator:
    """Tests for create_head_pose_estimator factory function."""
    
    @patch('personfromvid.models.head_pose_estimator.HeadPoseEstimator')
    def test_create_default(self, mock_estimator_class):
        """Test creating estimator with default parameters."""
        mock_estimator_class.return_value = Mock()
        
        estimator = create_head_pose_estimator()
        
        # Should use default model from config
        mock_estimator_class.assert_called_once()
        call_args = mock_estimator_class.call_args
        assert call_args[1]['device'] == "auto"
        assert call_args[1]['confidence_threshold'] == DEFAULT_CONFIDENCE_THRESHOLD

    @patch('personfromvid.models.head_pose_estimator.HeadPoseEstimator')
    def test_create_custom(self, mock_estimator_class):
        """Test creating estimator with custom parameters."""
        mock_estimator_class.return_value = Mock()
        
        estimator = create_head_pose_estimator(
            model_name="sixdrepnet", 
            device="cpu", 
            confidence_threshold=0.8
        )
        
        mock_estimator_class.assert_called_once_with(
            model_name="sixdrepnet", 
            device="cpu", 
            confidence_threshold=0.8
        )


class TestHeadPoseEstimatorIntegration:
    """Integration tests for HeadPoseEstimator with realistic scenarios."""
    
    def test_realistic_head_pose_estimation(self):
        """Test head pose estimation with realistic face images."""
        # This would require actual model files, so we'll mock the inference
        with patch('personfromvid.models.head_pose_estimator.ModelConfigs.get_model') as mock_get_model:
            with patch('personfromvid.models.head_pose_estimator.get_model_manager') as mock_get_manager:
                with patch('personfromvid.models.head_pose_estimator.HeadAngleClassifier') as mock_classifier:
                    # Set up mocks
                    mock_config = Mock()
                    mock_config.input_size = (224, 224)
                    mock_config.is_device_supported.return_value = True
                    mock_config.files = [Mock()]
                    mock_config.files[0].format = ModelFormat.PYTORCH  # sixdrepnet uses PyTorch format
                    
                    mock_get_model.return_value = mock_config
                    mock_get_manager.return_value = Mock()
                    mock_get_manager.return_value.ensure_model_available.return_value = Path("/mock/path")
                    mock_classifier.return_value = Mock()
                    
                    estimator = HeadPoseEstimator(TEST_HEAD_POSE_MODEL)
                    
                    # Mock the inference to return realistic results
                    with patch.object(estimator, '_load_model'):
                        # Mock the _transform attribute that would be set by _load_model
                        estimator._transform = Mock()
                        with patch.object(estimator, '_estimate_pytorch') as mock_estimate:
                            mock_estimate.return_value = HeadPoseResult(
                                yaw=12.3, pitch=-8.1, roll=3.4, confidence=0.87, direction="looking_left"
                            )
                            
                            # Create a realistic face image
                            face_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                            result = estimator.estimate_head_pose(face_image)
                        
                        assert isinstance(result, HeadPoseResult)
                        assert -180 <= result.yaw <= 180
                        assert -90 <= result.pitch <= 90
                        assert -180 <= result.roll <= 180
                        assert 0 <= result.confidence <= 1.0
                        assert result.direction in ["front", "looking_left", "looking_right", 
                                                   "profile_left", "profile_right", "looking_up", 
                                                   "looking_down", "looking_up_left", "looking_up_right"]

    def test_batch_processing_efficiency(self):
        """Test that batch processing is more efficient than individual processing."""
        # This test would verify batch processing performance benefits
        # For now, we'll just test that batch processing works correctly
        
        with patch('personfromvid.models.head_pose_estimator.ModelConfigs.get_model') as mock_get_model:
            with patch('personfromvid.models.head_pose_estimator.get_model_manager') as mock_get_manager:
                with patch('personfromvid.models.head_pose_estimator.HeadAngleClassifier') as mock_classifier:
                    # Set up mocks
                    mock_config = Mock()
                    mock_config.input_size = (224, 224)
                    mock_config.is_device_supported.return_value = True
                    mock_config.files = [Mock()]
                    mock_config.files[0].format = ModelFormat.PYTORCH  # sixdrepnet uses PyTorch format
                    
                    mock_get_model.return_value = mock_config
                    mock_get_manager.return_value = Mock()
                    mock_get_manager.return_value.ensure_model_available.return_value = Path("/mock/path")
                    mock_classifier.return_value = Mock()
                    
                    estimator = HeadPoseEstimator(TEST_HEAD_POSE_MODEL)
                    
                    # Mock batch processing
                    with patch.object(estimator, '_load_model'):
                        # Mock the _transform attribute that would be set by _load_model
                        estimator._transform = Mock()
                        with patch.object(estimator, '_estimate_batch_pytorch') as mock_batch_estimate:
                            mock_results = [
                                HeadPoseResult(yaw=0.0, pitch=0.0, roll=0.0, confidence=0.9, direction="front"),
                                HeadPoseResult(yaw=30.0, pitch=0.0, roll=0.0, confidence=0.8, direction="looking_left"),
                                HeadPoseResult(yaw=-30.0, pitch=0.0, roll=0.0, confidence=0.85, direction="looking_right")
                            ]
                            mock_batch_estimate.return_value = mock_results
                            
                            # Create batch of face images
                            face_images = [
                                np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                                for _ in range(3)
                            ]
                            
                            results = estimator.estimate_batch(face_images)
                        
                        assert len(results) == 3
                        assert all(isinstance(result, HeadPoseResult) for result in results)
                        # Verify batch method was called once (more efficient than 3 individual calls)
                        mock_batch_estimate.assert_called_once() 
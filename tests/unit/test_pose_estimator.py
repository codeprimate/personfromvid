"""Unit tests for PoseEstimator class."""

from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from personfromvid.data.detection_results import PoseDetection
from personfromvid.models.model_configs import ModelFormat
from personfromvid.models.pose_estimator import (
    COCO_KEYPOINT_NAMES,
    DEFAULT_CONFIDENCE_THRESHOLD,
    PoseEstimator,
    create_pose_estimator,
)
from personfromvid.utils.exceptions import PoseEstimationError

# Test constants to reduce duplication
TEST_POSE_MODEL = "yolov8n-pose"  # Use actual model name from config


class TestPoseEstimator:
    """Tests for PoseEstimator class."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create a sample image for testing
        self.test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        self.image_shape = (480, 640)

        # Mock model configuration
        self.mock_model_config = Mock()
        self.mock_model_config.input_size = (640, 640)
        self.mock_model_config.is_device_supported.return_value = True
        self.mock_model_config.files = [Mock()]
        self.mock_model_config.files[0].format = ModelFormat.PYTORCH

        # Mock model manager
        self.mock_model_manager = Mock()
        self.mock_model_manager.ensure_model_available.return_value = Path("/mock/path/model.pt")

        # Sample keypoints for testing
        self.sample_keypoints = {
            'nose': (320.0, 200.0, 0.9),
            'left_shoulder': (280.0, 300.0, 0.85),
            'right_shoulder': (360.0, 300.0, 0.88),
            'left_hip': (300.0, 500.0, 0.82),
            'right_hip': (340.0, 500.0, 0.84)
        }

    @patch('personfromvid.models.pose_estimator.ModelConfigs.get_model')
    @patch('personfromvid.models.pose_estimator.get_model_manager')
    def test_init_success(self, mock_get_manager, mock_get_model):
        """Test successful PoseEstimator initialization."""
        mock_get_model.return_value = self.mock_model_config
        mock_get_manager.return_value = self.mock_model_manager

        estimator = PoseEstimator(TEST_POSE_MODEL, device="cpu", confidence_threshold=0.8)

        assert estimator.model_name == TEST_POSE_MODEL
        assert estimator.device == "cpu"
        assert estimator.confidence_threshold == 0.8
        assert estimator._input_size == (640, 640)
        assert len(estimator._keypoint_names) == 17
        assert estimator._keypoint_names == COCO_KEYPOINT_NAMES
        mock_get_model.assert_called_once_with(TEST_POSE_MODEL)
        mock_get_manager.assert_called_once()

    @patch('personfromvid.models.pose_estimator.ModelConfigs.get_model')
    def test_init_unknown_model(self, mock_get_model):
        """Test initialization with unknown model."""
        mock_get_model.return_value = None

        with pytest.raises(PoseEstimationError, match="Unknown pose estimation model"):
            PoseEstimator("unknown_model")

    @patch('personfromvid.models.pose_estimator.ModelConfigs.get_model')
    @patch('personfromvid.models.pose_estimator.get_model_manager')
    def test_init_unsupported_device(self, mock_get_manager, mock_get_model):
        """Test initialization with unsupported device."""
        mock_config = Mock()
        mock_config.is_device_supported.return_value = False
        mock_get_model.return_value = mock_config
        mock_get_manager.return_value = self.mock_model_manager

        with pytest.raises(PoseEstimationError, match="does not support device"):
            PoseEstimator(TEST_POSE_MODEL, device="cuda")

    def test_resolve_device_auto_with_cuda(self):
        """Test device resolution with CUDA available."""
        with patch('personfromvid.models.pose_estimator.ModelConfigs.get_model') as mock_get_model, \
             patch('personfromvid.models.pose_estimator.get_model_manager') as mock_get_manager, \
             patch('torch.cuda.is_available', return_value=True):

            mock_get_model.return_value = self.mock_model_config
            mock_get_manager.return_value = self.mock_model_manager

            estimator = PoseEstimator(TEST_POSE_MODEL, device="auto")
            assert estimator.device == "cuda"

    def test_resolve_device_auto_without_cuda(self):
        """Test device resolution without CUDA."""
        with patch('personfromvid.models.pose_estimator.ModelConfigs.get_model') as mock_get_model, \
             patch('personfromvid.models.pose_estimator.get_model_manager') as mock_get_manager, \
             patch('torch.cuda.is_available', return_value=False):

            mock_get_model.return_value = self.mock_model_config
            mock_get_manager.return_value = self.mock_model_manager

            estimator = PoseEstimator(TEST_POSE_MODEL, device="auto")
            assert estimator.device == "cpu"

    @patch('personfromvid.models.pose_estimator.ModelConfigs.get_model')
    @patch('personfromvid.models.pose_estimator.get_model_manager')
    @patch('ultralytics.YOLO')
    def test_load_pytorch_yolo_model(self, mock_yolo, mock_get_manager, mock_get_model):
        """Test PyTorch YOLO model loading."""
        mock_get_model.return_value = self.mock_model_config
        mock_get_manager.return_value = self.mock_model_manager

        mock_model = Mock()
        mock_yolo.return_value = mock_model

        estimator = PoseEstimator(TEST_POSE_MODEL)
        estimator._load_model()

        assert estimator._model is not None
        mock_yolo.assert_called_once_with(str(estimator.model_path))

    @patch('personfromvid.models.pose_estimator.ModelConfigs.get_model')
    @patch('personfromvid.models.pose_estimator.get_model_manager')
    def test_load_pytorch_model_missing_dependency(self, mock_get_manager, mock_get_model):
        """Test PyTorch model loading with missing ultralytics."""
        mock_get_model.return_value = self.mock_model_config
        mock_get_manager.return_value = self.mock_model_manager

        estimator = PoseEstimator(TEST_POSE_MODEL)

        with patch('builtins.__import__', side_effect=ImportError("No module named 'ultralytics'")):
            with pytest.raises(PoseEstimationError, match="Required dependencies not installed"):
                estimator._load_model()

    @patch('personfromvid.models.pose_estimator.ModelConfigs.get_model')
    @patch('personfromvid.models.pose_estimator.get_model_manager')
    @patch('onnxruntime.InferenceSession')
    def test_load_onnx_model_success(self, mock_ort_session, mock_get_manager, mock_get_model):
        """Test successful ONNX model loading."""
        # Set up ONNX model config
        mock_config = Mock()
        mock_config.input_size = (640, 640)
        mock_config.is_device_supported.return_value = True
        mock_config.files = [Mock()]
        mock_config.files[0].format = ModelFormat.ONNX
        mock_get_model.return_value = mock_config
        mock_get_manager.return_value = self.mock_model_manager

        # Mock ONNX session
        mock_session = Mock()
        mock_input = Mock()
        mock_input.name = "input"
        mock_session.get_inputs.return_value = [mock_input]

        mock_output1 = Mock()
        mock_output1.name = "output1"
        mock_session.get_outputs.return_value = [mock_output1]

        mock_session.get_providers.return_value = ["CPUExecutionProvider"]
        mock_ort_session.return_value = mock_session

        estimator = PoseEstimator(TEST_POSE_MODEL)
        estimator._load_model()

        assert estimator._model is not None
        assert estimator._input_name == "input"
        assert estimator._output_names == ["output1"]
        mock_ort_session.assert_called_once()

    @patch('personfromvid.models.pose_estimator.ModelConfigs.get_model')
    @patch('personfromvid.models.pose_estimator.get_model_manager')
    def test_load_onnx_model_missing_dependency(self, mock_get_manager, mock_get_model):
        """Test ONNX model loading with missing onnxruntime."""
        # Set up ONNX model config
        mock_config = Mock()
        mock_config.input_size = (640, 640)
        mock_config.is_device_supported.return_value = True
        mock_config.files = [Mock()]
        mock_config.files[0].format = ModelFormat.ONNX
        mock_get_model.return_value = mock_config
        mock_get_manager.return_value = self.mock_model_manager

        estimator = PoseEstimator(TEST_POSE_MODEL)

        with patch('builtins.__import__', side_effect=ImportError("No module named 'onnxruntime'")):
            with pytest.raises(PoseEstimationError, match="onnxruntime not installed"):
                estimator._load_model()

    @patch('personfromvid.models.pose_estimator.ModelConfigs.get_model')
    @patch('personfromvid.models.pose_estimator.get_model_manager')
    @patch('ultralytics.YOLO')
    def test_estimate_pose_success(self, mock_yolo, mock_get_manager, mock_get_model):
        """Test successful pose estimation."""
        mock_get_model.return_value = self.mock_model_config
        mock_get_manager.return_value = self.mock_model_manager

        # Mock YOLO model and results
        mock_model = Mock()
        mock_result = Mock()
        mock_boxes = Mock()
        mock_keypoints = Mock()

        # Mock bounding boxes
        mock_boxes.xyxy = Mock()
        mock_boxes.xyxy.cpu.return_value.numpy.return_value = np.array([[100, 100, 300, 500]])
        mock_boxes.conf = Mock()
        mock_boxes.conf.cpu.return_value.numpy.return_value = np.array([0.9])

        # Mock keypoints (17 keypoints, 2D coordinates)
        mock_keypoints.xy = Mock()
        mock_keypoints.xy.cpu.return_value.numpy.return_value = np.array([[[320, 200], [310, 190], [330, 190],
                                                                           [300, 200], [340, 200], [280, 300],
                                                                           [360, 300], [260, 350], [380, 350],
                                                                           [250, 400], [390, 400], [300, 500],
                                                                           [340, 500], [290, 600], [350, 600],
                                                                           [280, 700], [360, 700]]])
        mock_keypoints.conf = Mock()
        mock_keypoints.conf.cpu.return_value.numpy.return_value = np.array([[0.9, 0.8, 0.8, 0.7, 0.7, 0.85, 0.88,
                                                                             0.6, 0.6, 0.5, 0.5, 0.82, 0.84, 0.75, 0.75,
                                                                             0.7, 0.7]])

        mock_result.boxes = mock_boxes
        mock_result.keypoints = mock_keypoints
        mock_model.return_value = [mock_result]
        mock_yolo.return_value = mock_model

        estimator = PoseEstimator(TEST_POSE_MODEL, confidence_threshold=0.7)
        poses = estimator.estimate_pose(self.test_image)

        assert len(poses) == 1
        pose = poses[0]
        assert isinstance(pose, PoseDetection)
        assert pose.confidence == 0.9
        assert len(pose.keypoints) == 17
        assert 'nose' in pose.keypoints
        assert 'left_shoulder' in pose.keypoints

    @patch('personfromvid.models.pose_estimator.ModelConfigs.get_model')
    @patch('personfromvid.models.pose_estimator.get_model_manager')
    def test_estimate_pose_empty_image(self, mock_get_manager, mock_get_model):
        """Test pose estimation with empty image."""
        mock_get_model.return_value = self.mock_model_config
        mock_get_manager.return_value = self.mock_model_manager

        estimator = PoseEstimator(TEST_POSE_MODEL)

        with pytest.raises(PoseEstimationError, match="Input image is empty or None"):
            estimator.estimate_pose(np.array([]))

    @patch('personfromvid.models.pose_estimator.ModelConfigs.get_model')
    @patch('personfromvid.models.pose_estimator.get_model_manager')
    @patch('ultralytics.YOLO')
    def test_estimate_batch_success(self, mock_yolo, mock_get_manager, mock_get_model):
        """Test successful batch pose estimation."""
        mock_get_model.return_value = self.mock_model_config
        mock_get_manager.return_value = self.mock_model_manager

        # Mock YOLO model and results for batch processing
        mock_model = Mock()
        mock_result1 = Mock()
        mock_result2 = Mock()

        # Set up mock results for each image
        for mock_result in [mock_result1, mock_result2]:
            mock_boxes = Mock()
            mock_keypoints = Mock()

            mock_boxes.xyxy = Mock()
            mock_boxes.xyxy.cpu.return_value.numpy.return_value = np.array([[100, 100, 300, 500]])
            mock_boxes.conf = Mock()
            mock_boxes.conf.cpu.return_value.numpy.return_value = np.array([0.9])

            mock_keypoints.xy = Mock()
            mock_keypoints.xy.cpu.return_value.numpy.return_value = np.array([[[320, 200], [310, 190], [330, 190],
                                                                               [300, 200], [340, 200], [280, 300],
                                                                               [360, 300], [260, 350], [380, 350],
                                                                               [250, 400], [390, 400], [300, 500],
                                                                               [340, 500], [290, 600], [350, 600],
                                                                               [280, 700], [360, 700]]])
            mock_keypoints.conf = Mock()
            mock_keypoints.conf.cpu.return_value.numpy.return_value = np.array([[0.9, 0.8, 0.8, 0.7, 0.7, 0.85, 0.88,
                                                                                 0.6, 0.6, 0.5, 0.5, 0.82, 0.84, 0.75, 0.75,
                                                                                 0.7, 0.7]])

            mock_result.boxes = mock_boxes
            mock_result.keypoints = mock_keypoints

        mock_model.return_value = [mock_result1, mock_result2]
        mock_yolo.return_value = mock_model

        estimator = PoseEstimator(TEST_POSE_MODEL, confidence_threshold=0.7)
        batch_poses = estimator.estimate_batch([self.test_image, self.test_image])

        assert len(batch_poses) == 2
        for poses in batch_poses:
            assert len(poses) == 1
            assert isinstance(poses[0], PoseDetection)

    @patch('personfromvid.models.pose_estimator.ModelConfigs.get_model')
    @patch('personfromvid.models.pose_estimator.get_model_manager')
    def test_estimate_batch_empty_list(self, mock_get_manager, mock_get_model):
        """Test batch estimation with empty list."""
        mock_get_model.return_value = self.mock_model_config
        mock_get_manager.return_value = self.mock_model_manager

        estimator = PoseEstimator(TEST_POSE_MODEL)
        batch_poses = estimator.estimate_batch([])

        assert batch_poses == []

    @patch('personfromvid.models.pose_estimator.ModelConfigs.get_model')
    @patch('personfromvid.models.pose_estimator.get_model_manager')
    def test_estimate_batch_with_invalid_image(self, mock_get_manager, mock_get_model):
        """Test batch estimation with invalid image in batch."""
        mock_get_model.return_value = self.mock_model_config
        mock_get_manager.return_value = self.mock_model_manager

        estimator = PoseEstimator(TEST_POSE_MODEL)

        with pytest.raises(PoseEstimationError, match="Input image at index 1 is empty or None"):
            estimator.estimate_batch([self.test_image, np.array([])])

    @patch('personfromvid.models.pose_estimator.ModelConfigs.get_model')
    @patch('personfromvid.models.pose_estimator.get_model_manager')
    def test_set_confidence_threshold(self, mock_get_manager, mock_get_model):
        """Test setting confidence threshold."""
        mock_get_model.return_value = self.mock_model_config
        mock_get_manager.return_value = self.mock_model_manager

        estimator = PoseEstimator(TEST_POSE_MODEL)

        # Test valid threshold
        estimator.set_confidence_threshold(0.8)
        assert estimator.confidence_threshold == 0.8

        # Test invalid thresholds
        with pytest.raises(ValueError, match="Confidence threshold must be between 0.0 and 1.0"):
            estimator.set_confidence_threshold(-0.1)

        with pytest.raises(ValueError, match="Confidence threshold must be between 0.0 and 1.0"):
            estimator.set_confidence_threshold(1.1)

    @patch('personfromvid.models.pose_estimator.ModelConfigs.get_model')
    @patch('personfromvid.models.pose_estimator.get_model_manager')
    def test_get_keypoint_names(self, mock_get_manager, mock_get_model):
        """Test getting keypoint names."""
        mock_get_model.return_value = self.mock_model_config
        mock_get_manager.return_value = self.mock_model_manager

        estimator = PoseEstimator(TEST_POSE_MODEL)
        keypoint_names = estimator.get_keypoint_names()

        assert keypoint_names == COCO_KEYPOINT_NAMES
        assert len(keypoint_names) == 17
        assert 'nose' in keypoint_names
        assert 'left_shoulder' in keypoint_names
        assert 'right_ankle' in keypoint_names

    @patch('personfromvid.models.pose_estimator.ModelConfigs.get_model')
    @patch('personfromvid.models.pose_estimator.get_model_manager')
    def test_get_model_info(self, mock_get_manager, mock_get_model):
        """Test getting model information."""
        mock_get_model.return_value = self.mock_model_config
        mock_get_manager.return_value = self.mock_model_manager

        estimator = PoseEstimator(TEST_POSE_MODEL, device="cpu", confidence_threshold=0.75)
        info = estimator.get_model_info()

        assert info['model_name'] == TEST_POSE_MODEL
        assert info['device'] == "cpu"
        assert info['confidence_threshold'] == 0.75
        assert info['input_size'] == (640, 640)
        assert info['model_loaded'] is False
        assert info['keypoint_names'] == COCO_KEYPOINT_NAMES
        assert info['num_keypoints'] == 17

    def test_validate_pose_detection(self):
        """Test pose detection validation."""
        valid_detection = PoseDetection(
            bbox=(100, 100, 200, 300),
            confidence=0.8,
            keypoints=self.sample_keypoints
        )

        # Mock estimator for validation
        estimator = Mock()
        estimator.confidence_threshold = 0.7

        # Bind the method to the mock
        from personfromvid.models.pose_estimator import PoseEstimator
        is_valid = PoseEstimator.validate_pose_detection(estimator, valid_detection, self.image_shape)
        assert is_valid is True

        # Test invalid detection (too small)
        invalid_detection = PoseDetection(
            bbox=(100, 100, 130, 130),  # Only 30x30 pixels
            confidence=0.8,
            keypoints=self.sample_keypoints
        )

        is_valid = PoseEstimator.validate_pose_detection(estimator, invalid_detection, self.image_shape)
        assert is_valid is False

        # Test detection with few high-confidence keypoints
        low_conf_keypoints = {
            'nose': (320.0, 200.0, 0.3),  # Low confidence
            'left_shoulder': (280.0, 300.0, 0.2),  # Low confidence
        }
        low_conf_detection = PoseDetection(
            bbox=(100, 100, 200, 300),
            confidence=0.8,
            keypoints=low_conf_keypoints
        )

        is_valid = PoseEstimator.validate_pose_detection(estimator, low_conf_detection, self.image_shape)
        assert is_valid is False

    def test_calculate_pose_confidence(self):
        """Test pose confidence calculation."""
        # Mock estimator for calculation
        estimator = Mock()

        # Test with good keypoints
        good_detection = PoseDetection(
            bbox=(100, 100, 200, 300),
            confidence=0.8,
            keypoints=self.sample_keypoints
        )

        from personfromvid.models.pose_estimator import PoseEstimator
        pose_conf = PoseEstimator.calculate_pose_confidence(estimator, good_detection)
        assert 0.7 < pose_conf < 1.0  # Should be high confidence

        # Test with no keypoints
        empty_detection = PoseDetection(
            bbox=(100, 100, 200, 300),
            confidence=0.8,
            keypoints={}
        )

        pose_conf = PoseEstimator.calculate_pose_confidence(estimator, empty_detection)
        assert pose_conf == 0.0

    def test_validate_and_normalize_keypoints(self):
        """Test keypoint validation and normalization."""
        # Mock estimator for testing
        estimator = Mock()

        # Test with valid keypoints
        from personfromvid.models.pose_estimator import PoseEstimator
        validated = PoseEstimator._validate_and_normalize_keypoints(
            estimator, self.sample_keypoints, self.image_shape
        )

        assert len(validated) == len(self.sample_keypoints)
        for _name, (x, y, conf) in validated.items():
            assert 0 <= x <= self.image_shape[1]
            assert 0 <= y <= self.image_shape[0]
            assert 0 <= conf <= 1.0

        # Test with out-of-bounds keypoints
        oob_keypoints = {
            'nose': (-10.0, 1000.0, 0.9),  # Out of bounds
            'left_shoulder': (280.0, 300.0, 0.85),  # Valid
        }

        validated = PoseEstimator._validate_and_normalize_keypoints(
            estimator, oob_keypoints, self.image_shape
        )

        # Out of bounds keypoint should be clamped with reduced confidence
        nose_kp = validated['nose']
        assert nose_kp[0] == 0  # Clamped to 0
        assert nose_kp[1] == self.image_shape[0]  # Clamped to height
        assert nose_kp[2] == 0.45  # Confidence reduced by half

    def test_preprocess_image(self):
        """Test image preprocessing."""
        # Mock estimator for testing
        estimator = Mock()
        estimator._input_size = (640, 640)

        from personfromvid.models.pose_estimator import PoseEstimator
        preprocessed = PoseEstimator._preprocess_image(estimator, self.test_image)

        # Check output shape and format
        assert preprocessed.shape == (1, 3, 640, 640)  # NCHW format
        assert preprocessed.dtype == np.float32
        assert 0 <= preprocessed.min() <= preprocessed.max() <= 1.0  # Normalized


class TestCreatePoseEstimator:
    """Tests for create_pose_estimator factory function."""

    @patch('personfromvid.models.pose_estimator.ModelConfigs.get_default_models')
    @patch('personfromvid.models.pose_estimator.PoseEstimator')
    def test_create_with_default_model(self, mock_pose_estimator, mock_get_defaults):
        """Test creating pose estimator with default model."""
        mock_get_defaults.return_value = {"pose_estimation": "yolov8s-pose"}

        create_pose_estimator()

        mock_get_defaults.assert_called_once()
        mock_pose_estimator.assert_called_once_with("yolov8s-pose", "auto", DEFAULT_CONFIDENCE_THRESHOLD, None)

    @patch('personfromvid.models.pose_estimator.PoseEstimator')
    def test_create_with_custom_params(self, mock_pose_estimator):
        """Test creating pose estimator with custom parameters."""
        create_pose_estimator(
            model_name="yolov8s-pose",
            device="cuda",
            confidence_threshold=0.85
        )

        mock_pose_estimator.assert_called_once_with("yolov8s-pose", "cuda", 0.85, None)


class TestIntegrationScenarios:
    """Integration test scenarios for PoseEstimator."""

    @patch('personfromvid.models.pose_estimator.ModelConfigs.get_model')
    @patch('personfromvid.models.pose_estimator.get_model_manager')
    @patch('ultralytics.YOLO')
    def test_typical_pipeline_usage(self, mock_yolo, mock_get_manager, mock_get_model):
        """Test typical pose estimation pipeline usage."""
        # Set up mocks
        mock_model_config = Mock()
        mock_model_config.input_size = (640, 640)
        mock_model_config.is_device_supported.return_value = True
        mock_model_config.files = [Mock()]
        mock_model_config.files[0].format = ModelFormat.PYTORCH
        mock_get_model.return_value = mock_model_config

        mock_model_manager = Mock()
        mock_model_manager.ensure_model_available.return_value = Path("/models/yolov8n-pose.pt")
        mock_get_manager.return_value = mock_model_manager

        # Mock YOLO model with realistic pose detection outputs
        mock_model = Mock()

        # Create multiple test images
        test_images = [
            np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(3)
        ]

        # Mock batch processing results
        batch_results = []
        for _i in range(3):
            mock_result = Mock()
            mock_boxes = Mock()
            mock_keypoints = Mock()

            # Two people per image
            mock_boxes.xyxy = Mock()
            mock_boxes.xyxy.cpu.return_value.numpy.return_value = np.array([
                [100, 100, 250, 450],
                [350, 120, 500, 470]
            ])
            mock_boxes.conf = Mock()
            mock_boxes.conf.cpu.return_value.numpy.return_value = np.array([0.9, 0.85])

            # Mock keypoints for two people
            mock_keypoints.xy = Mock()
            keypoints_data = np.array([
                # Person 1 keypoints
                [[175, 150], [170, 145], [180, 145], [165, 150], [185, 150],
                 [150, 200], [200, 200], [130, 250], [220, 250], [120, 300],
                 [230, 300], [160, 350], [190, 350], [150, 400], [200, 400],
                 [140, 450], [210, 450]],
                # Person 2 keypoints
                [[425, 170], [420, 165], [430, 165], [415, 170], [435, 170],
                 [400, 220], [450, 220], [380, 270], [470, 270], [370, 320],
                 [480, 320], [410, 370], [440, 370], [400, 420], [450, 420],
                 [390, 470], [460, 470]]
            ])
            mock_keypoints.xy.cpu.return_value.numpy.return_value = keypoints_data

            mock_keypoints.conf = Mock()
            # High confidence for most keypoints
            confidence_data = np.array([
                [0.9, 0.8, 0.8, 0.7, 0.7, 0.85, 0.88, 0.6, 0.6, 0.5, 0.5, 0.82, 0.84, 0.75, 0.75, 0.7, 0.7],
                [0.88, 0.82, 0.8, 0.72, 0.7, 0.87, 0.85, 0.62, 0.58, 0.52, 0.48, 0.8, 0.83, 0.73, 0.77, 0.68, 0.72]
            ])
            mock_keypoints.conf.cpu.return_value.numpy.return_value = confidence_data

            mock_result.boxes = mock_boxes
            mock_result.keypoints = mock_keypoints
            batch_results.append(mock_result)

        mock_model.return_value = batch_results
        mock_yolo.return_value = mock_model

        # Test typical usage
        estimator = PoseEstimator("yolov8n-pose", device="cpu", confidence_threshold=0.8)

        # Test batch processing
        batch_poses = estimator.estimate_batch(test_images)

        assert len(batch_poses) == 3
        for poses in batch_poses:
            assert len(poses) == 2  # Two people per image
            for pose in poses:
                assert isinstance(pose, PoseDetection)
                assert pose.confidence >= 0.8  # Above threshold
                assert len(pose.bbox) == 4  # Valid bbox format
                assert len(pose.keypoints) == 17  # All COCO keypoints

        # Test individual detection
        single_poses = estimator.estimate_pose(test_images[0])
        assert len(single_poses) == 2

    @patch('personfromvid.models.pose_estimator.ModelConfigs.get_model')
    @patch('personfromvid.models.pose_estimator.get_model_manager')
    def test_error_recovery_scenarios(self, mock_get_manager, mock_get_model):
        """Test error handling and recovery scenarios."""
        mock_model_config = Mock()
        mock_model_config.input_size = (640, 640)
        mock_model_config.is_device_supported.return_value = True
        mock_model_config.files = [Mock()]
        mock_model_config.files[0].format = ModelFormat.PYTORCH
        mock_get_model.return_value = mock_model_config

        mock_manager = Mock()
        mock_manager.ensure_model_available.return_value = Path("/models/yolov8n-pose.pt")
        mock_get_manager.return_value = mock_manager

        estimator = PoseEstimator(TEST_POSE_MODEL)

        # Test graceful handling of various error conditions
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Model loading failure should be handled gracefully
        with patch.object(estimator, '_load_pytorch_model', side_effect=Exception("Model load failed")):
            with pytest.raises(PoseEstimationError, match="Failed to load model"):
                estimator.estimate_pose(test_image)

        # Should be able to change thresholds and retry
        estimator.set_confidence_threshold(0.5)
        assert estimator.confidence_threshold == 0.5

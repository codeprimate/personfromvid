"""Unit tests for FaceDetector class."""

from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from personfromvid.data.detection_results import FaceDetection
from personfromvid.models.face_detector import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    FaceDetector,
    create_face_detector,
)
from personfromvid.models.model_configs import ModelFormat
from personfromvid.utils.exceptions import FaceDetectionError

# Test constants to reduce duplication
TEST_FACE_MODEL = "scrfd_10g"  # Use actual model name from config


class TestFaceDetector:
    """Tests for FaceDetector class."""

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
        self.mock_model_config.files[0].format = ModelFormat.ONNX

        # Mock model manager
        self.mock_model_manager = Mock()
        self.mock_model_manager.ensure_model_available.return_value = Path("/mock/path/model.onnx")

    @patch('personfromvid.models.face_detector.ModelConfigs.get_model')
    @patch('personfromvid.models.face_detector.get_model_manager')
    def test_init_success(self, mock_get_manager, mock_get_model):
        """Test successful FaceDetector initialization."""
        mock_get_model.return_value = self.mock_model_config
        mock_get_manager.return_value = self.mock_model_manager

        detector = FaceDetector(TEST_FACE_MODEL, device="cpu", confidence_threshold=0.8)

        assert detector.model_name == TEST_FACE_MODEL
        assert detector.device == "cpu"
        assert detector.confidence_threshold == 0.8
        assert detector._input_size == (640, 640)
        mock_get_model.assert_called_once_with(TEST_FACE_MODEL)
        mock_get_manager.assert_called_once()

    @patch('personfromvid.models.face_detector.ModelConfigs.get_model')
    def test_init_unknown_model(self, mock_get_model):
        """Test initialization with unknown model."""
        mock_get_model.return_value = None

        with pytest.raises(FaceDetectionError, match="Unknown face detection model"):
            FaceDetector("unknown_model")

    @patch('personfromvid.models.face_detector.ModelConfigs.get_model')
    @patch('personfromvid.models.face_detector.get_model_manager')
    def test_init_unsupported_device(self, mock_get_manager, mock_get_model):
        """Test initialization with unsupported device."""
        mock_config = Mock()
        mock_config.is_device_supported.return_value = False
        mock_get_model.return_value = mock_config
        mock_get_manager.return_value = self.mock_model_manager

        with pytest.raises(FaceDetectionError, match="does not support device"):
            FaceDetector("TEST_FACE_MODEL", device="cuda")

    def test_resolve_device_auto_with_cuda(self):
        """Test device resolution with CUDA available."""
        with patch('personfromvid.models.face_detector.ModelConfigs.get_model') as mock_get_model, \
             patch('personfromvid.models.face_detector.get_model_manager') as mock_get_manager, \
             patch('torch.cuda.is_available', return_value=True):

            mock_get_model.return_value = self.mock_model_config
            mock_get_manager.return_value = self.mock_model_manager

            detector = FaceDetector("TEST_FACE_MODEL", device="auto")
            assert detector.device == "cuda"

    def test_resolve_device_auto_without_cuda(self):
        """Test device resolution without CUDA."""
        with patch('personfromvid.models.face_detector.ModelConfigs.get_model') as mock_get_model, \
             patch('personfromvid.models.face_detector.get_model_manager') as mock_get_manager, \
             patch('torch.cuda.is_available', return_value=False):

            mock_get_model.return_value = self.mock_model_config
            mock_get_manager.return_value = self.mock_model_manager

            detector = FaceDetector("TEST_FACE_MODEL", device="auto")
            assert detector.device == "cpu"

    @patch('personfromvid.models.face_detector.ModelConfigs.get_model')
    @patch('personfromvid.models.face_detector.get_model_manager')
    @patch('onnxruntime.InferenceSession')
    def test_load_onnx_model_success(self, mock_ort_session, mock_get_manager, mock_get_model):
        """Test successful ONNX model loading."""
        mock_get_model.return_value = self.mock_model_config
        mock_get_manager.return_value = self.mock_model_manager

        # Mock ONNX session
        mock_session = Mock()
        mock_input = Mock()
        mock_input.name = "input"
        mock_session.get_inputs.return_value = [mock_input]

        mock_output1 = Mock()
        mock_output1.name = "output1"
        mock_output2 = Mock()
        mock_output2.name = "output2"
        mock_session.get_outputs.return_value = [mock_output1, mock_output2]

        mock_session.get_providers.return_value = ["CPUExecutionProvider"]
        mock_ort_session.return_value = mock_session

        detector = FaceDetector("TEST_FACE_MODEL")
        detector._load_model()

        assert detector._model is not None
        assert detector._input_name == "input"
        assert detector._output_names == ["output1", "output2"]
        mock_ort_session.assert_called_once()

    @patch('personfromvid.models.face_detector.ModelConfigs.get_model')
    @patch('personfromvid.models.face_detector.get_model_manager')
    def test_load_onnx_model_missing_dependency(self, mock_get_manager, mock_get_model):
        """Test ONNX model loading with missing onnxruntime."""
        mock_get_model.return_value = self.mock_model_config
        mock_get_manager.return_value = self.mock_model_manager

        detector = FaceDetector("TEST_FACE_MODEL")

        with patch('builtins.__import__', side_effect=ImportError("No module named 'onnxruntime'")):
            with pytest.raises(FaceDetectionError, match="onnxruntime not installed"):
                detector._load_model()

    @patch('personfromvid.models.face_detector.ModelConfigs.get_model')
    @patch('personfromvid.models.face_detector.get_model_manager')
    @patch('ultralytics.YOLO')
    def test_load_pytorch_yolo_model(self, mock_yolo, mock_get_manager, mock_get_model):
        """Test PyTorch YOLO model loading."""
        # Set up model config for PyTorch
        mock_config = Mock()
        mock_config.input_size = (640, 640)
        mock_config.is_device_supported.return_value = True
        mock_config.files = [Mock()]
        mock_config.files[0].format = ModelFormat.PYTORCH
        mock_get_model.return_value = mock_config
        mock_get_manager.return_value = self.mock_model_manager

        mock_model = Mock()
        mock_yolo.return_value = mock_model

        detector = FaceDetector("yolov8n-face")
        detector._load_model()

        assert detector._model is not None
        mock_yolo.assert_called_once()

    @patch('personfromvid.models.face_detector.ModelConfigs.get_model')
    @patch('personfromvid.models.face_detector.get_model_manager')
    @patch('onnxruntime.InferenceSession')
    def test_detect_faces_success(self, mock_ort_session, mock_get_manager, mock_get_model):
        """Test successful face detection."""
        mock_get_model.return_value = self.mock_model_config
        mock_get_manager.return_value = self.mock_model_manager

        # Mock ONNX session and outputs
        mock_session = Mock()
        mock_input = Mock()
        mock_input.name = "input"
        mock_session.get_inputs.return_value = [mock_input]

        mock_output1 = Mock()
        mock_output1.name = "bboxes"
        mock_output2 = Mock()
        mock_output2.name = "scores"
        mock_session.get_outputs.return_value = [mock_output1, mock_output2]

        mock_session.get_providers.return_value = ["CPUExecutionProvider"]

        # Mock detection outputs - note that these will be scaled by the preprocessing
        # Input size is 640x640, image is 480x640, so scale factors are 640/640=1.0 and 480/640=0.75
        mock_bboxes = np.array([[100, 133, 200, 267]])  # Adjusted for scaling
        mock_scores = np.array([0.9])  # High confidence
        mock_session.run.return_value = [mock_bboxes, mock_scores]

        mock_ort_session.return_value = mock_session

        detector = FaceDetector("TEST_FACE_MODEL", confidence_threshold=0.7)
        faces = detector.detect_faces(self.test_image)

        assert len(faces) == 1
        assert isinstance(faces[0], FaceDetection)
        assert faces[0].confidence == 0.9
        # Check bbox is reasonable (allowing for scaling/rounding differences)
        x1, y1, x2, y2 = faces[0].bbox
        assert 90 <= x1 <= 110  # Allow some tolerance for scaling
        assert 90 <= y1 <= 110
        assert 190 <= x2 <= 210
        assert 190 <= y2 <= 210

    @patch('personfromvid.models.face_detector.ModelConfigs.get_model')
    @patch('personfromvid.models.face_detector.get_model_manager')
    def test_detect_faces_empty_image(self, mock_get_manager, mock_get_model):
        """Test face detection with empty image."""
        mock_get_model.return_value = self.mock_model_config
        mock_get_manager.return_value = self.mock_model_manager

        detector = FaceDetector("TEST_FACE_MODEL")

        with pytest.raises(FaceDetectionError, match="Input image is empty"):
            detector.detect_faces(np.array([]))

    @patch('personfromvid.models.face_detector.ModelConfigs.get_model')
    @patch('personfromvid.models.face_detector.get_model_manager')
    @patch('onnxruntime.InferenceSession')
    def test_detect_batch_success(self, mock_ort_session, mock_get_manager, mock_get_model):
        """Test successful batch face detection."""
        mock_get_model.return_value = self.mock_model_config
        mock_get_manager.return_value = self.mock_model_manager

        # Mock ONNX session
        mock_session = Mock()
        mock_input = Mock()
        mock_input.name = "input"
        mock_session.get_inputs.return_value = [mock_input]

        mock_output1 = Mock()
        mock_output1.name = "bboxes"
        mock_output2 = Mock()
        mock_output2.name = "scores"
        mock_session.get_outputs.return_value = [mock_output1, mock_output2]

        mock_session.get_providers.return_value = ["CPUExecutionProvider"]

        # Mock outputs for two images
        mock_bboxes = np.array([[100, 133, 200, 267]])
        mock_scores = np.array([0.9])
        mock_session.run.return_value = [mock_bboxes, mock_scores]

        mock_ort_session.return_value = mock_session

        detector = FaceDetector("TEST_FACE_MODEL")
        images = [self.test_image, self.test_image]
        batch_faces = detector.detect_batch(images)

        assert len(batch_faces) == 2
        assert len(batch_faces[0]) == 1
        assert len(batch_faces[1]) == 1

    @patch('personfromvid.models.face_detector.ModelConfigs.get_model')
    @patch('personfromvid.models.face_detector.get_model_manager')
    def test_detect_batch_empty_list(self, mock_get_manager, mock_get_model):
        """Test batch detection with empty image list."""
        mock_get_model.return_value = self.mock_model_config
        mock_get_manager.return_value = self.mock_model_manager

        detector = FaceDetector("TEST_FACE_MODEL")
        result = detector.detect_batch([])

        assert result == []

    @patch('personfromvid.models.face_detector.ModelConfigs.get_model')
    @patch('personfromvid.models.face_detector.get_model_manager')
    def test_detect_batch_with_invalid_image(self, mock_get_manager, mock_get_model):
        """Test batch detection with invalid image in batch."""
        mock_get_model.return_value = self.mock_model_config
        mock_get_manager.return_value = self.mock_model_manager

        detector = FaceDetector("TEST_FACE_MODEL")
        images = [self.test_image, np.array([]), self.test_image]

        with pytest.raises(FaceDetectionError, match="Input image at index 1 is empty"):
            detector.detect_batch(images)

    @patch('personfromvid.models.face_detector.ModelConfigs.get_model')
    @patch('personfromvid.models.face_detector.get_model_manager')
    def test_set_confidence_threshold(self, mock_get_manager, mock_get_model):
        """Test setting confidence threshold."""
        mock_get_model.return_value = self.mock_model_config
        mock_get_manager.return_value = self.mock_model_manager

        detector = FaceDetector("TEST_FACE_MODEL")

        detector.set_confidence_threshold(0.85)
        assert detector.confidence_threshold == 0.85

        # Test invalid threshold
        with pytest.raises(ValueError, match="Confidence threshold must be between"):
            detector.set_confidence_threshold(1.5)

        with pytest.raises(ValueError, match="Confidence threshold must be between"):
            detector.set_confidence_threshold(-0.1)

    @patch('personfromvid.models.face_detector.ModelConfigs.get_model')
    @patch('personfromvid.models.face_detector.get_model_manager')
    def test_get_model_info(self, mock_get_manager, mock_get_model):
        """Test getting model information."""
        mock_get_model.return_value = self.mock_model_config
        mock_get_manager.return_value = self.mock_model_manager

        detector = FaceDetector("TEST_FACE_MODEL", device="cpu", confidence_threshold=0.8)
        info = detector.get_model_info()

        assert info["model_name"] == "TEST_FACE_MODEL"
        assert info["device"] == "cpu"
        assert info["confidence_threshold"] == 0.8
        assert info["input_size"] == (640, 640)
        assert info["model_loaded"] is False

    def test_validate_face_detection(self):
        """Test face detection validation."""
        with patch('personfromvid.models.face_detector.ModelConfigs.get_model') as mock_get_model, \
             patch('personfromvid.models.face_detector.get_model_manager') as mock_get_manager:

            mock_get_model.return_value = self.mock_model_config
            mock_get_manager.return_value = self.mock_model_manager

            detector = FaceDetector("TEST_FACE_MODEL", confidence_threshold=0.7)

            # Valid detection
            valid_detection = FaceDetection(bbox=(50, 50, 150, 150), confidence=0.8)
            assert detector.validate_face_detection(valid_detection, (480, 640)) is True

            # Invalid detection - out of bounds
            invalid_detection = FaceDetection(bbox=(600, 400, 700, 500), confidence=0.8)
            assert detector.validate_face_detection(invalid_detection, (480, 640)) is False

            # Invalid detection - low confidence
            low_conf_detection = FaceDetection(bbox=(50, 50, 150, 150), confidence=0.5)
            assert detector.validate_face_detection(low_conf_detection, (480, 640)) is False

            # Invalid detection - negative coordinates
            neg_detection = FaceDetection(bbox=(-10, 50, 150, 150), confidence=0.8)
            assert detector.validate_face_detection(neg_detection, (480, 640)) is False

            # Invalid detection - too small
            small_detection = FaceDetection(bbox=(100, 100, 105, 105), confidence=0.8)
            assert detector.validate_face_detection(small_detection, (480, 640)) is False

    def test_preprocess_image(self):
        """Test image preprocessing."""
        with patch('personfromvid.models.face_detector.ModelConfigs.get_model') as mock_get_model, \
             patch('personfromvid.models.face_detector.get_model_manager') as mock_get_manager:

            mock_get_model.return_value = self.mock_model_config
            mock_get_manager.return_value = self.mock_model_manager

            detector = FaceDetector("TEST_FACE_MODEL")

            # Test preprocessing
            processed = detector._preprocess_image(self.test_image)

            # Check output shape: (1, 3, 640, 640) for batch, channels, height, width
            assert processed.shape == (1, 3, 640, 640)
            assert processed.dtype == np.float32
            assert 0.0 <= processed.min() <= processed.max() <= 1.0


class TestCreateFaceDetector:
    """Tests for create_face_detector factory function."""

    @patch('personfromvid.models.face_detector.ModelConfigs.get_default_models')
    @patch('personfromvid.models.face_detector.FaceDetector')
    def test_create_with_default_model(self, mock_face_detector, mock_get_defaults):
        """Test creating detector with default model."""
        mock_get_defaults.return_value = {"face_detection": "TEST_FACE_MODEL"}
        mock_detector = Mock()
        mock_face_detector.return_value = mock_detector

        detector = create_face_detector()

        mock_face_detector.assert_called_once_with("TEST_FACE_MODEL", "auto", DEFAULT_CONFIDENCE_THRESHOLD, None)
        assert detector == mock_detector

    @patch('personfromvid.models.face_detector.FaceDetector')
    def test_create_with_custom_params(self, mock_face_detector):
        """Test creating detector with custom parameters."""
        mock_detector = Mock()
        mock_face_detector.return_value = mock_detector

        detector = create_face_detector(
            model_name="yolov8n-face",
            device="cuda",
            confidence_threshold=0.85
        )

        mock_face_detector.assert_called_once_with("yolov8n-face", "cuda", 0.85, None)
        assert detector == mock_detector


class TestIntegrationScenarios:
    """Integration-like tests for common usage scenarios."""

    @patch('personfromvid.models.face_detector.ModelConfigs.get_model')
    @patch('personfromvid.models.face_detector.get_model_manager')
    @patch('onnxruntime.InferenceSession')
    def test_typical_pipeline_usage(self, mock_ort_session, mock_get_manager, mock_get_model):
        """Test typical usage in a processing pipeline."""
        # Setup mocks
        mock_model_config = Mock()
        mock_model_config.input_size = (640, 640)
        mock_model_config.is_device_supported.return_value = True
        mock_model_config.files = [Mock()]
        mock_model_config.files[0].format = ModelFormat.ONNX
        mock_get_model.return_value = mock_model_config

        mock_manager = Mock()
        mock_manager.ensure_model_available.return_value = Path("/models/scrfd.onnx")
        mock_get_manager.return_value = mock_manager

        mock_session = Mock()
        mock_session.get_inputs.return_value = [Mock(name="input")]
        mock_session.get_outputs.return_value = [Mock(name="bboxes"), Mock(name="scores")]
        mock_session.get_providers.return_value = ["CPUExecutionProvider"]

        # Simulate multiple faces detected
        mock_bboxes = np.array([[100, 100, 200, 200], [300, 150, 400, 250]])
        mock_scores = np.array([0.95, 0.85])
        mock_session.run.return_value = [mock_bboxes, mock_scores]
        mock_ort_session.return_value = mock_session

        # Test pipeline usage
        detector = FaceDetector("TEST_FACE_MODEL", confidence_threshold=0.8)

        # Process multiple images as would happen in real pipeline
        test_images = [
            np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            for _ in range(5)
        ]

        batch_results = detector.detect_batch(test_images)

        # Verify results
        assert len(batch_results) == 5
        for faces in batch_results:
            assert len(faces) == 2  # Two faces per image
            for face in faces:
                assert isinstance(face, FaceDetection)
                assert face.confidence >= 0.8  # Above threshold
                assert len(face.bbox) == 4  # Valid bbox format

        # Test individual detection
        single_faces = detector.detect_faces(test_images[0])
        assert len(single_faces) == 2

    @patch('personfromvid.models.face_detector.ModelConfigs.get_model')
    @patch('personfromvid.models.face_detector.get_model_manager')
    def test_error_recovery_scenarios(self, mock_get_manager, mock_get_model):
        """Test error handling and recovery scenarios."""
        mock_model_config = Mock()
        mock_model_config.input_size = (640, 640)
        mock_model_config.is_device_supported.return_value = True
        mock_model_config.files = [Mock()]
        mock_model_config.files[0].format = ModelFormat.ONNX
        mock_get_model.return_value = mock_model_config

        mock_manager = Mock()
        mock_manager.ensure_model_available.return_value = Path("/models/scrfd.onnx")
        mock_get_manager.return_value = mock_manager

        detector = FaceDetector("TEST_FACE_MODEL")

        # Test graceful handling of various error conditions
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Model loading failure should be handled gracefully
        with patch.object(detector, '_load_onnx_model', side_effect=Exception("Model load failed")):
            with pytest.raises(FaceDetectionError, match="Failed to load model"):
                detector.detect_faces(test_image)

        # Should be able to change thresholds and retry
        detector.set_confidence_threshold(0.5)
        assert detector.confidence_threshold == 0.5

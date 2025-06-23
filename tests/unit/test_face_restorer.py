"""Unit tests for the FaceRestorer class."""

import unittest
from unittest.mock import Mock, MagicMock, patch, ANY
import numpy as np
from pathlib import Path

from personfromvid.models.face_restorer import FaceRestorer, create_face_restorer
from personfromvid.data.config import DeviceType
from personfromvid.utils.exceptions import FaceRestorationError


class TestFaceRestorer(unittest.TestCase):
    """Test cases for FaceRestorer class initialization and configuration."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_config = Mock()
        self.mock_model_config = Mock()
        self.mock_model_config.is_device_supported.return_value = True
        self.mock_model_config.supported_devices = [DeviceType.CPU, DeviceType.GPU]
        self.mock_model_config.input_size = (None, None)
        self.mock_model_config.description = "Test GFPGAN model"
        
        self.mock_model_manager = Mock()
        self.mock_model_manager.ensure_model_available.return_value = Path("/mock/path/model.pth")

    @patch('personfromvid.models.face_restorer.get_default_config')
    @patch('personfromvid.models.face_restorer.ModelConfigs')
    @patch('personfromvid.models.face_restorer.get_model_manager')
    def test_initialization_with_defaults(self, mock_get_manager, mock_configs, mock_get_config):
        """Test FaceRestorer initialization with default parameters."""
        # Arrange
        mock_get_config.return_value = self.mock_config
        mock_configs.get_model.return_value = self.mock_model_config
        mock_get_manager.return_value = self.mock_model_manager

        # Act
        restorer = FaceRestorer()

        # Assert
        self.assertEqual(restorer.model_name, "gfpgan_v1_4")
        self.assertEqual(restorer.device, "cpu")  # auto resolves to cpu without torch
        self.assertIsNone(restorer._gfpgan_restorer)
        mock_configs.get_model.assert_called_once_with("gfpgan_v1_4")
        mock_get_manager.assert_called_once()

    @patch('personfromvid.models.face_restorer.get_default_config')
    @patch('personfromvid.models.face_restorer.ModelConfigs')
    @patch('personfromvid.models.face_restorer.get_model_manager')
    def test_initialization_with_custom_params(self, mock_get_manager, mock_configs, mock_get_config):
        """Test FaceRestorer initialization with custom parameters."""
        # Arrange
        custom_config = Mock()
        mock_configs.get_model.return_value = self.mock_model_config
        mock_get_manager.return_value = self.mock_model_manager

        # Act
        restorer = FaceRestorer(
            model_name="custom_model",
            device="cuda",
            config=custom_config
        )

        # Assert
        self.assertEqual(restorer.model_name, "custom_model")
        self.assertEqual(restorer.device, "cuda")
        self.assertEqual(restorer.config, custom_config)
        mock_configs.get_model.assert_called_once_with("custom_model")
        mock_get_config.assert_not_called()

    @patch('personfromvid.models.face_restorer.get_default_config')
    @patch('personfromvid.models.face_restorer.ModelConfigs')
    @patch('personfromvid.models.face_restorer.get_model_manager')
    def test_initialization_unknown_model_error(self, mock_get_manager, mock_configs, mock_get_config):
        """Test FaceRestorer initialization with unknown model raises error."""
        # Arrange
        mock_get_config.return_value = self.mock_config
        mock_configs.get_model.return_value = None
        mock_get_manager.return_value = self.mock_model_manager

        # Act & Assert
        with self.assertRaises(FaceRestorationError) as context:
            FaceRestorer(model_name="unknown_model")
        
        self.assertIn("Unknown face restoration model: unknown_model", str(context.exception))

    @patch('personfromvid.models.face_restorer.get_default_config')
    @patch('personfromvid.models.face_restorer.ModelConfigs')
    @patch('personfromvid.models.face_restorer.get_model_manager')
    def test_initialization_unsupported_device_error(self, mock_get_manager, mock_configs, mock_get_config):
        """Test FaceRestorer initialization with unsupported device raises error."""
        # Arrange
        mock_get_config.return_value = self.mock_config
        mock_configs.get_model.return_value = self.mock_model_config
        mock_get_manager.return_value = self.mock_model_manager
        self.mock_model_config.is_device_supported.return_value = False

        # Act & Assert
        with self.assertRaises(FaceRestorationError) as context:
            FaceRestorer(device="cuda")
        
        self.assertIn("does not support device cuda", str(context.exception))

    def test_resolve_device_auto_without_torch(self):
        """Test device resolution with 'auto' when torch is not available."""
        # Arrange
        restorer = FaceRestorer.__new__(FaceRestorer)  # Create without __init__

        # Act - simulate ImportError when trying to import torch
        with patch('personfromvid.models.face_restorer.torch', create=True, side_effect=ImportError()):
            result = restorer._resolve_device("auto")

        # Assert
        self.assertEqual(result, "cpu")

    @patch('torch.cuda.is_available')
    def test_resolve_device_auto_with_cuda_available(self, mock_cuda_available):
        """Test device resolution with 'auto' when CUDA is available."""
        # Arrange
        mock_cuda_available.return_value = True
        restorer = FaceRestorer.__new__(FaceRestorer)  # Create without __init__

        # Act
        result = restorer._resolve_device("auto")

        # Assert
        self.assertEqual(result, "cuda")

    @patch('torch.cuda.is_available')
    def test_resolve_device_auto_with_cuda_unavailable(self, mock_cuda_available):
        """Test device resolution with 'auto' when CUDA is unavailable."""
        # Arrange
        mock_cuda_available.return_value = False
        restorer = FaceRestorer.__new__(FaceRestorer)  # Create without __init__

        # Act
        result = restorer._resolve_device("auto")

        # Assert
        self.assertEqual(result, "cpu")

    def test_resolve_device_explicit_cpu(self):
        """Test device resolution with explicit 'cpu'."""
        # Arrange
        restorer = FaceRestorer.__new__(FaceRestorer)  # Create without __init__

        # Act
        result = restorer._resolve_device("cpu")

        # Assert
        self.assertEqual(result, "cpu")

    def test_resolve_device_explicit_cuda(self):
        """Test device resolution with explicit 'cuda'."""
        # Arrange
        restorer = FaceRestorer.__new__(FaceRestorer)  # Create without __init__

        # Act
        result = restorer._resolve_device("cuda")

        # Assert
        self.assertEqual(result, "cuda")

    @patch('personfromvid.models.face_restorer.get_default_config')
    @patch('personfromvid.models.face_restorer.ModelConfigs')
    @patch('personfromvid.models.face_restorer.get_model_manager')
    def test_get_model_info(self, mock_get_manager, mock_configs, mock_get_config):
        """Test get_model_info method returns correct information."""
        # Arrange
        mock_get_config.return_value = self.mock_config
        mock_configs.get_model.return_value = self.mock_model_config
        mock_get_manager.return_value = self.mock_model_manager
        
        restorer = FaceRestorer(model_name="test_model", device="cpu")

        # Act
        info = restorer.get_model_info()

        # Assert
        self.assertEqual(info["model_name"], "test_model")
        self.assertEqual(info["device"], "cpu")
        self.assertIn("model_path", info)
        self.assertFalse(info["model_loaded"])  # Model not loaded yet
        self.assertIn("model_config", info)


class TestFaceRestorerModelLoading(unittest.TestCase):
    """Test cases for FaceRestorer model loading functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_config = Mock()
        self.mock_model_config = Mock()
        self.mock_model_config.is_device_supported.return_value = True
        self.mock_model_manager = Mock()
        self.mock_model_manager.ensure_model_available.return_value = Path("/mock/path/model.pth")

    @patch('personfromvid.models.face_restorer.get_default_config')
    @patch('personfromvid.models.face_restorer.ModelConfigs')
    @patch('personfromvid.models.face_restorer.get_model_manager')
    def test_load_model_skips_when_already_loaded(self, mock_get_manager, mock_configs, mock_get_config):
        """Test that _load_model skips loading when model is already loaded."""
        # Arrange
        mock_get_config.return_value = self.mock_config
        mock_configs.get_model.return_value = self.mock_model_config
        mock_get_manager.return_value = self.mock_model_manager

        restorer = FaceRestorer()
        restorer._gfpgan_restorer = Mock()  # Set as already loaded

        # Act
        restorer._load_model()  # Should return early

        # Assert - method should return without doing anything
        self.assertIsNotNone(restorer._gfpgan_restorer)

    @patch('personfromvid.models.face_restorer.get_default_config')
    @patch('personfromvid.models.face_restorer.ModelConfigs') 
    @patch('personfromvid.models.face_restorer.get_model_manager')
    def test_load_model_requires_gfpgan_not_loaded(self, mock_get_manager, mock_configs, mock_get_config):
        """Test that _load_model handles GFPGAN model loading errors."""
        # Arrange
        mock_get_config.return_value = self.mock_config
        mock_configs.get_model.return_value = self.mock_model_config
        mock_get_manager.return_value = self.mock_model_manager

        restorer = FaceRestorer()
        self.assertIsNone(restorer._gfpgan_restorer)  # Not loaded initially

        # Act & Assert - Trying to load with invalid model path should raise error
        with self.assertRaises(FaceRestorationError) as context:
            restorer._load_model()
        
        # With GFPGAN installed, we expect model file loading error
        self.assertIn("Failed to load GFPGAN model", str(context.exception))


class TestFaceRestorerRestoration(unittest.TestCase):
    """Test cases for FaceRestorer restoration functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_config = Mock()
        self.mock_model_config = Mock()
        self.mock_model_config.is_device_supported.return_value = True
        self.mock_model_manager = Mock()
        self.mock_model_manager.ensure_model_available.return_value = Path("/mock/path/model.pth")

        # Create test image (RGB format)
        self.test_image_rgb = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        self.test_image_float = self.test_image_rgb.astype(np.float32) / 255.0

    @patch('personfromvid.models.face_restorer.get_default_config')
    @patch('personfromvid.models.face_restorer.ModelConfigs')
    @patch('personfromvid.models.face_restorer.get_model_manager')
    def test_restore_face_empty_image_error(self, mock_get_manager, mock_configs, mock_get_config):
        """Test restore_face with empty image raises error."""
        # Arrange
        mock_get_config.return_value = self.mock_config
        mock_configs.get_model.return_value = self.mock_model_config
        mock_get_manager.return_value = self.mock_model_manager

        restorer = FaceRestorer()

        # Act & Assert
        with self.assertRaises(FaceRestorationError) as context:
            restorer.restore_face(None, 512)
        
        self.assertIn("Input image is empty or None", str(context.exception))

        # Test with empty array
        empty_image = np.array([])
        with self.assertRaises(FaceRestorationError) as context:
            restorer.restore_face(empty_image, 512)
        
        self.assertIn("Input image is empty or None", str(context.exception))

    @patch('personfromvid.models.face_restorer.get_default_config')
    @patch('personfromvid.models.face_restorer.ModelConfigs')
    @patch('personfromvid.models.face_restorer.get_model_manager')
    @patch('PIL.Image.fromarray')
    def test_restore_face_strength_zero(self, mock_from_array, mock_get_manager, mock_configs, mock_get_config):
        """Test restore_face with strength=0.0 returns original resized image."""
        # Arrange
        mock_get_config.return_value = self.mock_config
        mock_configs.get_model.return_value = self.mock_model_config
        mock_get_manager.return_value = self.mock_model_manager

        # Mock PIL operations
        mock_pil_image = Mock()
        mock_resized_pil = Mock()
        mock_resized_array = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        
        mock_from_array.return_value = mock_pil_image
        mock_pil_image.resize.return_value = mock_resized_pil
        mock_resized_pil.__array__ = Mock(return_value=mock_resized_array)
        
        # Mock np.array conversion
        with patch('numpy.array', return_value=mock_resized_array):
            restorer = FaceRestorer()

            # Act
            result = restorer.restore_face(self.test_image_rgb, target_size=512, strength=0.0)

            # Assert
            np.testing.assert_array_equal(result, mock_resized_array)
            mock_pil_image.resize.assert_called_once_with((512, 512), ANY)

    @patch('personfromvid.models.face_restorer.get_default_config')
    @patch('personfromvid.models.face_restorer.ModelConfigs')
    @patch('personfromvid.models.face_restorer.get_model_manager')
    @patch('PIL.Image.fromarray')
    def test_restore_face_strength_validation(self, mock_from_array, mock_get_manager, mock_configs, mock_get_config):
        """Test restore_face validates and clamps strength parameter."""
        # Arrange
        mock_get_config.return_value = self.mock_config
        mock_configs.get_model.return_value = self.mock_model_config
        mock_get_manager.return_value = self.mock_model_manager

        # Mock PIL operations
        mock_pil_image = Mock()
        mock_resized_pil = Mock()
        mock_resized_array = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        
        mock_from_array.return_value = mock_pil_image
        mock_pil_image.resize.return_value = mock_resized_pil
        
        with patch('numpy.array', return_value=mock_resized_array):
            restorer = FaceRestorer()

            # Test strength clamping
            result = restorer.restore_face(self.test_image_rgb, target_size=512, strength=-0.5)
            # Should clamp to 0.0 and return original
            np.testing.assert_array_equal(result, mock_resized_array)

            result = restorer.restore_face(self.test_image_rgb, target_size=512, strength=1.5)
            # Should clamp to 1.0 and proceed with restoration (but will fail at model loading)

    @patch('personfromvid.models.face_restorer.get_default_config')
    @patch('personfromvid.models.face_restorer.ModelConfigs')
    @patch('personfromvid.models.face_restorer.get_model_manager')
    def test_restore_face_fallback_on_error(self, mock_get_manager, mock_configs, mock_get_config):
        """Test restore_face falls back to original image on GFPGAN error."""
        # Arrange
        mock_get_config.return_value = self.mock_config
        mock_configs.get_model.return_value = self.mock_model_config
        mock_get_manager.return_value = self.mock_model_manager

        restorer = FaceRestorer()
        
        # Mock _load_model to raise an error
        restorer._load_model = Mock(side_effect=Exception("Model load failed"))

        # Mock fallback image
        mock_fallback_array = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        
        with patch.object(restorer, '_resize_to_target', return_value=mock_fallback_array) as mock_resize:
            # Act
            result = restorer.restore_face(self.test_image_rgb, target_size=512, strength=0.8)

            # Assert
            np.testing.assert_array_equal(result, mock_fallback_array)
            # Should have called resize for fallback
            mock_resize.assert_called_once_with(ANY, 512)


class TestCreateFaceRestorer(unittest.TestCase):
    """Test cases for create_face_restorer factory function."""

    @patch('personfromvid.models.face_restorer.FaceRestorer')
    def test_create_face_restorer_defaults(self, mock_face_restorer):
        """Test create_face_restorer with default parameters."""
        # Arrange
        mock_instance = Mock()
        mock_face_restorer.return_value = mock_instance

        # Act
        result = create_face_restorer()

        # Assert
        mock_face_restorer.assert_called_once_with(
            model_name="gfpgan_v1_4",
            device="auto", 
            config=None
        )
        self.assertEqual(result, mock_instance)

    @patch('personfromvid.models.face_restorer.FaceRestorer')
    def test_create_face_restorer_custom_params(self, mock_face_restorer):
        """Test create_face_restorer with custom parameters."""
        # Arrange
        mock_instance = Mock()
        mock_face_restorer.return_value = mock_instance
        custom_config = Mock()

        # Act
        result = create_face_restorer(
            model_name="custom_model",
            device="cuda",
            config=custom_config
        )

        # Assert
        mock_face_restorer.assert_called_once_with(
            model_name="custom_model",
            device="cuda",
            config=custom_config
        )
        self.assertEqual(result, mock_instance)

    @patch('personfromvid.models.face_restorer.FaceRestorer')
    def test_create_face_restorer_none_model_name(self, mock_face_restorer):
        """Test create_face_restorer with None model_name uses default."""
        # Arrange
        mock_instance = Mock()
        mock_face_restorer.return_value = mock_instance

        # Act
        result = create_face_restorer(model_name=None)

        # Assert
        mock_face_restorer.assert_called_once_with(
            model_name="gfpgan_v1_4",
            device="auto",
            config=None
        )


if __name__ == '__main__':
    unittest.main() 
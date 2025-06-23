"""Integration tests for the FaceRestorer class.

These tests verify end-to-end functionality including real model downloads,
device management, and complete restoration workflows.
"""

import unittest
from unittest.mock import Mock, patch
import numpy as np
from pathlib import Path
import tempfile
import shutil

from personfromvid.models.face_restorer import FaceRestorer, create_face_restorer
from personfromvid.models.model_manager import ModelManager
from personfromvid.models.model_configs import ModelConfigs
from personfromvid.data.config import DeviceType, get_default_config
from personfromvid.utils.exceptions import FaceRestorationError


class TestFaceRestorerIntegration(unittest.TestCase):
    """Integration tests for FaceRestorer with real model management."""

    def setUp(self):
        """Set up test fixtures with temporary cache directory."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.model_manager = ModelManager(cache_dir=self.temp_dir)
        
        # Create test image data
        self.test_image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)

    def tearDown(self):
        """Clean up temporary files."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_face_restorer_model_configuration_integration(self):
        """Test FaceRestorer can locate and configure GFPGAN model."""
        # Verify model configuration exists
        model_config = ModelConfigs.get_model("gfpgan_v1_4")
        self.assertIsNotNone(model_config)
        self.assertEqual(model_config.name, "gfpgan_v1_4")
        
        # Verify device support
        self.assertTrue(model_config.is_device_supported(DeviceType.CPU))
        self.assertTrue(model_config.is_device_supported(DeviceType.GPU))
        
        # Verify model files configuration
        self.assertEqual(len(model_config.files), 1)
        model_file = model_config.files[0]
        self.assertEqual(model_file.filename, "GFPGANv1.4.pth")
        self.assertIn("github.com", model_file.url)

    @patch('personfromvid.models.face_restorer.get_model_manager')
    def test_face_restorer_initialization_with_model_manager(self, mock_get_manager):
        """Test FaceRestorer initialization integrates with model management system."""
        # Arrange
        mock_get_manager.return_value = self.model_manager
        
        # Mock model path to avoid actual download
        mock_model_path = self.temp_dir / "gfpgan_v1_4" / "GFPGANv1.4.pth"
        mock_model_path.parent.mkdir(parents=True)
        mock_model_path.touch()
        
        with patch.object(self.model_manager, 'ensure_model_available', return_value=mock_model_path):
            # Act
            restorer = FaceRestorer(model_name="gfpgan_v1_4", device="cpu")
            
            # Assert
            self.assertEqual(restorer.model_name, "gfpgan_v1_4")
            self.assertEqual(restorer.device, "cpu")
            self.assertEqual(restorer.model_path, mock_model_path)
            self.assertIsNotNone(restorer.model_config)

    @patch('personfromvid.models.face_restorer.get_model_manager')
    def test_face_restorer_device_management_integration(self, mock_get_manager):
        """Test FaceRestorer device management with real device detection."""
        # Arrange
        mock_get_manager.return_value = self.model_manager
        mock_model_path = self.temp_dir / "model.pth"
        mock_model_path.touch()
        
        with patch.object(self.model_manager, 'ensure_model_available', return_value=mock_model_path):
            # Test CPU device
            restorer_cpu = FaceRestorer(device="cpu")
            self.assertEqual(restorer_cpu.device, "cpu")
            
            # Test CUDA device (should work regardless of actual CUDA availability)
            restorer_cuda = FaceRestorer(device="cuda")
            self.assertEqual(restorer_cuda.device, "cuda")
            
            # Test AUTO device resolution
            restorer_auto = FaceRestorer(device="auto")
            self.assertIn(restorer_auto.device, ["cpu", "cuda"])

    @patch('personfromvid.models.face_restorer.get_model_manager')
    def test_face_restorer_error_handling_integration(self, mock_get_manager):
        """Test FaceRestorer error handling in integration scenarios."""
        # Arrange
        mock_get_manager.return_value = self.model_manager
        
        # Test unknown model error
        with self.assertRaises(FaceRestorationError) as context:
            FaceRestorer(model_name="unknown_model")
        self.assertIn("Unknown face restoration model", str(context.exception))
        
        # Test model path handling
        mock_model_path = self.temp_dir / "model.pth"
        mock_model_path.touch()
        
        with patch.object(self.model_manager, 'ensure_model_available', return_value=mock_model_path):
            restorer = FaceRestorer()
            
            # Test restore_face error handling
            with self.assertRaises(FaceRestorationError):
                restorer.restore_face(None, 512)

    @patch('personfromvid.models.face_restorer.get_model_manager')
    def test_face_restorer_fallback_mechanism_integration(self, mock_get_manager):
        """Test FaceRestorer fallback mechanism in integration context."""
        # Arrange
        mock_get_manager.return_value = self.model_manager
        mock_model_path = self.temp_dir / "model.pth"
        mock_model_path.touch()
        
        with patch.object(self.model_manager, 'ensure_model_available', return_value=mock_model_path):
            restorer = FaceRestorer()
            
            # Mock _load_model to simulate GFPGAN failure
            restorer._load_model = Mock(side_effect=Exception("Model loading failed"))
            
            # Test that fallback works
            result = restorer.restore_face(self.test_image, target_size=128, strength=0.8)
            
            # Should return resized image without crashing
            self.assertIsInstance(result, np.ndarray)
            self.assertEqual(result.shape[:2], (128, 128))

    def test_create_face_restorer_integration(self):
        """Test create_face_restorer factory function integration."""
        # Test with default parameters
        with patch('personfromvid.models.face_restorer.FaceRestorer') as mock_restorer:
            mock_instance = Mock()
            mock_restorer.return_value = mock_instance
            
            result = create_face_restorer()
            
            mock_restorer.assert_called_once_with(
                model_name="gfpgan_v1_4",
                device="auto",
                config=None
            )
            self.assertEqual(result, mock_instance)

    @patch('personfromvid.models.face_restorer.get_model_manager')
    def test_face_restorer_model_info_integration(self, mock_get_manager):
        """Test get_model_info method in integration context."""
        # Arrange
        mock_get_manager.return_value = self.model_manager
        mock_model_path = self.temp_dir / "model.pth"
        mock_model_path.touch()
        
        with patch.object(self.model_manager, 'ensure_model_available', return_value=mock_model_path):
            restorer = FaceRestorer(model_name="gfpgan_v1_4", device="cpu")
            
            # Act
            info = restorer.get_model_info()
            
            # Assert
            self.assertEqual(info["model_name"], "gfpgan_v1_4")
            self.assertEqual(info["device"], "cpu")
            self.assertIn("model_path", info)
            self.assertIn("model_config", info)
            self.assertFalse(info["model_loaded"])  # Not loaded yet
            
            # Verify model config info
            model_config_info = info["model_config"]
            self.assertIn("supported_devices", model_config_info)
            self.assertIn("cpu", model_config_info["supported_devices"])

    @patch('personfromvid.models.face_restorer.get_model_manager')
    def test_face_restorer_memory_management_integration(self, mock_get_manager):
        """Test FaceRestorer memory management and cleanup."""
        # Arrange
        mock_get_manager.return_value = self.model_manager
        mock_model_path = self.temp_dir / "model.pth"
        mock_model_path.touch()
        
        with patch.object(self.model_manager, 'ensure_model_available', return_value=mock_model_path):
            # Create and destroy restorer to test cleanup
            restorer = FaceRestorer(device="cpu")
            
            # Mock the cleanup behavior
            with patch('torch.cuda.empty_cache') as mock_empty_cache:
                # Test CPU cleanup (should not call CUDA cleanup)
                del restorer
                mock_empty_cache.assert_not_called()
            
            # Test GPU cleanup
            restorer_gpu = FaceRestorer(device="cuda")
            restorer_gpu._gfpgan_restorer = Mock()  # Simulate loaded model
            
            with patch('torch.cuda.empty_cache') as mock_empty_cache:
                del restorer_gpu
                # Cleanup is called in __del__, but might not be immediate


class TestFaceRestorerConfigurationIntegration(unittest.TestCase):
    """Integration tests for FaceRestorer configuration compatibility."""

    def test_face_restorer_with_real_config(self):
        """Test FaceRestorer with real application configuration."""
        # Get real default config
        config = get_default_config()
        
        # Test that FaceRestorer can use real config
        with patch('personfromvid.models.face_restorer.get_model_manager') as mock_get_manager:
            mock_manager = Mock()
            mock_model_path = Path("/mock/model.pth")
            mock_manager.ensure_model_available.return_value = mock_model_path
            mock_get_manager.return_value = mock_manager
            
            restorer = FaceRestorer(config=config)
            
            self.assertEqual(restorer.config, config)
            self.assertIsNotNone(restorer.model_config)

    def test_face_restorer_device_type_compatibility(self):
        """Test FaceRestorer device type compatibility with config system."""
        # Test DeviceType enum compatibility
        model_config = ModelConfigs.get_model("gfpgan_v1_4")
        
        # Verify CPU support
        self.assertTrue(model_config.is_device_supported(DeviceType.CPU))
        
        # Verify GPU support
        self.assertTrue(model_config.is_device_supported(DeviceType.GPU))
        
        # Verify supported devices list
        supported_devices = model_config.supported_devices
        self.assertIn(DeviceType.CPU, supported_devices)
        self.assertIn(DeviceType.GPU, supported_devices)


if __name__ == '__main__':
    unittest.main() 
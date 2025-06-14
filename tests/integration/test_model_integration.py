"""Integration tests for model management system.

These tests demonstrate realistic usage patterns of the simplified ModelManager.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

from personfromvid.models.model_manager import ModelManager, get_model_manager
from personfromvid.models import model_utils
from personfromvid.models.model_configs import ModelConfigs
from personfromvid.utils.exceptions import ModelNotFoundError, ModelDownloadError

# Test constants to reduce duplication and ensure consistency
TEST_MODELS = {
    "FACE_DETECTION": "scrfd_10g",  # Use actual model name from config
    "POSE_ESTIMATION": "yolov8n-pose",
    "HEAD_POSE": "sixdrepnet"  # Use actual default model from config
}


class TestModelManagerIntegration:
    """Integration tests for the complete model management workflow."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        # Reset global manager to use our temp directory
        import personfromvid.models.model_manager as mm
        mm._model_manager = None
    
    def teardown_method(self):
        """Clean up test environment."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
        # Reset global manager
        import personfromvid.models.model_manager as mm
        mm._model_manager = None
    
    def test_typical_usage_workflow(self):
        """Test the typical workflow: check if cached, download if not, use model."""
        model_name = TEST_MODELS["FACE_DETECTION"]
        manager = ModelManager(cache_dir=self.temp_dir)
        
        # Step 1: Initially model is not cached
        assert not manager.is_model_cached(model_name)
        
        # Step 2: Try to get model path (should fail)
        with pytest.raises(ModelNotFoundError):
            manager.get_model_path(model_name)
        
        # Step 3: Mock successful download
        with patch.object(manager, '_download_file') as mock_download:
            # Create fake model file to simulate successful download
            model = ModelConfigs.get_model(model_name)
            model_dir = self.temp_dir / model_name
            model_dir.mkdir(parents=True)
            primary_file = model_dir / model.get_primary_file().filename
            primary_file.write_bytes(b"fake model content")
            
            # This would normally download, but we've pre-created the files
            model_path = manager.ensure_model_available(model_name)
            
            # Verify we got the right path
            assert model_path.exists()
            assert model_path == primary_file
            assert manager.is_model_cached(model_name)
    
    def test_ensure_model_available_idempotency(self):
        """Test that ensure_model_available is idempotent."""
        model_name = TEST_MODELS["FACE_DETECTION"]
        manager = ModelManager(cache_dir=self.temp_dir)
        
        # Create fake cached model
        model = ModelConfigs.get_model(model_name)
        model_dir = self.temp_dir / model_name
        model_dir.mkdir(parents=True)
        primary_file = model_dir / model.get_primary_file().filename
        primary_file.write_bytes(b"fake model content")
        
        # First call should return the cached model
        path1 = manager.ensure_model_available(model_name)
        
        # Second call should return the same path without re-downloading
        with patch.object(manager, 'download_model') as mock_download:
            path2 = manager.ensure_model_available(model_name)
            
            # Should not have called download
            mock_download.assert_not_called()
            assert path1 == path2
    
    def test_multiple_models_management(self):
        """Test managing multiple models simultaneously."""
        model_names = [TEST_MODELS["FACE_DETECTION"], TEST_MODELS["POSE_ESTIMATION"]]
        manager = ModelManager(cache_dir=self.temp_dir)
        
        # Create fake models
        for model_name in model_names:
            model = ModelConfigs.get_model(model_name)
            model_dir = self.temp_dir / model_name
            model_dir.mkdir(parents=True)
            
            # Create all required files for the model
            for file_info in model.files:
                file_path = model_dir / file_info.filename
                file_path.write_bytes(f"fake {file_info.filename} content".encode())
        
        # Verify both models are cached
        for model_name in model_names:
            assert manager.is_model_cached(model_name)
            model_path = manager.get_model_path(model_name)
            assert model_path.exists()
        
        # Verify listing shows both models
        cached_models = manager.list_cached_models()
        for model_name in model_names:
            assert model_name in cached_models
        
        # Verify cache size calculation
        cache_size = manager.get_cache_size()
        assert cache_size > 0
    
    def test_global_manager_usage(self):
        """Test using the global manager pattern."""
        # First call creates manager with default cache
        manager1 = get_model_manager()
        
        # Second call returns same instance
        manager2 = get_model_manager()
        assert manager1 is manager2
        
        # But providing cache_dir creates new instance
        manager3 = get_model_manager(cache_dir=self.temp_dir)
        assert manager3 is not manager1
        assert manager3.cache_dir == self.temp_dir
    
    def test_model_utilities_integration(self):
        """Test integration between ModelManager and model utilities."""
        model_name = TEST_MODELS["FACE_DETECTION"]
        manager = ModelManager(cache_dir=self.temp_dir)
        
        # Create fake model with known content for hash verification
        model = ModelConfigs.get_model(model_name)
        model_dir = self.temp_dir / model_name
        model_dir.mkdir(parents=True)
        
        test_content = b"fake model content for hash test"
        primary_file = model_dir / model.get_primary_file().filename
        primary_file.write_bytes(test_content)
        
        # Test utility functions
        download_time = model_utils.get_model_download_time(model_dir)
        assert download_time is not None
        
        # Test validation with mocked model config
        import hashlib
        expected_hash = hashlib.sha256(test_content).hexdigest()
        
        mock_file = MagicMock()
        mock_file.filename = model.get_primary_file().filename
        mock_file.sha256_hash = expected_hash
        
        mock_model = MagicMock()
        mock_model.files = [mock_file]
        
        with patch.object(ModelConfigs, 'get_model', return_value=mock_model):
            results = model_utils.validate_model_cache(self.temp_dir)
            assert results[model_name] is True
    
    def test_error_recovery_workflow(self):
        """Test error handling and recovery scenarios."""
        model_name = TEST_MODELS["FACE_DETECTION"]
        manager = ModelManager(cache_dir=self.temp_dir)
        
        # Test download failure
        with patch.object(manager, '_download_file', side_effect=Exception("Network error")):
            with pytest.raises(ModelDownloadError):
                manager.download_model(model_name)
            
            # Verify cleanup happened
            assert not manager.is_model_cached(model_name)
            model_dir = self.temp_dir / model_name
            assert not model_dir.exists()
        
        # Test partial model (missing files)
        model = ModelConfigs.get_model(model_name)
        if len(model.files) > 1:
            model_dir = self.temp_dir / model_name
            model_dir.mkdir(parents=True)
            
            # Create only first file
            (model_dir / model.files[0].filename).write_bytes(b"partial model")
            
            # Should not be considered cached
            assert not manager.is_model_cached(model_name)
    
    def test_cache_management_operations(self):
        """Test cache management and cleanup operations."""
        model_names = ["test_model_1", "test_model_2"]
        manager = ModelManager(cache_dir=self.temp_dir)
        
        # Create fake models
        for model_name in model_names:
            model_dir = self.temp_dir / model_name
            model_dir.mkdir(parents=True)
            (model_dir / "model.bin").write_bytes(b"fake model")
        
        # Test cache size before cleanup
        initial_size = manager.get_cache_size()
        assert initial_size > 0
        
        # Test cache clearing
        manager.clear_cache()
        
        # Verify cache is empty
        assert manager.get_cache_size() == 0
        assert len(manager.list_cached_models()) == 0
        
        # Verify cache directory still exists
        assert manager.cache_dir.exists()
    
    def test_realistic_ai_model_loading_scenario(self):
        """Test a realistic scenario of loading models for AI inference."""
        # Scenario: Application needs face detection model for processing
        manager = ModelManager(cache_dir=self.temp_dir)
        
        # Models typically needed for face detection pipeline
        required_models = [TEST_MODELS["FACE_DETECTION"]]  # Just one for simplicity
        
        model_paths = {}
        for model_name in required_models:
            # Create fake model to simulate it being available
            model = ModelConfigs.get_model(model_name)
            model_dir = self.temp_dir / model_name
            model_dir.mkdir(parents=True)
            
            for file_info in model.files:
                file_path = model_dir / file_info.filename
                file_path.write_bytes(f"fake {file_info.filename}".encode())
            
            # This is how the application would ensure models are available
            model_path = manager.ensure_model_available(model_name)
            model_paths[model_name] = model_path
            
            # Verify model is ready for use
            assert model_path.exists()
            assert model_path.is_file()
        
        # Simulate successful model loading
        for model_name, model_path in model_paths.items():
            # In real usage, this would be something like:
            # model = load_onnx_model(model_path)
            # or: model = torch.load(model_path)
            assert model_path.suffix in ['.onnx', '.pt', '.pth', '.pkl']
            
        # Verify all models are cached for future use
        cached_models = manager.list_cached_models()
        for model_name in required_models:
            assert model_name in cached_models 
"""Unit tests for model management system."""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
from datetime import datetime

from personfromvid.models.model_configs import (
    ModelConfigs, ModelMetadata, ModelFile, ModelFormat, ModelProvider
)
from personfromvid.models.model_manager import ModelManager, get_model_manager
from personfromvid.data.config import DeviceType
from personfromvid.utils.exceptions import ModelNotFoundError, ModelDownloadError

# Test constants to reduce duplication
TEST_MODELS = {
    "FACE_DETECTION": "scrfd_10g",  # Use actual model name from config
    "POSE_ESTIMATION": "yolov8n-pose",
    "HEAD_POSE": "hopenet_alpha1"  # Use actual model name from config
}


class TestModelConfigs:
    """Tests for ModelConfigs class."""
    
    def test_get_all_models(self):
        """Test getting all model configurations."""
        models = ModelConfigs.get_all_models()
        
        assert isinstance(models, dict)
        assert len(models) > 0
        
        # Check that we have expected model types using actual model names
        model_names = list(models.keys())
        assert TEST_MODELS["FACE_DETECTION"] in model_names
        assert TEST_MODELS["POSE_ESTIMATION"] in model_names
        assert TEST_MODELS["HEAD_POSE"] in model_names
    
    def test_get_model(self):
        """Test getting a specific model configuration."""
        # Test existing model using constant
        model = ModelConfigs.get_model(TEST_MODELS["FACE_DETECTION"])
        assert model is not None
        assert model.name == TEST_MODELS["FACE_DETECTION"]
        assert model.provider in [ModelProvider.DIRECT_URL, ModelProvider.HUGGINGFACE, ModelProvider.GITHUB, ModelProvider.ULTRALYTICS]
        assert len(model.files) > 0
        
        # Test non-existing model
        model = ModelConfigs.get_model("non_existent_model")
        assert model is None
    
    def test_get_models_by_type(self):
        """Test getting models by type."""
        face_models = ModelConfigs.get_models_by_type("face")
        pose_models = ModelConfigs.get_models_by_type("pose")
        head_pose_models = ModelConfigs.get_models_by_type("head_pose")
        
        assert len(face_models) >= 2  # scrfd, yoloface
        assert len(pose_models) >= 2  # yolov8n-pose, yolov8s-pose
        assert len(head_pose_models) >= 2  # hopenet, sixdrepnet
        
        # Test invalid type
        invalid_models = ModelConfigs.get_models_by_type("invalid_type")
        assert len(invalid_models) == 0
    
    def test_get_default_models(self):
        """Test getting default model names."""
        defaults = ModelConfigs.get_default_models()
        
        assert isinstance(defaults, dict)
        assert "face_detection" in defaults
        assert "pose_estimation" in defaults
        assert "head_pose_estimation" in defaults
        
        # Verify default models exist in configuration
        for model_name in defaults.values():
            model = ModelConfigs.get_model(model_name)
            assert model is not None
    
    def test_validate_model_config(self):
        """Test model configuration validation."""
        # Test valid model with supported device using constant
        valid = ModelConfigs.validate_model_config(TEST_MODELS["FACE_DETECTION"], DeviceType.CPU)
        assert valid is True
        
        # Test invalid model
        valid = ModelConfigs.validate_model_config("non_existent", DeviceType.CPU)
        assert valid is False


class TestModelMetadata:
    """Tests for ModelMetadata class."""
    
    def test_get_primary_file(self):
        """Test getting primary model file."""
        model = ModelConfigs.get_model(TEST_MODELS["FACE_DETECTION"])
        primary_file = model.get_primary_file()
        
        assert isinstance(primary_file, ModelFile)
        assert primary_file.filename.endswith(".onnx")
    
    def test_get_cache_key(self):
        """Test cache key generation."""
        model = ModelConfigs.get_model(TEST_MODELS["FACE_DETECTION"])
        cache_key = model.get_cache_key()
        
        assert isinstance(cache_key, str)
        assert len(cache_key) == 32  # MD5 hash length
    
    def test_is_device_supported(self):
        """Test device support checking."""
        model = ModelConfigs.get_model(TEST_MODELS["FACE_DETECTION"])
        
        assert model.is_device_supported(DeviceType.CPU) is True
        assert model.is_device_supported(DeviceType.GPU) is True
        assert model.is_device_supported(DeviceType.AUTO) is False  # Not explicitly supported
    
    def test_model_with_no_files_raises_error(self):
        """Test that models with no files raise appropriate errors."""
        # Create a model with no files
        empty_model = ModelMetadata(
            name="test_empty",
            version="1.0.0",
            provider=ModelProvider.HUGGINGFACE,
            files=[],  # Empty files list
            supported_devices=[DeviceType.CPU],
            input_size=(640, 640),
            description="Test model with no files",
            license="MIT"
        )
        
        with pytest.raises(ValueError, match="No files defined"):
            empty_model.get_primary_file()


class TestModelManager:
    """Tests for ModelManager class."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.model_manager = ModelManager(cache_dir=self.temp_dir)
    
    def teardown_method(self):
        """Clean up test environment."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test ModelManager initialization."""
        assert self.model_manager.cache_dir == self.temp_dir
        assert self.temp_dir.exists()
    
    def test_is_model_cached_false_for_uncached(self):
        """Test that uncached models return False."""
        assert self.model_manager.is_model_cached("TEST_FACE_MODEL") is False
    
    def test_is_model_cached_false_for_nonexistent_model(self):
        """Test that nonexistent models return False."""
        assert self.model_manager.is_model_cached("nonexistent_model") is False
    
    def test_get_model_path_raises_for_uncached(self):
        """Test that getting path for uncached model raises error."""
        with pytest.raises(ModelNotFoundError):
            self.model_manager.get_model_path("TEST_FACE_MODEL")
    
    def test_get_model_path_raises_for_nonexistent(self):
        """Test that getting path for nonexistent model raises error."""
        with pytest.raises(ModelNotFoundError):
            self.model_manager.get_model_path("nonexistent_model")
    
    def test_list_cached_models_empty_initially(self):
        """Test that cached models list is empty initially."""
        cached = self.model_manager.list_cached_models()
        assert cached == []
    
    def test_get_cache_size_zero_initially(self):
        """Test that cache size is zero initially."""
        size = self.model_manager.get_cache_size()
        assert size == 0
    
    def test_download_model_with_invalid_name(self):
        """Test downloading non-existent model raises error."""
        with pytest.raises(ModelNotFoundError):
            self.model_manager.download_model("non_existent_model")
    
    @patch('personfromvid.models.model_manager.requests.get')
    @patch('personfromvid.models.model_manager.hf_hub_download')
    def test_download_model_success(self, mock_hf_download, mock_requests):
        """Test successful model download."""
        # Setup mock responses using test constant
        model_name = TEST_MODELS["FACE_DETECTION"]
        model = ModelConfigs.get_model(model_name)
        
        # Mock file download
        test_file_content = b"fake model data"
        mock_response = MagicMock()
        mock_response.iter_content.return_value = [test_file_content]
        mock_response.headers = {'content-length': str(len(test_file_content))}
        mock_response.raise_for_status.return_value = None
        mock_requests.return_value = mock_response
        
        # Create fake file for hf_hub_download to return
        fake_file = self.temp_dir / model_name / model.get_primary_file().filename
        fake_file.parent.mkdir(parents=True, exist_ok=True)
        fake_file.write_bytes(test_file_content)
        mock_hf_download.return_value = str(fake_file)
        
        # Download the model
        result_path = self.model_manager.download_model(model_name)
        
        # Verify results
        assert result_path.exists()
        assert self.model_manager.is_model_cached(model_name)
        assert model_name in self.model_manager.list_cached_models()
    
    @patch('personfromvid.models.model_manager.requests.get')
    def test_download_model_failure_cleans_up(self, mock_requests):
        """Test that failed downloads clean up partial files."""
        model_name = TEST_MODELS["FACE_DETECTION"]
        
        # Mock failed download
        mock_requests.side_effect = Exception("Download failed")
        
        # Attempt download
        with pytest.raises(ModelDownloadError):
            self.model_manager.download_model(model_name)
        
        # Verify cleanup
        assert not self.model_manager.is_model_cached(model_name)
        model_dir = self.temp_dir / model_name
        assert not model_dir.exists()
    
    def test_ensure_model_available_downloads_if_missing(self):
        """Test that ensure_model_available downloads missing models."""
        model_name = "TEST_FACE_MODEL"
        
        with patch.object(self.model_manager, 'download_model') as mock_download:
            mock_download.return_value = Path("/fake/path")
            
            result = self.model_manager.ensure_model_available(model_name)
            
            mock_download.assert_called_once_with(model_name)
            assert result == Path("/fake/path")
    
    def test_ensure_model_available_returns_existing_path(self):
        """Test that ensure_model_available returns path for cached models."""
        model_name = TEST_MODELS["FACE_DETECTION"]
        
        # Create fake cached model
        model = ModelConfigs.get_model(model_name)
        model_dir = self.temp_dir / model_name
        model_dir.mkdir(parents=True)
        (model_dir / model.get_primary_file().filename).write_text("fake model")
        
        with patch.object(self.model_manager, 'download_model') as mock_download:
            result = self.model_manager.ensure_model_available(model_name)
            
            # Should not call download
            mock_download.assert_not_called()
            assert result.exists()
            assert result.name == model.get_primary_file().filename
    
    def test_is_model_cached_with_partial_files(self):
        """Test that models with missing files are not considered cached."""
        model_name = TEST_MODELS["FACE_DETECTION"]
        model = ModelConfigs.get_model(model_name)
        
        # Create model directory but only some files
        model_dir = self.temp_dir / model_name
        model_dir.mkdir(parents=True)
        
        # Create only the first file (assuming multiple files)
        if len(model.files) > 1:
            (model_dir / model.files[0].filename).write_text("fake")
            # Don't create the other files
            
            assert not self.model_manager.is_model_cached(model_name)
    
    def test_clear_cache(self):
        """Test clearing the model cache."""
        # Create some fake cached models
        model_names = ["model1", "model2"]
        for name in model_names:
            model_dir = self.temp_dir / name
            model_dir.mkdir(parents=True)
            (model_dir / "model.bin").write_text("fake model")
        
        # Verify models exist
        assert len(list(self.temp_dir.iterdir())) == 2
        
        # Clear cache
        self.model_manager.clear_cache()
        
        # Verify cache is empty but directory still exists
        assert self.temp_dir.exists()
        assert len(list(self.temp_dir.iterdir())) == 0
        assert self.model_manager.get_cache_size() == 0
    
    def test_get_cache_size_with_models(self):
        """Test cache size calculation with models."""
        # Create fake model files
        model_dir = self.temp_dir / "test_model"
        model_dir.mkdir(parents=True)
        
        file1 = model_dir / "model.bin"
        file2 = model_dir / "config.json"
        
        file1.write_text("x" * 1000)  # 1KB
        file2.write_text("y" * 500)   # 0.5KB
        
        size = self.model_manager.get_cache_size()
        assert size == 1500  # 1.5KB total

    # NEW TESTS TO IMPROVE COVERAGE
    
    @patch('personfromvid.models.model_manager.hf_hub_download')
    def test_download_from_huggingface_invalid_url(self, mock_hf_download):
        """Test handling of invalid Hugging Face URLs."""
        model_name = "TEST_FACE_MODEL"
        model = ModelConfigs.get_model(model_name)
        
        # Create mock file with invalid URL format
        mock_file = MagicMock()
        mock_file.url = "https://invalid-url"  # Too few parts
        mock_file.filename = "test.onnx"
        
        file_path = self.temp_dir / "test.onnx"
        
        # Mock the fallback download
        with patch.object(self.model_manager, '_download_from_url') as mock_fallback:
            self.model_manager._download_from_huggingface(mock_file, file_path)
            
            # Should call fallback download due to invalid URL
            mock_fallback.assert_called_once_with(mock_file, file_path)
    
    @patch('personfromvid.models.model_manager.hf_hub_download')
    @patch('personfromvid.models.model_manager.shutil.move')
    def test_download_from_huggingface_file_move(self, mock_move, mock_hf_download):
        """Test file moving when HF download returns different path."""
        mock_file = MagicMock()
        mock_file.url = "https://huggingface.co/user/model/resolve/main/test.onnx"
        mock_file.filename = "test.onnx"
        
        file_path = self.temp_dir / "test.onnx"
        different_path = self.temp_dir / "different.onnx"
        
        # Mock HF download returning different path
        mock_hf_download.return_value = str(different_path)
        
        self.model_manager._download_from_huggingface(mock_file, file_path)
        
        # Should call move to relocate file
        mock_move.assert_called_once_with(str(different_path), str(file_path))
    
    @patch('personfromvid.models.model_manager.requests.get')
    def test_download_from_url_no_content_length(self, mock_requests):
        """Test URL download when content-length header is missing."""
        mock_file = MagicMock()
        mock_file.url = "https://example.com/model.bin"
        mock_file.filename = "model.bin"
        
        file_path = self.temp_dir / "model.bin"
        
        # Mock response without content-length
        mock_response = MagicMock()
        mock_response.iter_content.return_value = [b"test content"]
        mock_response.headers = {}  # No content-length
        mock_response.raise_for_status.return_value = None
        mock_requests.return_value = mock_response
        
        # Should not raise error even without content-length
        self.model_manager._download_from_url(mock_file, file_path)
        
        # File should be created
        assert file_path.exists()
    
    def test_download_file_direct_provider(self):
        """Test _download_file with non-HuggingFace provider."""
        mock_file = MagicMock()
        mock_file.url = "https://example.com/model.bin"
        mock_file.filename = "model.bin"
        
        file_path = self.temp_dir / "model.bin"
        
        with patch.object(self.model_manager, '_download_from_url') as mock_url_download:
            # Test with DIRECT_URL provider (not HUGGINGFACE)
            self.model_manager._download_file(mock_file, file_path, ModelProvider.DIRECT_URL)
            
            mock_url_download.assert_called_once_with(mock_file, file_path)
    
    def test_get_model_path_with_cached_but_no_metadata(self):
        """Test get_model_path when model is cached but metadata is gone."""
        model_name = "test_model"
        
        # Create fake cached model
        model_dir = self.temp_dir / model_name
        model_dir.mkdir(parents=True)
        (model_dir / "model.bin").write_text("fake model")
        
        # Mock ModelConfigs to return None (metadata not found)
        # This affects is_model_cached() which will return False, so the error will be "not cached"
        with patch.object(ModelConfigs, 'get_model', return_value=None):
            with pytest.raises(ModelNotFoundError, match="is not cached"):
                self.model_manager.get_model_path(model_name)
    
    def test_list_cached_models_skips_non_directories(self):
        """Test that list_cached_models skips non-directory files."""
        # Create a regular file in cache dir
        (self.temp_dir / "not_a_model.txt").write_text("test")
        
        # Create a valid model directory using test constant
        model_name = "test_model"
        model = ModelConfigs.get_model(TEST_MODELS["FACE_DETECTION"])  # Use real model for metadata
        model_dir = self.temp_dir / model_name
        model_dir.mkdir(parents=True)
        
        # Mock the model to appear as "test_model"
        with patch.object(ModelConfigs, 'get_model') as mock_get_model:
            mock_get_model.side_effect = lambda name: model if name == model_name else None
            
            # Create all required files for the mocked model
            for file_info in model.files:
                (model_dir / file_info.filename).write_text("fake")
            
            cached_models = self.model_manager.list_cached_models()
            
            # Should only include the directory, not the text file
            assert model_name in cached_models
            assert len(cached_models) == 1
    
    def test_initialization_with_default_config(self):
        """Test ModelManager initialization with default config."""
        # Mock the default config path - need to patch the import within the function
        with patch('personfromvid.data.config.get_default_config') as mock_config:
            mock_config_obj = MagicMock()
            mock_config_obj.storage.cache_directory = self.temp_dir  # Use real temp dir
            mock_config.return_value = mock_config_obj
            
            # Create manager without cache_dir
            manager = ModelManager()
            
            # Should use path from config
            assert manager.cache_dir == self.temp_dir / "models"


class TestGetModelManager:
    """Tests for get_model_manager function."""
    
    def test_get_model_manager_singleton(self):
        """Test that get_model_manager returns singleton instance."""
        manager1 = get_model_manager()
        manager2 = get_model_manager()
        
        assert manager1 is manager2
    
    def test_get_model_manager_with_custom_cache_dir(self):
        """Test get_model_manager with custom cache directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = get_model_manager(cache_dir=temp_dir)
            
            assert manager.cache_dir == Path(temp_dir)
            assert manager.cache_dir.exists()


if __name__ == "__main__":
    pytest.main([__file__]) 
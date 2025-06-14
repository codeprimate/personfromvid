"""Unit tests for model utilities."""

import pytest
import tempfile
import shutil
import hashlib
from pathlib import Path
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

from personfromvid.models import model_utils
from personfromvid.models.model_configs import ModelConfigs, ModelMetadata, ModelFile, ModelProvider
from personfromvid.data.config import DeviceType


class TestVerifyFileIntegrity:
    """Tests for verify_file_integrity function."""
    
    def test_verify_correct_hash(self):
        """Test that correct hash verification passes."""
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            test_content = b"test file content"
            temp_file.write(test_content)
            temp_file.flush()
            
            # Calculate expected hash
            expected_hash = hashlib.sha256(test_content).hexdigest()
            
            try:
                result = model_utils.verify_file_integrity(Path(temp_file.name), expected_hash)
                assert result is True
            finally:
                Path(temp_file.name).unlink()
    
    def test_verify_incorrect_hash(self):
        """Test that incorrect hash verification fails."""
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            test_content = b"test file content"
            temp_file.write(test_content)
            temp_file.flush()
            
            # Use wrong hash
            wrong_hash = "0" * 64
            
            try:
                result = model_utils.verify_file_integrity(Path(temp_file.name), wrong_hash)
                assert result is False
            finally:
                Path(temp_file.name).unlink()
    
    def test_verify_nonexistent_file(self):
        """Test that nonexistent file verification fails."""
        nonexistent_path = Path("/path/that/does/not/exist")
        result = model_utils.verify_file_integrity(nonexistent_path, "somehash")
        assert result is False
    
    def test_verify_file_read_error(self):
        """Test handling of file read errors."""
        with tempfile.NamedTemporaryFile() as temp_file:
            temp_path = Path(temp_file.name)
            
            # Mock open to raise an exception
            with patch('builtins.open', side_effect=PermissionError("Access denied")):
                result = model_utils.verify_file_integrity(temp_path, "somehash")
                assert result is False


class TestGetModelDownloadTime:
    """Tests for get_model_download_time function."""
    
    def test_existing_directory(self):
        """Test getting download time for existing directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir) / "test_model"
            model_dir.mkdir()
            
            download_time = model_utils.get_model_download_time(model_dir)
            
            # Should be recent (within last minute)
            now = datetime.now()
            assert (now - download_time).total_seconds() < 60
    
    def test_nonexistent_directory(self):
        """Test getting download time for nonexistent directory."""
        nonexistent_dir = Path("/path/that/does/not/exist")
        download_time = model_utils.get_model_download_time(nonexistent_dir)
        
        # Should return current time
        now = datetime.now()
        assert (now - download_time).total_seconds() < 1


class TestCleanupOldModels:
    """Tests for cleanup_old_models function."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def teardown_method(self):
        """Clean up test environment."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_cleanup_old_models(self):
        """Test cleaning up models older than specified days."""
        import os
        
        # Create old model directory
        old_model_dir = self.temp_dir / "old_model"
        old_model_dir.mkdir()
        (old_model_dir / "model.bin").write_text("fake model")
        
        # Create recent model directory
        recent_model_dir = self.temp_dir / "recent_model"
        recent_model_dir.mkdir()
        (recent_model_dir / "model.bin").write_text("fake model")
        
        # Modify old directory timestamp to be old
        old_timestamp = (datetime.now() - timedelta(days=35)).timestamp()
        os.utime(old_model_dir, (old_timestamp, old_timestamp))
        
        # Mock ModelConfigs to return None for both (so they're treated as obsolete)
        with patch.object(ModelConfigs, 'get_model', return_value=None):
            removed = model_utils.cleanup_old_models(self.temp_dir, keep_days=30)
        
        # Both should be removed since they're not in configuration
        assert "old_model" in removed
        assert "recent_model" in removed
        assert not old_model_dir.exists()
        assert not recent_model_dir.exists()
    
    def test_cleanup_preserves_valid_recent_models(self):
        """Test that valid recent models are preserved."""
        # Create model directory
        model_name = "valid_model"
        model_dir = self.temp_dir / model_name
        model_dir.mkdir()
        (model_dir / "model.bin").write_text("fake model")
        
        # Mock ModelConfigs to return a valid model
        mock_model = MagicMock()
        with patch.object(ModelConfigs, 'get_model', return_value=mock_model):
            removed = model_utils.cleanup_old_models(self.temp_dir, keep_days=30)
        
        # Should not be removed
        assert model_name not in removed
        assert model_dir.exists()
    
    def test_cleanup_removes_obsolete_models(self):
        """Test that models not in configuration are removed regardless of age."""
        # Create model directory (recent)
        model_name = "obsolete_model"
        model_dir = self.temp_dir / model_name
        model_dir.mkdir()
        (model_dir / "model.bin").write_text("fake model")
        
        # Mock ModelConfigs to return None (not in configuration)
        with patch.object(ModelConfigs, 'get_model', return_value=None):
            removed = model_utils.cleanup_old_models(self.temp_dir, keep_days=30)
        
        # Should be removed even if recent
        assert model_name in removed
        assert not model_dir.exists()
    
    def test_cleanup_skips_non_directories(self):
        """Test that cleanup skips non-directory files."""
        # Create a regular file
        regular_file = self.temp_dir / "not_a_model_dir.txt"
        regular_file.write_text("some content")
        
        removed = model_utils.cleanup_old_models(self.temp_dir, keep_days=30)
        
        # File should still exist and not be in removed list
        assert regular_file.exists()
        assert "not_a_model_dir.txt" not in removed


class TestValidateModelCache:
    """Tests for validate_model_cache function."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def teardown_method(self):
        """Clean up test environment."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_validate_complete_model(self):
        """Test validation of complete model with correct files."""
        model_name = "test_model"
        model_dir = self.temp_dir / model_name
        model_dir.mkdir()
        
        # Create mock model metadata
        test_content = b"test model content"
        expected_hash = hashlib.sha256(test_content).hexdigest()
        
        mock_file = MagicMock()
        mock_file.filename = "model.bin"
        mock_file.sha256_hash = expected_hash
        
        mock_model = MagicMock()
        mock_model.files = [mock_file]
        
        # Create the actual file with correct content
        (model_dir / "model.bin").write_bytes(test_content)
        
        with patch.object(ModelConfigs, 'get_model', return_value=mock_model):
            results = model_utils.validate_model_cache(self.temp_dir)
        
        assert results[model_name] is True
    
    def test_validate_missing_files(self):
        """Test validation fails for models with missing files."""
        model_name = "incomplete_model"
        model_dir = self.temp_dir / model_name
        model_dir.mkdir()
        
        # Create mock model metadata with file that doesn't exist
        mock_file = MagicMock()
        mock_file.filename = "missing_file.bin"
        mock_file.sha256_hash = "somehash"
        
        mock_model = MagicMock()
        mock_model.files = [mock_file]
        
        with patch.object(ModelConfigs, 'get_model', return_value=mock_model):
            results = model_utils.validate_model_cache(self.temp_dir)
        
        assert results[model_name] is False
    
    def test_validate_corrupted_files(self):
        """Test validation fails for models with corrupted files."""
        model_name = "corrupted_model"
        model_dir = self.temp_dir / model_name
        model_dir.mkdir()
        
        # Create mock model metadata
        mock_file = MagicMock()
        mock_file.filename = "model.bin"
        mock_file.sha256_hash = "correct_hash_but_file_is_wrong"
        
        mock_model = MagicMock()
        mock_model.files = [mock_file]
        
        # Create file with wrong content
        (model_dir / "model.bin").write_bytes(b"wrong content")
        
        with patch.object(ModelConfigs, 'get_model', return_value=mock_model):
            results = model_utils.validate_model_cache(self.temp_dir)
        
        assert results[model_name] is False
    
    def test_validate_model_without_hashes(self):
        """Test validation passes for models without hash information."""
        model_name = "no_hash_model"
        model_dir = self.temp_dir / model_name
        model_dir.mkdir()
        
        # Create mock model metadata without hash
        mock_file = MagicMock()
        mock_file.filename = "model.bin"
        mock_file.sha256_hash = None  # No hash provided
        
        mock_model = MagicMock()
        mock_model.files = [mock_file]
        
        # Create the file
        (model_dir / "model.bin").write_bytes(b"some content")
        
        with patch.object(ModelConfigs, 'get_model', return_value=mock_model):
            results = model_utils.validate_model_cache(self.temp_dir)
        
        assert results[model_name] is True
    
    def test_validate_unknown_model(self):
        """Test validation fails for models not in configuration."""
        model_name = "unknown_model"
        model_dir = self.temp_dir / model_name
        model_dir.mkdir()
        (model_dir / "model.bin").write_text("fake model")
        
        # Mock ModelConfigs to return None
        with patch.object(ModelConfigs, 'get_model', return_value=None):
            results = model_utils.validate_model_cache(self.temp_dir)
        
        assert results[model_name] is False
    
    def test_validate_skips_non_directories(self):
        """Test that validation skips non-directory files."""
        # Create a regular file
        regular_file = self.temp_dir / "not_a_model.txt"
        regular_file.write_text("some content")
        
        results = model_utils.validate_model_cache(self.temp_dir)
        
        # Should not appear in results
        assert "not_a_model.txt" not in results
    
    def test_validate_empty_cache(self):
        """Test validation of empty cache directory."""
        results = model_utils.validate_model_cache(self.temp_dir)
        assert results == {}


class TestIntegration:
    """Integration tests for model utilities."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def teardown_method(self):
        """Clean up test environment."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_full_model_lifecycle(self):
        """Test a complete model lifecycle with utilities."""
        model_name = "lifecycle_test_model"
        model_dir = self.temp_dir / model_name
        model_dir.mkdir()
        
        # Create model file
        test_content = b"test model for lifecycle"
        model_file = model_dir / "model.bin"
        model_file.write_bytes(test_content)
        expected_hash = hashlib.sha256(test_content).hexdigest()
        
        # Create mock configuration
        mock_file = MagicMock()
        mock_file.filename = "model.bin"
        mock_file.sha256_hash = expected_hash
        
        mock_model = MagicMock()
        mock_model.files = [mock_file]
        
        with patch.object(ModelConfigs, 'get_model', return_value=mock_model):
            # 1. Verify integrity
            assert model_utils.verify_file_integrity(model_file, expected_hash) is True
            
            # 2. Get download time
            download_time = model_utils.get_model_download_time(model_dir)
            assert isinstance(download_time, datetime)
            
            # 3. Validate cache
            results = model_utils.validate_model_cache(self.temp_dir)
            assert results[model_name] is True
            
            # 4. Cleanup (should preserve recent valid model)
            removed = model_utils.cleanup_old_models(self.temp_dir, keep_days=30)
            assert model_name not in removed
            assert model_dir.exists()
        
        # 5. Test cleanup when model becomes obsolete
        with patch.object(ModelConfigs, 'get_model', return_value=None):
            removed = model_utils.cleanup_old_models(self.temp_dir, keep_days=30)
            assert model_name in removed
            assert not model_dir.exists() 
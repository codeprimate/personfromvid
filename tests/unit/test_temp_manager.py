"""Unit tests for TempManager."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from personfromvid.core import TempManager
from personfromvid.data.config import Config, StorageConfig
from personfromvid.utils.exceptions import TempDirectoryError


class TestTempManager:
    """Test TempManager functionality."""

    @pytest.fixture
    def temp_video_file(self):
        """Create a temporary video file for testing."""
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            # Write some dummy content
            f.write(b"dummy video content for testing")
            temp_path = Path(f.name)

        yield temp_path

        # Cleanup
        if temp_path.exists():
            temp_path.unlink()

    @pytest.fixture
    def temp_cache_dir(self):
        """Create a temporary cache directory for testing."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir

        # Cleanup
        import shutil
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

    @pytest.fixture
    def mock_config(self, temp_cache_dir):
        """Create a mock config with temporary cache directory."""
        storage_config = StorageConfig(cache_directory=temp_cache_dir)
        config = Config(storage=storage_config)
        return config

    @pytest.fixture
    def temp_manager(self, temp_video_file, mock_config):
        """Create a TempManager instance for testing."""
        manager = TempManager(str(temp_video_file), mock_config)
        yield manager

        # Cleanup temp directory if it exists
        if manager.temp_dir_path.exists():
            manager.cleanup_temp_files()

    def test_temp_manager_initialization(self, temp_video_file, mock_config):
        """Test TempManager initialization."""
        manager = TempManager(str(temp_video_file), mock_config)

        assert manager.video_path == temp_video_file
        expected_temp_name = f"temp_{temp_video_file.stem}"
        assert manager.temp_dir_name == expected_temp_name

        # Check that temp directory is in cache directory
        expected_temp_path = mock_config.storage.cache_directory / "temp" / expected_temp_name
        assert manager.temp_dir_path == expected_temp_path

        # Initially no subdirectories should be set
        assert manager.frames_dir is None

    def test_create_temp_structure(self, temp_manager):
        """Test creating temporary directory structure."""
        # Create structure
        temp_path = temp_manager.create_temp_structure()

        # Verify main directory was created
        assert temp_path.exists()
        assert temp_path.is_dir()
        assert temp_path == temp_manager.temp_dir_path

        # Verify subdirectories were created
        assert temp_manager.frames_dir.exists()

        # Verify subdirectories are correct paths
        assert temp_manager.frames_dir == temp_path / "frames"

    def test_cleanup_temp_files(self, temp_manager):
        """Test cleaning up temporary files."""
        # Create temp structure
        temp_manager.create_temp_structure()

        # Create some test files
        test_file = temp_manager.frames_dir / "test_frame.jpg"
        test_file.write_text("test content")

        # Verify file exists
        assert test_file.exists()
        assert temp_manager.temp_dir_path.exists()

        # Cleanup
        temp_manager.cleanup_temp_files()

        # Verify everything is gone
        assert not temp_manager.temp_dir_path.exists()
        assert not test_file.exists()

    def test_cleanup_nonexistent_directory(self, temp_manager):
        """Test cleanup when directory doesn't exist."""
        # Should not raise error
        temp_manager.cleanup_temp_files()
        assert not temp_manager.temp_dir_path.exists()

    def test_get_temp_path(self, temp_manager):
        """Test getting temp directory path."""
        # Should raise error before creation
        with pytest.raises(TempDirectoryError, match="Temp directory has not been created"):
            temp_manager.get_temp_path()

        # Should work after creation
        temp_manager.create_temp_structure()
        path = temp_manager.get_temp_path()
        assert path == temp_manager.temp_dir_path
        assert path.exists()

    def test_get_subdirectory_paths(self, temp_manager):
        """Test getting subdirectory paths."""
        # Should raise errors before creation
        with pytest.raises(TempDirectoryError, match="Frames directory has not been created"):
            temp_manager.get_frames_dir()

        # Should work after creation
        temp_manager.create_temp_structure()

        frames_dir = temp_manager.get_frames_dir()

        assert frames_dir.exists()

    def test_get_temp_file_path(self, temp_manager):
        """Test getting temporary file paths."""
        temp_manager.create_temp_structure()

        # Test without subdirectory
        file_path = temp_manager.get_temp_file_path("test.txt")
        assert file_path == temp_manager.temp_dir_path / "test.txt"

        # Test with subdirectories
        frame_path = temp_manager.get_temp_file_path("frame001.jpg", "frames")
        assert frame_path == temp_manager.frames_dir / "frame001.jpg"

        # Test invalid subdirectory
        with pytest.raises(ValueError, match="Unknown subdirectory"):
            temp_manager.get_temp_file_path("test.txt", "invalid")

    def test_monitor_disk_space(self, temp_manager):
        """Test disk space monitoring - now simplified."""
        temp_manager.create_temp_structure()

        # Always returns True now (disk checking removed)
        result = temp_manager.monitor_disk_space(1.0)
        assert result is True

    def test_get_temp_usage_info(self, temp_manager):
        """Test getting temporary directory usage information."""
        # Before creation
        info = temp_manager.get_temp_usage_info()
        assert info["exists"] is False
        assert info["size_mb"] == 0.0
        assert info["file_count"] == 0

        # After creation
        temp_manager.create_temp_structure()

        # Create some test files
        (temp_manager.frames_dir / "frame1.jpg").write_text("frame content")

        info = temp_manager.get_temp_usage_info()
        assert info["exists"] is True
        assert info["size_mb"] > 0
        assert info["file_count"] == 1

        # Check subdirectory info
        assert "frames" in info["subdirs"]
        assert info["subdirs"]["frames"]["file_count"] == 1

    def test_context_manager(self, temp_video_file, mock_config):
        """Test TempManager as context manager."""
        temp_dir_path = None

        # Use as context manager
        with TempManager(str(temp_video_file), mock_config) as manager:
            temp_dir_path = manager.temp_dir_path
            assert temp_dir_path.exists()
            assert manager.frames_dir.exists()

        # Should be cleaned up after exiting
        assert not temp_dir_path.exists()

    def test_context_manager_with_exception(self, temp_video_file, mock_config):
        """Test TempManager context manager with exception."""
        temp_dir_path = None

        try:
            with TempManager(str(temp_video_file), mock_config) as manager:
                temp_dir_path = manager.temp_dir_path
                assert temp_dir_path.exists()
                raise ValueError("Test exception")
        except ValueError:
            pass  # Expected exception

        # Should still be cleaned up after exception
        assert not temp_dir_path.exists()

    def test_force_cleanup(self, temp_manager):
        """Test force cleanup functionality."""
        temp_manager.create_temp_structure()

        # Create a test file
        test_file = temp_manager.frames_dir / "test.txt"
        test_file.write_text("test content")

        # Mock the initial rmtree to fail
        with patch('shutil.rmtree') as mock_rmtree:
            mock_rmtree.side_effect = [None, None]  # First call succeeds, second call (if any) succeeds
            temp_manager.cleanup_temp_files()

    def test_get_directory_size(self, temp_manager):
        """Test getting directory size."""
        temp_manager.create_temp_structure()

        # Create files with known sizes
        (temp_manager.frames_dir / "file1.txt").write_text("a" * 100)
        (temp_manager.frames_dir / "file2.txt").write_text("b" * 200)

        size = temp_manager._get_directory_size(temp_manager.frames_dir)
        assert size == 300

    def test_count_files(self, temp_manager):
        """Test counting files in directory."""
        temp_manager.create_temp_structure()

        # Create some files
        (temp_manager.frames_dir / "file1.txt").write_text("content1")
        (temp_manager.frames_dir / "file2.txt").write_text("content2")

        frames_count = temp_manager._count_files(temp_manager.frames_dir)

        assert frames_count == 2

    def test_create_temp_structure_permission_error(self, temp_video_file, mock_config):
        """Test handling permission errors during temp structure creation."""
        manager = TempManager(str(temp_video_file), mock_config)

        with patch.object(Path, 'mkdir', side_effect=PermissionError("Permission denied")):
            with pytest.raises(TempDirectoryError, match="Cannot create temp directory"):
                manager.create_temp_structure()

    def test_cleanup_with_locked_files(self, temp_manager):
        """Test cleanup when some files are locked."""
        temp_manager.create_temp_structure()

        # Create a test file
        test_file = temp_manager.frames_dir / "locked_file.txt"
        test_file.write_text("locked content")

        # Mock shutil.rmtree to simulate locked files
        with patch('shutil.rmtree') as mock_rmtree:
            mock_rmtree.side_effect = OSError("File is locked")

            # Should not raise exception (cleanup is best effort)
            temp_manager.cleanup_temp_files()

            # Verify rmtree was called
            mock_rmtree.assert_called()

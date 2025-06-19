"""Unit tests for validation utilities."""

import shutil
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from personfromvid.utils.exceptions import (
    ValidationError,
    VideoFileError,
)
from personfromvid.utils.validation import (
    MIN_DISK_SPACE_GB,
    MIN_PYTHON_VERSION,
    SUPPORTED_VIDEO_EXTENSIONS,
    SUPPORTED_VIDEO_MIMETYPES,
    sanitize_filename,
    validate_config_values,
    validate_model_path,
    validate_output_path,
    validate_system_requirements,
    validate_video_file,
)

# Test constants
TEST_FACE_MODEL = "scrfd_10g"


class TestValidateVideoFile:
    """Tests for validate_video_file function."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        """Clean up test environment."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_valid_video_file(self):
        """Test validation of a valid video file."""
        # Create a fake video file
        video_file = self.temp_dir / "test_video.mp4"
        video_file.write_text("fake video content" * 100)  # Make it non-empty

        with patch('personfromvid.utils.validation._get_video_metadata_ffprobe') as mock_metadata:
            mock_metadata.return_value = {
                'duration': 120.0,
                'width': 1920,
                'height': 1080,
                'fps': 30.0
            }

            result = validate_video_file(video_file)

        assert result['exists'] is True
        assert result['readable'] is True
        assert result['size_bytes'] > 0
        assert result['extension_valid'] is True
        assert result['mimetype_valid'] is True
        assert 'metadata' in result

    def test_nonexistent_video_file(self):
        """Test validation of non-existent video file."""
        nonexistent_file = self.temp_dir / "nonexistent.mp4"

        with pytest.raises(VideoFileError, match="does not exist"):
            validate_video_file(nonexistent_file)

    def test_empty_video_file(self):
        """Test validation of empty video file."""
        empty_file = self.temp_dir / "empty.mp4"
        empty_file.touch()  # Create empty file

        with pytest.raises(VideoFileError, match="is empty"):
            validate_video_file(empty_file)

    def test_suspiciously_small_video_file(self):
        """Test validation of suspiciously small video file."""
        small_file = self.temp_dir / "small.mp4"
        small_file.write_text("x")  # 1 byte file

        with pytest.raises(ValidationError, match="suspiciously small"):
            validate_video_file(small_file)

    def test_unsupported_extension(self):
        """Test validation of unsupported file extension."""
        unsupported_file = self.temp_dir / "test.txt"
        unsupported_file.write_text("fake content" * 100)

        with pytest.raises(ValidationError, match="Unsupported video file extension"):
            validate_video_file(unsupported_file)

    def test_unreadable_file(self):
        """Test validation of unreadable file."""
        video_file = self.temp_dir / "unreadable.mp4"
        video_file.write_text("fake content" * 100)

        # Make file unreadable
        with patch('os.access', return_value=False):
            with pytest.raises(VideoFileError, match="not readable"):
                validate_video_file(video_file)

    def test_file_access_error(self):
        """Test handling of OS errors during file access."""
        video_file = self.temp_dir / "error.mp4"
        video_file.write_text("fake content" * 100)

        # Mock exists to raise OSError
        with patch.object(Path, 'exists', side_effect=OSError("Access denied")):
            with pytest.raises(OSError, match="Access denied"):
                validate_video_file(video_file)

    def test_path_string_conversion(self):
        """Test that string paths are converted to Path objects."""
        video_file = self.temp_dir / "test.mp4"
        video_file.write_text("fake content" * 100)

        with patch('personfromvid.utils.validation._get_video_metadata_ffprobe') as mock_metadata:
            mock_metadata.return_value = {}

            # Pass string path instead of Path object
            result = validate_video_file(str(video_file))

            assert result['path'] == video_file
            assert result['exists'] is True

    def test_metadata_extraction_failure(self):
        """Test handling of metadata extraction failures."""
        video_file = self.temp_dir / "test.mp4"
        video_file.write_text("fake content" * 100)

        with patch('personfromvid.utils.validation._get_video_metadata_ffprobe') as mock_metadata:
            mock_metadata.side_effect = Exception("FFprobe failed")

            result = validate_video_file(video_file)

            # Should not fail, just record error in metadata
            assert 'error' in result['metadata']
            assert result['metadata']['error'] == "FFprobe failed"

    def test_unexpected_mimetype_warning(self):
        """Test warning for unexpected MIME types."""
        video_file = self.temp_dir / "test.mp4"
        video_file.write_text("fake content" * 100)

        with patch('mimetypes.guess_type', return_value=('application/octet-stream', None)):
            with patch('personfromvid.utils.validation._get_video_metadata_ffprobe', return_value={}):
                with pytest.warns(UserWarning, match="Unexpected MIME type"):
                    validate_video_file(video_file)


class TestValidateOutputPath:
    """Tests for validate_output_path function."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        """Clean up test environment."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_valid_existing_directory(self):
        """Test validation of existing writable directory."""
        output_path = self.temp_dir / "output"
        output_path.mkdir()

        result = validate_output_path(output_path)
        assert result is True

    def test_create_missing_directory(self):
        """Test creation of missing directory."""
        output_path = self.temp_dir / "new_output"

        result = validate_output_path(output_path, create_if_missing=True)
        assert result is True
        assert output_path.parent.exists()

    def test_missing_directory_no_create(self):
        """Test validation fails when directory missing and create_if_missing=False."""
        output_path = Path("/nonexistent/path/output")

        with pytest.raises(ValidationError, match="does not exist"):
            validate_output_path(output_path, create_if_missing=False)

    def test_unwritable_directory(self):
        """Test validation fails for unwritable directory."""
        output_path = self.temp_dir / "output"
        output_path.mkdir()

        with patch('os.access', return_value=False):
            with pytest.raises(ValidationError, match="not writable"):
                validate_output_path(output_path)

    def test_insufficient_disk_space(self):
        """Test validation fails with insufficient disk space."""
        output_path = self.temp_dir / "output"
        output_path.mkdir()

        # Mock disk usage to show insufficient space
        mock_usage = MagicMock()
        mock_usage.free = 100 * 1024 * 1024  # 100MB (less than MIN_DISK_SPACE_GB)

        with patch('shutil.disk_usage', return_value=mock_usage):
            with pytest.raises(ValidationError, match="Insufficient disk space"):
                validate_output_path(output_path)

    def test_disk_space_check_error(self):
        """Test handling of disk space check errors."""
        output_path = self.temp_dir / "output"
        output_path.mkdir()

        with patch('shutil.disk_usage', side_effect=OSError("Disk error")):
            with pytest.raises(ValidationError, match="Could not check disk space"):
                validate_output_path(output_path)

    def test_create_directory_error(self):
        """Test handling of directory creation errors."""
        output_path = Path("/invalid/path/that/cannot/be/created")

        with pytest.raises(ValidationError, match="Cannot create output directory"):
            validate_output_path(output_path, create_if_missing=True)

    def test_string_path_conversion(self):
        """Test that string paths are converted to Path objects."""
        output_path = self.temp_dir / "output"
        output_path.mkdir()

        # Pass string path
        result = validate_output_path(str(output_path))
        assert result is True


class TestValidateSystemRequirements:
    """Tests for validate_system_requirements function."""

    def test_valid_system(self):
        """Test validation of valid system."""
        # Mock system to meet all requirements
        with patch('sys.version_info', (3, 9, 0)):
            issues = validate_system_requirements()
            # GPU warning might still be present
            assert len(issues) <= 1

    def test_insufficient_python_version(self):
        """Test detection of insufficient Python version."""
        with patch('sys.version_info', (3, 7, 0)):  # Below minimum
            issues = validate_system_requirements()

            assert len(issues) > 0
            assert any("Python version" in issue for issue in issues)




class TestValidateConfigValues:
    """Tests for validate_config_values function."""

    def test_valid_config(self):
        """Test validation of valid configuration."""
        config = {
            'models': {
                'face_detection': 'TEST_FACE_MODEL',
                'pose_estimation': 'yolov8n-pose'
            },
            'device': 'cpu',
            'batch_size': 8
        }

        with patch('personfromvid.utils.validation.validate_model_path', return_value=True):
            issues = validate_config_values(config)
            assert len(issues) == 0

    def test_invalid_device_type(self):
        """Test that device type validation is not implemented."""
        config = {
            'device': 'invalid_device',
            'models': {}
        }

        issues = validate_config_values(config)
        # Current implementation doesn't validate device types
        assert len(issues) == 0

    def test_invalid_batch_size(self):
        """Test detection of invalid batch sizes."""
        invalid_configs = [
            {'models': {'batch_size': 0}},
            {'models': {'batch_size': -1}},
            {'models': {'batch_size': 'invalid'}},
            {'models': {'batch_size': 1.5}}
        ]

        for config in invalid_configs:
            issues = validate_config_values(config)
            assert len(issues) > 0

    def test_invalid_model_config(self):
        """Test that model config validation is not implemented."""
        config = {
            'models': {
                'face_detection': 'nonexistent_model'
            }
        }

        issues = validate_config_values(config)
        # Current implementation doesn't validate model configs
        assert len(issues) == 0


class TestSanitizeFilename:
    """Tests for sanitize_filename function."""

    def test_valid_filename(self):
        """Test that valid filenames pass through unchanged."""
        valid_name = "normal_filename.mp4"
        result = sanitize_filename(valid_name)
        assert result == "normal_filename.mp4"

    def test_invalid_characters_removed(self):
        """Test that invalid characters are removed."""
        invalid_name = "test<>:\"/\\|?*file.mp4"
        result = sanitize_filename(invalid_name)

        # Should replace all invalid characters with underscores
        assert result == "test_________file.mp4"

    def test_spaces_replaced(self):
        """Test that spaces are not replaced in the current implementation."""
        name_with_spaces = "test file name.mp4"
        result = sanitize_filename(name_with_spaces)

        # Current implementation preserves spaces
        assert result == "test file name.mp4"

    def test_unicode_characters(self):
        """Test handling of unicode characters."""
        unicode_name = "tëst_fïlé.mp4"
        result = sanitize_filename(unicode_name)

        # Current implementation preserves unicode characters
        assert result == "tëst_fïlé.mp4"

    def test_empty_filename(self):
        """Test handling of empty filename."""
        result = sanitize_filename("")
        assert result == "unnamed"

    def test_only_invalid_characters(self):
        """Test filename with only invalid characters."""
        invalid_only = "<>:\"/\\|?*"
        result = sanitize_filename(invalid_only)
        assert result == "_________"

    def test_preserve_extension(self):
        """Test that file extensions are preserved."""
        name = "test<file>.mp4"
        result = sanitize_filename(name)
        assert result == "test_file_.mp4"
        assert result.endswith(".mp4")


class TestValidateModelPath:
    """Tests for validate_model_path function."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        """Clean up test environment."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_valid_model_file(self):
        """Test validation of valid model file."""
        model_file = self.temp_dir / "model.onnx"
        model_file.write_text("fake model content" * 100)

        result = validate_model_path(model_file)
        assert result is True

    def test_nonexistent_model_file(self):
        """Test validation of non-existent model file."""
        nonexistent_file = self.temp_dir / "nonexistent.onnx"

        with pytest.raises(ValidationError, match="Model file does not exist"):
            validate_model_path(nonexistent_file)

    def test_empty_model_file(self):
        """Test validation of empty model file."""
        empty_file = self.temp_dir / "empty.onnx"
        empty_file.touch()

        with pytest.raises(ValidationError, match="Model file is empty"):
            validate_model_path(empty_file)

    def test_model_directory(self):
        """Test validation of model directory."""
        model_dir = self.temp_dir / "model_dir"
        model_dir.mkdir()
        (model_dir / "model.bin").write_text("fake model")
        (model_dir / "config.json").write_text("{}")

        with pytest.raises(ValidationError, match="Model path is not a file"):
            validate_model_path(model_dir)

    def test_empty_model_directory(self):
        """Test validation of empty model directory."""
        empty_dir = self.temp_dir / "empty_dir"
        empty_dir.mkdir()

        with pytest.raises(ValidationError, match="Model path is not a file"):
            validate_model_path(empty_dir)

    def test_unreadable_model_file(self):
        """Test validation of unreadable model file."""
        model_file = self.temp_dir / "unreadable.onnx"
        model_file.write_text("fake model")

        with patch('os.access', return_value=False):
            with pytest.raises(ValidationError, match="Model file is not readable"):
                validate_model_path(model_file)


class TestConstants:
    """Tests for module constants."""

    def test_supported_extensions(self):
        """Test that supported extensions are reasonable."""
        assert '.mp4' in SUPPORTED_VIDEO_EXTENSIONS
        assert '.avi' in SUPPORTED_VIDEO_EXTENSIONS
        assert '.mov' in SUPPORTED_VIDEO_EXTENSIONS
        assert '.mkv' in SUPPORTED_VIDEO_EXTENSIONS
        assert len(SUPPORTED_VIDEO_EXTENSIONS) >= 5

    def test_supported_mimetypes(self):
        """Test that supported MIME types are reasonable."""
        assert 'video/mp4' in SUPPORTED_VIDEO_MIMETYPES
        assert 'video/avi' in SUPPORTED_VIDEO_MIMETYPES
        assert 'video/quicktime' in SUPPORTED_VIDEO_MIMETYPES
        assert len(SUPPORTED_VIDEO_MIMETYPES) >= 5

    def test_minimum_requirements(self):
        """Test that minimum requirements are reasonable."""
        assert MIN_PYTHON_VERSION >= (3, 8)
        assert MIN_DISK_SPACE_GB > 0


class TestPrivateFunctions:
    """Tests for private helper functions."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        """Clean up test environment."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_get_video_metadata_ffprobe_success(self):
        """Test successful ffprobe metadata extraction."""
        from personfromvid.utils.validation import _get_video_metadata_ffprobe

        video_file = self.temp_dir / "test.mp4"
        video_file.write_text("fake video")

        mock_result = {
            'returncode': 0,
            'stdout': '{"streams":[{"codec_type":"video","width":1920,"height":1080,"r_frame_rate":"30/1","duration":"120.0"}],"format":{"duration":"120.0"}}'
        }

        with patch('subprocess.run', return_value=MagicMock(**mock_result)):
            metadata = _get_video_metadata_ffprobe(video_file)

            assert 'width' in metadata
            assert metadata['width'] == 1920

    def test_get_video_metadata_ffprobe_failure(self):
        """Test ffprobe failure handling."""
        from personfromvid.utils.validation import _get_video_metadata_ffprobe

        video_file = self.temp_dir / "test.mp4"
        video_file.write_text("fake video")

        with patch(
            "subprocess.run", side_effect=subprocess.SubprocessError("FFprobe not found")
        ):
            with pytest.raises(VideoFileError):
                _get_video_metadata_ffprobe(video_file)

    def test_check_executable_found(self):
        """Test executable check when executable is found."""
        from personfromvid.utils.validation import _check_executable

        with patch('shutil.which', return_value='/usr/bin/ffmpeg'):
            result = _check_executable('ffmpeg')
            assert result is True

    def test_check_executable_not_found(self):
        """Test executable check when executable is not found."""
        from personfromvid.utils.validation import _check_executable

        with patch('shutil.which', return_value=None):
            result = _check_executable('nonexistent')
            assert result is False

    def test_check_gpu_availability_success(self):
        """Test GPU availability check success."""
        from personfromvid.utils.validation import _check_gpu_availability

        # Mock successful PyTorch CUDA detection
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.device_count', return_value=1), \
             patch('torch.cuda.get_device_name', return_value="NVIDIA GeForce RTX 3080"), \
             patch('torch.cuda.get_device_properties') as mock_props:

            mock_props.return_value.total_memory = 8 * 1024**3  # 8GB
            gpu_info = _check_gpu_availability()

            assert gpu_info['available'] is True
            assert 'devices' in gpu_info

    def test_check_gpu_availability_failure(self):
        """Test GPU availability check failure."""
        from personfromvid.utils.validation import _check_gpu_availability

        with patch('torch.cuda.is_available', return_value=False):
            gpu_info = _check_gpu_availability()

            assert gpu_info['available'] is False
            assert 'issues' in gpu_info

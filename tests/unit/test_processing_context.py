"""Unit tests for ProcessingContext."""

import tempfile
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import patch

import pytest

from personfromvid.data import Config, ProcessingContext


class TestProcessingContext:
    """Test ProcessingContext functionality."""

    @pytest.fixture
    def temp_video_file(self):
        """Create a temporary video file for testing."""
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            f.write(b'dummy video content for testing')
            temp_path = Path(f.name)

        yield temp_path

        # Cleanup
        if temp_path.exists():
            temp_path.unlink()

    @pytest.fixture
    def config(self):
        """Create a Config instance for testing."""
        return Config()

    @pytest.fixture
    def processing_context(self, temp_video_file, config):
        """Create a ProcessingContext instance for testing."""
        output_dir = temp_video_file.parent / 'test_output'

        with patch('personfromvid.core.temp_manager.TempManager') as MockTempManager:
            mock_instance = MockTempManager.return_value
            mock_instance.get_temp_path.return_value = Path('/tmp/fake_temp')
            mock_instance.get_frames_dir.return_value = Path('/tmp/fake_temp/frames')

            context = ProcessingContext(
                video_path=temp_video_file,
                video_base_name=temp_video_file.stem,
                config=config,
                output_directory=output_dir
            )

            yield context, MockTempManager

        # Cleanup output directory
        if output_dir.exists():
            # Be careful with rmtree in tests
            import shutil
            shutil.rmtree(output_dir, ignore_errors=True)

    def test_processing_context_creation(self, processing_context, temp_video_file):
        """Test ProcessingContext creation with valid parameters."""
        context, MockTempManager = processing_context
        assert context.video_path == temp_video_file
        assert context.video_base_name == temp_video_file.stem
        assert isinstance(context.config, Config)
        assert context.output_directory.exists()

        # Verify TempManager was instantiated and used
        MockTempManager.assert_called_once_with(str(temp_video_file), context.config)
        context.temp_manager.create_temp_structure.assert_called_once()

    def test_processing_context_is_frozen(self, processing_context):
        """Test that ProcessingContext is immutable (frozen)."""
        context, _ = processing_context
        with pytest.raises(FrozenInstanceError):
            context.video_path = Path("different/path")

        with pytest.raises(FrozenInstanceError):
            context.config = Config()

    def test_video_properties(self, processing_context, temp_video_file):
        """Test video-related properties."""
        context, _ = processing_context
        assert context.video_name == temp_video_file.name
        assert context.video_stem == temp_video_file.stem
        assert context.video_suffix == temp_video_file.suffix
        assert context.video_suffix == '.mp4'

    def test_directory_properties(self, processing_context):
        """Test directory-related properties."""
        context, _ = processing_context
        assert context.temp_directory == Path('/tmp/fake_temp')
        assert context.frames_directory == Path('/tmp/fake_temp/frames')

    def test_output_directory_creation(self, temp_video_file, config):
        """Test that output directory is created during initialization."""
        output_dir = temp_video_file.parent / 'auto_created_output'

        # Ensure directory doesn't exist initially
        if output_dir.exists():
            import shutil
            shutil.rmtree(output_dir, ignore_errors=True)
        assert not output_dir.exists()

        # Create context - should create output directory
        with patch('personfromvid.core.temp_manager.TempManager'):
            ProcessingContext(
                video_path=temp_video_file,
                video_base_name=temp_video_file.stem,
                config=config,
                output_directory=output_dir
            )

        # Verify directory was created
        assert output_dir.exists()

        # Cleanup
        import shutil
        shutil.rmtree(output_dir, ignore_errors=True)

    def test_invalid_video_path(self, config):
        """Test ProcessingContext creation with invalid video path."""
        invalid_path = Path('nonexistent_video.mp4')
        output_dir = Path('output')

        with pytest.raises(FileNotFoundError):
            with patch('personfromvid.core.temp_manager.TempManager'):
                ProcessingContext(
                    video_path=invalid_path,
                    video_base_name='test',
                    config=config,
                    output_directory=output_dir
                )

    def test_directory_as_video_path(self, config):
        """Test ProcessingContext creation with directory instead of file."""
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            dir_path = Path(temp_dir)
            output_dir = Path('output')

            with pytest.raises(ValueError, match="Video path is not a file"):
                with patch('personfromvid.core.temp_manager.TempManager'):
                    ProcessingContext(
                        video_path=dir_path,
                        video_base_name='test',
                        config=config,
                        output_directory=output_dir
                    )

    def test_context_attributes_consistency(self, processing_context):
        """Test that context attributes are consistent."""
        context, _ = processing_context
        # video_base_name should match the stem of video_path
        assert context.video_base_name == context.video_path.stem

        # video_stem property should match video_base_name
        assert context.video_stem == context.video_base_name

    def test_string_representation(self, processing_context):
        """Test that ProcessingContext has a reasonable string representation."""
        context, _ = processing_context
        # Should not raise an exception
        str_repr = str(context)
        assert 'ProcessingContext' in str_repr

        # Should include the video path info
        assert context.video_name in str_repr or context.video_path.name in str_repr

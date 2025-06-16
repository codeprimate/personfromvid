"""Unit tests for output generation components."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from PIL import Image

from personfromvid.output.naming_convention import NamingConvention
from personfromvid.output.image_writer import ImageWriter
from personfromvid.data import Config, ProcessingContext
from personfromvid.data.config import OutputImageConfig, PngConfig, JpegConfig
from personfromvid.data.frame_data import FrameData, SourceInfo, ImageProperties, SelectionInfo
from personfromvid.data.detection_results import FaceDetection
from personfromvid.core import TempManager


@pytest.fixture
def temp_output_dir():
    """Create a temporary output directory for tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def processing_context(tmp_path):
    """Create a ProcessingContext for testing."""
    # Create a test video file
    video_file = tmp_path / "test_video.mp4"
    video_file.write_bytes(b'dummy video content')
    
    # Create components
    config = Config()
    output_dir = tmp_path / 'output'
    
    # Create ProcessingContext
    with patch('personfromvid.core.temp_manager.TempManager') as MockTempManager:
        # If tests need the temp_manager, you can configure the mock here
        # For NamingConvention and ImageWriter, they might need paths from it
        mock_temp_manager = MockTempManager.return_value
        mock_temp_manager.get_temp_path.return_value = tmp_path / '.temp'
        
        context = ProcessingContext(
            video_path=video_file,
            video_base_name=video_file.stem,
            config=config,
            output_directory=output_dir
        )
        
        yield context


@pytest.fixture
def sample_frame_data(tmp_path):
    """Create sample frame data for testing."""
    frame_file = tmp_path / "frame_001.jpg"
    frame_file.touch()

    # Create a mock image
    mock_image = np.zeros((1080, 1920, 3), dtype=np.uint8)
    
    frame = FrameData(
        frame_id="test_frame_001",
        file_path=frame_file,
        source_info=SourceInfo(
            video_timestamp=30.5,
            extraction_method="i_frame",
            original_frame_number=915,
            video_fps=30.0
        ),
        image_properties=ImageProperties(
            width=1920,
            height=1080,
            channels=3,
            file_size_bytes=100000,
            format="JPEG"
        ),
        face_detections=[
            FaceDetection(
                bbox=(500, 300, 700, 500),
                confidence=0.92,
                landmarks=[(520, 320), (680, 320), (600, 380), (540, 450), (660, 450)]
            )
        ],
        selections=SelectionInfo(
            selected_for_poses=["standing"],
            selected_for_head_angles=["front"],
            selection_rank=1
        )
    )
    # Mock the _load_image method to return our test image
    frame._image = mock_image  # Set the private field directly for testing
    return frame


class TestNamingConvention:
    """Tests for NamingConvention class."""

    def test_full_frame_filename(self, processing_context, sample_frame_data):
        """Test full frame filename generation."""
        naming = NamingConvention(context=processing_context)
        filename = naming.get_full_frame_filename(sample_frame_data, "standing", 1, "png")
        assert filename == "test_video_standing_001.png"

    def test_face_crop_filename(self, processing_context, sample_frame_data):
        """Test face crop filename generation."""
        naming = NamingConvention(context=processing_context)
        filename = naming.get_face_crop_filename(sample_frame_data, "front", 1, "jpg")
        assert filename == "test_video_face_front_001.jpg"

    def test_crop_suffixed_filename(self, processing_context):
        """Test crop suffix filename generation."""
        naming = NamingConvention(context=processing_context)
        
        # Test various filename formats
        assert naming.get_crop_suffixed_filename("video_pose_001.jpg") == "video_pose_001_crop.jpg"
        assert naming.get_crop_suffixed_filename("test.png") == "test_crop.png"
        assert naming.get_crop_suffixed_filename("complex_name_with_underscores.jpeg") == "complex_name_with_underscores_crop.jpeg"


class TestImageWriter:
    """Tests for ImageWriter class."""

    @patch('personfromvid.output.image_writer.Image.fromarray')
    def test_save_full_frame(self, mock_fromarray, processing_context, sample_frame_data):
        """Test saving a full frame image."""
        mock_pil_image = Mock()
        mock_fromarray.return_value = mock_pil_image
        mock_pil_image.convert.return_value = mock_pil_image

        # Update config to enable full frame output
        processing_context.config.output.image.full_frame_enabled = True
        processing_context.config.output.image.face_crop_enabled = False
        processing_context.config.output.image.format = 'png'
        
        # Set up enhanced selection data for pose category
        sample_frame_data.selections.primary_selection_category = "pose_standing"
        sample_frame_data.selections.selection_rank = 1
        
        writer = ImageWriter(context=processing_context)

        output_files = writer.save_frame_outputs(sample_frame_data)
        
        assert len(output_files) == 1
        mock_pil_image.save.assert_called_once()
        assert output_files[0].endswith('.png')

    @patch('personfromvid.output.image_writer.Image.fromarray')
    def test_save_face_crop(self, mock_fromarray, processing_context, sample_frame_data):
        """Test saving a cropped face image."""
        mock_pil_image = Mock()
        mock_fromarray.return_value = mock_pil_image
        mock_fromarray.return_value = mock_pil_image
        mock_pil_image.convert.return_value = mock_pil_image
        mock_pil_image.resize.return_value = mock_pil_image

        # Update config to enable face crop output
        processing_context.config.output.image.full_frame_enabled = False
        processing_context.config.output.image.face_crop_enabled = True
        processing_context.config.output.image.format = 'jpeg'
        
        # Set up enhanced selection data for head angle category
        sample_frame_data.selections.primary_selection_category = "head_angle_front"
        sample_frame_data.selections.selection_rank = 1
        
        writer = ImageWriter(context=processing_context)

        output_files = writer.save_frame_outputs(sample_frame_data)
        
        assert len(output_files) == 1
        mock_pil_image.save.assert_called_once()
        assert output_files[0].endswith('.jpg')

    @patch('personfromvid.output.image_writer.Image.fromarray')
    def test_resize_full_frame(self, mock_fromarray, processing_context, sample_frame_data):
        """Test resizing full frame images."""
        mock_pil_image = Mock()
        mock_fromarray.return_value = mock_pil_image
        mock_pil_image.convert.return_value = mock_pil_image
        mock_pil_image.resize.return_value = mock_pil_image

        # Configure resize to 1024px
        processing_context.config.output.image.resize = 1024
        processing_context.config.output.image.full_frame_enabled = True
        processing_context.config.output.image.face_crop_enabled = False
        
        # Set up enhanced selection data for pose category
        sample_frame_data.selections.primary_selection_category = "pose_standing"
        sample_frame_data.selections.selection_rank = 1
        
        writer = ImageWriter(context=processing_context)
        output_files = writer.save_frame_outputs(sample_frame_data)
        
        # Should call resize since original image is 1920x1080 (larger than 1024)
        mock_pil_image.resize.assert_called_once()
        resize_call = mock_pil_image.resize.call_args[0][0]  # Get the size tuple
        
        # Check that the larger dimension (1920) was scaled to 1024
        # Expected: scale_factor = 1024/1920 = 0.533...
        # New dimensions: width=1024, height=576
        assert resize_call == (1024, 576)

    @patch('personfromvid.output.image_writer.Image.fromarray')
    def test_resize_face_crop_with_resize_config(self, mock_fromarray, processing_context, sample_frame_data):
        """Test face crop respects resize configuration instead of default 512."""
        mock_pil_image = Mock()
        mock_fromarray.return_value = mock_pil_image
        mock_fromarray.return_value = mock_pil_image
        mock_pil_image.convert.return_value = mock_pil_image
        mock_pil_image.resize.return_value = mock_pil_image

        # Configure resize to 256px (smaller than default 512)
        processing_context.config.output.image.resize = 256
        processing_context.config.output.image.face_crop_enabled = True
        processing_context.config.output.image.full_frame_enabled = False
        
        # Set up enhanced selection data for head angle category
        sample_frame_data.selections.primary_selection_category = "head_angle_front"
        sample_frame_data.selections.selection_rank = 1
        
        writer = ImageWriter(context=processing_context)
        
        # Create a small face crop that would normally be upscaled to 512
        # but should now be upscaled to 256
        with patch.object(writer, '_crop_face') as mock_crop_face:
            # Mock a small face crop (100x100) that needs upscaling
            small_face_crop = np.zeros((100, 100, 3), dtype=np.uint8)
            mock_crop_face.return_value = small_face_crop
            
            output_files = writer.save_frame_outputs(sample_frame_data)
            
            # _crop_face should have been called
            mock_crop_face.assert_called_once()

    @patch('personfromvid.output.image_writer.Image.fromarray')
    def test_no_resize_when_image_smaller_than_target(self, mock_fromarray, processing_context, sample_frame_data):
        """Test that resize is not applied when image is smaller than target."""
        mock_pil_image = Mock()
        mock_fromarray.return_value = mock_pil_image
        mock_pil_image.convert.return_value = mock_pil_image

        # Configure resize to 4096px (larger than test image 1920x1080)
        processing_context.config.output.image.resize = 4096
        processing_context.config.output.image.full_frame_enabled = True
        processing_context.config.output.image.face_crop_enabled = False
        
        # Set up enhanced selection data for pose category
        sample_frame_data.selections.primary_selection_category = "pose_standing"
        sample_frame_data.selections.selection_rank = 1
        
        writer = ImageWriter(context=processing_context)
        output_files = writer.save_frame_outputs(sample_frame_data)
        
        # Should NOT call resize since image is smaller than target
        mock_pil_image.resize.assert_not_called()

    def test_apply_resize_method_downscaling(self, processing_context):
        """Test _apply_resize method for downscaling."""
        processing_context.config.output.image.resize = 512
        writer = ImageWriter(context=processing_context)
        
        # Create a large test image
        large_image = np.zeros((1080, 1920, 3), dtype=np.uint8)
        
        with patch('personfromvid.output.image_writer.Image.fromarray') as mock_fromarray:
            mock_pil_image = Mock()
            mock_fromarray.return_value = mock_pil_image
            mock_pil_image.resize.return_value = mock_pil_image
            
            result = writer._apply_resize(large_image)
            
            # Should call resize with calculated dimensions
            mock_pil_image.resize.assert_called_once()
            resize_call = mock_pil_image.resize.call_args[0][0]
            
            # Scale factor should be 512/1920 = 0.2667
            # New dimensions: width=512, height=288
            assert resize_call == (512, 288)

    def test_apply_resize_method_no_resize_needed(self, processing_context):
        """Test _apply_resize method when no resize is needed."""
        processing_context.config.output.image.resize = 2048
        writer = ImageWriter(context=processing_context)
        
        # Create a small test image
        small_image = np.zeros((512, 768, 3), dtype=np.uint8)
        
        result = writer._apply_resize(small_image)
        
        # Should return original image unchanged
        assert np.array_equal(result, small_image)

    def test_apply_resize_method_no_resize_config(self, processing_context):
        """Test _apply_resize method when resize is not configured."""
        processing_context.config.output.image.resize = None
        writer = ImageWriter(context=processing_context)
        
        test_image = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result = writer._apply_resize(test_image)
        
        # Should return original image unchanged
        assert np.array_equal(result, test_image)

    def test_face_crop_uses_resize_value(self, processing_context, sample_frame_data):
        """Test that face crop uses resize value instead of hardcoded 512."""
        processing_context.config.output.image.resize = 256
        writer = ImageWriter(context=processing_context)
        
        # Create a test image with a small face crop area
        test_image = np.zeros((1080, 1920, 3), dtype=np.uint8)
        face_detection = FaceDetection(
            bbox=(500, 300, 600, 400),  # 100x100 face
            confidence=0.9,
            landmarks=[]
        )
        
        with patch('personfromvid.output.image_writer.Image.fromarray') as mock_fromarray:
            mock_pil_image = Mock()
            mock_fromarray.return_value = mock_pil_image
            mock_pil_image.resize.return_value = mock_pil_image
            
            result = writer._crop_face(test_image, face_detection)
            
            # Should call resize because face crop will be small and need upscaling to 256
            mock_pil_image.resize.assert_called_once()

    def test_face_crop_uses_default_512_when_no_resize_config(self, processing_context, sample_frame_data):
        """Test that face crop uses default 512 when resize is not configured."""
        processing_context.config.output.image.resize = None
        writer = ImageWriter(context=processing_context)
        
        # Create a test image with a small face crop area
        test_image = np.zeros((1080, 1920, 3), dtype=np.uint8)
        face_detection = FaceDetection(
            bbox=(500, 300, 600, 400),  # 100x100 face
            confidence=0.9,
            landmarks=[]
        )
        
        with patch('personfromvid.output.image_writer.Image.fromarray') as mock_fromarray:
            mock_pil_image = Mock()
            mock_fromarray.return_value = mock_pil_image
            mock_pil_image.resize.return_value = mock_pil_image
            
            result = writer._crop_face(test_image, face_detection)
            
            # Should call resize to upscale to default 512
            mock_pil_image.resize.assert_called_once()

    def test_config_validation_resize(self, processing_context):
        """Test configuration validation for resize values."""
        from personfromvid.utils.exceptions import ImageWriteError
        
        # Test valid resize value
        processing_context.config.output.image.resize = 1024
        writer = ImageWriter(context=processing_context)  # Should not raise
        
        # Test invalid resize values
        processing_context.config.output.image.resize = 100  # Too small
        with pytest.raises(ValueError, match="Resize dimension must be between 256 and 4096"):
            writer = ImageWriter(context=processing_context)
        
        processing_context.config.output.image.resize = 5000  # Too large
        with pytest.raises(ValueError, match="Resize dimension must be between 256 and 4096"):
            writer = ImageWriter(context=processing_context)

    def test_get_output_statistics_includes_resize(self, processing_context):
        """Test that output statistics include resize information."""
        processing_context.config.output.image.resize = 1024
        writer = ImageWriter(context=processing_context)
        
        stats = writer.get_output_statistics()
        
        assert "resize" in stats
        assert stats["resize"] == 1024
        
        # Test with no resize configured
        processing_context.config.output.image.resize = None
        writer = ImageWriter(context=processing_context)
        
        stats = writer.get_output_statistics()
        assert stats["resize"] is None

    @patch('personfromvid.output.image_writer.Image.fromarray')
    def test_pose_cropping_feature_enabled(self, mock_fromarray, processing_context, sample_frame_data):
        """Test pose cropping feature when enabled."""
        mock_pil_image = Mock()
        mock_fromarray.return_value = mock_pil_image
        mock_pil_image.convert.return_value = mock_pil_image

        # Enable pose cropping feature
        processing_context.config.output.image.enable_pose_cropping = True
        processing_context.config.output.image.pose_crop_padding = 0.2
        processing_context.config.output.image.full_frame_enabled = True
        processing_context.config.output.image.face_crop_enabled = False
        processing_context.config.output.image.format = "jpeg"
        
        # Set up enhanced selection data for pose category
        sample_frame_data.selections.primary_selection_category = "pose_standing"
        sample_frame_data.selections.selection_rank = 1
        
        # Add required pose detection data for pose cropping
        from personfromvid.data.detection_results import PoseDetection
        pose_detection = PoseDetection(
            bbox=(400, 200, 800, 900),  # x1, y1, x2, y2
            confidence=0.9,
            keypoints={},  # Not needed for cropping
            pose_classifications=[("standing", 0.95)]  # Pose type with confidence
        )
        sample_frame_data.pose_detections.append(pose_detection)
        
        writer = ImageWriter(context=processing_context)

        with patch.object(writer, '_crop_region') as mock_crop_region:
            mock_crop_region.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
            
            output_files = writer.save_frame_outputs(sample_frame_data)
            
            # Should have 2 files: full frame + pose crop
            assert len(output_files) == 2
            
            # Should call _crop_region for pose cropping
            mock_crop_region.assert_called_once()
            
            # Should call PIL save twice (once for full frame, once for crop)
            assert mock_pil_image.save.call_count == 2
            
            # Verify we have both a regular file and a crop file
            crop_files = [f for f in output_files if '_crop.jpg' in f]
            regular_files = [f for f in output_files if '_crop.jpg' not in f]
            assert len(crop_files) == 1
            assert len(regular_files) == 1

    @patch('personfromvid.output.image_writer.Image.fromarray')
    def test_pose_cropping_feature_disabled(self, mock_fromarray, processing_context, sample_frame_data):
        """Test pose cropping feature when disabled."""
        mock_pil_image = Mock()
        mock_fromarray.return_value = mock_pil_image
        mock_pil_image.convert.return_value = mock_pil_image

        # Disable pose cropping feature (default)
        processing_context.config.output.image.enable_pose_cropping = False
        processing_context.config.output.image.full_frame_enabled = True
        processing_context.config.output.image.face_crop_enabled = False
        processing_context.config.output.image.format = "jpeg"
        
        # Set up enhanced selection data for pose category
        sample_frame_data.selections.primary_selection_category = "pose_standing"
        sample_frame_data.selections.selection_rank = 1
        
        writer = ImageWriter(context=processing_context)

        with patch.object(writer, '_crop_region') as mock_crop_region:
            output_files = writer.save_frame_outputs(sample_frame_data)
            
            # Should have only 1 file: full frame
            assert len(output_files) == 1
            
            # Should NOT call _crop_region
            mock_crop_region.assert_not_called()
            
            # Should call PIL save only once (for full frame)
            mock_pil_image.save.assert_called_once()
            
            # No files should have _crop suffix
            crop_files = [f for f in output_files if '_crop.jpg' in f]
            assert len(crop_files) == 0 
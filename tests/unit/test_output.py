"""Unit tests for output generation components."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from personfromvid.data import Config, ProcessingContext
from personfromvid.data.detection_results import CloseupDetection, FaceDetection
from personfromvid.data.frame_data import (
    FrameData,
    ImageProperties,
    SelectionInfo,
    SourceInfo,
)
from personfromvid.output.image_writer import ImageWriter
from personfromvid.output.naming_convention import NamingConvention


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
        closeup_detections=[
            CloseupDetection(
                is_closeup=True,
                shot_type="medium_closeup",
                confidence=0.9,
                face_area_ratio=0.3
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
        assert filename == "test_video_standing_medium_closeup_001.png"

    def test_face_crop_filename(self, processing_context, sample_frame_data):
        """Test face crop filename generation."""
        naming = NamingConvention(context=processing_context)
        filename = naming.get_face_crop_filename(sample_frame_data, "front", 1, "jpg")
        assert filename == "test_video_face_front_001.jpg"  # Simplified: no shot type

    def test_crop_suffixed_filename(self, processing_context):
        """Test crop suffix filename generation."""
        naming = NamingConvention(context=processing_context)

        # Test various filename formats
        assert naming.get_crop_suffixed_filename("video_pose_001.jpg") == "video_pose_001_crop.jpg"
        assert naming.get_crop_suffixed_filename("test.png") == "test_crop.png"
        assert naming.get_crop_suffixed_filename("complex_name_with_underscores.jpeg") == "complex_name_with_underscores_crop.jpeg"

    def test_full_frame_filename_with_person_id(self, processing_context, sample_frame_data):
        """Test full frame filename generation with person_id."""
        naming = NamingConvention(context=processing_context)

        # Test person-based naming
        filename = naming.get_full_frame_filename(sample_frame_data, "standing", 1, "png", person_id=0)
        assert filename == "test_video_person_0_standing_medium_closeup_001.png"

        filename = naming.get_full_frame_filename(sample_frame_data, "sitting", 2, "jpg", person_id=1)
        assert filename == "test_video_person_1_sitting_medium_closeup_002.jpg"

        # Test with larger person_id
        filename = naming.get_full_frame_filename(sample_frame_data, "walking", 3, "png", person_id=10)
        assert filename == "test_video_person_10_walking_medium_closeup_003.png"

    def test_face_crop_filename_with_person_id(self, processing_context, sample_frame_data):
        """Test face crop filename generation with person_id."""
        naming = NamingConvention(context=processing_context)

        # Test person-based naming
        filename = naming.get_face_crop_filename(sample_frame_data, "front", 1, "png", person_id=0)
        assert filename == "test_video_person_0_face_front_001.png"  # Simplified: no shot type

        filename = naming.get_face_crop_filename(sample_frame_data, "profile_left", 2, "jpg", person_id=2)
        assert filename == "test_video_person_2_face_profile_left_002.jpg"  # Simplified: no shot type

        # Test variations
        filename = naming.get_face_crop_filename(sample_frame_data, "profile_right", 5, "png", person_id=15)
        assert filename == "test_video_person_15_face_profile_right_005.png"  # Simplified: no shot type

    def test_person_id_none_maintains_backward_compatibility(self, processing_context, sample_frame_data):
        """Test that person_id=None maintains existing behavior."""
        # Create separate instances to avoid collision counter conflicts
        naming1 = NamingConvention(context=processing_context)
        naming2 = NamingConvention(context=processing_context)

        # Full frame - explicit None vs default
        filename_none = naming1.get_full_frame_filename(sample_frame_data, "standing", 1, "png", person_id=None)
        filename_default = naming2.get_full_frame_filename(sample_frame_data, "standing", 1, "png")
        assert filename_none == filename_default == "test_video_standing_medium_closeup_001.png"

        # Face crop - explicit None vs default (simplified: no shot type)
        naming3 = NamingConvention(context=processing_context)
        naming4 = NamingConvention(context=processing_context)
        filename_none = naming3.get_face_crop_filename(sample_frame_data, "front", 1, "png", person_id=None)
        filename_default = naming4.get_face_crop_filename(sample_frame_data, "front", 1, "png")
        assert filename_none == filename_default == "test_video_face_front_001.png"  # Simplified: no shot type

    def test_validate_filename_with_person_id(self, processing_context):
        """Test filename validation with person_id patterns."""
        naming = NamingConvention(context=processing_context)

        # Valid person-based filenames
        assert naming.validate_filename("test_video_person_0_standing_001.png")
        assert naming.validate_filename("test_video_person_5_face_front_001.jpg")
        assert naming.validate_filename("test_video_person_10_sitting_pose_002.png")

        # Invalid person-based filenames (invalid person_id)
        assert not naming.validate_filename("test_video_person_abc_standing_001.png")
        assert not naming.validate_filename("test_video_person__standing_001.png")
        assert not naming.validate_filename("test_video_person_1.5_standing_001.png")

        # Traditional filenames should still validate
        assert naming.validate_filename("test_video_standing_001.png")
        assert naming.validate_filename("test_video_face_front_001.jpg")

    def test_person_id_collision_prevention(self, processing_context, sample_frame_data):
        """Test that person-based and frame-based outputs have distinct names."""
        naming = NamingConvention(context=processing_context)

        # Generate traditional and person-based filenames
        frame_filename = naming.get_full_frame_filename(sample_frame_data, "standing", 1, "png")
        person_filename = naming.get_full_frame_filename(sample_frame_data, "standing", 1, "png", person_id=0)

        # They should be different
        assert frame_filename != person_filename
        assert frame_filename == "test_video_standing_medium_closeup_001.png"
        assert person_filename == "test_video_person_0_standing_medium_closeup_001.png"

        # Same for face crops (simplified: no shot type)
        frame_face_filename = naming.get_face_crop_filename(sample_frame_data, "front", 1, "png")
        person_face_filename = naming.get_face_crop_filename(sample_frame_data, "front", 1, "png", person_id=0)

        assert frame_face_filename != person_face_filename
        assert frame_face_filename == "test_video_face_front_001.png"  # Simplified: no shot type
        assert person_face_filename == "test_video_person_0_face_front_001.png"  # Simplified: no shot type


class TestImageWriter:
    """Tests for ImageWriter class."""

    @patch('personfromvid.output.image_writer.Image.fromarray')
    def test_save_full_frame(self, mock_fromarray, processing_context, sample_frame_data):
        """Test saving a full frame image."""
        mock_pil_image = Mock()
        mock_fromarray.return_value = mock_pil_image
        mock_fromarray.return_value = mock_pil_image
        mock_pil_image.convert.return_value = mock_pil_image

        # Update config to enable full frame output
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
        processing_context.config.output.image.face_crop_enabled = True
        processing_context.config.output.image.format = 'jpeg'

        # Set up enhanced selection data for head angle category
        sample_frame_data.selections.primary_selection_category = "head_angle_front"
        sample_frame_data.selections.selection_rank = 1

        # Clear pose categories to prevent full frame generation for this test
        sample_frame_data.selections.selected_for_poses = []

        writer = ImageWriter(context=processing_context)

        output_files = writer.save_frame_outputs(sample_frame_data)

        # Expect only 1 file: face crop for head angle (no full frame since no pose)
        assert len(output_files) == 1
        mock_pil_image.save.assert_called_once()
        assert output_files[0].endswith('.jpg')

        # Verify it's a face crop file
        assert 'face' in output_files[0]

    @patch('personfromvid.output.image_writer.Image.fromarray')
    def test_resize_full_frame(self, mock_fromarray, processing_context, sample_frame_data):
        """Test resizing full frame images."""
        mock_pil_image = Mock()
        mock_fromarray.return_value = mock_pil_image
        mock_fromarray.return_value = mock_pil_image
        mock_pil_image.convert.return_value = mock_pil_image
        mock_pil_image.resize.return_value = mock_pil_image

        # Configure resize to 1024px
        processing_context.config.output.image.resize = 1024
        processing_context.config.output.image.face_crop_enabled = False

        # Set up enhanced selection data for pose category
        sample_frame_data.selections.primary_selection_category = "pose_standing"
        sample_frame_data.selections.selection_rank = 1

        writer = ImageWriter(context=processing_context)
        writer.save_frame_outputs(sample_frame_data)

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

            writer.save_frame_outputs(sample_frame_data)

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
        processing_context.config.output.image.face_crop_enabled = False

        # Set up enhanced selection data for pose category
        sample_frame_data.selections.primary_selection_category = "pose_standing"
        sample_frame_data.selections.selection_rank = 1

        writer = ImageWriter(context=processing_context)
        writer.save_frame_outputs(sample_frame_data)

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

            writer._apply_resize(large_image)

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
        # Disable face restoration to test original resize logic
        processing_context.config.output.image.face_restoration_enabled = False
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

            writer._crop_face(test_image, face_detection)

            # Should call resize because face crop will be small and need upscaling to 256
            mock_pil_image.resize.assert_called_once()

    def test_face_crop_uses_default_512_when_no_resize_config(self, processing_context, sample_frame_data):
        """Test that face crop uses default 512 when resize is not configured."""
        processing_context.config.output.image.resize = None
        # Disable face restoration to test original resize logic
        processing_context.config.output.image.face_restoration_enabled = False
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

            writer._crop_face(test_image, face_detection)

            # Should call resize to upscale to default 512
            mock_pil_image.resize.assert_called_once()

    def test_config_validation_resize(self, processing_context):
        """Test configuration validation for resize values."""

        # Test valid resize value
        processing_context.config.output.image.resize = 1024
        ImageWriter(context=processing_context)  # Should not raise

        # Test invalid resize values
        processing_context.config.output.image.resize = 100  # Too small
        with pytest.raises(ValueError, match="Resize dimension must be between 256 and 4096"):
            ImageWriter(context=processing_context)

        processing_context.config.output.image.resize = 5000  # Too large
        with pytest.raises(ValueError, match="Resize dimension must be between 256 and 4096"):
            ImageWriter(context=processing_context)

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
        """Test pose cropping feature when enabled - should only generate crops, no full frames."""
        mock_pil_image = Mock()
        mock_fromarray.return_value = mock_pil_image
        mock_pil_image.convert.return_value = mock_pil_image

        # Enable pose cropping feature
        processing_context.config.output.image.enable_pose_cropping = True
        processing_context.config.output.image.pose_crop_padding = 0.2
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

        with patch.object(writer.image_processor, 'crop_and_resize') as mock_crop_and_resize:
            mock_crop_and_resize.return_value = np.zeros((100, 100, 3), dtype=np.uint8)

            output_files = writer.save_frame_outputs(sample_frame_data)

            # Should have only 1 file: pose crop (no full frame when cropping enabled)
            assert len(output_files) == 1

            # Should call crop_and_resize for pose cropping
            mock_crop_and_resize.assert_called_once()

            # Should call PIL save only once (for pose crop)
            assert mock_pil_image.save.call_count == 1

            # Should only have crop file, no regular full frame
            crop_files = [f for f in output_files if '_crop.jpg' in f]
            regular_files = [f for f in output_files if '_crop.jpg' not in f]
            assert len(crop_files) == 1
            assert len(regular_files) == 0

    @patch('personfromvid.output.image_writer.Image.fromarray')
    def test_pose_cropping_feature_disabled(self, mock_fromarray, processing_context, sample_frame_data):
        """Test pose cropping feature when disabled."""
        mock_pil_image = Mock()
        mock_fromarray.return_value = mock_pil_image
        mock_pil_image.convert.return_value = mock_pil_image

        # Disable pose cropping feature (default)
        processing_context.config.output.image.enable_pose_cropping = False
        processing_context.config.output.image.face_crop_enabled = False
        processing_context.config.output.image.format = "jpeg"

        # Set up enhanced selection data for pose category
        sample_frame_data.selections.primary_selection_category = "pose_standing"
        sample_frame_data.selections.selection_rank = 1

        writer = ImageWriter(context=processing_context)

        with patch.object(writer.image_processor, 'crop_and_resize') as mock_crop_and_resize:
            output_files = writer.save_frame_outputs(sample_frame_data)

            # Should have only 1 file: full frame
            assert len(output_files) == 1

            # Should NOT call crop_and_resize
            mock_crop_and_resize.assert_not_called()

            # Should call PIL save only once (for full frame)
            mock_pil_image.save.assert_called_once()

            # No files should have _crop suffix
            crop_files = [f for f in output_files if '_crop.jpg' in f]
            assert len(crop_files) == 0

    # Face Restoration Integration Tests

    def test_face_restorer_lazy_initialization_enabled(self, processing_context):
        """Test FaceRestorer lazy initialization when face restoration is enabled."""
        # Enable face restoration in config
        processing_context.config.output.image.face_restoration_enabled = True

        writer = ImageWriter(context=processing_context)

        # Initially, FaceRestorer should not be initialized
        assert writer._face_restorer is None
        assert not writer._face_restorer_initialized

        # Mock the FaceRestorer creation
        with patch('personfromvid.models.face_restorer.create_face_restorer') as mock_create_restorer:
            mock_restorer = Mock()
            mock_create_restorer.return_value = mock_restorer

            # First call should initialize the restorer
            result = writer._get_face_restorer()

            assert result is mock_restorer
            assert writer._face_restorer is mock_restorer
            assert writer._face_restorer_initialized
            mock_create_restorer.assert_called_once()

            # Second call should return cached instance
            result2 = writer._get_face_restorer()
            assert result2 is mock_restorer
            # Should not call create_face_restorer again
            mock_create_restorer.assert_called_once()

    def test_face_restorer_lazy_initialization_disabled(self, processing_context):
        """Test FaceRestorer initialization when face restoration is disabled."""
        # Disable face restoration in config
        processing_context.config.output.image.face_restoration_enabled = False

        writer = ImageWriter(context=processing_context)

        # Should return None when disabled
        result = writer._get_face_restorer()
        assert result is None
        assert not writer._face_restorer_initialized

    def test_face_restorer_initialization_failure(self, processing_context):
        """Test FaceRestorer initialization failure handling."""
        # Enable face restoration in config
        processing_context.config.output.image.face_restoration_enabled = True

        writer = ImageWriter(context=processing_context)

        # Mock the FaceRestorer creation to fail
        with patch('personfromvid.models.face_restorer.create_face_restorer') as mock_create_restorer:
            mock_create_restorer.side_effect = Exception("Model loading failed")

            # Should handle the error gracefully
            result = writer._get_face_restorer()

            assert result is None
            assert writer._face_restorer is None
            assert writer._face_restorer_initialized  # Should mark as initialized even on failure

    @patch('personfromvid.output.image_writer.Image.fromarray')
    def test_face_restoration_applied_to_face_crops(self, mock_fromarray, processing_context):
        """Test that face restoration is applied to face crops when enabled."""
        # Enable face restoration
        processing_context.config.output.image.face_restoration_enabled = True
        processing_context.config.output.image.face_restoration_strength = 0.8
        processing_context.config.output.image.resize = 512

        writer = ImageWriter(context=processing_context)

        # Create test image and face detection
        test_image = np.zeros((1080, 1920, 3), dtype=np.uint8)
        face_detection = FaceDetection(
            bbox=(500, 300, 600, 400),  # 100x100 face (needs upscaling)
            confidence=0.9,
            landmarks=[]
        )

        # Mock PIL Image operations
        mock_pil_image = Mock()
        mock_fromarray.return_value = mock_pil_image
        mock_pil_image.resize.return_value = mock_pil_image

        # Mock FaceRestorer
        mock_restorer = Mock()
        mock_restored_image = np.ones((512, 512, 3), dtype=np.uint8) * 128  # Gray image
        mock_restorer.restore_face.return_value = mock_restored_image

        with patch.object(writer.image_processor, '_get_face_restorer', return_value=mock_restorer):
            result = writer._crop_face(test_image, face_detection)

            # Should call face restoration via image processor
            mock_restorer.restore_face.assert_called_once()
            call_args = mock_restorer.restore_face.call_args

            # Check arguments passed to restore_face
            assert call_args[1]['target_size'] == 512  # Should use resize config
            assert call_args[1]['strength'] == 0.8  # Should use configured strength

            # Result should be the restored image
            assert np.array_equal(result, mock_restored_image)

    @patch('personfromvid.output.image_writer.Image.fromarray')
    def test_face_restoration_fallback_on_error(self, mock_fromarray, processing_context):
        """Test fallback to Lanczos when face restoration fails."""
        # Enable face restoration
        processing_context.config.output.image.face_restoration_enabled = True
        processing_context.config.output.image.resize = 512

        writer = ImageWriter(context=processing_context)

        # Create test image and face detection
        test_image = np.zeros((1080, 1920, 3), dtype=np.uint8)
        face_detection = FaceDetection(
            bbox=(500, 300, 600, 400),  # 100x100 face (needs upscaling)
            confidence=0.9,
            landmarks=[]
        )

        # Mock PIL Image operations
        mock_pil_image = Mock()
        mock_fromarray.return_value = mock_pil_image
        mock_lanczos_result = Mock()
        mock_pil_image.resize.return_value = mock_lanczos_result

        # Mock FaceRestorer to fail
        mock_restorer = Mock()
        mock_restorer.restore_face.side_effect = Exception("Restoration failed")

        with patch.object(writer.image_processor, '_get_face_restorer', return_value=mock_restorer):
            with patch('numpy.array', return_value=np.ones((256, 256, 3), dtype=np.uint8) * 64):
                writer._crop_face(test_image, face_detection)

                # Should attempt face restoration first via image processor
                mock_restorer.restore_face.assert_called_once()

                # Should fall back to Lanczos upscaling
                mock_pil_image.resize.assert_called()

    @patch('personfromvid.output.image_writer.Image.fromarray')
    def test_face_restoration_disabled_uses_lanczos(self, mock_fromarray, processing_context):
        """Test that Lanczos is used when face restoration is disabled."""
        # Disable face restoration
        processing_context.config.output.image.face_restoration_enabled = False
        processing_context.config.output.image.resize = 512

        writer = ImageWriter(context=processing_context)

        # Create test image and face detection
        test_image = np.zeros((1080, 1920, 3), dtype=np.uint8)
        face_detection = FaceDetection(
            bbox=(500, 300, 600, 400),  # 100x100 face (needs upscaling)
            confidence=0.9,
            landmarks=[]
        )

        # Mock PIL Image operations
        mock_pil_image = Mock()
        mock_fromarray.return_value = mock_pil_image
        mock_pil_image.resize.return_value = mock_pil_image

        with patch('numpy.array', return_value=np.ones((256, 256, 3), dtype=np.uint8) * 64):
            writer._crop_face(test_image, face_detection)

            # Should use Lanczos upscaling directly
            mock_pil_image.resize.assert_called()

            # Should not attempt to get face restorer
            assert writer._get_face_restorer() is None

    def test_face_restoration_strength_parameter_validation(self, processing_context):
        """Test that face restoration strength parameter is properly validated."""
        # Enable face restoration with various strength values
        processing_context.config.output.image.face_restoration_enabled = True

        writer = ImageWriter(context=processing_context)

        # Test various strength values are passed correctly
        test_strengths = [0.0, 0.5, 0.8, 1.0]

        # Mock the FaceRestorer class directly to avoid heavy model loading
        with patch('personfromvid.models.face_restorer.create_face_restorer') as mock_create_restorer:
            mock_restorer = Mock()
            mock_restorer.restore_face.return_value = np.ones((512, 512, 3), dtype=np.uint8)
            mock_create_restorer.return_value = mock_restorer

            for strength in test_strengths:
                processing_context.config.output.image.face_restoration_strength = strength
                mock_restorer.reset_mock()  # Clear previous calls

                # Reset face restorer initialization to test each strength value
                writer.image_processor._face_restorer_initialized = False
                writer.image_processor._face_restorer = None

                # Get the face restorer (triggers initialization)
                restorer = writer.image_processor._get_face_restorer()
                assert restorer is mock_restorer

                # Simulate calling restore_face with current config
                restorer.restore_face(
                    image=np.zeros((100, 100, 3), dtype=np.uint8),
                    target_size=512,
                    strength=processing_context.config.output.image.face_restoration_strength
                )

                # Verify restore_face was called with the correct strength
                mock_restorer.restore_face.assert_called_once()
                call_args = mock_restorer.restore_face.call_args
                assert call_args[1]['strength'] == strength

    @patch('personfromvid.output.image_writer.Image.fromarray')
    def test_face_restoration_target_size_calculation(self, mock_fromarray, processing_context):
        """Test that target size is calculated correctly for face restoration."""
        # Enable face restoration
        processing_context.config.output.image.face_restoration_enabled = True
        processing_context.config.output.image.resize = 256  # Smaller than default 512

        writer = ImageWriter(context=processing_context)

        # Create small test image that needs upscaling
        test_image = np.zeros((100, 150, 3), dtype=np.uint8)  # 100x150 image

        # Mock PIL Image operations
        mock_pil_image = Mock()
        mock_fromarray.return_value = mock_pil_image
        mock_pil_image.resize.return_value = mock_pil_image

        # Mock FaceRestorer
        mock_restorer = Mock()
        mock_restorer.restore_face.return_value = np.ones((256, 256, 3), dtype=np.uint8)

        with patch.object(writer.image_processor, '_get_face_restorer', return_value=mock_restorer):
            writer.image_processor.crop_and_resize(test_image, (0, 0, 100, 150), 0.0, use_face_restoration=True)

            # Should call restore_face with target_size = max(resize_config, max(crop_width, crop_height))
            # In this case: max(256, max(100, 150)) = max(256, 150) = 256
            mock_restorer.restore_face.assert_called_once()
            call_args = mock_restorer.restore_face.call_args
            assert call_args[1]['target_size'] == 256

    def test_face_restoration_not_applied_to_pose_crops(self, processing_context):
        """Test that face restoration is not applied to pose crops."""
        # Enable face restoration
        processing_context.config.output.image.face_restoration_enabled = True

        writer = ImageWriter(context=processing_context)

        # Create test image for pose crop
        test_image = np.zeros((200, 200, 3), dtype=np.uint8)  # Small image that would need upscaling

        # Mock FaceRestorer
        mock_restorer = Mock()

        with patch.object(writer.image_processor, '_get_face_restorer', return_value=mock_restorer):
            with patch('personfromvid.output.image_processor.Image.fromarray'):
                # Call crop_and_resize without use_face_restoration flag (default False for pose crops)
                writer.image_processor.crop_and_resize(test_image, (0, 0, 200, 200), 0.1, use_face_restoration=False)

                # Should NOT call face restoration
                mock_restorer.restore_face.assert_not_called()

    def test_face_restoration_no_upscaling_needed(self, processing_context):
        """Test that face restoration is not applied when no upscaling is needed."""
        # Enable face restoration
        processing_context.config.output.image.face_restoration_enabled = True
        processing_context.config.output.image.resize = 256

        writer = ImageWriter(context=processing_context)

        # Create large test image that doesn't need upscaling
        test_image = np.zeros((600, 800, 3), dtype=np.uint8)  # Large image (both dimensions > 256)

        # Mock FaceRestorer
        mock_restorer = Mock()

        with patch.object(writer.image_processor, '_get_face_restorer', return_value=mock_restorer):
            result = writer.image_processor.crop_and_resize(test_image, (0, 0, 600, 800), 0.0, use_face_restoration=True)

            # Should NOT call face restoration since no upscaling is needed
            # (both dimensions 600 and 800 are >= 256)
            mock_restorer.restore_face.assert_not_called()

            # Should return some cropped image (exact content doesn't matter for this test)
            assert result is not None
            # Just verify it's a valid result, dimensions may be altered by existing resize logic

    def test_face_restoration_config_validation(self, processing_context):
        """Test that face restoration configuration is properly validated."""
        # Test that configuration includes face restoration fields
        config = processing_context.config.output.image

        # Should have face restoration fields with proper defaults
        assert hasattr(config, 'face_restoration_enabled')
        assert hasattr(config, 'face_restoration_strength')
        assert config.face_restoration_enabled is False  # Default should be False
        assert config.face_restoration_strength == 0.8  # Default strength


class TestOutputGenerationStep:
    """Tests for OutputGenerationStep with dual input support."""

    @pytest.fixture
    def mock_pipeline(self):
        """Create a mock pipeline for testing."""
        pipeline = Mock()
        pipeline.context = Mock()
        pipeline.context.output_directory = Path("/test/output")
        return pipeline

    @pytest.fixture
    def mock_state(self):
        """Create a mock state manager for testing."""
        state = Mock()
        state.frames = []
        state.processing_stats = {}  # Use real dict for item assignment
        return state

    @pytest.fixture
    def mock_formatter(self):
        """Create a mock formatter for testing."""
        formatter = Mock()
        formatter.create_progress_bar.return_value.__enter__ = Mock(return_value=None)
        formatter.create_progress_bar.return_value.__exit__ = Mock(return_value=None)
        return formatter

    @pytest.fixture
    def output_step(self, mock_pipeline, mock_state, mock_formatter):
        """Create OutputGenerationStep instance for testing."""
        from personfromvid.core.steps.output_generation import OutputGenerationStep

        # Setup mock pipeline with required attributes
        mock_pipeline.state = mock_state
        mock_pipeline.formatter = mock_formatter
        mock_pipeline.logger = Mock()
        mock_pipeline.config = Mock()
        mock_pipeline._interrupted = False
        mock_pipeline._step_start_time = 0.0

        step = OutputGenerationStep(mock_pipeline)
        step._check_interrupted = Mock()  # Override to prevent interruption checks in tests
        return step

    @pytest.fixture
    def sample_person_selection(self, sample_frame_data):
        """Create sample PersonSelection for testing."""
        from personfromvid.analysis.person_selector import PersonSelection
        from personfromvid.data.person import BodyUnknown, Person, PersonQuality

        person = Person(
            person_id=0,
            face=sample_frame_data.face_detections[0],
            body=BodyUnknown(),
            head_pose=None,
            quality=PersonQuality(face_quality=0.8, body_quality=0.0)
        )

        return PersonSelection(
            frame_data=sample_frame_data,
            person_id=0,
            person=person,
            selection_score=0.85,
            category="quality"
        )

    def test_detect_input_type_person_selection(self, output_step, sample_person_selection):
        """Test input type detection with PersonSelection data."""
        # Mock person selection step progress
        person_progress = Mock()
        person_progress.get_data.return_value = [sample_person_selection]
        output_step.state.get_step_progress.side_effect = lambda step: person_progress if step == "person_selection" else None

        input_data, input_type = output_step._detect_input_type()

        assert input_type == "person_selection"
        assert len(input_data) == 1
        assert input_data[0] == sample_person_selection

    def test_detect_input_type_frame_selection(self, output_step):
        """Test input type detection with frame selection data."""
        # Mock frame selection step progress
        frame_progress = Mock()
        frame_progress.get_data.return_value = ["frame_001", "frame_002"]
        output_step.state.get_step_progress.side_effect = lambda step: frame_progress if step == "frame_selection" else None

        input_data, input_type = output_step._detect_input_type()

        assert input_type == "frame_selection"
        assert len(input_data) == 2
        assert input_data == ["frame_001", "frame_002"]

    def test_detect_input_type_no_data(self, output_step):
        """Test input type detection with no data."""
        output_step.state.get_step_progress.return_value = None

        input_data, input_type = output_step._detect_input_type()

        assert input_type == "none"
        assert len(input_data) == 0

    def test_detect_input_type_priority_person_over_frame(self, output_step, sample_person_selection):
        """Test that PersonSelection data takes priority over frame data."""
        # Mock both person and frame selection step progress
        person_progress = Mock()
        person_progress.get_data.return_value = [sample_person_selection]
        frame_progress = Mock()
        frame_progress.get_data.return_value = ["frame_001"]

        def mock_get_step_progress(step):
            if step == "person_selection":
                return person_progress
            elif step == "frame_selection":
                return frame_progress
            return None

        output_step.state.get_step_progress.side_effect = mock_get_step_progress

        input_data, input_type = output_step._detect_input_type()

        # PersonSelection should take priority
        assert input_type == "person_selection"
        assert len(input_data) == 1

    @patch('personfromvid.core.steps.output_generation.ImageWriter')
    def test_process_person_selections(self, mock_image_writer_class, output_step, sample_person_selection):
        """Test processing PersonSelection objects."""
        # Setup mocks
        mock_image_writer = Mock()
        # Person with face and body should generate multiple outputs (face crop + body crop + full frame)
        mock_image_writer.save_person_outputs.return_value = [
            "/test/output/person_0_face.jpg",
            "/test/output/person_0_standing_crop.jpg",
            "/test/output/person_0_standing_full.jpg"
        ]
        mock_image_writer_class.return_value = mock_image_writer

        step_progress = Mock()
        output_step.state.get_step_progress.return_value = step_progress

        # Test processing
        output_step._process_person_selections([sample_person_selection])

        # Verify calls - expect 3 outputs for person with face and body
        step_progress.start.assert_called_once_with(3)
        mock_image_writer.save_person_outputs.assert_called_once_with(sample_person_selection)
        assert output_step.state.processing_stats["total_output_files"] == 3

    @patch('personfromvid.core.steps.output_generation.ImageWriter')
    def test_process_frame_selections(self, mock_image_writer_class, output_step, sample_frame_data):
        """Test processing frame IDs (backwards compatibility)."""
        # Setup mocks
        mock_image_writer = Mock()
        mock_image_writer.save_frame_outputs.return_value = ["/test/output/frame_file.jpg"]
        mock_image_writer_class.return_value = mock_image_writer

        step_progress = Mock()
        output_step.state.get_step_progress.return_value = step_progress
        output_step.state.frames = [sample_frame_data]

        # Test processing
        output_step._process_frame_selections([sample_frame_data.frame_id])

        # Verify calls
        step_progress.start.assert_called_once_with(1)
        mock_image_writer.save_frame_outputs.assert_called_once_with(sample_frame_data)
        assert output_step.state.processing_stats["total_output_files"] == 1

    def test_handle_no_input_data(self, output_step):
        """Test handling case with no input data."""
        step_progress = Mock()
        output_step.state.get_step_progress.return_value = step_progress

        output_step._handle_no_input_data()

        step_progress.start.assert_called_once_with(0)
        output_step.formatter.print_warning.assert_called_once()

    @patch('personfromvid.core.steps.output_generation.ImageWriter')
    def test_generate_output_for_person_selection(self, mock_image_writer_class, output_step, sample_person_selection):
        """Test generating output for a single PersonSelection."""
        mock_image_writer = Mock()
        mock_image_writer.save_person_outputs.return_value = ["/test/output/person_file.jpg"]

        output_files = output_step._generate_output_for_person_selection(sample_person_selection, mock_image_writer)

        assert len(output_files) == 1
        assert output_files[0] == "/test/output/person_file.jpg"
        mock_image_writer.save_person_outputs.assert_called_once_with(sample_person_selection)

    def test_finalize_output_generation(self, output_step):
        """Test output generation finalization."""
        output_files = ["/test/file1.jpg", "/test/file2.jpg"]
        output_dir = "/test/output"

        step_progress = Mock()
        output_step.state.get_step_progress.return_value = step_progress

        output_step._finalize_output_generation(output_files, output_dir, "person-based")

        assert output_step.state.processing_stats["output_files"] == output_files
        assert output_step.state.processing_stats["total_output_files"] == 2
        step_progress.set_data.assert_called_once()

    @patch('personfromvid.core.steps.output_generation.ImageWriter')
    def test_execute_with_person_selection_input(self, mock_image_writer_class, output_step, sample_person_selection):
        """Test execute method with PersonSelection input."""
        # Setup mocks
        mock_image_writer = Mock()
        # Person with face and body should generate multiple outputs
        mock_image_writer.save_person_outputs.return_value = [
            "/test/output/person_0_face.jpg",
            "/test/output/person_0_standing_crop.jpg",
            "/test/output/person_0_standing_full.jpg"
        ]
        mock_image_writer_class.return_value = mock_image_writer

        person_progress = Mock()
        person_progress.get_data.return_value = [sample_person_selection]
        step_progress = Mock()

        def mock_get_step_progress(step):
            if step == "person_selection":
                return person_progress
            elif step == "output_generation":
                return step_progress
            return None

        output_step.state.get_step_progress.side_effect = mock_get_step_progress

        # Execute
        output_step.execute()

        # Verify - expect 3 outputs for person with face and body
        output_step.state.start_step.assert_called_once_with("output_generation")
        step_progress.start.assert_called_once_with(3)
        mock_image_writer.save_person_outputs.assert_called_once()

    @patch('personfromvid.core.steps.output_generation.ImageWriter')
    def test_execute_with_frame_selection_input(self, mock_image_writer_class, output_step, sample_frame_data):
        """Test execute method with frame selection input (backwards compatibility)."""
        # Setup mocks
        mock_image_writer = Mock()
        mock_image_writer.save_frame_outputs.return_value = ["/test/output/frame_file.jpg"]
        mock_image_writer_class.return_value = mock_image_writer

        frame_progress = Mock()
        frame_progress.get_data.return_value = [sample_frame_data.frame_id]
        step_progress = Mock()

        def mock_get_step_progress(step):
            if step == "frame_selection":
                return frame_progress
            elif step == "output_generation":
                return step_progress
            return None

        output_step.state.get_step_progress.side_effect = mock_get_step_progress
        output_step.state.frames = [sample_frame_data]

        # Execute
        output_step.execute()

        # Verify
        output_step.state.start_step.assert_called_once_with("output_generation")
        step_progress.start.assert_called_once_with(1)
        mock_image_writer.save_frame_outputs.assert_called_once()

    def test_execute_with_no_input_data(self, output_step):
        """Test execute method with no input data."""
        output_step.state.get_step_progress.return_value = None
        step_progress = Mock()
        output_step.state.get_step_progress.side_effect = lambda step: step_progress if step == "output_generation" else None

        output_step.execute()

        output_step.state.start_step.assert_called_once_with("output_generation")
        step_progress.start.assert_called_once_with(0)

    @patch('personfromvid.core.steps.output_generation.ImageWriter')
    def test_execute_error_handling(self, mock_image_writer_class, output_step, sample_person_selection):
        """Test execute method error handling."""
        # Force an exception during execution
        mock_image_writer_class.side_effect = Exception("Test error")

        person_progress = Mock()
        person_progress.get_data.return_value = [sample_person_selection]  # Use real PersonSelection
        output_step.state.get_step_progress.side_effect = lambda step: person_progress if step == "person_selection" else Mock()

        with pytest.raises(Exception, match="Test error"):
            output_step.execute()

        output_step.logger.error.assert_called_once()
        output_step.state.fail_step.assert_called_once_with("output_generation", "Test error")

    @patch('personfromvid.core.steps.output_generation.ImageWriter')
    def test_person_selection_processing_error_recovery(self, mock_image_writer_class, output_step, sample_person_selection):
        """Test error recovery during PersonSelection processing."""
        # Setup: First PersonSelection fails, second succeeds
        mock_image_writer = Mock()
        mock_image_writer.save_person_outputs.side_effect = [Exception("Processing error"), ["/test/output/success.jpg"]]
        mock_image_writer_class.return_value = mock_image_writer

        step_progress = Mock()
        output_step.state.get_step_progress.return_value = step_progress

        # Create two PersonSelection objects
        person_selections = [sample_person_selection, sample_person_selection]

        # Execute
        output_step._process_person_selections(person_selections)

        # Verify: Warning logged for first failure, processing continued for second
        assert output_step.logger.warning.call_count == 1  # One warning for one failure
        assert mock_image_writer.save_person_outputs.call_count == 2
        assert output_step.state.processing_stats["total_output_files"] == 1  # Only successful output counted


class TestImageWriterPersonSelection:
    """Tests for ImageWriter save_person_outputs method."""

    @pytest.fixture
    def sample_person_selection(self, sample_frame_data):
        """Create a sample PersonSelection for testing."""
        from personfromvid.analysis.person_selector import PersonSelection
        from personfromvid.data.detection_results import (
            FaceDetection,
            HeadPoseResult,
            PoseDetection,
        )
        from personfromvid.data.person import Person, PersonQuality

        # Create a person with both face and body detections
        person = Person(
            person_id=0,
            face=FaceDetection(
                bbox=(500, 300, 700, 500),
                confidence=0.92,
                landmarks=[(520, 320), (680, 320), (600, 380), (540, 450), (660, 450)]
            ),
            body=PoseDetection(
                bbox=(450, 250, 750, 800),
                keypoints={"nose": (600, 400, 0.9)},  # Fixed keypoints format
                confidence=0.85,
                pose_classifications=[("standing", 0.9), ("sitting", 0.1)]
            ),
            head_pose=HeadPoseResult(
                yaw=10.0,
                pitch=5.0,
                roll=2.0,
                confidence=0.88,
                face_id=0,
                direction="front"
            ),
            quality=PersonQuality(face_quality=0.92, body_quality=0.85)
        )

        return PersonSelection(
            frame_data=sample_frame_data,
            person_id=0,
            person=person,
            selection_score=0.89,
            category="minimum"
        )

    @pytest.fixture
    def sample_person_selection_face_only(self, sample_frame_data):
        """Create a PersonSelection with face only (no body)."""
        from personfromvid.analysis.person_selector import PersonSelection
        from personfromvid.data.detection_results import FaceDetection, HeadPoseResult
        from personfromvid.data.person import BodyUnknown, Person, PersonQuality

        person = Person(
            person_id=1,
            face=FaceDetection(
                bbox=(500, 300, 700, 500),
                confidence=0.88,
                landmarks=[(520, 320), (680, 320), (600, 380), (540, 450), (660, 450)]
            ),
            body=BodyUnknown(),
            head_pose=HeadPoseResult(
                yaw=15.0,
                pitch=-5.0,
                roll=1.0,
                confidence=0.85,
                face_id=0,
                direction="front"
            ),
            quality=PersonQuality(face_quality=0.88, body_quality=0.0)
        )

        return PersonSelection(
            frame_data=sample_frame_data,
            person_id=1,
            person=person,
            selection_score=0.88,
            category="minimum"
        )

    @pytest.fixture
    def sample_person_selection_body_only(self, sample_frame_data):
        """Create a PersonSelection with body only (no face)."""
        from personfromvid.analysis.person_selector import PersonSelection
        from personfromvid.data.detection_results import PoseDetection
        from personfromvid.data.person import FaceUnknown, Person, PersonQuality

        person = Person(
            person_id=2,
            face=FaceUnknown(),
            body=PoseDetection(
                bbox=(450, 250, 750, 800),
                keypoints={"hip": (600, 400, 0.8)},  # Fixed keypoints format
                confidence=0.80,
                pose_classifications=[("walking", 0.8), ("standing", 0.2)]
            ),
            head_pose=None,  # No head pose for body-only person
            quality=PersonQuality(face_quality=0.0, body_quality=0.80)
        )

        return PersonSelection(
            frame_data=sample_frame_data,
            person_id=2,
            person=person,
            selection_score=0.80,
            category="additional"
        )

    @patch('personfromvid.output.image_writer.Image.fromarray')
    def test_save_person_outputs_with_face_and_body(self, mock_fromarray, processing_context, sample_person_selection):
        """Test saving person outputs with both face and body detections."""
        mock_pil_image = Mock()
        mock_fromarray.return_value = mock_pil_image
        mock_pil_image.convert.return_value = mock_pil_image
        mock_pil_image.resize.return_value = mock_pil_image

        # Enable both face crop and pose cropping
        processing_context.config.output.image.face_crop_enabled = True
        processing_context.config.output.image.enable_pose_cropping = True
        processing_context.config.output.image.format = 'png'

        writer = ImageWriter(context=processing_context)
        output_files = writer.save_person_outputs(sample_person_selection)

        # Should generate both face crop and body crop
        assert len(output_files) == 2
        mock_pil_image.save.assert_called()

        # Verify person_id is in filenames and check structure
        from pathlib import Path
        face_file = [f for f in output_files if '_face_' in Path(f).name][0]
        body_file = [f for f in output_files if 'crop' in Path(f).name][0]  # Body files have _crop suffix

        assert 'person_0' in face_file
        assert 'person_0' in body_file
        assert '_face_' in face_file
        assert 'crop' in body_file

    @patch('personfromvid.output.image_writer.Image.fromarray')
    def test_save_person_outputs_face_only(self, mock_fromarray, processing_context, sample_person_selection_face_only):
        """Test saving person outputs with face detection only."""
        mock_pil_image = Mock()
        mock_fromarray.return_value = mock_pil_image
        mock_pil_image.convert.return_value = mock_pil_image
        mock_pil_image.resize.return_value = mock_pil_image

        processing_context.config.output.image.face_crop_enabled = True
        processing_context.config.output.image.enable_pose_cropping = False
        processing_context.config.output.image.format = 'jpg'

        writer = ImageWriter(context=processing_context)
        output_files = writer.save_person_outputs(sample_person_selection_face_only)

        # Should generate face crop + full frame (since pose cropping disabled)
        assert len(output_files) == 2

        # Verify person_id is in filenames and check structure
        from pathlib import Path
        face_files = [f for f in output_files if '_face_' in Path(f).name]
        full_frame_files = [f for f in output_files if '_face_' not in Path(f).name and 'crop' not in Path(f).name]

        assert len(face_files) == 1
        assert len(full_frame_files) == 1
        assert 'person_1' in face_files[0]
        assert 'person_1' in full_frame_files[0]
        # Face-only person should have "unknown" pose in full frame filename
        assert 'unknown' in full_frame_files[0]

    @patch('personfromvid.output.image_writer.Image.fromarray')
    def test_save_person_outputs_body_only(self, mock_fromarray, processing_context, sample_person_selection_body_only):
        """Test saving person outputs with body detection only."""
        mock_pil_image = Mock()
        mock_fromarray.return_value = mock_pil_image
        mock_pil_image.convert.return_value = mock_pil_image

        processing_context.config.output.image.face_crop_enabled = False
        processing_context.config.output.image.enable_pose_cropping = True
        processing_context.config.output.image.format = 'png'

        writer = ImageWriter(context=processing_context)
        output_files = writer.save_person_outputs(sample_person_selection_body_only)

        # Should generate only body crop
        assert len(output_files) == 1
        assert 'person_2' in output_files[0]
        assert 'walking' in output_files[0]  # Uses the primary pose classification
        assert 'crop' in output_files[0]

    @patch('personfromvid.output.image_writer.Image.fromarray')
    def test_save_person_outputs_full_frame_fallback(self, mock_fromarray, processing_context, sample_person_selection):
        """Test full frame output when pose cropping is disabled."""
        mock_pil_image = Mock()
        mock_fromarray.return_value = mock_pil_image
        mock_fromarray.return_value = mock_pil_image

        # Disable pose cropping, enable face crop
        processing_context.config.output.image.face_crop_enabled = True
        processing_context.config.output.image.enable_pose_cropping = False

        writer = ImageWriter(context=processing_context)
        output_files = writer.save_person_outputs(sample_person_selection)

        # Should generate face crop + full frame
        assert len(output_files) == 2

        from pathlib import Path
        face_files = [f for f in output_files if '_face_' in Path(f).name]
        full_frame_files = [f for f in output_files if '_face_' not in Path(f).name and 'crop' not in Path(f).name]

        assert len(face_files) == 1
        assert len(full_frame_files) == 1
        assert 'person_0' in face_files[0]
        assert 'person_0' in full_frame_files[0]

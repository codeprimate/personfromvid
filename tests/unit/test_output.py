"""Unit tests for output generation components."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np

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


class TestImageWriter:
    """Tests for ImageWriter class."""

    @patch('personfromvid.output.image_writer.cv2.imread')
    @patch('personfromvid.output.image_writer.cv2.cvtColor')
    @patch('personfromvid.output.image_writer.Image.fromarray')
    def test_save_full_frame(self, mock_fromarray, mock_cvtcolor, mock_imread, processing_context, sample_frame_data):
        """Test saving a full frame image."""
        mock_imread.return_value = np.zeros((1080, 1920, 3), dtype=np.uint8)
        mock_cvtcolor.return_value = np.zeros((1080, 1920, 3), dtype=np.uint8)
        mock_pil_image = Mock()
        mock_fromarray.return_value = mock_pil_image
        mock_pil_image.convert.return_value = mock_pil_image

        # Update config to enable full frame output
        processing_context.config.output.image.full_frame_enabled = True
        processing_context.config.output.image.face_crop_enabled = False
        processing_context.config.output.image.format = 'png'
        
        writer = ImageWriter(context=processing_context)

        output_files = writer.save_frame_outputs(sample_frame_data, pose_categories=["standing"], head_angle_categories=[])
        
        assert len(output_files) == 1
        mock_imread.assert_called_once_with(str(sample_frame_data.file_path))
        mock_pil_image.save.assert_called_once()
        assert output_files[0].endswith('.png')

    @patch('personfromvid.output.image_writer.cv2.imread')
    @patch('personfromvid.output.image_writer.cv2.cvtColor')
    @patch('personfromvid.output.image_writer.Image.fromarray')
    def test_save_face_crop(self, mock_fromarray, mock_cvtcolor, mock_imread, processing_context, sample_frame_data):
        """Test saving a cropped face image."""
        mock_imread.return_value = np.zeros((1080, 1920, 3), dtype=np.uint8)
        mock_cvtcolor.return_value = np.zeros((1080, 1920, 3), dtype=np.uint8)
        mock_pil_image = Mock()
        mock_fromarray.return_value = mock_pil_image
        mock_pil_image.convert.return_value = mock_pil_image

        # Update config to enable face crop output
        processing_context.config.output.image.full_frame_enabled = False
        processing_context.config.output.image.face_crop_enabled = True
        processing_context.config.output.image.format = 'jpeg'
        
        writer = ImageWriter(context=processing_context)

        output_files = writer.save_frame_outputs(sample_frame_data, pose_categories=[], head_angle_categories=["front"])
        
        assert len(output_files) == 1
        mock_imread.assert_called_once_with(str(sample_frame_data.file_path))
        mock_pil_image.save.assert_called_once()
        assert output_files[0].endswith('.jpeg') 
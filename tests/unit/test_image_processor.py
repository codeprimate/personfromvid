"""Tests for ImageProcessor class."""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock, ANY

from personfromvid.output.image_processor import ImageProcessor
from personfromvid.data.config import OutputImageConfig


class TestImageProcessor:
    """Test ImageProcessor functionality."""

    @pytest.fixture
    def basic_config(self):
        """Basic configuration for testing."""
        return OutputImageConfig(
            default_crop_size=640,
            resize=None,
            face_restoration_enabled=False,
            face_restoration_strength=0.8
        )

    @pytest.fixture
    def face_restoration_config(self):
        """Configuration with face restoration enabled."""
        return OutputImageConfig(
            default_crop_size=640,
            resize=None,
            face_restoration_enabled=True,
            face_restoration_strength=0.8
        )

    @pytest.fixture
    def resize_config(self):
        """Configuration with resize parameter."""
        return OutputImageConfig(
            default_crop_size=640,
            resize=1024,
            face_restoration_enabled=False
        )

    @pytest.fixture
    def test_image(self):
        """Create a test image."""
        return np.random.randint(0, 255, (800, 800, 3), dtype=np.uint8)

    @pytest.fixture
    def small_test_image(self):
        """Create a small test image."""
        return np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)

    def test_initialization(self, basic_config):
        """Test ImageProcessor initialization."""
        processor = ImageProcessor(basic_config)
        
        assert processor.config == basic_config
        assert processor._face_restorer is None
        assert processor._face_restorer_initialized is False

    def test_basic_cropping_no_padding(self, basic_config, test_image):
        """Test basic cropping without padding."""
        processor = ImageProcessor(basic_config)
        bbox = (100, 100, 300, 300)  # 200x200 crop
        
        result = processor.crop_and_resize(test_image, bbox, padding=0.0)
        
        # 200x200 crop is smaller than default_crop_size (640), so it gets upscaled
        assert result.shape == (640, 640, 3)

    def test_cropping_with_padding(self, basic_config, test_image):
        """Test cropping with padding."""
        processor = ImageProcessor(basic_config)
        bbox = (200, 200, 400, 400)  # 200x200 base crop
        
        result = processor.crop_and_resize(test_image, bbox, padding=0.2)
        
        # With 20% padding, crop should be larger than base bbox and upscaled
        assert result.shape == (640, 640, 3)

    def test_boundary_handling(self, basic_config, test_image):
        """Test cropping near image boundaries."""
        processor = ImageProcessor(basic_config)
        # Bbox near edge of 800x800 image
        bbox = (700, 700, 800, 800)  # 100x100 at edge
        
        result = processor.crop_and_resize(test_image, bbox, padding=0.5)
        
        # Should handle boundary gracefully and upscale
        assert result.shape == (640, 640, 3)

    def test_upscaling_needed(self, basic_config, test_image):
        """Test upscaling when crop is smaller than minimum dimension."""
        processor = ImageProcessor(basic_config)
        # Small bbox that will need upscaling
        bbox = (100, 100, 200, 200)  # 100x100 crop (< 640 default_crop_size)
        
        with patch('PIL.Image.fromarray') as mock_from_array:
            mock_pil_image = Mock()
            mock_from_array.return_value = mock_pil_image
            mock_pil_image.resize.return_value = mock_pil_image
            mock_array_result = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            
            with patch('numpy.array', return_value=mock_array_result):
                result = processor.crop_and_resize(test_image, bbox, padding=0.0)
                
                # Should call PIL resize for upscaling
                mock_pil_image.resize.assert_called_once()

    def test_no_upscaling_needed(self, basic_config, test_image):
        """Test when crop is already large enough."""
        processor = ImageProcessor(basic_config)
        # Large bbox that doesn't need upscaling
        bbox = (0, 0, 700, 700)  # 700x700 crop (> 640 default_crop_size)
        
        result = processor.crop_and_resize(test_image, bbox, padding=0.0)
        
        # Should not upscale
        assert result.shape == (700, 700, 3)

    def test_resize_config_priority(self, resize_config, test_image):
        """Test that resize config takes priority over default_crop_size."""
        processor = ImageProcessor(resize_config)
        bbox = (100, 100, 200, 200)  # 100x100 crop
        
        with patch('PIL.Image.fromarray') as mock_from_array:
            mock_pil_image = Mock()
            mock_from_array.return_value = mock_pil_image
            mock_pil_image.resize.return_value = mock_pil_image
            
            processor.crop_and_resize(test_image, bbox, padding=0.0)
            
            # Should use resize value (1024) not default_crop_size (640)
            # Scale factor should be 1024/100 = 10.24
            expected_size = (1024, 1024)
            mock_pil_image.resize.assert_called_with(expected_size, ANY)

    @patch('personfromvid.models.face_restorer.create_face_restorer')
    def test_face_restoration_lazy_loading(self, mock_create_restorer, face_restoration_config):
        """Test face restorer lazy loading."""
        mock_restorer = Mock()
        mock_create_restorer.return_value = mock_restorer
        
        processor = ImageProcessor(face_restoration_config)
        
        # First call should initialize
        restorer1 = processor._get_face_restorer()
        assert restorer1 == mock_restorer
        assert processor._face_restorer_initialized is True
        mock_create_restorer.assert_called_once()
        
        # Second call should reuse
        restorer2 = processor._get_face_restorer()
        assert restorer2 == mock_restorer
        assert mock_create_restorer.call_count == 1

    def test_face_restoration_disabled(self, basic_config):
        """Test face restoration when disabled in config."""
        processor = ImageProcessor(basic_config)
        
        restorer = processor._get_face_restorer()
        assert restorer is None

    @patch('personfromvid.models.face_restorer.create_face_restorer')
    def test_face_restoration_initialization_failure(self, mock_create_restorer, face_restoration_config):
        """Test graceful handling of face restoration initialization failure."""
        mock_create_restorer.side_effect = Exception("Model loading failed")
        
        processor = ImageProcessor(face_restoration_config)
        
        restorer = processor._get_face_restorer()
        assert restorer is None
        assert processor._face_restorer_initialized is True

    @patch('personfromvid.models.face_restorer.create_face_restorer')
    def test_face_restoration_applied(self, mock_create_restorer, face_restoration_config, small_test_image):
        """Test face restoration application."""
        mock_restorer = Mock()
        mock_restored_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        mock_restorer.restore_face.return_value = mock_restored_image
        mock_create_restorer.return_value = mock_restorer
        
        processor = ImageProcessor(face_restoration_config)
        bbox = (50, 50, 150, 150)  # 100x100 crop (< 640, needs upscaling)
        
        result = processor.crop_and_resize(small_test_image, bbox, padding=0.0, use_face_restoration=True)
        
        # Should use face restoration result
        assert np.array_equal(result, mock_restored_image)
        mock_restorer.restore_face.assert_called_once()

    @patch('personfromvid.models.face_restorer.create_face_restorer')
    def test_face_restoration_fallback(self, mock_create_restorer, face_restoration_config, small_test_image):
        """Test fallback to Lanczos when face restoration fails."""
        mock_restorer = Mock()
        mock_restorer.restore_face.side_effect = Exception("Restoration failed")
        mock_create_restorer.return_value = mock_restorer
        
        processor = ImageProcessor(face_restoration_config)
        bbox = (50, 50, 150, 150)  # 100x100 crop
        
        with patch('PIL.Image.fromarray') as mock_from_array:
            mock_pil_image = Mock()
            mock_from_array.return_value = mock_pil_image
            mock_upscaled = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            mock_pil_image.resize.return_value = mock_pil_image
            
            with patch('numpy.array', return_value=mock_upscaled):
                result = processor.crop_and_resize(small_test_image, bbox, padding=0.0, use_face_restoration=True)
                
                # Should fallback to Lanczos upscaling
                mock_pil_image.resize.assert_called_once()

    def test_face_restoration_target_size_calculation(self, face_restoration_config):
        """Test face restoration target size uses config values correctly."""
        # Test with default_crop_size smaller than min_dimension
        config = OutputImageConfig(
            default_crop_size=512,
            resize=1024,
            face_restoration_enabled=True
        )
        processor = ImageProcessor(config)
        
        with patch.object(processor, '_get_face_restorer') as mock_get_restorer:
            mock_restorer = Mock()
            mock_get_restorer.return_value = mock_restorer
            mock_restorer.restore_face.return_value = np.zeros((512, 512, 3), dtype=np.uint8)
            
            small_image = np.zeros((100, 100, 3), dtype=np.uint8)
            bbox = (10, 10, 90, 90)  # Small bbox needing upscaling
            
            processor.crop_and_resize(small_image, bbox, padding=0.0, use_face_restoration=True)
            
            # Target size should be min(resize=1024, default_crop_size=512) = 512
            mock_restorer.restore_face.assert_called_once()
            call_args = mock_restorer.restore_face.call_args
            assert call_args[1]['target_size'] == 512

    def test_face_restoration_strength_configuration(self, face_restoration_config):
        """Test face restoration uses configured strength."""
        with patch.object(face_restoration_config, 'face_restoration_strength', 0.7):
            processor = ImageProcessor(face_restoration_config)
            
            with patch.object(processor, '_get_face_restorer') as mock_get_restorer:
                mock_restorer = Mock()
                mock_get_restorer.return_value = mock_restorer
                mock_restorer.restore_face.return_value = np.zeros((640, 640, 3), dtype=np.uint8)
                
                small_image = np.zeros((100, 100, 3), dtype=np.uint8)
                bbox = (10, 10, 90, 90)
                
                processor.crop_and_resize(small_image, bbox, padding=0.0, use_face_restoration=True)
                
                # Should use configured strength
                call_args = mock_restorer.restore_face.call_args
                assert call_args[1]['strength'] == 0.7

    def test_edge_case_very_small_bbox(self, basic_config, test_image):
        """Test handling of very small bounding boxes."""
        processor = ImageProcessor(basic_config)
        bbox = (100, 100, 102, 102)  # 2x2 bbox
        
        result = processor.crop_and_resize(test_image, bbox, padding=0.0)
        
        # Should handle gracefully and upscale significantly
        assert result.shape[0] > 2
        assert result.shape[1] > 2

    def test_edge_case_bbox_at_image_edge(self, basic_config, test_image):
        """Test bounding box exactly at image edge."""
        processor = ImageProcessor(basic_config)
        bbox = (750, 750, 800, 800)  # 50x50 at bottom-right corner
        
        result = processor.crop_and_resize(test_image, bbox, padding=0.2)
        
        # Should handle edge position without error
        assert result.shape[0] > 0
        assert result.shape[1] > 0

    def test_large_bbox_no_upscaling(self, basic_config, test_image):
        """Test large bbox that doesn't need upscaling."""
        processor = ImageProcessor(basic_config)
        bbox = (50, 50, 750, 750)  # 700x700 crop (larger than default_crop_size)
        
        result = processor.crop_and_resize(test_image, bbox, padding=0.0)
        
        # Should not upscale, return original crop size
        assert result.shape == (700, 700, 3)

    def test_config_default_crop_size_used(self, basic_config, test_image):
        """Test that config.default_crop_size is used instead of hardcoded values."""
        # Test with custom default_crop_size
        config = OutputImageConfig(default_crop_size=768)
        processor = ImageProcessor(config)
        
        bbox = (100, 100, 200, 200)  # 100x100 crop (< 768)
        
        with patch('PIL.Image.fromarray') as mock_from_array:
            mock_pil_image = Mock()
            mock_from_array.return_value = mock_pil_image
            mock_pil_image.resize.return_value = mock_pil_image
            
            processor.crop_and_resize(test_image, bbox, padding=0.0)
            
            # Scale factor should be 768/100 = 7.68
            expected_size = (768, 768)
            mock_pil_image.resize.assert_called_with(expected_size, ANY)

    def test_expand_bbox_any_case(self, basic_config, test_image):
        """Test expand_bbox_to_aspect_ratio with crop_ratio='any'."""
        # Configure for "any" case
        config_dict = basic_config.model_dump()
        config_dict['crop_ratio'] = "any"
        config_dict['enable_pose_cropping'] = True
        config = OutputImageConfig(**config_dict)
        processor = ImageProcessor(config)
        
        bbox = (100, 100, 300, 300)  # 200x200 base bbox
        padding = 0.2  # 20% padding
        
        result_bbox = processor.expand_bbox_to_aspect_ratio(bbox, test_image, padding)
        
        # Should apply padding but skip aspect ratio calculation
        # With 20% padding: 200x200 base, padding_x = 40, padding_y = 40
        # Expected: (60, 60, 340, 340)
        expected_bbox = (60, 60, 340, 340)
        assert result_bbox == expected_bbox

    def test_expand_bbox_any_various_padding(self, basic_config, test_image):
        """Test expand_bbox_to_aspect_ratio with crop_ratio='any' and various padding values."""
        config_dict = basic_config.model_dump()
        config_dict['crop_ratio'] = "any"
        config_dict['enable_pose_cropping'] = True
        config = OutputImageConfig(**config_dict)
        processor = ImageProcessor(config)
        
        bbox = (200, 200, 400, 400)  # 200x200 base bbox at (300, 300) center
        
        # Test different padding values
        test_cases = [
            (0.0, (200, 200, 400, 400)),  # No padding
            (0.1, (180, 180, 420, 420)),  # 10% padding
            (0.5, (100, 100, 500, 500)),  # 50% padding
        ]
        
        for padding, expected_bbox in test_cases:
            result_bbox = processor.expand_bbox_to_aspect_ratio(bbox, test_image, padding)
            assert result_bbox == expected_bbox, f"Failed for padding {padding}"

    def test_expand_bbox_any_vs_fixed_ratio(self, basic_config, test_image):
        """Test that 'any' case skips aspect ratio calculation compared to fixed ratios."""
        bbox = (100, 150, 300, 400)  # 200x250 base bbox (aspect ratio 4:5)
        
        # Test with "any" - should preserve natural aspect ratio with padding
        config_any_dict = basic_config.model_dump()
        config_any_dict['crop_ratio'] = "any"
        config_any_dict['enable_pose_cropping'] = True
        config_any = OutputImageConfig(**config_any_dict)
        processor_any = ImageProcessor(config_any)
        result_any = processor_any.expand_bbox_to_aspect_ratio(bbox, test_image, padding=0.1)
        
        # Test with "1:1" - should enforce square aspect ratio
        config_square_dict = basic_config.model_dump()
        config_square_dict['crop_ratio'] = "1:1"
        config_square_dict['enable_pose_cropping'] = True
        config_square = OutputImageConfig(**config_square_dict)
        processor_square = ImageProcessor(config_square)
        result_square = processor_square.expand_bbox_to_aspect_ratio(bbox, test_image, padding=0.1)
        
        # "any" should preserve original aspect ratio (just with padding)
        any_width = result_any[2] - result_any[0]
        any_height = result_any[3] - result_any[1]
        
        # "1:1" should create square bbox
        square_width = result_square[2] - result_square[0] 
        square_height = result_square[3] - result_square[1]
        
        # "any" should maintain original proportions (200:250 = 4:5)
        assert abs(any_width / any_height - 4/5) < 0.1
        
        # "1:1" should be square
        assert abs(square_width / square_height - 1.0) < 0.1
        
        # Results should be different
        assert result_any != result_square

    def test_crop_and_resize_any_case_integration(self, basic_config, test_image):
        """Test full crop_and_resize pipeline with crop_ratio='any'."""
        config_dict = basic_config.model_dump()
        config_dict['crop_ratio'] = "any"
        config_dict['enable_pose_cropping'] = True
        config_dict['default_crop_size'] = 640
        config = OutputImageConfig(**config_dict)
        processor = ImageProcessor(config)
        
        bbox = (100, 100, 300, 400)  # 200x300 bbox (2:3 aspect ratio)
        padding = 0.2
        
        result = processor.crop_and_resize(test_image, bbox, padding=padding)
        
        # Should apply padding and preserve aspect ratio, then upscale if needed
        # With padding: 200x300 -> 240x360
        # Since max(240, 360) = 360 < 640, should upscale to fit 640px minimum
        # Scale factor: 640/360 ≈ 1.78
        # Expected size: 240*1.78 ≈ 427, 360*1.78 ≈ 640
        assert result.shape[2] == 3  # RGB channels
        
        # Height should be close to 640 (the constraining dimension)
        # Width should maintain aspect ratio
        height, width = result.shape[:2]
        aspect_ratio = width / height
        expected_aspect_ratio = 240 / 360  # 2:3
        assert abs(aspect_ratio - expected_aspect_ratio) < 0.1 
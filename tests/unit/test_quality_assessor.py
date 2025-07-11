"""Unit tests for QualityAssessor."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from personfromvid.analysis.quality_assessor import (
    QualityAssessor,
    create_quality_assessor,
)
from personfromvid.data.detection_results import QualityMetrics
from personfromvid.data.frame_data import FrameData, ImageProperties, SourceInfo


class TestQualityAssessor:
    """Test cases for QualityAssessor class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.assessor = QualityAssessor()

    def _create_test_frame_with_image(self, frame_id: str, image: np.ndarray) -> FrameData:
        """Helper to create a test FrameData object with an actual image file."""
        # Create a temporary file
        temp_fd, temp_path = tempfile.mkstemp(suffix='.jpg', prefix=f'{frame_id}_')
        os.close(temp_fd)  # Close the file descriptor, we'll use cv2 to write

        # Save the image to the temporary file
        cv2.imwrite(temp_path, image)

        # Store the temp path for cleanup
        if not hasattr(self, '_temp_files'):
            self._temp_files = []
        self._temp_files.append(temp_path)

        return FrameData(
            frame_id=frame_id,
            file_path=Path(temp_path),
            source_info=SourceInfo(
                video_timestamp=1.0,
                extraction_method='test',
                original_frame_number=30,
                video_fps=30.0
            ),
            image_properties=ImageProperties(
                width=image.shape[1],
                height=image.shape[0],
                channels=image.shape[2] if len(image.shape) == 3 else 1,
                file_size_bytes=os.path.getsize(temp_path),
                format='JPG'
            )
        )

    def teardown_method(self):
        """Clean up temporary files."""
        if hasattr(self, '_temp_files'):
            for temp_file in self._temp_files:
                try:
                    os.unlink(temp_file)
                except FileNotFoundError:
                    pass

    def test_initialization(self):
        """Test quality assessor initialization."""
        assert self.assessor is not None
        assert hasattr(self.assessor, 'logger')

    def test_create_quality_assessor_factory(self):
        """Test factory function."""
        assessor = create_quality_assessor()
        assert isinstance(assessor, QualityAssessor)

    def test_assess_quality_in_frame_sharp_image(self):
        """Test quality assessment on a sharp synthetic image."""
        # Create a synthetic sharp image with clear edges
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        image[:240, :320] = [255, 255, 255]  # White square
        image[240:, 320:] = [128, 128, 128]  # Gray square
        image[120:360, 160:480] = [64, 64, 64]  # Dark rectangle

        frame = self._create_test_frame_with_image('sharp_test', image)
        self.assessor.assess_quality_in_frame(frame)
        quality = frame.quality_metrics

        assert isinstance(quality, QualityMetrics)
        assert quality.laplacian_variance > 0
        assert quality.sobel_variance > 0
        assert 0 <= quality.brightness_score <= 255
        assert quality.contrast_score > 0
        assert 0 <= quality.overall_quality <= 1.0
        assert isinstance(quality.quality_issues, list)
        assert isinstance(quality.usable, bool)

    def test_assess_quality_in_frame_blurry_image(self):
        """Test quality assessment on a blurry image."""
        # Create a synthetic blurry image (uniform color)
        image = np.full((480, 640, 3), 128, dtype=np.uint8)

        frame = self._create_test_frame_with_image('blurry_test', image)
        self.assessor.assess_quality_in_frame(frame)
        quality = frame.quality_metrics

        assert isinstance(quality, QualityMetrics)
        # Blurry image should have low variance
        assert quality.laplacian_variance < 100  # Below our MIN_LAPLACIAN_VARIANCE
        assert quality.sobel_variance < 100
        assert "blurry" in quality.quality_issues
        # Usability is determined solely by overall quality score, not individual issues
        assert quality.usable == (quality.overall_quality >= self.assessor.min_quality_threshold)

    def test_assess_quality_in_frame_dark_image(self):
        """Test quality assessment on a dark image."""
        # Create a very dark image
        image = np.full((480, 640, 3), 20, dtype=np.uint8)

        frame = self._create_test_frame_with_image('dark_test', image)
        self.assessor.assess_quality_in_frame(frame)
        quality = frame.quality_metrics

        assert quality.brightness_score < 30  # Below MIN_BRIGHTNESS
        assert "dark" in quality.quality_issues

    def test_assess_quality_in_frame_bright_image(self):
        """Test quality assessment on a bright image."""
        # Create a very bright image
        image = np.full((480, 640, 3), 240, dtype=np.uint8)

        frame = self._create_test_frame_with_image('bright_test', image)
        self.assessor.assess_quality_in_frame(frame)
        quality = frame.quality_metrics

        assert quality.brightness_score > 225  # Above MAX_BRIGHTNESS
        assert "overexposed" in quality.quality_issues

    def test_assess_quality_in_frame_optimal_image(self):
        """Test quality assessment on an optimal image."""
        # Create an image with good sharpness, brightness, and contrast
        image = np.random.randint(80, 180, (480, 640, 3), dtype=np.uint8)
        # Add some sharp edges
        cv2.rectangle(image, (100, 100), (300, 300), (255, 255, 255), 2)
        cv2.rectangle(image, (350, 150), (550, 350), (50, 50, 50), 2)

        frame = self._create_test_frame_with_image('optimal_test', image)
        self.assessor.assess_quality_in_frame(frame)
        quality = frame.quality_metrics

        # Should have good metrics
        assert quality.laplacian_variance > 100  # Good sharpness
        assert 80 <= quality.brightness_score <= 180  # Good brightness
        assert quality.contrast_score > 20  # Good contrast
        assert quality.overall_quality > 0.5  # Decent overall quality
        assert quality.usable

    def test_calculate_laplacian_variance(self):
        """Test Laplacian variance calculation."""
        # Sharp image with edges
        image = np.zeros((100, 100), dtype=np.uint8)
        image[:50, :] = 255

        variance = self.assessor.calculate_laplacian_variance(image)
        assert variance > 0
        assert isinstance(variance, float)

    def test_calculate_sobel_variance(self):
        """Test Sobel variance calculation."""
        # Image with edges
        image = np.zeros((100, 100), dtype=np.uint8)
        image[:, :50] = 255

        variance = self.assessor.calculate_sobel_variance(image)
        assert variance > 0
        assert isinstance(variance, float)

    def test_assess_brightness(self):
        """Test brightness assessment."""
        # Test different brightness levels
        dark_image = np.full((100, 100), 30, dtype=np.uint8)
        bright_image = np.full((100, 100), 200, dtype=np.uint8)

        dark_brightness = self.assessor.assess_brightness(dark_image)
        bright_brightness = self.assessor.assess_brightness(bright_image)

        assert dark_brightness < bright_brightness
        assert 25 <= dark_brightness <= 35
        assert 195 <= bright_brightness <= 205

    def test_assess_contrast(self):
        """Test contrast assessment."""
        # Low contrast image
        low_contrast = np.full((100, 100), 128, dtype=np.uint8)

        # High contrast image
        high_contrast = np.zeros((100, 100), dtype=np.uint8)
        high_contrast[:50, :] = 255

        low_score = self.assessor.assess_contrast(low_contrast)
        high_score = self.assessor.assess_contrast(high_contrast)

        assert high_score > low_score
        assert low_score < 5  # Very low contrast
        assert high_score > 100  # High contrast

    def test_identify_quality_issues(self):
        """Test quality issue identification."""
        # Test various scenarios
        issues_blur = self.assessor.identify_quality_issues(50, 128, 30)  # Blurry
        assert "blurry" in issues_blur

        issues_dark = self.assessor.identify_quality_issues(150, 20, 30)  # Too dark
        assert "dark" in issues_dark

        issues_bright = self.assessor.identify_quality_issues(150, 240, 30)  # Too bright
        assert "overexposed" in issues_bright

        issues_low_contrast = self.assessor.identify_quality_issues(150, 128, 10)  # Low contrast
        assert "low_contrast" in issues_low_contrast

        # Good quality should have no issues
        issues_good = self.assessor.identify_quality_issues(200, 128, 50)
        assert len(issues_good) == 0

    def test_error_handling_no_image(self):
        """Test error handling for a frame with no image data."""
        # Create a FrameData with a non-existent file path
        frame = FrameData(
            frame_id='no_image_test',
            file_path=Path('/nonexistent/path.jpg'),
            source_info=SourceInfo(
                video_timestamp=1.0,
                extraction_method='test',
                original_frame_number=30,
                video_fps=30.0
            ),
            image_properties=ImageProperties(
                width=640,
                height=480,
                channels=3,
                file_size_bytes=0,
                format='JPG'
            )
        )

        # The image property will return None due to file not found
        with pytest.raises(ValueError, match="has no image data loaded"):
            self.assessor.assess_quality_in_frame(frame)

    def test_assess_quality_handles_exception(self):
        """Test that _assess_quality returns default on exception."""
        with patch.object(self.assessor, 'calculate_laplacian_variance', side_effect=Exception("Test Error")):
            # Create a simple test image
            image = np.zeros((10, 10, 3), dtype=np.uint8)
            frame = self._create_test_frame_with_image('exception_test', image)

            self.assessor.assess_quality_in_frame(frame)
            quality = frame.quality_metrics

            assert quality is not None
            assert quality.overall_quality == 0.0
            assert not quality.usable
            assert "assessment_failed" in quality.quality_issues

    def test_assess_quality_of_bbox_valid_region(self):
        """Test bbox quality assessment on a valid region."""
        # Create a test image with a sharp region
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        # Add a sharp region in the bbox area
        image[100:200, 100:200] = 255  # White square
        image[150:250, 150:250] = 128  # Gray square overlapping

        # Define a bbox covering the sharp region
        bbox = (100, 100, 250, 250)

        quality_score, quality_metrics = self.assessor.assess_quality_of_bbox(image, bbox)

        assert isinstance(quality_score, float)
        assert 0.0 <= quality_score <= 1.0
        assert isinstance(quality_metrics, QualityMetrics)
        assert quality_metrics.laplacian_variance > 0
        assert quality_metrics.overall_quality == quality_score

    def test_assess_quality_of_bbox_empty_region(self):
        """Test bbox quality assessment on an empty region."""
        image = np.zeros((480, 640, 3), dtype=np.uint8)

        # Define a bbox with invalid coordinates (empty region)
        bbox = (100, 100, 100, 100)  # Zero-area bbox

        quality_score, quality_metrics = self.assessor.assess_quality_of_bbox(image, bbox)

        # Should handle empty region gracefully - creates 1x1 black pixel with low quality
        assert quality_score == 0.0
        assert quality_metrics.overall_quality == 0.0
        assert not quality_metrics.usable
        # Black pixel has natural quality issues rather than assessment_failed
        assert any(issue in quality_metrics.quality_issues for issue in ["blurry", "dark", "low_contrast"])

    def test_assess_quality_of_bbox_out_of_bounds(self):
        """Test bbox quality assessment with out-of-bounds coordinates."""
        image = np.zeros((480, 640, 3), dtype=np.uint8)

        # Define a bbox that extends beyond image boundaries
        bbox = (500, 400, 800, 600)  # Partially outside image

        quality_score, quality_metrics = self.assessor.assess_quality_of_bbox(image, bbox)

        # Should handle out-of-bounds gracefully by clipping
        assert isinstance(quality_score, float)
        assert 0.0 <= quality_score <= 1.0
        assert isinstance(quality_metrics, QualityMetrics)

    def test_assess_quality_of_bbox_small_region(self):
        """Test bbox quality assessment on a very small region."""
        # Create a detailed test image
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Define a small bbox
        bbox = (100, 100, 110, 110)  # 10x10 pixel region

        quality_score, quality_metrics = self.assessor.assess_quality_of_bbox(image, bbox)

        assert isinstance(quality_score, float)
        assert 0.0 <= quality_score <= 1.0
        assert isinstance(quality_metrics, QualityMetrics)

    def test_assess_quality_of_bbox_comparison_with_full_frame(self):
        """Test that bbox assessment uses the same quality metrics as full frame."""
        # Create a uniform image
        image = np.full((200, 200, 3), 128, dtype=np.uint8)

        # Assess full image
        full_frame_metrics = self.assessor._assess_quality(image)

        # Assess same region as bbox (entire image)
        bbox = (0, 0, 200, 200)
        bbox_score, bbox_metrics = self.assessor.assess_quality_of_bbox(image, bbox)

        # Should produce identical results
        assert abs(bbox_score - full_frame_metrics.overall_quality) < 0.001
        assert abs(bbox_metrics.laplacian_variance - full_frame_metrics.laplacian_variance) < 0.001
        assert abs(bbox_metrics.brightness_score - full_frame_metrics.brightness_score) < 0.001

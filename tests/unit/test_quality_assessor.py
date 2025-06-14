"""Unit tests for QualityAssessor."""

import pytest
import numpy as np
import cv2
from unittest.mock import Mock, patch
from pathlib import Path

from personfromvid.analysis.quality_assessor import QualityAssessor, create_quality_assessor
from personfromvid.data.detection_results import QualityMetrics


class TestQualityAssessor:
    """Test cases for QualityAssessor class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.assessor = QualityAssessor()

    def test_initialization(self):
        """Test quality assessor initialization."""
        assert self.assessor is not None
        assert hasattr(self.assessor, 'logger')

    def test_create_quality_assessor_factory(self):
        """Test factory function."""
        assessor = create_quality_assessor()
        assert isinstance(assessor, QualityAssessor)

    def test_assess_quality_sharp_image(self):
        """Test quality assessment on a sharp synthetic image."""
        # Create a synthetic sharp image with clear edges
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        image[:240, :320] = [255, 255, 255]  # White square
        image[240:, 320:] = [128, 128, 128]  # Gray square
        image[120:360, 160:480] = [64, 64, 64]  # Dark rectangle
        
        quality = self.assessor.assess_quality(image)
        
        assert isinstance(quality, QualityMetrics)
        assert quality.laplacian_variance > 0
        assert quality.sobel_variance > 0
        assert 0 <= quality.brightness_score <= 255
        assert quality.contrast_score > 0
        assert 0 <= quality.overall_quality <= 1.0
        assert isinstance(quality.quality_issues, list)
        assert isinstance(quality.usable, bool)

    def test_assess_quality_blurry_image(self):
        """Test quality assessment on a blurry image."""
        # Create a synthetic blurry image (uniform color)
        image = np.full((480, 640, 3), 128, dtype=np.uint8)
        
        quality = self.assessor.assess_quality(image)
        
        assert isinstance(quality, QualityMetrics)
        # Blurry image should have low variance
        assert quality.laplacian_variance < 100  # Below our MIN_LAPLACIAN_VARIANCE
        assert quality.sobel_variance < 100
        assert "blurry" in quality.quality_issues
        assert not quality.usable  # Should not be usable due to blur

    def test_assess_quality_dark_image(self):
        """Test quality assessment on a dark image."""
        # Create a very dark image
        image = np.full((480, 640, 3), 20, dtype=np.uint8)
        
        quality = self.assessor.assess_quality(image)
        
        assert quality.brightness_score < 30  # Below MIN_BRIGHTNESS
        assert "too_dark" in quality.quality_issues

    def test_assess_quality_bright_image(self):
        """Test quality assessment on a bright image."""
        # Create a very bright image
        image = np.full((480, 640, 3), 240, dtype=np.uint8)
        
        quality = self.assessor.assess_quality(image)
        
        assert quality.brightness_score > 225  # Above MAX_BRIGHTNESS
        assert "too_bright" in quality.quality_issues

    def test_assess_quality_optimal_image(self):
        """Test quality assessment on an optimal image."""
        # Create an image with good sharpness, brightness, and contrast
        image = np.random.randint(80, 180, (480, 640, 3), dtype=np.uint8)
        # Add some sharp edges
        cv2.rectangle(image, (100, 100), (300, 300), (255, 255, 255), 2)
        cv2.rectangle(image, (350, 150), (550, 350), (50, 50, 50), 2)
        
        quality = self.assessor.assess_quality(image)
        
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
        assert "too_dark" in issues_dark
        
        issues_bright = self.assessor.identify_quality_issues(150, 240, 30)  # Too bright
        assert "too_bright" in issues_bright
        
        issues_low_contrast = self.assessor.identify_quality_issues(150, 128, 10)  # Low contrast
        assert "low_contrast" in issues_low_contrast
        
        # Good quality should have no issues
        issues_good = self.assessor.identify_quality_issues(200, 128, 50)
        assert len(issues_good) == 0

    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        # Test with invalid image
        invalid_image = None
        
        with patch.object(self.assessor.logger, 'error') as mock_logger:
            quality = self.assessor.assess_quality(invalid_image)
            
            # Should return default poor quality metrics
            assert quality.overall_quality == 0.0
            assert not quality.usable
            assert "assessment_failed" in quality.quality_issues
            mock_logger.assert_called_once()

    @patch('cv2.imread')
    def test_process_frame_batch(self, mock_imread):
        """Test batch processing of frames."""
        from personfromvid.data.frame_data import FrameData, SourceInfo, ImageProperties
        from personfromvid.data.detection_results import FaceDetection, PoseDetection
        from pathlib import Path
        
        # Mock successful image loading
        mock_image = np.random.randint(80, 180, (480, 640, 3), dtype=np.uint8)
        mock_imread.return_value = mock_image
        
        # Create FrameData objects
        frames_data = [
            FrameData(
                frame_id='frame_001',
                file_path=Path('/path/to/frame1.jpg'),
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
                    file_size_bytes=1024000,
                    format='JPEG'
                ),
                face_detections=[
                    FaceDetection(bbox=(100, 100, 200, 200), confidence=0.9)
                ],
                pose_detections=[
                    PoseDetection(
                        bbox=(50, 50, 300, 400),
                        confidence=0.8,
                        keypoints={}
                    )
                ]
            ),
            FrameData(
                frame_id='frame_002',
                file_path=Path('/path/to/frame2.jpg'),
                source_info=SourceInfo(
                    video_timestamp=2.0,
                    extraction_method='test',
                    original_frame_number=60,
                    video_fps=30.0
                ),
                image_properties=ImageProperties(
                    width=640,
                    height=480,
                    channels=3,
                    file_size_bytes=1024000,
                    format='JPEG'
                ),
                face_detections=[
                    FaceDetection(bbox=(150, 150, 250, 250), confidence=0.85)
                ],
                pose_detections=[
                    PoseDetection(
                        bbox=(75, 75, 325, 425),
                        confidence=0.75,
                        keypoints={}
                    )
                ]
            )
        ]
        
        # Mock Path.exists to return True
        with patch.object(Path, 'exists', return_value=True):
            progress_calls = []
            
            def progress_callback(count):
                progress_calls.append(count)
            
            quality_stats, total_assessed = self.assessor.process_frame_batch(
                frames_data, progress_callback
            )
            
            # Verify results
            assert total_assessed == 2
            assert isinstance(quality_stats, dict)
            assert 'usable' in quality_stats
            assert 'high_quality' in quality_stats
            assert 'total' in quality_stats
            
            # Verify frame data was updated in-place
            for frame in frames_data:
                assert frame.quality_metrics is not None
                metrics = frame.quality_metrics
                assert hasattr(metrics, 'laplacian_variance')
                assert hasattr(metrics, 'overall_quality')
                assert hasattr(metrics, 'usable')
            
            # Verify progress was called
            assert len(progress_calls) == 2
            assert progress_calls == [1, 2]

    def test_process_frame_batch_empty(self):
        """Test batch processing with empty input."""
        quality_stats, total_assessed = self.assessor.process_frame_batch([])
        
        assert quality_stats == {}
        assert total_assessed == 0

    @patch('cv2.imread')
    def test_process_frame_batch_missing_files(self, mock_imread):
        """Test batch processing with missing image files."""
        from personfromvid.data.frame_data import FrameData, SourceInfo, ImageProperties
        from personfromvid.data.detection_results import FaceDetection, PoseDetection
        from pathlib import Path
        
        mock_imread.return_value = None  # Simulate failed image loading
        
        frames_data = [
            FrameData(
                frame_id='frame_001',
                file_path=Path('/nonexistent/frame1.jpg'),
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
                    file_size_bytes=1024000,
                    format='JPEG'
                ),
                face_detections=[
                    FaceDetection(bbox=(100, 100, 200, 200), confidence=0.9)
                ],
                pose_detections=[
                    PoseDetection(
                        bbox=(50, 50, 300, 400),
                        confidence=0.8,
                        keypoints={}
                    )
                ]
            )
        ]
        
        with patch.object(Path, 'exists', return_value=False):
            quality_stats, total_assessed = self.assessor.process_frame_batch(frames_data)
            
            # Should handle missing files gracefully
            assert total_assessed == 0
            assert quality_stats['total'] == 0 
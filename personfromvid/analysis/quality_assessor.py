"""Image quality assessment for frame selection.

This module provides image quality evaluation using essential metrics
with configurable thresholds for easy adjustment.
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
import time

from ..data.detection_results import QualityMetrics
from ..utils.logging import get_logger

# =============================================================================
# ADJUSTABLE QUALITY CONSTANTS
# =============================================================================

# Blur Detection (Laplacian Variance) - Primary metric
MIN_LAPLACIAN_VARIANCE = 100.0  # Minimum sharpness threshold
HIGH_QUALITY_LAPLACIAN = 200.0  # High quality sharpness threshold

# Brightness Assessment
MIN_BRIGHTNESS = 30.0  # Too dark threshold
MAX_BRIGHTNESS = 225.0  # Too bright threshold
OPTIMAL_BRIGHTNESS_MIN = 80.0  # Optimal range start
OPTIMAL_BRIGHTNESS_MAX = 180.0  # Optimal range end

# Contrast Assessment (Simple std deviation)
MIN_CONTRAST_STD = 20.0  # Minimum contrast standard deviation

# Overall Quality Scoring Weights
LAPLACIAN_WEIGHT = 0.7  # Primary blur metric weight
BRIGHTNESS_WEIGHT = 0.2  # Brightness score weight
CONTRAST_WEIGHT = 0.1  # Contrast score weight

# Quality Classification Thresholds
HIGH_QUALITY_THRESHOLD = 0.7  # Overall score for high quality
# Usable quality threshold will be passed from config

# Sobel Variance (Secondary sharpness metric)
MIN_SOBEL_VARIANCE = 50.0  # Minimum edge sharpness
HIGH_QUALITY_SOBEL = 100.0  # High quality edge sharpness

# =============================================================================


logger = get_logger("quality_assessor")


class QualityAssessor:
    """Evaluates image quality using essential metrics for frame selection."""

    def __init__(self):
        """Initialize quality assessor."""
        self.logger = get_logger("quality_assessor")
        
        # Get minimum quality threshold from app config
        from ..data.config import get_default_config
        config = get_default_config()
        self.min_quality_threshold = config.frame_selection.min_quality_threshold

    def assess_quality_in_frame(self, frame: "FrameData"):
        """Assess quality for a single frame and update it in place.

        Args:
            frame: The FrameData object to assess. The `image` attribute
                   must be populated with the loaded image data.
        """
        if frame.image is None:
            raise ValueError(
                f"Frame {frame.frame_id} has no image data loaded. Image must be loaded before quality assessment."
            )

        quality_metrics = self._assess_quality(frame.image)
        frame.quality_metrics = quality_metrics

    def _assess_quality(self, image: np.ndarray) -> QualityMetrics:
        """Assess overall image quality using multiple metrics.

        Args:
            image: Input image as numpy array (BGR or RGB)

        Returns:
            QualityMetrics with comprehensive quality assessment
        """
        start_time = time.time()

        try:
            # Convert to grayscale for analysis
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()

            # Primary metric: Laplacian variance (blur detection)
            laplacian_variance = self.calculate_laplacian_variance(gray)

            # Secondary metric: Sobel variance (edge sharpness)
            sobel_variance = self.calculate_sobel_variance(gray)

            # Brightness assessment
            brightness_score = self.assess_brightness(gray)

            # Contrast assessment (simple std deviation)
            contrast_score = self.assess_contrast(gray)

            # Calculate overall quality score
            overall_quality = self.calculate_overall_score(
                laplacian_variance, sobel_variance, brightness_score, contrast_score
            )

            # Identify quality issues
            quality_issues = self.identify_quality_issues(
                laplacian_variance, brightness_score, contrast_score
            )

            # Determine if image is usable
            # DISABLED:Frame is unusable if it has critical quality issues, regardless of overall score
            #   critical_issues = {"blurry", "dark", "overexposed"}
            #   has_critical_issues = any(issue in quality_issues for issue in critical_issues)
            usable = overall_quality >= self.min_quality_threshold # and not has_critical_issues

            processing_time = (time.time() - start_time) * 1000  # Convert to ms

            self.logger.debug(
                f"Quality assessment: laplacian={laplacian_variance:.1f}, "
                f"brightness={brightness_score:.1f}, overall={overall_quality:.3f}, "
                f"usable={usable}, time={processing_time:.1f}ms"
            )

            return QualityMetrics(
                laplacian_variance=laplacian_variance,
                sobel_variance=sobel_variance,
                brightness_score=brightness_score,
                contrast_score=contrast_score,
                overall_quality=overall_quality,
                quality_issues=quality_issues,
                usable=usable,
            )

        except Exception as e:
            self.logger.error(f"Quality assessment failed: {e}")
            # Return default metrics indicating poor quality
            return QualityMetrics(
                laplacian_variance=0.0,
                sobel_variance=0.0,
                brightness_score=0.0,
                contrast_score=0.0,
                overall_quality=0.0,
                quality_issues=["assessment_failed"],
                usable=False,
            )

    def calculate_laplacian_variance(self, gray_image: np.ndarray) -> float:
        """Calculate Laplacian variance for blur detection.

        This is the primary sharpness metric. Higher values indicate sharper images.

        Args:
            gray_image: Grayscale image

        Returns:
            Laplacian variance value
        """
        try:
            laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
            variance = laplacian.var()
            return float(variance)
        except Exception as e:
            self.logger.warning(f"Laplacian calculation failed: {e}")
            return 0.0

    def calculate_sobel_variance(self, gray_image: np.ndarray) -> float:
        """Calculate Sobel variance for edge sharpness assessment.

        Secondary sharpness metric focusing on edge detection.

        Args:
            gray_image: Grayscale image

        Returns:
            Sobel variance value
        """
        try:
            sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
            sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
            variance = sobel_magnitude.var()
            return float(variance)
        except Exception as e:
            self.logger.warning(f"Sobel calculation failed: {e}")
            return 0.0

    def assess_brightness(self, gray_image: np.ndarray) -> float:
        """Assess image brightness and return normalized score.

        Returns a score from 0-255 representing average brightness.

        Args:
            gray_image: Grayscale image

        Returns:
            Average brightness value (0-255)
        """
        try:
            mean_brightness = np.mean(gray_image)
            return float(mean_brightness)
        except Exception as e:
            self.logger.warning(f"Brightness assessment failed: {e}")
            return 0.0

    def assess_contrast(self, gray_image: np.ndarray) -> float:
        """Assess image contrast using standard deviation.

        Higher standard deviation indicates better contrast.

        Args:
            gray_image: Grayscale image

        Returns:
            Standard deviation of pixel values
        """
        try:
            contrast = np.std(gray_image)
            return float(contrast)
        except Exception as e:
            self.logger.warning(f"Contrast assessment failed: {e}")
            return 0.0

    def calculate_overall_score(
        self,
        laplacian_variance: float,
        sobel_variance: float,
        brightness_score: float,
        contrast_score: float,
    ) -> float:
        """Calculate weighted overall quality score.

        Combines multiple metrics into a single quality score (0.0 to 1.0).

        Args:
            laplacian_variance: Blur detection score
            sobel_variance: Edge sharpness score
            brightness_score: Brightness assessment (0-255)
            contrast_score: Contrast assessment

        Returns:
            Overall quality score (0.0 to 1.0)
        """
        try:
            # Normalize laplacian score (primary metric)
            laplacian_normalized = min(laplacian_variance / HIGH_QUALITY_LAPLACIAN, 1.0)

            # Normalize brightness score (optimal range gets score of 1.0)
            if OPTIMAL_BRIGHTNESS_MIN <= brightness_score <= OPTIMAL_BRIGHTNESS_MAX:
                brightness_normalized = 1.0
            elif brightness_score < MIN_BRIGHTNESS or brightness_score > MAX_BRIGHTNESS:
                brightness_normalized = 0.0
            else:
                # Partial score for suboptimal but acceptable brightness
                if brightness_score < OPTIMAL_BRIGHTNESS_MIN:
                    brightness_normalized = (brightness_score - MIN_BRIGHTNESS) / (
                        OPTIMAL_BRIGHTNESS_MIN - MIN_BRIGHTNESS
                    )
                else:
                    brightness_normalized = (MAX_BRIGHTNESS - brightness_score) / (
                        MAX_BRIGHTNESS - OPTIMAL_BRIGHTNESS_MAX
                    )
                brightness_normalized = max(0.0, min(1.0, brightness_normalized))

            # Normalize contrast score
            contrast_normalized = min(contrast_score / (MIN_CONTRAST_STD * 3), 1.0)

            # Calculate weighted overall score
            overall_score = (
                LAPLACIAN_WEIGHT * laplacian_normalized
                + BRIGHTNESS_WEIGHT * brightness_normalized
                + CONTRAST_WEIGHT * contrast_normalized
            )

            return max(0.0, min(1.0, overall_score))

        except Exception as e:
            self.logger.warning(f"Overall score calculation failed: {e}")
            return 0.0

    def identify_quality_issues(
        self, laplacian_variance: float, brightness_score: float, contrast_score: float
    ) -> List[str]:
        """Identify specific quality issues based on metrics.

        Args:
            laplacian_variance: Blur detection score
            brightness_score: Brightness assessment
            contrast_score: Contrast assessment

        Returns:
            List of quality issue identifiers (e.g., "blurry", "dark")
        """
        issues = []
        if laplacian_variance < MIN_LAPLACIAN_VARIANCE:
            issues.append("blurry")

        if brightness_score < MIN_BRIGHTNESS:
            issues.append("dark")
        elif brightness_score > MAX_BRIGHTNESS:
            issues.append("overexposed")

        if contrast_score < MIN_CONTRAST_STD:
            issues.append("low_contrast")

        return issues


def create_quality_assessor() -> QualityAssessor:
    """Factory function to create a QualityAssessor instance.

    Returns:
        A new instance of QualityAssessor.
    """
    return QualityAssessor()

"""
Image Quality Analyzer

CPU-only image quality assessment using OpenCV.
Checks blur, noise, resolution, and overall quality.

Author: Animation AI Studio
Date: 2025-12-02
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from ..common import ImageQualityMetrics

logger = logging.getLogger(__name__)


class ImageQualityAnalyzer:
    """
    CPU-only image quality analyzer

    Assesses technical image quality using:
    - Blur detection (Laplacian variance)
    - Noise estimation (RGB std deviation)
    - Resolution validation (minimum dimensions)
    - Overall quality scoring

    Features:
    - 100% CPU-only (OpenCV + NumPy)
    - Fast batch processing
    - Configurable thresholds
    - Detailed metrics output

    Example:
        analyzer = ImageQualityAnalyzer(
            min_resolution=(512, 512),
            blur_threshold=100.0,
            noise_threshold=50.0
        )

        metrics = analyzer.analyze_image("path/to/image.jpg")
        if not metrics.is_valid:
            print(f"Quality issues: blur={metrics.is_blurry}, noise={metrics.is_noisy}")
    """

    def __init__(
        self,
        min_resolution: Tuple[int, int] = (512, 512),
        blur_threshold: float = 100.0,
        noise_threshold: float = 50.0,
        quality_weights: Optional[dict] = None
    ):
        """
        Initialize Image Quality Analyzer

        Args:
            min_resolution: Minimum (width, height) in pixels
            blur_threshold: Laplacian variance threshold (lower = blurrier)
            noise_threshold: Noise std threshold (higher = noisier)
            quality_weights: Custom weights for overall score
        """
        self.min_resolution = min_resolution
        self.blur_threshold = blur_threshold
        self.noise_threshold = noise_threshold

        # Default weights for overall quality score
        self.quality_weights = quality_weights or {
            "blur": 0.4,
            "noise": 0.3,
            "resolution": 0.3
        }

        logger.info(f"ImageQualityAnalyzer initialized: min_res={min_resolution}, "
                   f"blur_threshold={blur_threshold}, noise_threshold={noise_threshold}")

    def analyze_image(self, image_path: str) -> ImageQualityMetrics:
        """
        Analyze single image quality

        Args:
            image_path: Path to image file

        Returns:
            ImageQualityMetrics with all quality assessments
        """
        image_path = Path(image_path)

        # Try to load image
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                logger.error(f"Failed to load image: {image_path}")
                return self._create_invalid_metrics(str(image_path), "Failed to load")

            # Get dimensions
            height, width = image.shape[:2]
            aspect_ratio = width / height if height > 0 else 0.0

            # Calculate blur
            blur_variance = self._calculate_blur(image)
            blur_score = self._normalize_blur_score(blur_variance)
            is_blurry = blur_variance < self.blur_threshold

            # Calculate noise
            noise_std = self._calculate_noise(image)
            noise_score = self._normalize_noise_score(noise_std)
            is_noisy = noise_std > self.noise_threshold

            # Check resolution
            is_low_resolution = not self._check_resolution(width, height)
            resolution_score = 100.0 if not is_low_resolution else 50.0

            # Calculate overall score
            overall_score = self._calculate_overall_score(
                blur_score, noise_score, resolution_score
            )

            # Determine validity
            is_valid = not (is_blurry or is_noisy or is_low_resolution)

            return ImageQualityMetrics(
                file_path=str(image_path),
                width=width,
                height=height,
                blur_score=blur_score,
                noise_score=noise_score,
                overall_score=overall_score,
                is_blurry=is_blurry,
                is_noisy=is_noisy,
                is_low_resolution=is_low_resolution,
                is_valid=is_valid,
                blur_variance=blur_variance,
                noise_std=noise_std,
                aspect_ratio=aspect_ratio
            )

        except Exception as e:
            logger.error(f"Error analyzing {image_path}: {e}")
            return self._create_invalid_metrics(str(image_path), str(e))

    def _calculate_blur(self, image: np.ndarray) -> float:
        """
        Calculate blur using Laplacian variance

        Higher variance = sharper image
        Lower variance = blurrier image

        Args:
            image: BGR image array

        Returns:
            Laplacian variance (float)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Laplacian operator
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)

        # Calculate variance
        variance = laplacian.var()

        return float(variance)

    def _calculate_noise(self, image: np.ndarray) -> float:
        """
        Estimate noise using std deviation

        Higher std = more noise
        Lower std = cleaner image (or flat/uniform image)

        Args:
            image: BGR image array

        Returns:
            Average std deviation across channels
        """
        # Calculate std per channel
        b_std = np.std(image[:, :, 0])
        g_std = np.std(image[:, :, 1])
        r_std = np.std(image[:, :, 2])

        # Average across channels
        avg_std = (b_std + g_std + r_std) / 3.0

        return float(avg_std)

    def _check_resolution(self, width: int, height: int) -> bool:
        """
        Check if resolution meets minimum requirements

        Args:
            width: Image width in pixels
            height: Image height in pixels

        Returns:
            True if resolution is acceptable
        """
        min_width, min_height = self.min_resolution
        return width >= min_width and height >= min_height

    def _normalize_blur_score(self, blur_variance: float) -> float:
        """
        Normalize blur variance to 0-100 score

        Higher score = better (sharper)

        Args:
            blur_variance: Raw Laplacian variance

        Returns:
            Normalized score (0-100)
        """
        # Sigmoid-like normalization
        # threshold = 100 → score 50
        # variance > threshold → score > 50
        # variance < threshold → score < 50

        if blur_variance >= self.blur_threshold:
            # Good: map [threshold, threshold*10] → [50, 100]
            normalized = 50 + 50 * min(blur_variance / (self.blur_threshold * 10), 1.0)
        else:
            # Bad: map [0, threshold] → [0, 50]
            normalized = 50 * (blur_variance / self.blur_threshold)

        return min(100.0, max(0.0, normalized))

    def _normalize_noise_score(self, noise_std: float) -> float:
        """
        Normalize noise std to 0-100 score

        Higher score = better (less noise)

        Args:
            noise_std: Raw std deviation

        Returns:
            Normalized score (0-100)
        """
        # Inverse normalization
        # noise < threshold → score high
        # noise > threshold → score low

        if noise_std <= self.noise_threshold:
            # Good: map [0, threshold] → [100, 50]
            normalized = 100 - 50 * (noise_std / self.noise_threshold)
        else:
            # Bad: map [threshold, threshold*2] → [50, 0]
            excess = min(noise_std - self.noise_threshold, self.noise_threshold)
            normalized = 50 - 50 * (excess / self.noise_threshold)

        return min(100.0, max(0.0, normalized))

    def _calculate_overall_score(
        self,
        blur_score: float,
        noise_score: float,
        resolution_score: float
    ) -> float:
        """
        Calculate weighted overall quality score

        Args:
            blur_score: Blur quality (0-100)
            noise_score: Noise quality (0-100)
            resolution_score: Resolution quality (0-100)

        Returns:
            Weighted average score (0-100)
        """
        weighted_sum = (
            blur_score * self.quality_weights["blur"] +
            noise_score * self.quality_weights["noise"] +
            resolution_score * self.quality_weights["resolution"]
        )

        return min(100.0, max(0.0, weighted_sum))

    def _create_invalid_metrics(
        self,
        file_path: str,
        reason: str
    ) -> ImageQualityMetrics:
        """
        Create metrics for invalid/unreadable image

        Args:
            file_path: Path to image
            reason: Reason for invalidity

        Returns:
            ImageQualityMetrics with is_valid=False
        """
        logger.warning(f"Invalid image {file_path}: {reason}")

        return ImageQualityMetrics(
            file_path=file_path,
            width=0,
            height=0,
            blur_score=0.0,
            noise_score=0.0,
            overall_score=0.0,
            is_blurry=True,
            is_noisy=True,
            is_low_resolution=True,
            is_valid=False,
            blur_variance=0.0,
            noise_std=0.0,
            aspect_ratio=0.0
        )

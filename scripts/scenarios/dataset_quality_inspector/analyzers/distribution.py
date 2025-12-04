"""
Distribution Analyzer

Analyzes dataset balance and diversity.
Checks category distribution, image diversity, and imbalance issues.

Author: Animation AI Studio
Date: 2025-12-02
"""

import logging
from pathlib import Path
from typing import Dict, List, Any
from collections import Counter

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class DistributionAnalyzer:
    """
    Dataset distribution and diversity analyzer

    Analyzes:
    - Category distribution (if categorized)
    - Image count per category
    - Visual diversity (histogram comparison)
    - Balance across categories

    Features:
    - CPU-only analysis
    - Configurable thresholds
    - Diversity scoring via histogram similarity
    - Imbalance detection

    Example:
        analyzer = DistributionAnalyzer(
            min_category_size=50,
            imbalance_ratio=3.0
        )

        results = analyzer.analyze_distribution("/path/to/dataset")
        if results["imbalanced"]:
            print(f"Dataset is imbalanced: {results['imbalance_details']}")
    """

    def __init__(
        self,
        min_category_size: int = 50,
        imbalance_ratio: float = 3.0,
        diversity_sample_size: int = 100
    ):
        """
        Initialize Distribution Analyzer

        Args:
            min_category_size: Minimum images per category
            imbalance_ratio: Max ratio between largest/smallest category
            diversity_sample_size: Number of images to sample for diversity check
        """
        self.min_category_size = min_category_size
        self.imbalance_ratio = imbalance_ratio
        self.diversity_sample_size = diversity_sample_size

        logger.info(f"DistributionAnalyzer initialized: min_size={min_category_size}, "
                   f"imbalance_ratio={imbalance_ratio}")

    def analyze_distribution(self, dataset_path: str) -> Dict[str, Any]:
        """
        Analyze dataset distribution

        Args:
            dataset_path: Path to dataset directory

        Returns:
            Dictionary with distribution analysis results
        """
        dataset_path = Path(dataset_path)

        if not dataset_path.exists():
            logger.error(f"Dataset path does not exist: {dataset_path}")
            return {"error": "Dataset path not found"}

        # Scan dataset for images
        images = self._scan_images(dataset_path)

        if not images:
            logger.warning(f"No images found in {dataset_path}")
            return {
                "total_images": 0,
                "categories": {},
                "balanced": True,
                "diversity_score": 0.0,
                "issues": ["No images found"]
            }

        # Count by category
        category_counts = self._count_by_category(images, dataset_path)

        # Check balance
        balanced, imbalance_details = self._check_balance(category_counts)

        # Calculate diversity
        diversity_score = self._calculate_diversity(images)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            category_counts, balanced, diversity_score
        )

        return {
            "total_images": len(images),
            "categories": category_counts,
            "balanced": balanced,
            "imbalance_details": imbalance_details,
            "diversity_score": diversity_score,
            "recommendations": recommendations
        }

    def _scan_images(self, dataset_path: Path) -> List[Path]:
        """
        Scan directory for image files

        Args:
            dataset_path: Path to dataset

        Returns:
            List of image file paths
        """
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
        images = []

        for ext in image_extensions:
            images.extend(dataset_path.rglob(f"*{ext}"))

        logger.info(f"Found {len(images)} images in {dataset_path}")
        return images

    def _count_by_category(
        self,
        images: List[Path],
        dataset_path: Path
    ) -> Dict[str, int]:
        """
        Count images by category

        Categories are inferred from subdirectory structure.
        If no subdirectories, all images are in "default" category.

        Args:
            images: List of image paths
            dataset_path: Root dataset path

        Returns:
            Dictionary mapping category name to count
        """
        category_counts = Counter()

        for img_path in images:
            # Get relative path from dataset root
            rel_path = img_path.relative_to(dataset_path)

            # Extract category (first subdirectory)
            if len(rel_path.parts) > 1:
                category = rel_path.parts[0]
            else:
                category = "default"

            category_counts[category] += 1

        return dict(category_counts)

    def _check_balance(
        self,
        category_counts: Dict[str, int]
    ) -> tuple[bool, Dict[str, Any]]:
        """
        Check if dataset is balanced

        A dataset is considered imbalanced if:
        1. Any category is below min_category_size
        2. Ratio between largest/smallest exceeds imbalance_ratio

        Args:
            category_counts: Category name to count mapping

        Returns:
            (is_balanced, imbalance_details)
        """
        if not category_counts:
            return True, {}

        counts = list(category_counts.values())
        min_count = min(counts)
        max_count = max(counts)

        # Check minimum size
        small_categories = [
            cat for cat, count in category_counts.items()
            if count < self.min_category_size
        ]

        # Check ratio
        ratio = max_count / min_count if min_count > 0 else float('inf')
        ratio_exceeded = ratio > self.imbalance_ratio

        balanced = not small_categories and not ratio_exceeded

        imbalance_details = {
            "min_count": min_count,
            "max_count": max_count,
            "ratio": ratio,
            "small_categories": small_categories,
            "ratio_exceeded": ratio_exceeded
        }

        if not balanced:
            logger.warning(f"Dataset imbalance detected: {imbalance_details}")

        return balanced, imbalance_details

    def _calculate_diversity(self, images: List[Path]) -> float:
        """
        Calculate visual diversity using histogram comparison

        Higher diversity = more varied images
        Lower diversity = similar/repetitive images

        Args:
            images: List of image paths

        Returns:
            Diversity score (0-100)
        """
        if not images:
            return 0.0

        # Sample images for efficiency
        sample_size = min(self.diversity_sample_size, len(images))
        sample = np.random.choice(images, size=sample_size, replace=False)

        # Extract histograms
        histograms = []
        for img_path in sample:
            try:
                hist = self._compute_histogram(img_path)
                if hist is not None:
                    histograms.append(hist)
            except Exception as e:
                logger.debug(f"Failed to compute histogram for {img_path}: {e}")

        if len(histograms) < 2:
            return 0.0

        # Compute pairwise similarities
        similarities = []
        for i in range(len(histograms)):
            for j in range(i + 1, len(histograms)):
                sim = cv2.compareHist(
                    histograms[i],
                    histograms[j],
                    cv2.HISTCMP_CORREL
                )
                similarities.append(sim)

        # Average similarity
        avg_similarity = np.mean(similarities)

        # Convert to diversity score (inverse of similarity)
        # High similarity → Low diversity
        # Low similarity → High diversity
        diversity_score = (1.0 - avg_similarity) * 100

        logger.info(f"Diversity score: {diversity_score:.2f} "
                   f"(avg similarity: {avg_similarity:.3f})")

        return float(diversity_score)

    def _compute_histogram(self, img_path: Path) -> np.ndarray:
        """
        Compute color histogram for image

        Args:
            img_path: Path to image

        Returns:
            Normalized histogram or None on error
        """
        img = cv2.imread(str(img_path))
        if img is None:
            return None

        # Convert to HSV for better color representation
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Compute histogram (Hue channel)
        hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])

        # Normalize
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

        return hist

    def _generate_recommendations(
        self,
        category_counts: Dict[str, int],
        balanced: bool,
        diversity_score: float
    ) -> List[str]:
        """
        Generate recommendations based on analysis

        Args:
            category_counts: Category distribution
            balanced: Whether dataset is balanced
            diversity_score: Diversity score (0-100)

        Returns:
            List of recommendation strings
        """
        recommendations = []

        # Balance recommendations
        if not balanced:
            small_cats = [
                f"{cat} ({count})"
                for cat, count in category_counts.items()
                if count < self.min_category_size
            ]
            if small_cats:
                recommendations.append(
                    f"Increase images in small categories: {', '.join(small_cats)}"
                )

            counts = list(category_counts.values())
            if counts:
                ratio = max(counts) / min(counts)
                if ratio > self.imbalance_ratio:
                    recommendations.append(
                        f"Balance category sizes (current ratio: {ratio:.1f}:1, "
                        f"target: <{self.imbalance_ratio}:1)"
                    )

        # Diversity recommendations
        if diversity_score < 30:
            recommendations.append(
                "Very low diversity detected. Add more varied images to improve training."
            )
        elif diversity_score < 50:
            recommendations.append(
                "Moderate diversity. Consider adding more varied angles, poses, or backgrounds."
            )

        if not recommendations:
            recommendations.append("Dataset distribution looks good!")

        return recommendations

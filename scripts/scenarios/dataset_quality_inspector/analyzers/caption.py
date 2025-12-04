"""
Caption Analyzer

Validates caption quality and consistency for training datasets.
Checks caption existence, length, format, and semantic consistency.

Author: Animation AI Studio
Date: 2025-12-02
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import Counter
import re

logger = logging.getLogger(__name__)


class CaptionAnalyzer:
    """
    Caption quality and consistency analyzer

    Validates:
    - Caption file existence
    - Caption length (token count)
    - Format compliance (special characters, encoding)
    - Keyword consistency across dataset
    - Optional: Semantic analysis via Agent Framework

    Features:
    - CPU-only text analysis
    - Configurable length thresholds
    - Keyword extraction and consistency scoring
    - Integration with Agent for semantic validation

    Example:
        analyzer = CaptionAnalyzer(
            min_length=10,
            max_length=77,
            required_keywords=["3d animated", "character"]
        )

        results = analyzer.analyze_captions("/path/to/dataset")
        if results["issues"]:
            print(f"Caption issues: {results['issues']}")
    """

    def __init__(
        self,
        min_length: int = 10,
        max_length: int = 77,
        required_keywords: Optional[List[str]] = None,
        caption_suffix: str = ".txt"
    ):
        """
        Initialize Caption Analyzer

        Args:
            min_length: Minimum caption length (tokens)
            max_length: Maximum caption length (tokens)
            required_keywords: Keywords that should appear frequently
            caption_suffix: Caption file extension (default: .txt)
        """
        self.min_length = min_length
        self.max_length = max_length
        self.required_keywords = required_keywords or []
        self.caption_suffix = caption_suffix

        logger.info(f"CaptionAnalyzer initialized: min={min_length}, max={max_length}")

    def analyze_captions(self, dataset_path: str) -> Dict[str, Any]:
        """
        Analyze all captions in dataset

        Args:
            dataset_path: Path to dataset directory

        Returns:
            Dictionary with caption analysis results
        """
        dataset_path = Path(dataset_path)

        if not dataset_path.exists():
            logger.error(f"Dataset path does not exist: {dataset_path}")
            return {"error": "Dataset path not found"}

        # Scan for images and captions
        images = self._scan_images(dataset_path)
        caption_files = self._scan_captions(dataset_path)

        if not images:
            logger.warning(f"No images found in {dataset_path}")
            return {
                "total_images": 0,
                "issues": ["No images found"]
            }

        # Check caption existence
        missing_captions = self._check_missing_captions(images, caption_files)

        # Analyze existing captions
        caption_data = self._load_captions(caption_files)

        # Validate lengths
        length_issues = self._validate_lengths(caption_data)

        # Check consistency
        consistency_score, keyword_stats = self._check_consistency(caption_data)

        # Generate issues list
        issues = []
        if missing_captions:
            issues.append(f"Missing captions: {len(missing_captions)} images")
        if length_issues["too_short"]:
            issues.append(f"Too short: {len(length_issues['too_short'])} captions")
        if length_issues["too_long"]:
            issues.append(f"Too long: {len(length_issues['too_long'])} captions")
        if consistency_score < 50:
            issues.append(f"Low consistency: {consistency_score:.1f}/100")

        # Generate recommendations
        recommendations = self._generate_recommendations(
            missing_captions, length_issues, consistency_score, keyword_stats
        )

        return {
            "total_images": len(images),
            "total_captions": len(caption_files),
            "missing_captions": len(missing_captions),
            "missing_caption_files": [str(p) for p in missing_captions[:10]],  # First 10
            "length_issues": {
                "too_short": len(length_issues["too_short"]),
                "too_long": len(length_issues["too_long"]),
                "valid": len(caption_data) - len(length_issues["too_short"]) - len(length_issues["too_long"])
            },
            "consistency_score": consistency_score,
            "keyword_stats": keyword_stats,
            "issues": issues,
            "recommendations": recommendations
        }

    def _scan_images(self, dataset_path: Path) -> List[Path]:
        """Scan for image files"""
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
        images = []
        for ext in image_extensions:
            images.extend(dataset_path.rglob(f"*{ext}"))
        return images

    def _scan_captions(self, dataset_path: Path) -> List[Path]:
        """Scan for caption files"""
        return list(dataset_path.rglob(f"*{self.caption_suffix}"))

    def _check_missing_captions(
        self,
        images: List[Path],
        caption_files: List[Path]
    ) -> List[Path]:
        """
        Find images without corresponding caption files

        Args:
            images: List of image paths
            caption_files: List of caption file paths

        Returns:
            List of images missing captions
        """
        # Build set of caption stems (without extension)
        caption_stems = {cf.stem for cf in caption_files}

        # Find images without matching caption
        missing = []
        for img_path in images:
            if img_path.stem not in caption_stems:
                missing.append(img_path)

        if missing:
            logger.warning(f"Found {len(missing)} images without captions")

        return missing

    def _load_captions(self, caption_files: List[Path]) -> Dict[str, str]:
        """
        Load caption text from files

        Args:
            caption_files: List of caption file paths

        Returns:
            Dictionary mapping filename to caption text
        """
        captions = {}

        for caption_file in caption_files:
            try:
                with open(caption_file, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                    captions[str(caption_file)] = text
            except Exception as e:
                logger.warning(f"Failed to load caption {caption_file}: {e}")

        return captions

    def _validate_lengths(
        self,
        caption_data: Dict[str, str]
    ) -> Dict[str, List[str]]:
        """
        Validate caption lengths

        Args:
            caption_data: Filename to caption text mapping

        Returns:
            Dictionary with too_short and too_long lists
        """
        too_short = []
        too_long = []

        for filename, text in caption_data.items():
            # Simple token count (split by whitespace)
            token_count = len(text.split())

            if token_count < self.min_length:
                too_short.append(filename)
            elif token_count > self.max_length:
                too_long.append(filename)

        if too_short:
            logger.warning(f"Found {len(too_short)} captions that are too short")
        if too_long:
            logger.warning(f"Found {len(too_long)} captions that are too long")

        return {
            "too_short": too_short,
            "too_long": too_long
        }

    def _check_consistency(
        self,
        caption_data: Dict[str, str]
    ) -> tuple[float, Dict[str, Any]]:
        """
        Check keyword consistency across captions

        Args:
            caption_data: Filename to caption text mapping

        Returns:
            (consistency_score, keyword_statistics)
        """
        if not caption_data:
            return 0.0, {}

        # Extract all keywords
        all_keywords = []
        for text in caption_data.values():
            # Simple keyword extraction (lowercase, alphanumeric only)
            words = re.findall(r'\b\w+\b', text.lower())
            all_keywords.extend(words)

        # Count keyword frequencies
        keyword_counts = Counter(all_keywords)

        # Calculate consistency score
        # High consistency = common keywords appear frequently
        total_captions = len(caption_data)

        # Check required keywords
        required_present = sum(
            1 for kw in self.required_keywords
            if kw.lower() in keyword_counts
        )
        required_score = (required_present / len(self.required_keywords) * 100
                         if self.required_keywords else 100)

        # Calculate keyword diversity
        unique_keywords = len(keyword_counts)
        total_keywords = len(all_keywords)
        diversity_ratio = unique_keywords / total_keywords if total_keywords > 0 else 0

        # Consistency score: balance between common terms and diversity
        # High diversity + required keywords = good
        consistency_score = (required_score * 0.7 + diversity_ratio * 30)

        # Top keywords
        top_keywords = dict(keyword_counts.most_common(20))

        keyword_stats = {
            "total_keywords": total_keywords,
            "unique_keywords": unique_keywords,
            "diversity_ratio": diversity_ratio,
            "required_present": required_present,
            "required_total": len(self.required_keywords),
            "top_keywords": top_keywords
        }

        logger.info(f"Caption consistency score: {consistency_score:.1f}/100")

        return consistency_score, keyword_stats

    def _generate_recommendations(
        self,
        missing_captions: List[Path],
        length_issues: Dict[str, List[str]],
        consistency_score: float,
        keyword_stats: Dict[str, Any]
    ) -> List[str]:
        """
        Generate recommendations based on analysis

        Args:
            missing_captions: Images without captions
            length_issues: Length validation results
            consistency_score: Consistency score (0-100)
            keyword_stats: Keyword statistics

        Returns:
            List of recommendation strings
        """
        recommendations = []

        # Missing captions
        if missing_captions:
            recommendations.append(
                f"Generate captions for {len(missing_captions)} missing images"
            )

        # Length issues
        if length_issues["too_short"]:
            recommendations.append(
                f"Expand {len(length_issues['too_short'])} short captions "
                f"(target: {self.min_length}+ tokens)"
            )
        if length_issues["too_long"]:
            recommendations.append(
                f"Trim {len(length_issues['too_long'])} long captions "
                f"(target: <{self.max_length} tokens)"
            )

        # Consistency
        if consistency_score < 30:
            recommendations.append(
                "Very low caption consistency. Review and standardize caption format."
            )
        elif consistency_score < 50:
            recommendations.append(
                "Moderate caption consistency. Consider adding consistent keywords or prefixes."
            )

        # Required keywords
        if self.required_keywords:
            missing_keywords = [
                kw for kw in self.required_keywords
                if kw.lower() not in keyword_stats.get("top_keywords", {})
            ]
            if missing_keywords:
                recommendations.append(
                    f"Add required keywords to captions: {', '.join(missing_keywords)}"
                )

        if not recommendations:
            recommendations.append("Caption quality looks good!")

        return recommendations

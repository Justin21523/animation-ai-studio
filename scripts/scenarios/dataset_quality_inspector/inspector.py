"""
Dataset Inspector

Main orchestrator coordinating all quality analysis components.
Performs comprehensive dataset inspection and generates detailed reports.

Author: Animation AI Studio
Date: 2025-12-02
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import time

from .common import (
    Issue, IssueCategory, IssueSeverity,
    ImageQualityMetrics, DatasetSummary, InspectionReport
)
from .analyzers import ImageQualityAnalyzer, DistributionAnalyzer, CaptionAnalyzer
from .detectors import DuplicateDetector, CorruptionDetector, FormatValidator

logger = logging.getLogger(__name__)


class DatasetInspector:
    """
    Dataset Quality Inspector

    Orchestrates all quality analysis components to perform
    comprehensive dataset inspection.

    Components:
    - ImageQualityAnalyzer: Technical image quality
    - DistributionAnalyzer: Dataset balance and diversity
    - CaptionAnalyzer: Caption quality and consistency
    - DuplicateDetector: Duplicate detection
    - CorruptionDetector: File integrity
    - FormatValidator: Structure validation

    Features:
    - CPU-only operation
    - Comprehensive quality assessment
    - Detailed issue reporting
    - Actionable recommendations
    - Progress tracking

    Example:
        inspector = DatasetInspector(
            dataset_path="/path/to/dataset",
            config={
                "min_resolution": (512, 512),
                "blur_threshold": 100.0,
                "enable_format_validation": True
            }
        )

        report = inspector.inspect()

        print(f"Overall Score: {report.overall_score:.1f}/100")
        print(f"Issues Found: {report.total_issues}")

        for issue in report.issues:
            if issue.severity == IssueSeverity.CRITICAL:
                print(f"CRITICAL: {issue.description}")
    """

    def __init__(
        self,
        dataset_path: str,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Dataset Inspector

        Args:
            dataset_path: Path to dataset directory
            config: Configuration dictionary with component settings
        """
        self.dataset_path = Path(dataset_path)
        self.config = config or {}

        # Initialize components
        logger.info("Initializing DatasetInspector components...")

        self.image_analyzer = ImageQualityAnalyzer(
            min_resolution=self.config.get("min_resolution", (512, 512)),
            blur_threshold=self.config.get("blur_threshold", 100.0),
            noise_threshold=self.config.get("noise_threshold", 50.0)
        )

        self.dist_analyzer = DistributionAnalyzer(
            min_category_size=self.config.get("min_category_size", 50),
            imbalance_ratio=self.config.get("imbalance_ratio", 3.0)
        )

        self.caption_analyzer = CaptionAnalyzer(
            min_length=self.config.get("min_caption_length", 10),
            max_length=self.config.get("max_caption_length", 77),
            required_keywords=self.config.get("required_keywords", [])
        )

        self.dup_detector = DuplicateDetector(
            hash_size=self.config.get("hash_size", 8),
            similarity_threshold=self.config.get("similarity_threshold", 5)
        )

        self.corr_detector = CorruptionDetector(
            min_file_size=self.config.get("min_file_size", 1024)
        )

        self.format_validator = FormatValidator(
            required_dirs=self.config.get("required_dirs", []),
            metadata_required=self.config.get("metadata_required", True),
            captions_required=self.config.get("captions_required", True)
        )

        logger.info("DatasetInspector initialized successfully")

    def inspect(self, enable_recommendations: bool = True) -> InspectionReport:
        """
        Perform comprehensive dataset inspection

        Args:
            enable_recommendations: Generate AI-powered recommendations

        Returns:
            InspectionReport with all analysis results
        """
        logger.info(f"Starting inspection of {self.dataset_path}")
        start_time = time.time()

        # Step 1: Scan dataset
        logger.info("Step 1/7: Scanning dataset...")
        images = self._scan_dataset()

        if not images:
            logger.error("No images found in dataset")
            return self._create_empty_report("No images found")

        # Step 2: Analyze image quality
        logger.info(f"Step 2/7: Analyzing image quality ({len(images)} images)...")
        quality_metrics, quality_issues = self._analyze_images(images)

        # Step 3: Detect duplicates
        logger.info("Step 3/7: Detecting duplicates...")
        duplicate_issues = self._detect_duplicates()

        # Step 4: Check corruption
        logger.info("Step 4/7: Checking file integrity...")
        corruption_issues = self._detect_corruption()

        # Step 5: Validate format
        logger.info("Step 5/7: Validating dataset structure...")
        format_issues = self._validate_format()

        # Step 6: Analyze distribution
        logger.info("Step 6/7: Analyzing distribution...")
        distribution_issues = self._analyze_distribution()

        # Step 7: Analyze captions
        logger.info("Step 7/7: Analyzing captions...")
        caption_issues = self._analyze_captions()

        # Collect all issues
        all_issues = (
            quality_issues +
            duplicate_issues +
            corruption_issues +
            format_issues +
            distribution_issues +
            caption_issues
        )

        # Build dataset summary
        dataset_summary = self._build_dataset_summary(images, quality_metrics)

        # Calculate scores
        overall_score, category_scores = self._calculate_scores(all_issues, quality_metrics)

        # Count issues by severity
        severity_counts = self._count_by_severity(all_issues)

        # Generate recommendations (placeholder for Agent integration)
        recommendations = []
        if enable_recommendations:
            recommendations = self._generate_recommendations(all_issues, category_scores)

        # Build final report
        report = InspectionReport(
            dataset_summary=dataset_summary,
            issues=all_issues,
            quality_metrics=quality_metrics,
            overall_score=overall_score,
            category_scores=category_scores,
            total_issues=len(all_issues),
            critical_issues=severity_counts[IssueSeverity.CRITICAL],
            high_issues=severity_counts[IssueSeverity.HIGH],
            medium_issues=severity_counts[IssueSeverity.MEDIUM],
            low_issues=severity_counts[IssueSeverity.LOW],
            recommendations=recommendations
        )

        elapsed_time = time.time() - start_time
        logger.info(f"Inspection completed in {elapsed_time:.1f}s")
        logger.info(f"Overall Score: {overall_score:.1f}/100")
        logger.info(f"Total Issues: {len(all_issues)}")

        return report

    def _scan_dataset(self) -> List[Path]:
        """Scan dataset for image files"""
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
        images = []
        for ext in image_extensions:
            images.extend(self.dataset_path.rglob(f"*{ext}"))
        return images

    def _analyze_images(
        self,
        images: List[Path]
    ) -> tuple[List[ImageQualityMetrics], List[Issue]]:
        """
        Analyze image quality for all images

        Args:
            images: List of image paths

        Returns:
            (quality_metrics, issues)
        """
        quality_metrics = []
        issues = []

        for img_path in images:
            try:
                metrics = self.image_analyzer.analyze_image(str(img_path))
                quality_metrics.append(metrics)

                # Generate issues for problematic images
                if not metrics.is_valid:
                    severity = IssueSeverity.HIGH if metrics.is_blurry else IssueSeverity.MEDIUM

                    problems = []
                    if metrics.is_blurry:
                        problems.append(f"blurry (score: {metrics.blur_score:.1f})")
                    if metrics.is_noisy:
                        problems.append(f"noisy (score: {metrics.noise_score:.1f})")
                    if metrics.is_low_resolution:
                        problems.append(f"low resolution ({metrics.width}x{metrics.height})")

                    issues.append(Issue(
                        category=IssueCategory.IMAGE_QUALITY,
                        severity=severity,
                        description=f"Quality issues: {', '.join(problems)}",
                        affected_files=[str(img_path)],
                        details=metrics.to_dict()
                    ))

            except Exception as e:
                logger.error(f"Error analyzing {img_path}: {e}")

        return quality_metrics, issues

    def _detect_duplicates(self) -> List[Issue]:
        """Detect duplicate images"""
        results = self.dup_detector.find_duplicates(str(self.dataset_path))
        issues = []

        if "error" in results:
            return issues

        # Exact duplicates
        if results["exact_duplicates"] > 0:
            issues.append(Issue(
                category=IssueCategory.DUPLICATES,
                severity=IssueSeverity.HIGH,
                description=f"Found {results['exact_duplicates']} exact duplicate images",
                affected_files=[],
                details={"groups": results["exact_duplicate_groups"]},
                recommendation="Remove duplicate images to reduce redundancy"
            ))

        # Near duplicates
        if results["near_duplicates"] > 0:
            issues.append(Issue(
                category=IssueCategory.DUPLICATES,
                severity=IssueSeverity.MEDIUM,
                description=f"Found {results['near_duplicates']} near-duplicate images",
                affected_files=[],
                details={"groups": results["near_duplicate_groups"]},
                recommendation="Review near-duplicates and consider removing very similar images"
            ))

        return issues

    def _detect_corruption(self) -> List[Issue]:
        """Detect corrupted files"""
        results = self.corr_detector.scan_for_corruption(str(self.dataset_path))
        issues = []

        if "error" in results:
            return issues

        if results["corrupted_count"] > 0:
            issues.append(Issue(
                category=IssueCategory.CORRUPTION,
                severity=IssueSeverity.CRITICAL,
                description=f"Found {results['corrupted_count']} corrupted files",
                affected_files=results["corrupted_files"],
                details=results["corruption_details"],
                recommendation="Remove or repair corrupted files before training"
            ))

        return issues

    def _validate_format(self) -> List[Issue]:
        """Validate dataset format"""
        results = self.format_validator.validate_structure(str(self.dataset_path))

        if "error" in results:
            return [Issue(
                category=IssueCategory.FORMAT,
                severity=IssueSeverity.CRITICAL,
                description=results["error"],
                affected_files=[]
            )]

        return results.get("issues", [])

    def _analyze_distribution(self) -> List[Issue]:
        """Analyze dataset distribution"""
        results = self.dist_analyzer.analyze_distribution(str(self.dataset_path))
        issues = []

        if "error" in results:
            return issues

        if not results.get("balanced", True):
            issues.append(Issue(
                category=IssueCategory.DISTRIBUTION,
                severity=IssueSeverity.MEDIUM,
                description="Dataset is imbalanced across categories",
                affected_files=[],
                details=results["imbalance_details"],
                recommendation="\n".join(results.get("recommendations", []))
            ))

        if results.get("diversity_score", 100) < 30:
            issues.append(Issue(
                category=IssueCategory.DISTRIBUTION,
                severity=IssueSeverity.HIGH,
                description=f"Low diversity score: {results['diversity_score']:.1f}/100",
                affected_files=[],
                recommendation="Add more varied images to improve training diversity"
            ))

        return issues

    def _analyze_captions(self) -> List[Issue]:
        """Analyze caption quality"""
        results = self.caption_analyzer.analyze_captions(str(self.dataset_path))
        issues = []

        if "error" in results:
            return issues

        if results.get("missing_captions", 0) > 0:
            issues.append(Issue(
                category=IssueCategory.CAPTIONS,
                severity=IssueSeverity.CRITICAL,
                description=f"Missing captions for {results['missing_captions']} images",
                affected_files=results.get("missing_caption_files", []),
                recommendation="Generate captions for all images"
            ))

        length_issues = results.get("length_issues", {})
        if length_issues.get("too_short", 0) > 0:
            issues.append(Issue(
                category=IssueCategory.CAPTIONS,
                severity=IssueSeverity.MEDIUM,
                description=f"{length_issues['too_short']} captions are too short",
                affected_files=[],
                recommendation="Expand short captions to be more descriptive"
            ))

        if results.get("consistency_score", 100) < 50:
            issues.append(Issue(
                category=IssueCategory.CAPTIONS,
                severity=IssueSeverity.MEDIUM,
                description=f"Low caption consistency: {results['consistency_score']:.1f}/100",
                affected_files=[],
                recommendation="Standardize caption format and keywords"
            ))

        return issues

    def _build_dataset_summary(
        self,
        images: List[Path],
        quality_metrics: List[ImageQualityMetrics]
    ) -> DatasetSummary:
        """Build dataset summary from analysis results"""

        total_size = sum(img.stat().st_size for img in images) / (1024 * 1024)  # MB

        file_types = {}
        for img in images:
            ext = img.suffix.lower()
            file_types[ext] = file_types.get(ext, 0) + 1

        widths = [m.width for m in quality_metrics if m.width > 0]
        heights = [m.height for m in quality_metrics if m.height > 0]

        valid_count = sum(1 for m in quality_metrics if m.is_valid)
        low_quality_count = sum(1 for m in quality_metrics if not m.is_valid)

        return DatasetSummary(
            dataset_path=str(self.dataset_path),
            total_images=len(images),
            total_size_mb=total_size,
            file_types=file_types,
            min_width=min(widths) if widths else 0,
            max_width=max(widths) if widths else 0,
            min_height=min(heights) if heights else 0,
            max_height=max(heights) if heights else 0,
            avg_width=sum(widths) / len(widths) if widths else 0,
            avg_height=sum(heights) / len(heights) if heights else 0,
            valid_images=valid_count,
            corrupted_images=0,  # Updated by corruption detector
            low_quality_images=low_quality_count
        )

    def _calculate_scores(
        self,
        issues: List[Issue],
        quality_metrics: List[ImageQualityMetrics]
    ) -> tuple[float, Dict[str, float]]:
        """Calculate overall and category scores"""

        # Category scores (0-100)
        category_scores = {
            "image_quality": 100.0,
            "duplicates": 100.0,
            "corruption": 100.0,
            "format": 100.0,
            "captions": 100.0,
            "distribution": 100.0
        }

        # Deduct points based on issues
        for issue in issues:
            category = issue.category.value

            # Deduction based on severity
            deduction = {
                IssueSeverity.CRITICAL: 25,
                IssueSeverity.HIGH: 15,
                IssueSeverity.MEDIUM: 10,
                IssueSeverity.LOW: 5,
                IssueSeverity.INFO: 0
            }.get(issue.severity, 0)

            category_scores[category] = max(0, category_scores[category] - deduction)

        # Image quality score from metrics
        if quality_metrics:
            avg_quality = sum(m.overall_score for m in quality_metrics) / len(quality_metrics)
            category_scores["image_quality"] = avg_quality

        # Overall score (weighted average)
        weights = {
            "image_quality": 0.25,
            "corruption": 0.20,
            "captions": 0.20,
            "duplicates": 0.15,
            "distribution": 0.10,
            "format": 0.10
        }

        overall_score = sum(
            category_scores[cat] * weight
            for cat, weight in weights.items()
        )

        return overall_score, category_scores

    def _count_by_severity(self, issues: List[Issue]) -> Dict[IssueSeverity, int]:
        """Count issues by severity"""
        counts = {severity: 0 for severity in IssueSeverity}
        for issue in issues:
            counts[issue.severity] += 1
        return counts

    def _generate_recommendations(
        self,
        issues: List[Issue],
        category_scores: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """
        Generate recommendations based on analysis

        Placeholder for future Agent Framework integration
        """
        recommendations = []

        # Extract issue recommendations
        for issue in issues:
            if issue.recommendation and issue.severity in [IssueSeverity.CRITICAL, IssueSeverity.HIGH]:
                recommendations.append({
                    "priority": issue.severity.value,
                    "category": issue.category.value,
                    "action": issue.recommendation,
                    "affected_count": len(issue.affected_files)
                })

        # Add category-based recommendations
        for category, score in category_scores.items():
            if score < 50:
                recommendations.append({
                    "priority": "high",
                    "category": category,
                    "action": f"Improve {category} (current score: {score:.1f}/100)",
                    "affected_count": 0
                })

        return recommendations

    def _create_empty_report(self, reason: str) -> InspectionReport:
        """Create empty report for error cases"""
        empty_summary = DatasetSummary(
            dataset_path=str(self.dataset_path),
            total_images=0,
            total_size_mb=0.0,
            file_types={},
            min_width=0,
            max_width=0,
            min_height=0,
            max_height=0,
            avg_width=0.0,
            avg_height=0.0,
            valid_images=0,
            corrupted_images=0,
            low_quality_images=0
        )

        return InspectionReport(
            dataset_summary=empty_summary,
            issues=[Issue(
                category=IssueCategory.FORMAT,
                severity=IssueSeverity.CRITICAL,
                description=reason,
                affected_files=[]
            )],
            quality_metrics=[],
            overall_score=0.0,
            category_scores={},
            total_issues=1,
            critical_issues=1,
            high_issues=0,
            medium_issues=0,
            low_issues=0
        )

"""
Common data structures for Dataset Quality Inspector

Author: Animation AI Studio
Date: 2025-12-02
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
from pathlib import Path


class IssueSeverity(Enum):
    """Issue severity levels"""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class IssueCategory(Enum):
    """Issue categories"""
    IMAGE_QUALITY = "image_quality"
    DUPLICATES = "duplicates"
    CORRUPTION = "corruption"
    FORMAT = "format"
    CAPTIONS = "captions"
    DISTRIBUTION = "distribution"


@dataclass
class Issue:
    """Single quality issue"""
    category: IssueCategory
    severity: IssueSeverity
    description: str
    affected_files: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)
    recommendation: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category.value,
            "severity": self.severity.value,
            "description": self.description,
            "affected_files": self.affected_files,
            "details": self.details,
            "recommendation": self.recommendation
        }


@dataclass
class ImageQualityMetrics:
    """Image quality assessment metrics"""
    file_path: str
    width: int
    height: int

    # Quality scores (0-100)
    blur_score: float
    noise_score: float
    overall_score: float

    # Checks
    is_blurry: bool
    is_noisy: bool
    is_low_resolution: bool
    is_valid: bool

    # Details
    blur_variance: float
    noise_std: float
    aspect_ratio: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_path": self.file_path,
            "dimensions": {"width": self.width, "height": self.height},
            "scores": {
                "blur": self.blur_score,
                "noise": self.noise_score,
                "overall": self.overall_score
            },
            "checks": {
                "is_blurry": self.is_blurry,
                "is_noisy": self.is_noisy,
                "is_low_resolution": self.is_low_resolution,
                "is_valid": self.is_valid
            },
            "metrics": {
                "blur_variance": self.blur_variance,
                "noise_std": self.noise_std,
                "aspect_ratio": self.aspect_ratio
            }
        }


@dataclass
class DatasetSummary:
    """Dataset overview summary"""
    dataset_path: str
    total_images: int
    total_size_mb: float

    # File types
    file_types: Dict[str, int]  # {".jpg": 450, ".png": 50}

    # Dimensions
    min_width: int
    max_width: int
    min_height: int
    max_height: int
    avg_width: float
    avg_height: float

    # Quality summary
    valid_images: int
    corrupted_images: int
    low_quality_images: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_path": self.dataset_path,
            "total_images": self.total_images,
            "total_size_mb": self.total_size_mb,
            "file_types": self.file_types,
            "dimensions": {
                "min": {"width": self.min_width, "height": self.min_height},
                "max": {"width": self.max_width, "height": self.max_height},
                "avg": {"width": self.avg_width, "height": self.avg_height}
            },
            "quality_summary": {
                "valid": self.valid_images,
                "corrupted": self.corrupted_images,
                "low_quality": self.low_quality_images
            }
        }


@dataclass
class InspectionReport:
    """Complete inspection report"""
    dataset_summary: DatasetSummary
    issues: List[Issue]
    quality_metrics: List[ImageQualityMetrics]

    # Scores
    overall_score: float
    category_scores: Dict[str, float]

    # Counts
    total_issues: int
    critical_issues: int
    high_issues: int
    medium_issues: int
    low_issues: int

    # Recommendations
    recommendations: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_summary": self.dataset_summary.to_dict(),
            "overall_score": self.overall_score,
            "category_scores": self.category_scores,
            "issue_counts": {
                "total": self.total_issues,
                "critical": self.critical_issues,
                "high": self.high_issues,
                "medium": self.medium_issues,
                "low": self.low_issues
            },
            "issues": [issue.to_dict() for issue in self.issues],
            "quality_metrics": [m.to_dict() for m in self.quality_metrics],
            "recommendations": self.recommendations
        }

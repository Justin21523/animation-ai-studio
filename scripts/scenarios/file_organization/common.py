"""
File Organization - Common Data Structures

Shared enums, dataclasses, and types for file organization scenario.

Author: Animation AI Studio
Date: 2025-12-03
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from datetime import datetime


class FileType(Enum):
    """File type classification"""
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    DOCUMENT = "document"
    CODE = "code"
    ARCHIVE = "archive"
    MODEL = "model"  # AI model files (.safetensors, .pt, .ckpt)
    DATASET = "dataset"  # Training datasets
    CONFIG = "config"  # Configuration files
    LOG = "log"
    OTHER = "other"


class OrganizationIssue(Enum):
    """Types of organization issues detected"""
    DUPLICATE = "duplicate"
    MISPLACED = "misplaced"
    NAMING_INCONSISTENT = "naming_inconsistent"
    ORPHANED = "orphaned"
    NESTED_EXCESSIVE = "nested_excessive"
    SIZE_ANOMALY = "size_anomaly"
    PERMISSION_ISSUE = "permission_issue"
    SYMLINK_BROKEN = "symlink_broken"


class IssueSeverity(Enum):
    """Severity levels for organization issues"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class OrganizationStrategy(Enum):
    """File organization strategies"""
    BY_TYPE = "by_type"  # Group by file type (images/, videos/, etc.)
    BY_DATE = "by_date"  # Group by creation/modification date
    BY_PROJECT = "by_project"  # Group by detected project structure
    BY_SIZE = "by_size"  # Group by file size ranges
    CUSTOM = "custom"  # User-defined rules
    SMART = "smart"  # AI-powered organization


@dataclass
class FileMetadata:
    """
    Metadata for a single file

    Attributes:
        path: Absolute path to file
        file_type: Classified file type
        size_bytes: File size in bytes
        created_time: Creation timestamp
        modified_time: Last modification timestamp
        accessed_time: Last access timestamp
        hash_sha256: SHA256 content hash (optional)
        hash_perceptual: Perceptual hash for images/videos (optional)
        mime_type: MIME type string
        extension: File extension (with dot)
        is_symlink: Whether file is a symbolic link
        is_hidden: Whether file is hidden (starts with .)
        permissions: Unix permission string (e.g., "0644")
        owner_uid: Owner user ID
        owner_gid: Owner group ID
    """
    path: Path
    file_type: FileType
    size_bytes: int
    created_time: datetime
    modified_time: datetime
    accessed_time: datetime
    hash_sha256: Optional[str] = None
    hash_perceptual: Optional[str] = None
    mime_type: Optional[str] = None
    extension: str = ""
    is_symlink: bool = False
    is_hidden: bool = False
    permissions: Optional[str] = None
    owner_uid: Optional[int] = None
    owner_gid: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "path": str(self.path),
            "file_type": self.file_type.value,
            "size_bytes": self.size_bytes,
            "created_time": self.created_time.isoformat(),
            "modified_time": self.modified_time.isoformat(),
            "accessed_time": self.accessed_time.isoformat(),
            "hash_sha256": self.hash_sha256,
            "hash_perceptual": self.hash_perceptual,
            "mime_type": self.mime_type,
            "extension": self.extension,
            "is_symlink": self.is_symlink,
            "is_hidden": self.is_hidden,
            "permissions": self.permissions,
            "owner_uid": self.owner_uid,
            "owner_gid": self.owner_gid
        }


@dataclass
class DuplicateGroup:
    """
    Group of duplicate files

    Attributes:
        files: List of file paths that are duplicates
        hash: Common content hash
        total_size_bytes: Combined size of all duplicates
        savings_bytes: Potential space savings if duplicates removed
        strategy: Recommended deduplication strategy
    """
    files: List[Path]
    hash: str
    total_size_bytes: int
    savings_bytes: int
    strategy: str = "keep_oldest"  # keep_oldest, keep_newest, keep_largest

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "files": [str(f) for f in self.files],
            "hash": self.hash,
            "total_size_bytes": self.total_size_bytes,
            "savings_bytes": self.savings_bytes,
            "strategy": self.strategy
        }


@dataclass
class Issue:
    """
    Single organization issue detected

    Attributes:
        category: Type of issue
        severity: Severity level
        description: Human-readable description
        affected_files: Files affected by this issue
        recommendation: Suggested fix
        auto_fixable: Whether this can be fixed automatically
    """
    category: OrganizationIssue
    severity: IssueSeverity
    description: str
    affected_files: List[Path] = field(default_factory=list)
    recommendation: Optional[str] = None
    auto_fixable: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "category": self.category.value,
            "severity": self.severity.value,
            "description": self.description,
            "affected_files": [str(f) for f in self.affected_files],
            "recommendation": self.recommendation,
            "auto_fixable": self.auto_fixable
        }


@dataclass
class StructureAnalysis:
    """
    Analysis of directory structure

    Attributes:
        total_directories: Total number of directories
        max_depth: Maximum nesting depth
        avg_depth: Average nesting depth
        directories_with_issues: Directories flagged for issues
        suggested_structure: AI-recommended structure
    """
    total_directories: int
    max_depth: int
    avg_depth: float
    directories_with_issues: List[Path] = field(default_factory=list)
    suggested_structure: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "total_directories": self.total_directories,
            "max_depth": self.max_depth,
            "avg_depth": self.avg_depth,
            "directories_with_issues": [str(d) for d in self.directories_with_issues],
            "suggested_structure": self.suggested_structure
        }


@dataclass
class OrganizationReport:
    """
    Comprehensive file organization report

    Attributes:
        root_path: Root directory analyzed
        timestamp: Analysis timestamp
        total_files: Total number of files scanned
        total_size_bytes: Total size of all files
        file_type_counts: Count of files per type
        duplicate_groups: Groups of duplicate files
        issues: List of detected issues
        structure_analysis: Directory structure analysis
        organization_score: Overall organization quality (0-100)
        recommendations: AI-powered recommendations
        potential_savings_bytes: Potential space savings from cleanup
    """
    root_path: Path
    timestamp: datetime
    total_files: int
    total_size_bytes: int
    file_type_counts: Dict[FileType, int]
    duplicate_groups: List[DuplicateGroup] = field(default_factory=list)
    issues: List[Issue] = field(default_factory=list)
    structure_analysis: Optional[StructureAnalysis] = None
    organization_score: float = 0.0
    recommendations: List[Dict[str, Any]] = field(default_factory=list)
    potential_savings_bytes: int = 0

    @property
    def total_duplicates(self) -> int:
        """Count total duplicate files"""
        return sum(len(group.files) - 1 for group in self.duplicate_groups)

    @property
    def critical_issues(self) -> int:
        """Count critical issues"""
        return sum(1 for issue in self.issues if issue.severity == IssueSeverity.CRITICAL)

    @property
    def high_issues(self) -> int:
        """Count high severity issues"""
        return sum(1 for issue in self.issues if issue.severity == IssueSeverity.HIGH)

    @property
    def medium_issues(self) -> int:
        """Count medium severity issues"""
        return sum(1 for issue in self.issues if issue.severity == IssueSeverity.MEDIUM)

    @property
    def low_issues(self) -> int:
        """Count low severity issues"""
        return sum(1 for issue in self.issues if issue.severity == IssueSeverity.LOW)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "root_path": str(self.root_path),
            "timestamp": self.timestamp.isoformat(),
            "total_files": self.total_files,
            "total_size_bytes": self.total_size_bytes,
            "file_type_counts": {ft.value: count for ft, count in self.file_type_counts.items()},
            "duplicate_groups": [dg.to_dict() for dg in self.duplicate_groups],
            "issues": [issue.to_dict() for issue in self.issues],
            "structure_analysis": self.structure_analysis.to_dict() if self.structure_analysis else None,
            "organization_score": self.organization_score,
            "recommendations": self.recommendations,
            "potential_savings_bytes": self.potential_savings_bytes,
            "summary": {
                "total_duplicates": self.total_duplicates,
                "critical_issues": self.critical_issues,
                "high_issues": self.high_issues,
                "medium_issues": self.medium_issues,
                "low_issues": self.low_issues
            }
        }

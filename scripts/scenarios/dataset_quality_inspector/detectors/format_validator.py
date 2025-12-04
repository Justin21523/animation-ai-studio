"""
Format Validator

Validates dataset structure, naming conventions, and metadata integrity.
Ensures dataset follows expected organization for training.

Author: Animation AI Studio
Date: 2025-12-02
"""

import logging
from pathlib import Path
from typing import List, Dict, Any
import json
import re

from ..common import Issue, IssueCategory, IssueSeverity

logger = logging.getLogger(__name__)


class FormatValidator:
    """
    Dataset format and structure validator

    Validates:
    - Directory structure
    - File naming conventions
    - Metadata file existence and validity
    - Caption file correspondence
    - Required file organization

    Features:
    - Configurable structure requirements
    - Flexible naming pattern matching
    - Metadata schema validation
    - Detailed issue reporting

    Example:
        validator = FormatValidator(
            required_dirs=["images", "captions"],
            naming_pattern=r"^\w+_\d{4}\.(jpg|png)$"
        )

        results = validator.validate_structure("/path/to/dataset")
        for issue in results["issues"]:
            print(f"{issue.severity.value}: {issue.description}")
    """

    def __init__(
        self,
        required_dirs: List[str] = None,
        optional_dirs: List[str] = None,
        naming_pattern: str = None,
        metadata_required: bool = True,
        captions_required: bool = True
    ):
        """
        Initialize Format Validator

        Args:
            required_dirs: List of required subdirectories
            optional_dirs: List of optional subdirectories
            naming_pattern: Regex pattern for image file names
            metadata_required: Whether metadata.json is required
            captions_required: Whether caption files are required
        """
        self.required_dirs = required_dirs or []
        self.optional_dirs = optional_dirs or []
        self.naming_pattern = naming_pattern
        self.metadata_required = metadata_required
        self.captions_required = captions_required

        logger.info(f"FormatValidator initialized: "
                   f"required_dirs={len(self.required_dirs)}, "
                   f"metadata_required={metadata_required}")

    def validate_structure(self, dataset_path: str) -> Dict[str, Any]:
        """
        Validate dataset structure

        Args:
            dataset_path: Path to dataset directory

        Returns:
            Dictionary with validation results and issues
        """
        dataset_path = Path(dataset_path)

        if not dataset_path.exists():
            logger.error(f"Dataset path does not exist: {dataset_path}")
            return {
                "error": "Dataset path not found",
                "issues": [
                    Issue(
                        category=IssueCategory.FORMAT,
                        severity=IssueSeverity.CRITICAL,
                        description="Dataset path does not exist",
                        affected_files=[str(dataset_path)]
                    )
                ]
            }

        issues = []

        # Check directory structure
        dir_issues = self._check_directories(dataset_path)
        issues.extend(dir_issues)

        # Check naming conventions
        naming_issues = self._validate_naming(dataset_path)
        issues.extend(naming_issues)

        # Check metadata
        if self.metadata_required:
            metadata_issues = self._check_metadata(dataset_path)
            issues.extend(metadata_issues)

        # Check caption correspondence
        if self.captions_required:
            caption_issues = self._check_captions(dataset_path)
            issues.extend(caption_issues)

        # Calculate severity counts
        severity_counts = {
            "critical": sum(1 for i in issues if i.severity == IssueSeverity.CRITICAL),
            "high": sum(1 for i in issues if i.severity == IssueSeverity.HIGH),
            "medium": sum(1 for i in issues if i.severity == IssueSeverity.MEDIUM),
            "low": sum(1 for i in issues if i.severity == IssueSeverity.LOW),
            "info": sum(1 for i in issues if i.severity == IssueSeverity.INFO)
        }

        # Determine if structure is valid
        is_valid = severity_counts["critical"] == 0 and severity_counts["high"] == 0

        return {
            "is_valid": is_valid,
            "total_issues": len(issues),
            "severity_counts": severity_counts,
            "issues": issues
        }

    def _check_directories(self, dataset_path: Path) -> List[Issue]:
        """
        Check required and optional directories exist

        Args:
            dataset_path: Root dataset path

        Returns:
            List of directory-related issues
        """
        issues = []

        # Check required directories
        for dir_name in self.required_dirs:
            dir_path = dataset_path / dir_name
            if not dir_path.exists():
                issues.append(Issue(
                    category=IssueCategory.FORMAT,
                    severity=IssueSeverity.CRITICAL,
                    description=f"Required directory missing: {dir_name}",
                    affected_files=[str(dir_path)]
                ))
            elif not dir_path.is_dir():
                issues.append(Issue(
                    category=IssueCategory.FORMAT,
                    severity=IssueSeverity.HIGH,
                    description=f"Required path is not a directory: {dir_name}",
                    affected_files=[str(dir_path)]
                ))

        # Check optional directories (info only)
        for dir_name in self.optional_dirs:
            dir_path = dataset_path / dir_name
            if not dir_path.exists():
                issues.append(Issue(
                    category=IssueCategory.FORMAT,
                    severity=IssueSeverity.INFO,
                    description=f"Optional directory not found: {dir_name}",
                    affected_files=[str(dir_path)],
                    recommendation=f"Consider adding {dir_name} directory for better organization"
                ))

        return issues

    def _validate_naming(self, dataset_path: Path) -> List[Issue]:
        """
        Validate file naming conventions

        Args:
            dataset_path: Root dataset path

        Returns:
            List of naming-related issues
        """
        if not self.naming_pattern:
            return []

        issues = []
        pattern = re.compile(self.naming_pattern)

        # Find all image files
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
        images = []
        for ext in image_extensions:
            images.extend(dataset_path.rglob(f"*{ext}"))

        # Check each filename
        invalid_names = []
        for img_path in images:
            filename = img_path.name
            if not pattern.match(filename):
                invalid_names.append(str(img_path))

        if invalid_names:
            issues.append(Issue(
                category=IssueCategory.FORMAT,
                severity=IssueSeverity.MEDIUM,
                description=f"Files with invalid naming: {len(invalid_names)}",
                affected_files=invalid_names[:10],  # First 10
                details={"pattern": self.naming_pattern, "total_invalid": len(invalid_names)},
                recommendation=f"Rename files to match pattern: {self.naming_pattern}"
            ))

        return issues

    def _check_metadata(self, dataset_path: Path) -> List[Issue]:
        """
        Check metadata file existence and validity

        Args:
            dataset_path: Root dataset path

        Returns:
            List of metadata-related issues
        """
        issues = []
        metadata_path = dataset_path / "metadata.json"

        # Check existence
        if not metadata_path.exists():
            issues.append(Issue(
                category=IssueCategory.FORMAT,
                severity=IssueSeverity.HIGH,
                description="metadata.json not found",
                affected_files=[str(metadata_path)],
                recommendation="Create metadata.json with dataset information"
            ))
            return issues

        # Check validity
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            # Basic schema validation
            expected_fields = ["dataset_name", "created_at", "total_images"]
            missing_fields = [
                field for field in expected_fields
                if field not in metadata
            ]

            if missing_fields:
                issues.append(Issue(
                    category=IssueCategory.FORMAT,
                    severity=IssueSeverity.MEDIUM,
                    description=f"metadata.json missing fields: {', '.join(missing_fields)}",
                    affected_files=[str(metadata_path)],
                    details={"missing_fields": missing_fields},
                    recommendation=f"Add missing fields to metadata.json: {', '.join(missing_fields)}"
                ))

        except json.JSONDecodeError as e:
            issues.append(Issue(
                category=IssueCategory.FORMAT,
                severity=IssueSeverity.HIGH,
                description=f"metadata.json invalid JSON: {str(e)}",
                affected_files=[str(metadata_path)],
                recommendation="Fix JSON syntax errors in metadata.json"
            ))
        except Exception as e:
            issues.append(Issue(
                category=IssueCategory.FORMAT,
                severity=IssueSeverity.HIGH,
                description=f"Error reading metadata.json: {str(e)}",
                affected_files=[str(metadata_path)]
            ))

        return issues

    def _check_captions(self, dataset_path: Path) -> List[Issue]:
        """
        Check caption file correspondence with images

        Args:
            dataset_path: Root dataset path

        Returns:
            List of caption-related issues
        """
        issues = []

        # Find all images
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
        images = []
        for ext in image_extensions:
            images.extend(dataset_path.rglob(f"*{ext}"))

        # Find all caption files
        captions = list(dataset_path.rglob("*.txt"))
        caption_stems = {c.stem for c in captions}

        # Check correspondence
        missing_captions = []
        for img_path in images:
            if img_path.stem not in caption_stems:
                missing_captions.append(str(img_path))

        if missing_captions:
            issues.append(Issue(
                category=IssueCategory.FORMAT,
                severity=IssueSeverity.HIGH,
                description=f"Images missing caption files: {len(missing_captions)}",
                affected_files=missing_captions[:10],  # First 10
                details={"total_missing": len(missing_captions)},
                recommendation="Generate caption files for all images"
            ))

        # Check for orphan captions (captions without images)
        image_stems = {img.stem for img in images}
        orphan_captions = []
        for caption_path in captions:
            if caption_path.stem not in image_stems:
                orphan_captions.append(str(caption_path))

        if orphan_captions:
            issues.append(Issue(
                category=IssueCategory.FORMAT,
                severity=IssueSeverity.MEDIUM,
                description=f"Caption files without matching images: {len(orphan_captions)}",
                affected_files=orphan_captions[:10],
                details={"total_orphan": len(orphan_captions)},
                recommendation="Remove orphan caption files or add corresponding images"
            ))

        return issues

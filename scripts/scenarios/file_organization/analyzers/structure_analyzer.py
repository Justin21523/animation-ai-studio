"""
Structure Analyzer

Analyzes directory structure and identifies organizational issues:
- Directory depth analysis
- Project structure pattern detection
- Orphaned directory identification
- Naming convention analysis
- Excessive nesting detection

Author: Animation AI Studio
Date: 2025-12-03
"""

import logging
import re
from pathlib import Path
from typing import List, Dict, Optional, Set, Tuple
from collections import defaultdict, Counter

from ..common import (
    StructureAnalysis,
    Issue,
    OrganizationIssue,
    IssueSeverity,
    FileMetadata
)

logger = logging.getLogger(__name__)


class StructureAnalyzer:
    """
    Directory structure analyzer

    Features:
    - Calculate nesting depth statistics
    - Detect common project structures
    - Find orphaned/empty directories
    - Analyze naming consistency
    - Identify excessive nesting
    - Detect misplaced files

    Example:
        analyzer = StructureAnalyzer(max_depth=5)
        analysis = analyzer.analyze_structure(Path("/path/to/dir"))

        print(f"Max depth: {analysis.max_depth}")
        print(f"Directories with issues: {len(analysis.directories_with_issues)}")
    """

    # Common project structure patterns
    PROJECT_PATTERNS = {
        'python': {'src', 'tests', 'docs', 'requirements.txt', 'setup.py'},
        'nodejs': {'src', 'node_modules', 'package.json', 'package-lock.json'},
        'java': {'src', 'target', 'pom.xml', 'build.gradle'},
        'web': {'css', 'js', 'images', 'index.html'},
        'ml_project': {'data', 'models', 'notebooks', 'scripts', 'requirements.txt'},
        'dataset': {'train', 'test', 'val', 'images', 'labels'},
    }

    # Naming convention patterns
    NAMING_PATTERNS = {
        'snake_case': re.compile(r'^[a-z0-9]+(_[a-z0-9]+)*$'),
        'kebab-case': re.compile(r'^[a-z0-9]+(-[a-z0-9]+)*$'),
        'camelCase': re.compile(r'^[a-z][a-zA-Z0-9]*$'),
        'PascalCase': re.compile(r'^[A-Z][a-zA-Z0-9]*$'),
    }

    def __init__(
        self,
        max_depth: int = 10,
        min_files_per_dir: int = 1,
        excessive_nesting_threshold: int = 5
    ):
        """
        Initialize structure analyzer

        Args:
            max_depth: Maximum allowed nesting depth
            min_files_per_dir: Minimum files per directory
            excessive_nesting_threshold: Depth threshold for excessive nesting warning
        """
        self.max_depth = max_depth
        self.min_files_per_dir = min_files_per_dir
        self.excessive_nesting_threshold = excessive_nesting_threshold

        logger.info(
            f"StructureAnalyzer initialized "
            f"(max_depth={max_depth}, "
            f"min_files={min_files_per_dir})"
        )

    def analyze_structure(
        self,
        root: Path,
        files: Optional[List[FileMetadata]] = None
    ) -> Tuple[StructureAnalysis, List[Issue]]:
        """
        Analyze directory structure

        Args:
            root: Root directory to analyze
            files: Optional pre-scanned file list

        Returns:
            Tuple of (StructureAnalysis, List[Issue])
        """
        logger.info(f"Analyzing directory structure: {root}")

        if not root.exists() or not root.is_dir():
            logger.error(f"Invalid root directory: {root}")
            return StructureAnalysis(
                total_directories=0,
                max_depth=0,
                avg_depth=0.0
            ), []

        # Collect all directories
        directories = self._collect_directories(root)

        # Calculate depth statistics
        depths = [self.calculate_depth(d, root) for d in directories]
        max_depth = max(depths) if depths else 0
        avg_depth = sum(depths) / len(depths) if depths else 0.0

        logger.info(
            f"Found {len(directories)} directories "
            f"(max_depth={max_depth}, avg_depth={avg_depth:.1f})"
        )

        # Detect project structure
        project_type = self.detect_project_structure(root)
        if project_type:
            logger.info(f"Detected project type: {project_type}")

        # Find issues
        issues: List[Issue] = []

        # Check for excessive nesting
        issues.extend(self._check_excessive_nesting(directories, root))

        # Check for orphaned directories
        issues.extend(self._check_orphaned_directories(directories))

        # Check for naming inconsistencies
        issues.extend(self._check_naming_consistency(directories, root))

        # Check for misplaced files (if file list provided)
        if files:
            issues.extend(self._check_misplaced_files(files, project_type))

        # Collect directories with issues
        directories_with_issues = list(set(
            issue.affected_files[0]
            for issue in issues
            if issue.affected_files
        ))

        # Build suggested structure (if issues found)
        suggested_structure = None
        if issues and project_type:
            suggested_structure = self._suggest_structure(project_type)

        analysis = StructureAnalysis(
            total_directories=len(directories),
            max_depth=max_depth,
            avg_depth=avg_depth,
            directories_with_issues=directories_with_issues,
            suggested_structure=suggested_structure
        )

        logger.info(
            f"Structure analysis complete: "
            f"{len(issues)} issues found in {len(directories_with_issues)} directories"
        )

        return analysis, issues

    def _collect_directories(self, root: Path) -> List[Path]:
        """Collect all directories under root"""
        directories = []

        try:
            for path in root.rglob('*'):
                if path.is_dir():
                    # Skip hidden directories and common exclusions
                    if not self._should_skip_directory(path):
                        directories.append(path)
        except PermissionError as e:
            logger.warning(f"Permission denied accessing {root}: {e}")

        return directories

    def _should_skip_directory(self, path: Path) -> bool:
        """Check if directory should be skipped"""
        # Skip hidden directories
        if path.name.startswith('.'):
            return True

        # Skip common system/build directories
        skip_patterns = {
            '__pycache__', 'node_modules', '.git', '.svn',
            'build', 'dist', 'target', '.idea', '.vscode',
            'venv', 'env', '.env', 'virtualenv'
        }

        return path.name in skip_patterns

    def calculate_depth(self, path: Path, root: Path) -> int:
        """
        Calculate directory depth relative to root

        Args:
            path: Directory path
            root: Root directory

        Returns:
            Depth level (root = 0)
        """
        try:
            relative = path.relative_to(root)
            return len(relative.parts)
        except ValueError:
            # Path not relative to root
            return 0

    def detect_project_structure(self, root: Path) -> Optional[str]:
        """
        Detect project type based on directory structure

        Args:
            root: Root directory

        Returns:
            Project type string or None
        """
        # Get immediate children
        children = set()
        try:
            for item in root.iterdir():
                if not item.name.startswith('.'):
                    children.add(item.name)
        except PermissionError:
            return None

        # Check against known patterns
        best_match = None
        best_score = 0

        for project_type, indicators in self.PROJECT_PATTERNS.items():
            # Calculate match score
            matches = len(children & indicators)
            score = matches / len(indicators) if indicators else 0

            if score > best_score and score >= 0.5:  # At least 50% match
                best_score = score
                best_match = project_type

        return best_match

    def _check_excessive_nesting(
        self,
        directories: List[Path],
        root: Path
    ) -> List[Issue]:
        """Check for excessive directory nesting"""
        issues = []

        for directory in directories:
            depth = self.calculate_depth(directory, root)

            if depth > self.excessive_nesting_threshold:
                issues.append(Issue(
                    category=OrganizationIssue.NESTED_EXCESSIVE,
                    severity=IssueSeverity.MEDIUM if depth <= self.max_depth else IssueSeverity.HIGH,
                    description=f"Excessive nesting depth ({depth} levels): {directory.relative_to(root)}",
                    affected_files=[directory],
                    recommendation=f"Consider flattening directory structure (current depth: {depth})",
                    auto_fixable=False
                ))

        return issues

    def _check_orphaned_directories(
        self,
        directories: List[Path]
    ) -> List[Issue]:
        """Check for orphaned/empty directories"""
        issues = []

        for directory in directories:
            try:
                # Check if directory is empty
                has_files = any(directory.iterdir())

                if not has_files:
                    issues.append(Issue(
                        category=OrganizationIssue.ORPHANED,
                        severity=IssueSeverity.LOW,
                        description=f"Empty directory: {directory.name}",
                        affected_files=[directory],
                        recommendation="Remove empty directory or add files",
                        auto_fixable=True
                    ))

                # Check if directory has very few files
                else:
                    file_count = sum(1 for _ in directory.iterdir() if _.is_file())

                    if file_count < self.min_files_per_dir and file_count > 0:
                        issues.append(Issue(
                            category=OrganizationIssue.ORPHANED,
                            severity=IssueSeverity.LOW,
                            description=f"Directory with very few files ({file_count}): {directory.name}",
                            affected_files=[directory],
                            recommendation=f"Consider consolidating with parent directory (only {file_count} files)",
                            auto_fixable=False
                        ))

            except PermissionError:
                continue

        return issues

    def _check_naming_consistency(
        self,
        directories: List[Path],
        root: Path
    ) -> List[Issue]:
        """Check for naming convention consistency"""
        issues = []

        # Detect predominant naming convention
        conventions = Counter()

        for directory in directories:
            name = directory.name

            for convention, pattern in self.NAMING_PATTERNS.items():
                if pattern.match(name):
                    conventions[convention] += 1
                    break

        if not conventions:
            return issues

        # Most common convention
        predominant_convention = conventions.most_common(1)[0][0]

        # Find directories not following predominant convention
        for directory in directories:
            name = directory.name

            # Check if follows predominant convention
            pattern = self.NAMING_PATTERNS[predominant_convention]

            if not pattern.match(name):
                # Check if it matches any convention
                matches_any = any(
                    p.match(name)
                    for p in self.NAMING_PATTERNS.values()
                )

                if matches_any:
                    issues.append(Issue(
                        category=OrganizationIssue.NAMING_INCONSISTENT,
                        severity=IssueSeverity.LOW,
                        description=f"Inconsistent naming convention: '{name}' (expected {predominant_convention})",
                        affected_files=[directory],
                        recommendation=f"Rename to follow {predominant_convention} convention",
                        auto_fixable=False
                    ))

        return issues

    def _check_misplaced_files(
        self,
        files: List[FileMetadata],
        project_type: Optional[str]
    ) -> List[Issue]:
        """Check for misplaced files based on project structure"""
        if not project_type:
            return []

        issues = []

        # Define expected locations for file types by project
        expected_locations = {
            'python': {
                'CODE': ['src', 'lib', 'scripts'],
                'DOCUMENT': ['docs', 'documentation'],
                'CONFIG': ['.'],
                'LOG': ['logs', 'log'],
            },
            'ml_project': {
                'MODEL': ['models', 'checkpoints'],
                'DATASET': ['data', 'datasets'],
                'CODE': ['scripts', 'src'],
                'DOCUMENT': ['notebooks', 'docs'],
            },
            'dataset': {
                'IMAGE': ['images', 'train', 'test', 'val'],
                'DOCUMENT': ['labels', 'annotations'],
            }
        }

        # Get expected locations for this project type
        expected = expected_locations.get(project_type, {})

        for file_meta in files:
            file_type_str = file_meta.file_type.value.upper()

            if file_type_str not in expected:
                continue

            # Check if file is in expected location
            expected_dirs = expected[file_type_str]
            file_dir = file_meta.path.parent.name

            if file_dir not in expected_dirs:
                issues.append(Issue(
                    category=OrganizationIssue.MISPLACED,
                    severity=IssueSeverity.MEDIUM,
                    description=f"{file_meta.file_type.value} file in unexpected location: {file_meta.path.name}",
                    affected_files=[file_meta.path],
                    recommendation=f"Consider moving to: {', '.join(expected_dirs)}",
                    auto_fixable=True
                ))

        return issues

    def _suggest_structure(self, project_type: str) -> Dict[str, any]:
        """Suggest ideal structure for project type"""
        suggestions = {
            'python': {
                'src/': 'Source code',
                'tests/': 'Unit tests',
                'docs/': 'Documentation',
                'scripts/': 'Utility scripts',
                'requirements.txt': 'Dependencies',
                'README.md': 'Project overview'
            },
            'ml_project': {
                'data/': 'Datasets',
                'models/': 'Trained models',
                'notebooks/': 'Jupyter notebooks',
                'scripts/': 'Training/inference scripts',
                'src/': 'Source code',
                'docs/': 'Documentation',
                'configs/': 'Configuration files'
            },
            'dataset': {
                'train/images/': 'Training images',
                'train/labels/': 'Training labels',
                'val/images/': 'Validation images',
                'val/labels/': 'Validation labels',
                'test/images/': 'Test images',
                'metadata.json': 'Dataset metadata'
            }
        }

        return suggestions.get(project_type, {})

    def find_orphaned_directories(self, root: Path) -> List[Path]:
        """
        Find orphaned (empty or nearly empty) directories

        Args:
            root: Root directory to search

        Returns:
            List of orphaned directory paths
        """
        orphaned = []

        for directory in self._collect_directories(root):
            try:
                # Count files (not directories)
                file_count = sum(1 for _ in directory.iterdir() if _.is_file())

                if file_count < self.min_files_per_dir:
                    orphaned.append(directory)

            except PermissionError:
                continue

        return orphaned

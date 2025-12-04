"""
File Organizer

Main orchestrator coordinating all file organization components.
Performs comprehensive analysis and smart file organization.

Author: Animation AI Studio
Date: 2025-12-03
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import time

from .common import (
    FileMetadata, FileType, OrganizationStrategy, Issue, IssueSeverity,
    OrganizationReport, StructureAnalysis, DuplicateGroup
)
from .analyzers import FileClassifier, DuplicateDetector, StructureAnalyzer
from .processors import SmartOrganizer

logger = logging.getLogger(__name__)


class FileOrganizer:
    """
    File Organization Orchestrator

    Orchestrates all file organization components to perform
    comprehensive analysis and intelligent organization.

    Components:
    - FileClassifier: Multi-layer file type detection
    - DuplicateDetector: Exact and near-duplicate detection
    - StructureAnalyzer: Directory structure analysis
    - SmartOrganizer: AI-powered file organization

    Features:
    - CPU-only operation
    - Comprehensive file analysis
    - Multiple organization strategies
    - Dry-run preview mode
    - Backup and rollback support
    - Progress tracking

    Example:
        organizer = FileOrganizer(
            root_path="/path/to/organize",
            config={
                "min_file_size": 1024,
                "enable_perceptual_hashing": True,
                "create_backup": True
            }
        )

        # Analyze current state
        report = organizer.analyze()

        print(f"Organization Score: {report.organization_score:.1f}/100")
        print(f"Issues Found: {len(report.issues)}")

        # Plan organization
        plan = organizer.plan_organization(
            strategy=OrganizationStrategy.SMART
        )

        print(f"Will organize {plan.total_files} files")

        # Execute (dry-run first)
        result = organizer.organize(plan, dry_run=True)

        if result.success:
            # Actually execute
            result = organizer.organize(plan, dry_run=False)
    """

    def __init__(
        self,
        root_path: str,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize File Organizer

        Args:
            root_path: Path to root directory to analyze/organize
            config: Configuration dictionary with component settings
        """
        self.root_path = Path(root_path)
        self.config = config or {}

        # Initialize components
        logger.info("Initializing FileOrganizer components...")

        self.classifier = FileClassifier()

        self.dup_detector = DuplicateDetector(
            min_file_size=self.config.get("min_file_size", 1024),
            enable_perceptual=self.config.get("enable_perceptual_hashing", True),
            similarity_threshold=self.config.get("similarity_threshold", 5)
        )

        self.structure_analyzer = StructureAnalyzer(
            max_depth=self.config.get("max_depth", 10),
            min_files_per_dir=self.config.get("min_files_per_dir", 3),
            excessive_nesting_threshold=self.config.get("excessive_nesting_threshold", 5)
        )

        self.smart_organizer = SmartOrganizer(
            create_backup=self.config.get("create_backup", True),
            backup_dir=Path(self.config.get("backup_dir", ".backup")) if self.config.get("backup_dir") else None
        )

        logger.info("FileOrganizer initialized successfully")

    def analyze(self, enable_recommendations: bool = True) -> OrganizationReport:
        """
        Perform comprehensive file organization analysis

        Args:
            enable_recommendations: Generate AI-powered recommendations

        Returns:
            OrganizationReport with all analysis results
        """
        logger.info(f"Starting analysis of {self.root_path}")
        start_time = time.time()

        # Step 1: Scan and classify files
        logger.info("Step 1/5: Scanning and classifying files...")
        files_metadata = self._scan_and_classify()

        if not files_metadata:
            logger.error("No files found in directory")
            return self._create_empty_report("No files found")

        # Step 2: Detect duplicates
        logger.info(f"Step 2/5: Detecting duplicates ({len(files_metadata)} files)...")
        duplicate_groups = self._detect_duplicates(files_metadata)

        # Step 3: Analyze directory structure
        logger.info("Step 3/5: Analyzing directory structure...")
        structure_analysis, structure_issues = self._analyze_structure(files_metadata)

        # Step 4: Detect organization issues
        logger.info("Step 4/5: Detecting organization issues...")
        organization_issues = self._detect_organization_issues(
            files_metadata,
            structure_analysis
        )

        # Step 5: Generate recommendations
        logger.info("Step 5/5: Generating recommendations...")
        recommendations = []
        if enable_recommendations:
            recommendations = self._generate_recommendations(
                files_metadata,
                duplicate_groups,
                structure_issues + organization_issues
            )

        # Collect all issues
        all_issues = structure_issues + organization_issues

        # Calculate statistics
        total_size = sum(f.size_bytes for f in files_metadata)
        file_type_counts = self._count_by_file_type(files_metadata)

        # Calculate organization score
        organization_score = self._calculate_organization_score(
            structure_analysis,
            all_issues,
            duplicate_groups
        )

        # Build report
        report = OrganizationReport(
            root_path=self.root_path,
            timestamp=time.time(),
            total_files=len(files_metadata),
            total_size_bytes=total_size,
            file_type_counts=file_type_counts,
            duplicate_groups=duplicate_groups,
            issues=all_issues,
            structure_analysis=structure_analysis,
            organization_score=organization_score,
            recommendations=recommendations
        )

        elapsed_time = time.time() - start_time
        logger.info(f"Analysis completed in {elapsed_time:.1f}s")
        logger.info(f"Organization Score: {organization_score:.1f}/100")
        logger.info(f"Total Issues: {len(all_issues)}")
        logger.info(f"Duplicate Groups: {len(duplicate_groups)}")

        return report

    def plan_organization(
        self,
        strategy: OrganizationStrategy = OrganizationStrategy.SMART,
        custom_rules: Optional[List] = None
    ):
        """
        Plan file organization without executing

        Args:
            strategy: Organization strategy to use
            custom_rules: Custom organization rules (for CUSTOM strategy)

        Returns:
            OrganizationPlan with proposed moves
        """
        logger.info(f"Planning organization with strategy: {strategy.value}")

        # Scan and classify files
        files_metadata = self._scan_and_classify()

        if not files_metadata:
            logger.error("No files to organize")
            return None

        # Use smart organizer to plan
        plan = self.smart_organizer.plan_organization(
            files=files_metadata,
            strategy=strategy,
            root=self.root_path,
            custom_rules=custom_rules
        )

        logger.info(
            f"Organization plan ready: {plan.total_files} files, "
            f"{plan.total_size_bytes / 1024 / 1024:.1f} MB"
        )

        return plan

    def organize(self, plan, dry_run: bool = True):
        """
        Execute file organization

        Args:
            plan: OrganizationPlan from plan_organization()
            dry_run: If True, only simulate (don't actually move files)

        Returns:
            OrganizationResult with execution details
        """
        logger.info(f"Organizing files (dry_run={dry_run})...")

        # Get files metadata
        files_metadata = self._scan_and_classify()

        # Execute organization
        result = self.smart_organizer.organize(
            files=files_metadata,
            plan=plan,
            dry_run=dry_run
        )

        mode = "preview" if dry_run else "execution"
        logger.info(
            f"Organization {mode} complete: "
            f"{result.moved_files} moved, {result.failed_moves} failed"
        )

        return result

    def _scan_and_classify(self) -> List[FileMetadata]:
        """
        Scan directory and classify all files

        Returns:
            List of FileMetadata objects
        """
        files_metadata = []

        for path in self.root_path.rglob("*"):
            if not path.is_file():
                continue

            try:
                # Classify file type
                file_type = self.classifier.classify_file(path)

                # Get file stats
                stat = path.stat()

                # Create metadata
                metadata = FileMetadata(
                    path=path,
                    file_type=file_type,
                    size_bytes=stat.st_size,
                    created_time=stat.st_ctime,
                    modified_time=stat.st_mtime,
                    accessed_time=stat.st_atime
                )

                files_metadata.append(metadata)

            except Exception as e:
                logger.error(f"Error processing {path}: {e}")

        logger.info(f"Scanned {len(files_metadata)} files")
        return files_metadata

    def _detect_duplicates(
        self,
        files_metadata: List[FileMetadata]
    ) -> List[DuplicateGroup]:
        """
        Detect duplicate files

        Args:
            files_metadata: List of file metadata

        Returns:
            List of DuplicateGroup objects
        """
        duplicate_groups = self.dup_detector.detect_duplicates(files_metadata)

        exact_count = sum(1 for g in duplicate_groups if g.is_exact)
        near_count = sum(1 for g in duplicate_groups if not g.is_exact)

        logger.info(
            f"Found {len(duplicate_groups)} duplicate groups "
            f"({exact_count} exact, {near_count} near-duplicates)"
        )

        return duplicate_groups

    def _analyze_structure(
        self,
        files_metadata: List[FileMetadata]
    ) -> tuple[StructureAnalysis, List[Issue]]:
        """
        Analyze directory structure

        Args:
            files_metadata: List of file metadata

        Returns:
            (StructureAnalysis, list of Issue objects)
        """
        analysis, issues = self.structure_analyzer.analyze_structure(
            root=self.root_path,
            files=files_metadata
        )

        logger.info(
            f"Structure analysis: {analysis.total_directories} directories, "
            f"max depth {analysis.max_depth}, {len(issues)} issues"
        )

        return analysis, issues

    def _detect_organization_issues(
        self,
        files_metadata: List[FileMetadata],
        structure_analysis: StructureAnalysis
    ) -> List[Issue]:
        """
        Detect organization-specific issues

        Args:
            files_metadata: List of file metadata
            structure_analysis: Structure analysis results

        Returns:
            List of Issue objects
        """
        issues = []

        # Check for mixed file types in same directory
        dir_file_types: Dict[Path, set] = {}
        for file_meta in files_metadata:
            parent = file_meta.path.parent
            if parent not in dir_file_types:
                dir_file_types[parent] = set()
            dir_file_types[parent].add(file_meta.file_type)

        for dir_path, file_types in dir_file_types.items():
            if len(file_types) > 5:  # Too many different types
                issues.append(Issue(
                    category="organization_mixed_types",
                    severity=IssueSeverity.MEDIUM,
                    description=f"Directory contains {len(file_types)} different file types",
                    affected_paths=[dir_path],
                    recommendation="Consider organizing files by type"
                ))

        # Check for scattered file types
        type_locations: Dict[FileType, set] = {}
        for file_meta in files_metadata:
            if file_meta.file_type not in type_locations:
                type_locations[file_meta.file_type] = set()
            type_locations[file_meta.file_type].add(file_meta.path.parent)

        for file_type, locations in type_locations.items():
            if len(locations) > 10:  # Too scattered
                issues.append(Issue(
                    category="organization_scattered",
                    severity=IssueSeverity.LOW,
                    description=f"{file_type.value} files scattered across {len(locations)} directories",
                    affected_paths=list(locations)[:5],  # Sample
                    recommendation=f"Consolidate {file_type.value} files into dedicated directory"
                ))

        logger.info(f"Detected {len(issues)} organization issues")
        return issues

    def _count_by_file_type(
        self,
        files_metadata: List[FileMetadata]
    ) -> Dict[FileType, int]:
        """Count files by type"""
        counts = {}
        for file_meta in files_metadata:
            counts[file_meta.file_type] = counts.get(file_meta.file_type, 0) + 1
        return counts

    def _calculate_organization_score(
        self,
        structure_analysis: StructureAnalysis,
        issues: List[Issue],
        duplicate_groups: List[DuplicateGroup]
    ) -> float:
        """
        Calculate overall organization score (0-100)

        Args:
            structure_analysis: Structure analysis results
            issues: List of all issues
            duplicate_groups: List of duplicate groups

        Returns:
            Score from 0-100
        """
        score = 100.0

        # Deduct for excessive nesting
        if structure_analysis.max_depth > 5:
            score -= (structure_analysis.max_depth - 5) * 5

        # Deduct for issues
        for issue in issues:
            deduction = {
                IssueSeverity.CRITICAL: 20,
                IssueSeverity.HIGH: 10,
                IssueSeverity.MEDIUM: 5,
                IssueSeverity.LOW: 2
            }.get(issue.severity, 0)
            score -= deduction

        # Deduct for duplicates
        if duplicate_groups:
            duplicate_count = sum(len(g.files) for g in duplicate_groups)
            score -= min(20, duplicate_count * 0.5)

        return max(0.0, score)

    def _generate_recommendations(
        self,
        files_metadata: List[FileMetadata],
        duplicate_groups: List[DuplicateGroup],
        issues: List[Issue]
    ) -> List[Dict[str, Any]]:
        """
        Generate recommendations based on analysis

        Placeholder for future Agent Framework integration

        Args:
            files_metadata: List of file metadata
            duplicate_groups: List of duplicate groups
            issues: List of all issues

        Returns:
            List of recommendation dictionaries
        """
        recommendations = []

        # Add issue-based recommendations
        for issue in issues:
            if issue.severity in [IssueSeverity.CRITICAL, IssueSeverity.HIGH]:
                if issue.recommendation:
                    recommendations.append({
                        "priority": issue.severity.value,
                        "category": issue.category,
                        "action": issue.recommendation,
                        "affected_count": len(issue.affected_paths)
                    })

        # Add duplicate recommendations
        if duplicate_groups:
            total_duplicates = sum(len(g.files) - 1 for g in duplicate_groups)
            potential_savings = sum(
                g.total_size_bytes * (len(g.files) - 1) / len(g.files)
                for g in duplicate_groups
            )

            recommendations.append({
                "priority": "high",
                "category": "duplicates",
                "action": f"Remove {total_duplicates} duplicate files to save {potential_savings / 1024 / 1024:.1f} MB",
                "affected_count": total_duplicates
            })

        # Add organization strategy recommendation
        file_types_count = len(self._count_by_file_type(files_metadata))

        if file_types_count > 5:
            recommendations.append({
                "priority": "medium",
                "category": "organization",
                "action": f"Organize {file_types_count} file types using BY_TYPE or SMART strategy",
                "affected_count": len(files_metadata)
            })

        logger.info(f"Generated {len(recommendations)} recommendations")
        return recommendations

    def _create_empty_report(self, reason: str) -> OrganizationReport:
        """Create empty report for error cases"""
        return OrganizationReport(
            root_path=self.root_path,
            timestamp=time.time(),
            total_files=0,
            total_size_bytes=0,
            file_type_counts={},
            duplicate_groups=[],
            issues=[Issue(
                category="system_error",
                severity=IssueSeverity.CRITICAL,
                description=reason,
                affected_paths=[]
            )],
            structure_analysis=None,
            organization_score=0.0,
            recommendations=[]
        )

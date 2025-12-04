"""
File Organization Scenario

Intelligent file system analysis and automated organization for Animation AI Studio.

This scenario provides:
- File type classification with magic bytes detection
- Duplicate file detection (exact and near-duplicate)
- Directory structure analysis
- AI-powered organization recommendations
- Safe file operations with dry-run mode

Author: Animation AI Studio
Date: 2025-12-03
Version: 1.0.0
"""

from .common import (
    # Enums
    FileType,
    OrganizationIssue,
    IssueSeverity,
    OrganizationStrategy,

    # Dataclasses
    FileMetadata,
    DuplicateGroup,
    Issue,
    StructureAnalysis,
    OrganizationReport
)

from .organizer import FileOrganizer
from .analyzers import FileClassifier, DuplicateDetector, StructureAnalyzer
from .processors import SmartOrganizer
from .integration import (
    AgentIntegration,
    RAGIntegration,
    create_agent_integration,
    create_rag_integration
)

__version__ = "1.0.0"

__all__ = [
    # Enums
    "FileType",
    "OrganizationIssue",
    "IssueSeverity",
    "OrganizationStrategy",

    # Dataclasses
    "FileMetadata",
    "DuplicateGroup",
    "Issue",
    "StructureAnalysis",
    "OrganizationReport",

    # Main Orchestrator
    "FileOrganizer",

    # Components
    "FileClassifier",
    "DuplicateDetector",
    "StructureAnalyzer",
    "SmartOrganizer",

    # Integration
    "AgentIntegration",
    "RAGIntegration",
    "create_agent_integration",
    "create_rag_integration"
]

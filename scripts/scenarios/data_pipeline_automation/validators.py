"""
Data Pipeline Automation - Validation Utilities

Additional validation utilities for pipeline configurations.

Author: Animation AI Studio
Date: 2025-12-03
"""

from pathlib import Path
from typing import Dict, List, Set

from .common import Pipeline, PipelineStage


def has_circular_dependencies(stages: List[PipelineStage]) -> bool:
    """
    Check if stages have circular dependencies using DFS

    Args:
        stages: List of pipeline stages

    Returns:
        True if circular dependencies exist
    """
    # Build adjacency list
    stage_map = {s.id: s for s in stages}
    adjacency: Dict[str, List[str]] = {s.id: s.depends_on.copy() for s in stages}

    visited: Set[str] = set()
    rec_stack: Set[str] = set()

    def dfs(node: str) -> bool:
        """DFS cycle detection"""
        visited.add(node)
        rec_stack.add(node)

        for neighbor in adjacency.get(node, []):
            if neighbor not in visited:
                if dfs(neighbor):
                    return True
            elif neighbor in rec_stack:
                return True

        rec_stack.remove(node)
        return False

    # Check each connected component
    for stage_id in stage_map:
        if stage_id not in visited:
            if dfs(stage_id):
                return True

    return False


def validate_dependencies_exist(stages: List[PipelineStage]) -> bool:
    """
    Validate all stage dependencies exist

    Args:
        stages: List of pipeline stages

    Returns:
        True if all dependencies exist

    Raises:
        ValueError: If dependency missing
    """
    stage_ids = {s.id for s in stages}

    for stage in stages:
        for dep in stage.depends_on:
            if dep not in stage_ids:
                raise ValueError(
                    f"Stage '{stage.id}' depends on unknown stage '{dep}'"
                )

    return True


def validate_no_duplicate_ids(stages: List[PipelineStage]) -> bool:
    """
    Validate no duplicate stage IDs

    Args:
        stages: List of pipeline stages

    Returns:
        True if no duplicates

    Raises:
        ValueError: If duplicates found
    """
    stage_ids = [s.id for s in stages]
    unique_ids = set(stage_ids)

    if len(unique_ids) != len(stage_ids):
        duplicates = [sid for sid in unique_ids if stage_ids.count(sid) > 1]
        raise ValueError(f"Duplicate stage IDs found: {duplicates}")

    return True


def validate_path_exists(path: Path, required: bool = True) -> bool:
    """
    Validate path exists

    Args:
        path: Path to validate
        required: Whether path is required to exist

    Returns:
        True if valid

    Raises:
        ValueError: If path invalid and required
    """
    if required and not path.exists():
        raise ValueError(f"Required path does not exist: {path}")

    return True


def validate_config_schema(config: dict, required_fields: List[str]) -> bool:
    """
    Validate configuration has required fields

    Args:
        config: Configuration dict
        required_fields: List of required field names

    Returns:
        True if valid

    Raises:
        ValueError: If required fields missing
    """
    missing_fields = [f for f in required_fields if f not in config]

    if missing_fields:
        raise ValueError(f"Missing required fields: {missing_fields}")

    return True

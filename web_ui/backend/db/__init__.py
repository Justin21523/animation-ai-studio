"""
Database layer for Web UI.
Extends existing workflows.db with job management tables.
"""

from .schema import init_database
from .operations import JobDatabase

__all__ = ['init_database', 'JobDatabase']

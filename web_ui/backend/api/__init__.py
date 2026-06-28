"""
API endpoints for Web UI.
"""

from .jobs import router as jobs_router

__all__ = ['jobs_router']

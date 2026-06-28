"""
Adapters for integrating Web UI with existing infrastructure.
"""

from .bash_runner import BashRunnerAdapter, BashScriptConfig

__all__ = ['BashRunnerAdapter', 'BashScriptConfig']

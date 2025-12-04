"""
Configuration and Utility Modules
Provides core configuration and utility functions for animation-ai-studio.

Components:
- config_loader: OmegaConf-based configuration management
- logger: Colorized logging setup
- path_utils: Path manipulation and validation
- model_paths: Model discovery and path management
- checkpoint_manager: Checkpoint/resume functionality
- prompt_loader: Prompt template loading and management
"""

from .config_loader import load_config, get_config
from .logger import setup_logger
from .path_utils import get_project_root, ensure_dir
from .checkpoint_manager import CheckpointManager
from .prompt_loader import PromptLoader

__all__ = [
    "load_config",
    "get_config",
    "setup_logger",
    "get_project_root",
    "ensure_dir",
    "CheckpointManager",
    "PromptLoader",
]

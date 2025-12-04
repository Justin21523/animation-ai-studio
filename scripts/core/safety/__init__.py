"""
Safety Module

Comprehensive safety infrastructure for CPU-only automation workflows.
Provides GPU isolation, memory monitoring, preflight checks, and runtime monitoring.

Modules:
  - gpu_isolation: CPU-only enforcement and GPU usage verification
  - memory_monitor: Real-time memory monitoring with tiered thresholds
  - preflight_checks: Pre-execution safety verification
  - runtime_monitor: Background monitoring during workflow execution

Quick Start:
  >>> from scripts.core.safety import run_preflight, RuntimeMonitor
  >>>
  >>> # Run preflight checks before starting
  >>> run_preflight()
  >>>
  >>> # Start runtime monitoring
  >>> with RuntimeMonitor(check_interval=30.0) as monitor:
  >>>     # Your workflow code here
  >>>     process_data()

Author: Animation AI Studio Team
Last Modified: 2025-12-02
"""

# GPU Isolation
from .gpu_isolation import (
    enforce_cpu_only,
    verify_no_gpu_usage,
    safe_import_torch,
    set_cpu_affinity,
    create_cpu_only_subprocess_env,
    run_cpu_only_subprocess,
    get_gpu_status,
    print_gpu_status,
    restore_environment,
    GPUIsolationError,
    EnvironmentSetupError,
)

# Memory Monitoring
from .memory_monitor import (
    MemoryMonitor,
    MemoryStats,
    MemoryThresholds,
    load_thresholds_from_yaml,
    print_memory_report,
)

# Preflight Checks
from .preflight_checks import (
    run_preflight,
    PreflightResult,
    PreflightError,
)

# Runtime Monitoring
from .runtime_monitor import (
    RuntimeMonitor,
    MonitoringStats,
    create_checkpoint_callback,
    create_batch_size_callback,
)

__all__ = [
    # GPU Isolation
    'enforce_cpu_only',
    'verify_no_gpu_usage',
    'safe_import_torch',
    'set_cpu_affinity',
    'create_cpu_only_subprocess_env',
    'run_cpu_only_subprocess',
    'get_gpu_status',
    'print_gpu_status',
    'restore_environment',
    'GPUIsolationError',
    'EnvironmentSetupError',

    # Memory Monitoring
    'MemoryMonitor',
    'MemoryStats',
    'MemoryThresholds',
    'load_thresholds_from_yaml',
    'print_memory_report',

    # Preflight Checks
    'run_preflight',
    'PreflightResult',
    'PreflightError',

    # Runtime Monitoring
    'RuntimeMonitor',
    'MonitoringStats',
    'create_checkpoint_callback',
    'create_batch_size_callback',
]

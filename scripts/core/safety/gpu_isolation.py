"""
GPU Isolation Module

Ensures CPU-only execution for automation tasks to prevent interference with GPU training.
Implements 4-layer isolation strategy:
  Layer 1: Environment variables (CUDA_VISIBLE_DEVICES, TORCH_DEVICE)
  Layer 2: Python import guards (verify PyTorch CPU-only)
  Layer 3: Process affinity (CPU cores only)
  Layer 4: Subprocess sandboxing (inherit CPU-only environment)

Critical Requirements:
  - Must NEVER allow GPU access during training workflows
  - Must fail fast with clear error messages
  - Must be importable before any ML library
  - Must provide verification utilities for runtime checks

Author: Animation AI Studio Team
Last Modified: 2025-12-02
"""

import os
import sys
import subprocess
import warnings
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging

# Setup logger
logger = logging.getLogger(__name__)


# ============================================================================
# Custom Exceptions
# ============================================================================

class GPUIsolationError(Exception):
    """
    Raised when GPU usage is detected in CPU-only mode.

    This exception indicates a critical safety violation where GPU resources
    were accessed despite enforcement of CPU-only execution. This typically
    means environment isolation failed or a library bypassed the checks.
    """
    pass


class EnvironmentSetupError(Exception):
    """
    Raised when CPU-only environment setup fails.

    This indicates the system could not properly configure CPU-only mode,
    possibly due to missing dependencies or permission issues.
    """
    pass


# ============================================================================
# Layer 1: Environment Variable Management
# ============================================================================

def enforce_cpu_only(strict: bool = True) -> Dict[str, str]:
    """
    Enforce CPU-only execution by setting environment variables.

    This function configures the process environment to disable GPU access
    for all major ML frameworks (PyTorch, JAX, TensorFlow, etc.).

    Args:
        strict: If True, raises error if variables can't be set.
                If False, logs warnings only.

    Returns:
        Dict mapping variable names to their previous values (for restoration)

    Raises:
        EnvironmentSetupError: If strict=True and setup fails

    Example:
        >>> previous_env = enforce_cpu_only()
        >>> # Run CPU-only code
        >>> restore_environment(previous_env)  # Optional restoration
    """
    previous_values = {}

    # Critical variables to set
    cpu_only_vars = {
        # PyTorch
        'CUDA_VISIBLE_DEVICES': '',  # Hide all CUDA devices
        'TORCH_DEVICE': 'cpu',       # Force CPU device
        'PYTORCH_CUDA_ALLOC_CONF': '',  # Disable CUDA memory config

        # JAX
        'JAX_PLATFORMS': 'cpu',      # JAX CPU-only mode
        'JAX_ENABLE_X64': 'True',    # Use 64-bit precision on CPU

        # TensorFlow (if needed in future)
        'CUDA_DEVICE_ORDER': 'PCI_BUS_ID',  # Consistent device ordering
        'TF_FORCE_GPU_ALLOW_GROWTH': 'false',  # Disable GPU growth

        # NumPy/OpenBLAS threading (prevent CPU overload)
        'OMP_NUM_THREADS': '8',      # Limit OpenMP threads
        'MKL_NUM_THREADS': '8',      # Limit MKL threads
        'OPENBLAS_NUM_THREADS': '8', # Limit OpenBLAS threads

        # HuggingFace/Cache locations (respect data_model_structure.md)
        'HF_HOME': '/mnt/c/ai_cache/huggingface',
        'TRANSFORMERS_CACHE': '/mnt/c/ai_cache/huggingface',
        'TORCH_HOME': '/mnt/c/ai_cache/torch',
        'XDG_CACHE_HOME': '/mnt/c/ai_cache',
    }

    # Store previous values and set new ones
    for var_name, var_value in cpu_only_vars.items():
        previous_values[var_name] = os.environ.get(var_name)
        try:
            os.environ[var_name] = var_value
            logger.debug(f"Set {var_name}={var_value}")
        except Exception as e:
            msg = f"Failed to set environment variable {var_name}: {e}"
            if strict:
                raise EnvironmentSetupError(msg)
            else:
                logger.warning(msg)

    # Verify critical variables are set correctly
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        msg = "Failed to hide CUDA devices (CUDA_VISIBLE_DEVICES not empty)"
        if strict:
            raise EnvironmentSetupError(msg)
        else:
            logger.warning(msg)

    logger.info("✓ CPU-only environment enforced successfully")
    return previous_values


def restore_environment(previous_values: Dict[str, Optional[str]]) -> None:
    """
    Restore environment variables to their previous state.

    Args:
        previous_values: Dict from enforce_cpu_only() return value

    Example:
        >>> prev = enforce_cpu_only()
        >>> # ... CPU-only work ...
        >>> restore_environment(prev)
    """
    for var_name, var_value in previous_values.items():
        if var_value is None:
            os.environ.pop(var_name, None)
        else:
            os.environ[var_name] = var_value

    logger.info("Environment variables restored")


# ============================================================================
# Layer 2: Python Import Guards
# ============================================================================

def verify_no_gpu_usage(raise_on_violation: bool = True) -> Tuple[bool, str]:
    """
    Verify that GPU is not accessible or in use.

    This function checks multiple conditions to ensure GPU isolation:
      1. PyTorch: torch.cuda.is_available() == False
      2. Environment: CUDA_VISIBLE_DEVICES is empty
      3. Process: No CUDA libraries loaded (check /proc/self/maps)

    Args:
        raise_on_violation: If True, raises GPUIsolationError on violation.
                           If False, returns (False, error_msg).

    Returns:
        Tuple of (is_safe, message):
          - is_safe: True if no GPU detected, False otherwise
          - message: Human-readable status message

    Raises:
        GPUIsolationError: If raise_on_violation=True and GPU detected

    Example:
        >>> enforce_cpu_only()
        >>> is_safe, msg = verify_no_gpu_usage(raise_on_violation=False)
        >>> if not is_safe:
        >>>     logger.error(f"GPU isolation violated: {msg}")
    """
    violations = []

    # Check 1: Environment variables
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', None)
    if cuda_visible != '':
        violations.append(
            f"CUDA_VISIBLE_DEVICES is '{cuda_visible}' (expected empty string)"
        )

    torch_device = os.environ.get('TORCH_DEVICE', None)
    if torch_device != 'cpu':
        violations.append(
            f"TORCH_DEVICE is '{torch_device}' (expected 'cpu')"
        )

    # Check 2: PyTorch availability (only if already imported)
    if 'torch' in sys.modules:
        try:
            import torch
            if torch.cuda.is_available():
                violations.append(
                    "PyTorch reports CUDA is available (torch.cuda.is_available() == True)"
                )

            # Check default device
            try:
                default_device = torch.tensor([1.0]).device
                if default_device.type != 'cpu':
                    violations.append(
                        f"PyTorch default device is '{default_device}' (expected 'cpu')"
                    )
            except Exception as e:
                logger.debug(f"Could not check default device: {e}")

        except ImportError:
            pass  # PyTorch not available, that's fine

    # Check 3: Process memory maps (Linux only)
    if sys.platform.startswith('linux'):
        try:
            maps_path = Path('/proc/self/maps')
            if maps_path.exists():
                maps_content = maps_path.read_text()
                cuda_libs = ['libcuda.so', 'libcudart.so', 'libcublas.so']
                loaded_cuda_libs = [lib for lib in cuda_libs if lib in maps_content]

                if loaded_cuda_libs:
                    violations.append(
                        f"CUDA libraries loaded in process: {', '.join(loaded_cuda_libs)}"
                    )
        except Exception as e:
            logger.debug(f"Could not check /proc/self/maps: {e}")

    # Evaluate results
    if violations:
        error_msg = "GPU isolation violated:\n  - " + "\n  - ".join(violations)
        if raise_on_violation:
            raise GPUIsolationError(error_msg)
        else:
            return False, error_msg
    else:
        return True, "✓ No GPU usage detected (CPU-only mode verified)"


def safe_import_torch(cpu_only: bool = True) -> 'torch':
    """
    Safely import PyTorch with CPU-only guarantee.

    This function ensures PyTorch is imported in CPU-only mode, or raises
    an error if GPU is detected after import.

    Args:
        cpu_only: If True, enforces CPU-only and verifies after import.
                 If False, allows GPU (for testing only).

    Returns:
        torch module

    Raises:
        GPUIsolationError: If cpu_only=True and GPU detected after import
        ImportError: If PyTorch not installed

    Example:
        >>> torch = safe_import_torch()
        >>> # Now safe to use torch on CPU only
    """
    # Enforce CPU-only before import
    if cpu_only:
        enforce_cpu_only(strict=True)

    # Import PyTorch
    try:
        import torch
    except ImportError as e:
        raise ImportError(
            f"PyTorch not installed. Install with: pip install torch --index-url "
            f"https://download.pytorch.org/whl/cpu"
        ) from e

    # Verify after import
    if cpu_only:
        is_safe, msg = verify_no_gpu_usage(raise_on_violation=False)
        if not is_safe:
            raise GPUIsolationError(
                f"GPU detected after importing PyTorch:\n{msg}\n\n"
                f"This may indicate PyTorch was installed with CUDA support. "
                f"Reinstall CPU-only version:\n"
                f"  pip uninstall torch\n"
                f"  pip install torch --index-url https://download.pytorch.org/whl/cpu"
            )

    logger.info(f"✓ PyTorch imported safely (version: {torch.__version__})")
    return torch


# ============================================================================
# Layer 3: Process Affinity
# ============================================================================

def set_cpu_affinity(cpu_ids: Optional[List[int]] = None) -> List[int]:
    """
    Bind process to specific CPU cores.

    This prevents the automation process from competing with training for
    CPU resources, and ensures predictable performance.

    Args:
        cpu_ids: List of CPU core IDs to bind to. If None, auto-select
                 cores 0-7 (leaving higher cores for training).

    Returns:
        List of CPU IDs the process is now bound to

    Raises:
        RuntimeError: If platform doesn't support CPU affinity

    Example:
        >>> # Bind to CPUs 0-7 (first 8 cores)
        >>> assigned_cpus = set_cpu_affinity([0,1,2,3,4,5,6,7])
    """
    if sys.platform.startswith('linux'):
        try:
            import psutil

            # Auto-select first 8 CPUs if not specified
            if cpu_ids is None:
                total_cpus = psutil.cpu_count(logical=True)
                cpu_ids = list(range(min(8, total_cpus)))

            # Set affinity
            process = psutil.Process()
            process.cpu_affinity(cpu_ids)

            # Verify
            actual_cpus = process.cpu_affinity()
            logger.info(f"✓ Process bound to CPU cores: {actual_cpus}")
            return actual_cpus

        except ImportError:
            logger.warning(
                "psutil not installed, cannot set CPU affinity. "
                "Install with: pip install psutil"
            )
            return []
        except Exception as e:
            logger.warning(f"Failed to set CPU affinity: {e}")
            return []

    else:
        logger.warning(f"CPU affinity not supported on {sys.platform}")
        return []


# ============================================================================
# Layer 4: Subprocess Sandboxing
# ============================================================================

def create_cpu_only_subprocess_env() -> Dict[str, str]:
    """
    Create environment dict for CPU-only subprocesses.

    Returns a copy of current environment with CPU-only enforcement applied.
    Use this when spawning subprocesses to ensure they inherit isolation.

    Returns:
        Dict suitable for subprocess.run(..., env=<this dict>)

    Example:
        >>> env = create_cpu_only_subprocess_env()
        >>> subprocess.run(['python', 'script.py'], env=env)
    """
    # Start with current environment
    env = os.environ.copy()

    # Apply CPU-only overrides
    env.update({
        'CUDA_VISIBLE_DEVICES': '',
        'TORCH_DEVICE': 'cpu',
        'JAX_PLATFORMS': 'cpu',
        'OMP_NUM_THREADS': '8',
        'MKL_NUM_THREADS': '8',
        'OPENBLAS_NUM_THREADS': '8',
        'HF_HOME': '/mnt/c/ai_cache/huggingface',
        'TRANSFORMERS_CACHE': '/mnt/c/ai_cache/huggingface',
        'TORCH_HOME': '/mnt/c/ai_cache/torch',
        'XDG_CACHE_HOME': '/mnt/c/ai_cache',
    })

    return env


def run_cpu_only_subprocess(
    cmd: List[str],
    check: bool = True,
    capture_output: bool = False,
    **kwargs
) -> subprocess.CompletedProcess:
    """
    Run subprocess with CPU-only environment enforced.

    Args:
        cmd: Command and arguments (e.g., ['python', 'script.py'])
        check: Raise CalledProcessError on non-zero exit (default True)
        capture_output: Capture stdout/stderr (default False)
        **kwargs: Additional arguments passed to subprocess.run

    Returns:
        CompletedProcess instance

    Example:
        >>> result = run_cpu_only_subprocess(
        ...     ['python', '-c', 'import torch; print(torch.cuda.is_available())'],
        ...     capture_output=True
        ... )
        >>> print(result.stdout)  # Should print "False"
    """
    # Ensure env is CPU-only
    env = kwargs.pop('env', None)
    if env is None:
        env = create_cpu_only_subprocess_env()
    else:
        # User provided env, overlay CPU-only settings
        cpu_env = create_cpu_only_subprocess_env()
        env = {**env, **cpu_env}  # CPU-only settings take precedence

    # Run subprocess
    return subprocess.run(cmd, env=env, check=check, capture_output=capture_output, **kwargs)


# ============================================================================
# Verification Utilities
# ============================================================================

def get_gpu_status() -> Dict[str, any]:
    """
    Get current GPU status for debugging.

    Returns dict with keys:
      - cuda_visible_devices: Value of CUDA_VISIBLE_DEVICES env var
      - torch_device: Value of TORCH_DEVICE env var
      - torch_available: Whether PyTorch is importable
      - torch_cuda_available: Whether torch.cuda.is_available() (if imported)
      - nvidia_smi_available: Whether nvidia-smi command works

    Returns:
        Dict with GPU status information
    """
    status = {
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES', '<not set>'),
        'torch_device': os.environ.get('TORCH_DEVICE', '<not set>'),
        'torch_available': False,
        'torch_cuda_available': None,
        'nvidia_smi_available': False,
    }

    # Check PyTorch
    try:
        import torch
        status['torch_available'] = True
        status['torch_cuda_available'] = torch.cuda.is_available()
        if status['torch_cuda_available']:
            try:
                status['torch_cuda_device_count'] = torch.cuda.device_count()
            except:
                pass
    except ImportError:
        pass

    # Check nvidia-smi
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            status['nvidia_smi_available'] = True
            status['nvidia_smi_gpus'] = result.stdout.strip().split('\n')
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    return status


def print_gpu_status() -> None:
    """
    Print formatted GPU status for debugging.

    Example:
        >>> print_gpu_status()
        GPU Status Report:
          CUDA_VISIBLE_DEVICES: ''
          TORCH_DEVICE: 'cpu'
          PyTorch available: True
          PyTorch CUDA available: False
          nvidia-smi available: True
          GPUs detected by nvidia-smi: ['NVIDIA GeForce RTX 4090']
    """
    status = get_gpu_status()

    print("GPU Status Report:")
    print(f"  CUDA_VISIBLE_DEVICES: '{status['cuda_visible_devices']}'")
    print(f"  TORCH_DEVICE: '{status['torch_device']}'")
    print(f"  PyTorch available: {status['torch_available']}")
    print(f"  PyTorch CUDA available: {status['torch_cuda_available']}")
    print(f"  nvidia-smi available: {status['nvidia_smi_available']}")

    if status['nvidia_smi_available']:
        gpus = status.get('nvidia_smi_gpus', [])
        print(f"  GPUs detected by nvidia-smi: {gpus}")


# ============================================================================
# Main Entry Point (for testing)
# ============================================================================

if __name__ == '__main__':
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("=" * 80)
    print("GPU Isolation Module - Self Test")
    print("=" * 80)

    # Test 1: Enforce CPU-only
    print("\n[Test 1] Enforcing CPU-only environment...")
    previous_env = enforce_cpu_only()
    print("✓ Success")

    # Test 2: Verify no GPU usage
    print("\n[Test 2] Verifying no GPU usage...")
    is_safe, msg = verify_no_gpu_usage(raise_on_violation=False)
    if is_safe:
        print(f"✓ {msg}")
    else:
        print(f"✗ {msg}")

    # Test 3: Safe PyTorch import
    print("\n[Test 3] Importing PyTorch safely...")
    try:
        torch = safe_import_torch()
        print(f"✓ PyTorch version {torch.__version__} imported")
        print(f"  CUDA available: {torch.cuda.is_available()}")
    except ImportError as e:
        print(f"⚠ PyTorch not installed: {e}")
    except GPUIsolationError as e:
        print(f"✗ GPU detected: {e}")

    # Test 4: CPU affinity
    print("\n[Test 4] Setting CPU affinity...")
    cpus = set_cpu_affinity()
    if cpus:
        print(f"✓ Bound to CPUs: {cpus}")
    else:
        print("⚠ CPU affinity not set (may not be supported)")

    # Test 5: Subprocess isolation
    print("\n[Test 5] Testing subprocess isolation...")
    try:
        result = run_cpu_only_subprocess(
            ['python', '-c', 'import os; print(os.environ.get("CUDA_VISIBLE_DEVICES"))'],
            capture_output=True,
            text=True
        )
        cuda_devices = result.stdout.strip()
        if cuda_devices == '':
            print(f"✓ Subprocess has CUDA_VISIBLE_DEVICES='' (empty)")
        else:
            print(f"✗ Subprocess has CUDA_VISIBLE_DEVICES='{cuda_devices}' (expected empty)")
    except Exception as e:
        print(f"✗ Subprocess test failed: {e}")

    # Test 6: GPU status report
    print("\n[Test 6] GPU status report:")
    print_gpu_status()

    print("\n" + "=" * 80)
    print("Self-test complete")
    print("=" * 80)

"""
BashRunnerAdapter: Wraps bash scripts as executable tasks for Web UI.

This adapter executes existing batch scripts via async subprocess,
parses their JSON output, and emits progress events.

Design:
- Async subprocess execution via asyncio
- Real-time stdout/stderr streaming
- Progress event parsing and emission
- JSON output parsing from scripts
- Error handling and retry support
"""
import asyncio
import json
import os
import re
import signal
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class BashScriptConfig:
    """Configuration for bash script execution."""
    script_path: str
    args: List[str]
    env: Optional[Dict[str, str]] = None
    timeout: Optional[int] = None  # seconds
    parse_json_output: bool = True
    emit_progress_events: bool = True
    log_dir: Optional[str] = None


@dataclass
class ScriptResult:
    """Result from bash script execution."""
    success: bool
    return_code: int
    outputs: Dict[str, Any]  # Parsed outputs (e.g., JSON files)
    stdout_lines: List[str]
    stderr_lines: List[str]
    error: Optional[str] = None
    execution_time: float = 0.0
    stdout_log_path: Optional[str] = None
    stderr_log_path: Optional[str] = None


class BashRunnerAdapter:
    """
    Adapter to execute bash scripts asynchronously.

    Features:
    - Async subprocess execution
    - Real-time stdout/stderr streaming
    - Progress event callbacks
    - JSON output parsing
    - Conda environment support
    """

    def __init__(
        self,
        job_id: str,
        progress_callback: Optional[Callable] = None,
        conda_env: str = "ai_env"
    ):
        """
        Initialize adapter.

        Args:
            job_id: Unique job identifier
            progress_callback: Async callback for progress events
            conda_env: Conda environment name (default: ai_env)
        """
        self.job_id = job_id
        self.progress_callback = progress_callback
        self.conda_env = conda_env
        self.process: Optional[asyncio.subprocess.Process] = None

    async def execute(self, config: BashScriptConfig) -> ScriptResult:
        """
        Execute bash script and return structured result.

        Args:
            config: Script execution configuration

        Returns:
            ScriptResult with parsed outputs and metrics
        """
        script_path = Path(config.script_path)
        if not script_path.exists():
            return ScriptResult(
                success=False,
                return_code=-1,
                outputs={},
                stdout_lines=[],
                stderr_lines=[],
                error=f"Script not found: {config.script_path}"
            )

        # Build command
        cmd = ["bash", str(script_path)] + config.args

        # Setup environment (include conda env in PATH)
        env = self._build_environment(config.env or {})

        # Setup log files (optional)
        stdout_log_path: Optional[str] = None
        stderr_log_path: Optional[str] = None
        stdout_fh = None
        stderr_fh = None
        if config.log_dir:
            log_dir = Path(config.log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            stdout_log_path = str(log_dir / "stdout.log")
            stderr_log_path = str(log_dir / "stderr.log")
            stdout_fh = open(stdout_log_path, "a", encoding="utf-8")
            stderr_fh = open(stderr_log_path, "a", encoding="utf-8")

        # Execute subprocess
        start_time = asyncio.get_event_loop().time()

        try:
            self.process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=script_path.parent,
                start_new_session=True,  # allow killing the full process group on cancel
            )

            # Stream output with progress parsing
            stdout_lines = []
            stderr_lines = []

            async def read_stream(stream, line_buffer, is_stderr=False, log_fh=None):
                """Read stream line by line and emit progress events."""
                while True:
                    line_bytes = await stream.readline()
                    if not line_bytes:
                        break

                    try:
                        line = line_bytes.decode('utf-8').strip()
                    except UnicodeDecodeError:
                        line = line_bytes.decode('utf-8', errors='ignore').strip()

                    line_buffer.append(line)
                    if log_fh is not None:
                        try:
                            log_fh.write(line + "\n")
                            log_fh.flush()
                        except Exception:
                            pass

                    # Emit progress events if enabled
                    if config.emit_progress_events and not is_stderr:
                        await self._parse_and_emit_progress(line)

            # Read stdout and stderr concurrently
            await asyncio.gather(
                read_stream(self.process.stdout, stdout_lines, log_fh=stdout_fh),
                read_stream(self.process.stderr, stderr_lines, is_stderr=True, log_fh=stderr_fh)
            )

            # Wait for completion with timeout
            try:
                if config.timeout:
                    return_code = await asyncio.wait_for(
                        self.process.wait(),
                        timeout=config.timeout
                    )
                else:
                    return_code = await self.process.wait()
            except asyncio.TimeoutError:
                self.process.kill()
                await self.process.wait()
                return ScriptResult(
                    success=False,
                    return_code=-1,
                    outputs={},
                    stdout_lines=stdout_lines,
                    stderr_lines=stderr_lines,
                    error=f"Script execution timeout after {config.timeout}s",
                    stdout_log_path=stdout_log_path,
                    stderr_log_path=stderr_log_path,
                )

            end_time = asyncio.get_event_loop().time()
            execution_time = end_time - start_time

            # Parse result
            if return_code == 0:
                outputs = self._parse_outputs(stdout_lines, config) if config.parse_json_output else {}
                return ScriptResult(
                    success=True,
                    return_code=0,
                    outputs=outputs,
                    stdout_lines=stdout_lines,
                    stderr_lines=stderr_lines,
                    execution_time=execution_time,
                    stdout_log_path=stdout_log_path,
                    stderr_log_path=stderr_log_path,
                )
            else:
                error_msg = "\n".join(stderr_lines[-10:]) if stderr_lines else "Unknown error"
                return ScriptResult(
                    success=False,
                    return_code=return_code,
                    outputs={},
                    stdout_lines=stdout_lines,
                    stderr_lines=stderr_lines,
                    error=f"Script failed with code {return_code}: {error_msg}",
                    execution_time=execution_time,
                    stdout_log_path=stdout_log_path,
                    stderr_log_path=stderr_log_path,
                )

        except Exception as e:
            logger.exception(f"Error executing script {config.script_path}")
            return ScriptResult(
                success=False,
                return_code=-1,
                outputs={},
                stdout_lines=[],
                stderr_lines=[],
                error=f"Execution error: {str(e)}",
                stdout_log_path=stdout_log_path,
                stderr_log_path=stderr_log_path,
            )
        finally:
            if stdout_fh is not None:
                try:
                    stdout_fh.close()
                except Exception:
                    pass
            if stderr_fh is not None:
                try:
                    stderr_fh.close()
                except Exception:
                    pass

    def _build_environment(self, extra_env: Dict[str, str]) -> Dict[str, str]:
        """
        Build execution environment with conda support.

        Args:
            extra_env: Additional environment variables

        Returns:
            Complete environment dict
        """
        env = os.environ.copy()

        # Add conda environment to PATH (prepend to ensure it takes precedence)
        conda_bin = f"/home/justin/miniconda3/envs/{self.conda_env}/bin"
        conda_lib = f"/home/justin/miniconda3/envs/{self.conda_env}/lib"

        env['PATH'] = f"{conda_bin}:{env.get('PATH', '/usr/bin:/bin')}"

        # Set conda environment variables
        env['CONDA_DEFAULT_ENV'] = self.conda_env
        env['CONDA_PREFIX'] = f"/home/justin/miniconda3/envs/{self.conda_env}"

        # Set Python-related environment variables
        env['PYTHONPATH'] = f"/mnt/c/ai_projects/animation-ai-studio:{env.get('PYTHONPATH', '')}"

        # Set library path for shared libraries
        env['LD_LIBRARY_PATH'] = f"{conda_lib}:{env.get('LD_LIBRARY_PATH', '')}"

        # Add extra environment variables from config
        env.update(extra_env)

        return env

    def _parse_outputs(self, stdout_lines: List[str], config: BashScriptConfig) -> Dict[str, Any]:
        """
        Parse script output to extract structured data.

        Existing bash scripts output JSON summaries like:
        - "Summary: /path/to/stage1_summary.json"
        - "Output: /path/to/output_dir"

        Args:
            stdout_lines: stdout lines from script
            config: Script configuration

        Returns:
            Dictionary of parsed outputs
        """
        outputs = {}

        # Look for JSON file paths in output
        for line in stdout_lines:
            # Pattern: "Summary: /path/to/summary.json"
            match = re.search(r'Summary:\s+(.+\.json)', line)
            if match:
                json_path = Path(match.group(1).strip())
                if json_path.exists():
                    try:
                        with open(json_path, 'r') as f:
                            data = json.load(f)
                            outputs[json_path.stem] = data
                            logger.info(f"Parsed JSON output: {json_path.name}")
                    except Exception as e:
                        logger.warning(f"Failed to parse JSON {json_path}: {e}")
                        outputs['parse_error'] = str(e)

        # Also capture output directories
        for line in stdout_lines:
            if "Output:" in line or "output_dir:" in line:
                match = re.search(r'(?:Output:|output_dir:)\s+(.+)$', line)
                if match:
                    output_dir = match.group(1).strip()
                    outputs['output_dir'] = output_dir
                    logger.info(f"Captured output directory: {output_dir}")

        return outputs

    async def _parse_and_emit_progress(self, line: str):
        """
        Parse progress information from stdout and emit events.

        Existing bash scripts output progress like:
        - "[INFO] Processing stage 1/3..."
        - "[✓] Frame extraction complete (1234 frames)"
        - "[GPU] VRAM: 8500MB / 16384MB (51.9%)"

        Args:
            line: stdout line to parse
        """
        if not self.progress_callback:
            return

        # Stage progress pattern: "Processing stage 1/3"
        stage_match = re.search(r'Processing stage (\d+)/(\d+)', line, re.IGNORECASE)
        if stage_match:
            current = int(stage_match.group(1))
            total = int(stage_match.group(2))
            progress = current / total

            await self.progress_callback({
                'type': 'progress',
                'job_id': self.job_id,
                'progress': progress,
                'stage': f"{current}/{total}",
                'message': line
            })
            return

        # Completion markers: "[✓]" or "complete"
        if "[✓]" in line or "complete" in line.lower():
            await self.progress_callback({
                'type': 'stage_complete',
                'job_id': self.job_id,
                'message': line
            })
            return

        # GPU metrics: "VRAM: 8500MB / 16384MB"
        gpu_match = re.search(r'VRAM:\s+(\d+)MB\s+/\s+(\d+)MB', line)
        if gpu_match:
            used = int(gpu_match.group(1))
            total = int(gpu_match.group(2))
            await self.progress_callback({
                'type': 'gpu_metrics',
                'job_id': self.job_id,
                'vram_used_mb': used,
                'vram_total_mb': total
            })
            return

    async def cancel(self):
        """Cancel running process."""
        if self.process and self.process.returncode is None:
            logger.info(f"Cancelling job {self.job_id}")
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
            except Exception:
                self.process.terminate()
            await asyncio.sleep(2)
            if self.process.returncode is None:
                try:
                    os.killpg(self.process.pid, signal.SIGKILL)
                except Exception:
                    self.process.kill()
                await self.process.wait()

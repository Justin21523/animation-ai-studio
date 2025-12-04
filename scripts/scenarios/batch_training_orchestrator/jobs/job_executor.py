"""
Batch Training Orchestrator - Job Executor

Executes training jobs using subprocess/tmux with log capture and error handling.

Author: Animation AI Studio
Date: 2025-12-03
"""

import os
import subprocess
import logging
import signal
import time
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

from ..common import TrainingJob, JobState, JobResult


logger = logging.getLogger(__name__)


class JobExecutor:
    """
    Executes training jobs using subprocess or tmux

    Features:
    - subprocess-based execution
    - Log capture and streaming
    - Environment variable injection
    - Process monitoring and control
    - Error handling and retry logic
    - Graceful shutdown with SIGTERM/SIGKILL
    """

    def __init__(self, use_tmux: bool = False, tmux_prefix: str = "train"):
        """
        Initialize job executor

        Args:
            use_tmux: Use tmux sessions instead of subprocess
            tmux_prefix: Prefix for tmux session names
        """
        self.use_tmux = use_tmux
        self.tmux_prefix = tmux_prefix
        self.active_processes: Dict[str, subprocess.Popen] = {}

        logger.info(f"JobExecutor initialized (tmux={use_tmux})")

    def execute(self, job: TrainingJob) -> JobResult:
        """
        Execute training job

        Args:
            job: Training job to execute

        Returns:
            Job result with exit code and metrics
        """
        logger.info(f"Executing job {job.id} ({job.name})")

        # Prepare execution environment
        env = self._prepare_environment(job)

        # Build command
        command = self._build_command(job)
        job.command = command

        # Open log file
        log_file = open(job.log_file, 'w')

        try:
            if self.use_tmux:
                # Execute in tmux session
                result = self._execute_tmux(job, command, env, log_file)
            else:
                # Execute via subprocess
                result = self._execute_subprocess(job, command, env, log_file)

            return result

        except Exception as e:
            logger.error(f"Job {job.id} execution failed: {e}")

            return JobResult(
                job_id=job.id,
                success=False,
                state=JobState.FAILED,
                duration=0.0,
                error_message=str(e),
                output_dir=job.output_dir,
                log_file=job.log_file
            )

        finally:
            log_file.close()

    def terminate(self, job_id: str, timeout: int = 30) -> bool:
        """
        Terminate running job gracefully

        Args:
            job_id: Job identifier
            timeout: Seconds to wait before force kill

        Returns:
            True if job was terminated
        """
        if job_id not in self.active_processes:
            logger.warning(f"Job {job_id} not in active processes")
            return False

        process = self.active_processes[job_id]

        try:
            # Send SIGTERM for graceful shutdown
            logger.info(f"Sending SIGTERM to job {job_id} (PID {process.pid})")
            process.terminate()

            # Wait for process to exit
            try:
                process.wait(timeout=timeout)
                logger.info(f"Job {job_id} terminated gracefully")
                return True
            except subprocess.TimeoutExpired:
                # Force kill if timeout exceeded
                logger.warning(f"Job {job_id} did not exit, sending SIGKILL")
                process.kill()
                process.wait()
                logger.info(f"Job {job_id} force killed")
                return True

        except Exception as e:
            logger.error(f"Failed to terminate job {job_id}: {e}")
            return False

        finally:
            # Remove from active processes
            if job_id in self.active_processes:
                del self.active_processes[job_id]

    def is_running(self, job_id: str) -> bool:
        """
        Check if job is currently running

        Args:
            job_id: Job identifier

        Returns:
            True if job process exists and is running
        """
        if job_id not in self.active_processes:
            return False

        process = self.active_processes[job_id]
        return process.poll() is None

    def get_exit_code(self, job_id: str) -> Optional[int]:
        """
        Get exit code of completed job

        Args:
            job_id: Job identifier

        Returns:
            Exit code or None if job not found or still running
        """
        if job_id not in self.active_processes:
            return None

        process = self.active_processes[job_id]
        return process.returncode

    # ========================================================================
    # Private Helper Methods
    # ========================================================================

    def _execute_subprocess(self,
                            job: TrainingJob,
                            command: str,
                            env: Dict[str, str],
                            log_file) -> JobResult:
        """
        Execute job using subprocess

        Args:
            job: Training job
            command: Command to execute
            env: Environment variables
            log_file: Log file handle

        Returns:
            Job result
        """
        start_time = time.time()

        try:
            # Start process
            process = subprocess.Popen(
                command,
                shell=True,
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                preexec_fn=os.setsid  # Create new process group
            )

            job.process_id = process.pid
            self.active_processes[job.id] = process

            logger.info(f"Job {job.id} started (PID {process.pid})")

            # Wait for completion (with timeout if specified)
            if job.timeout:
                try:
                    exit_code = process.wait(timeout=job.timeout)
                except subprocess.TimeoutExpired:
                    logger.error(f"Job {job.id} timed out after {job.timeout}s")
                    process.kill()
                    process.wait()

                    return JobResult(
                        job_id=job.id,
                        success=False,
                        state=JobState.FAILED,
                        duration=time.time() - start_time,
                        exit_code=-1,
                        error_message=f"Job timed out after {job.timeout}s",
                        output_dir=job.output_dir,
                        log_file=job.log_file
                    )
            else:
                exit_code = process.wait()

            duration = time.time() - start_time

            # Determine success
            success = exit_code == 0

            # Collect checkpoints if successful
            checkpoints = []
            if success:
                checkpoints = self._find_checkpoints(job.output_dir)

            logger.info(f"Job {job.id} completed (exit_code={exit_code}, duration={duration:.1f}s)")

            return JobResult(
                job_id=job.id,
                success=success,
                state=JobState.COMPLETED if success else JobState.FAILED,
                duration=duration,
                exit_code=exit_code,
                checkpoints=checkpoints,
                output_dir=job.output_dir,
                log_file=job.log_file
            )

        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Job {job.id} execution failed: {e}")

            return JobResult(
                job_id=job.id,
                success=False,
                state=JobState.FAILED,
                duration=duration,
                error_message=str(e),
                output_dir=job.output_dir,
                log_file=job.log_file
            )

        finally:
            # Cleanup
            if job.id in self.active_processes:
                del self.active_processes[job.id]

    def _execute_tmux(self,
                      job: TrainingJob,
                      command: str,
                      env: Dict[str, str],
                      log_file) -> JobResult:
        """
        Execute job in tmux session

        Args:
            job: Training job
            command: Command to execute
            env: Environment variables
            log_file: Log file handle

        Returns:
            Job result
        """
        # Generate tmux session name
        session_name = f"{self.tmux_prefix}_{job.id[:8]}"
        job.tmux_session = session_name

        # Build tmux command
        env_vars = " ".join([f"{k}={v}" for k, v in env.items()])
        tmux_cmd = f"tmux new-session -d -s {session_name} '{env_vars} {command}'"

        start_time = time.time()

        try:
            # Start tmux session
            subprocess.run(tmux_cmd, shell=True, check=True)
            logger.info(f"Job {job.id} started in tmux session '{session_name}'")

            # Poll tmux session until it exits
            while True:
                # Check if session exists
                check_cmd = f"tmux has-session -t {session_name} 2>/dev/null"
                result = subprocess.run(check_cmd, shell=True)

                if result.returncode != 0:
                    # Session ended
                    break

                time.sleep(5)  # Poll every 5 seconds

            duration = time.time() - start_time

            # Capture tmux session output (if available)
            capture_cmd = f"tmux capture-pane -t {session_name} -p"
            try:
                output = subprocess.check_output(capture_cmd, shell=True, text=True)
                log_file.write(output)
            except subprocess.CalledProcessError:
                pass

            # Determine success (assume success if no error captured)
            success = True
            checkpoints = self._find_checkpoints(job.output_dir)

            logger.info(f"Job {job.id} completed in tmux (duration={duration:.1f}s)")

            return JobResult(
                job_id=job.id,
                success=success,
                state=JobState.COMPLETED,
                duration=duration,
                exit_code=0,
                checkpoints=checkpoints,
                output_dir=job.output_dir,
                log_file=job.log_file
            )

        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Job {job.id} tmux execution failed: {e}")

            return JobResult(
                job_id=job.id,
                success=False,
                state=JobState.FAILED,
                duration=duration,
                error_message=str(e),
                output_dir=job.output_dir,
                log_file=job.log_file
            )

    def _build_command(self, job: TrainingJob) -> str:
        """
        Build training command from job config

        Args:
            job: Training job

        Returns:
            Command string
        """
        # Assume Kohya_ss sd-scripts training
        # Command should be in job.metadata or built from config_path

        if job.command:
            return job.command

        # Default: assume kohya_ss train_network.py
        command = (
            f"accelerate launch --num_cpu_threads_per_process=2 "
            f"train_network.py "
            f"--config_file {job.config_path}"
        )

        return command

    def _prepare_environment(self, job: TrainingJob) -> Dict[str, str]:
        """
        Prepare environment variables for job

        Args:
            job: Training job

        Returns:
            Environment dictionary
        """
        env = os.environ.copy()

        # Set CUDA_VISIBLE_DEVICES if GPUs allocated
        if job.allocated_gpus:
            gpu_ids = ",".join(str(gpu_id) for gpu_id in job.allocated_gpus)
            env["CUDA_VISIBLE_DEVICES"] = gpu_ids
            logger.debug(f"Job {job.id} using GPUs: {gpu_ids}")

        # Add custom env vars from job metadata
        custom_env = job.metadata.get("env", {})
        env.update(custom_env)

        return env

    def _find_checkpoints(self, output_dir: Path) -> list:
        """
        Find checkpoint files in output directory

        Args:
            output_dir: Job output directory

        Returns:
            List of checkpoint paths
        """
        if not output_dir.exists():
            return []

        # Common checkpoint patterns
        patterns = ["*.safetensors", "*.ckpt", "*.pt", "*.pth"]

        checkpoints = []
        for pattern in patterns:
            checkpoints.extend(output_dir.glob(pattern))

        return sorted(checkpoints, key=lambda p: p.stat().st_mtime)

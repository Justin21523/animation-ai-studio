"""
Batch Training Orchestrator - CLI

Command-line interface for batch training orchestration.

Author: Animation AI Studio
Date: 2025-12-03

Usage:
    python -m scripts.scenarios.batch_training_orchestrator submit --config training.toml --name my_job
    python -m scripts.scenarios.batch_training_orchestrator list
    python -m scripts.scenarios.batch_training_orchestrator status JOB_ID
    python -m scripts.scenarios.batch_training_orchestrator cancel JOB_ID
    python -m scripts.scenarios.batch_training_orchestrator stats
"""

import argparse
import logging
import sys
from pathlib import Path

from .batch_training_orchestrator import BatchTrainingOrchestrator
from .common import JobState, ResourceRequirements, OrchestratorConfig, format_duration, format_memory


logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def cmd_submit(args):
    """Submit training job"""
    orchestrator = BatchTrainingOrchestrator()

    # Create resource requirements
    requirements = ResourceRequirements(
        gpu_count=args.gpu_count,
        gpu_memory=args.gpu_memory,
        cpu_cores=args.cpu_cores,
        system_memory=args.system_memory
    )

    # Submit job
    job_id = orchestrator.submit_job_from_config(
        name=args.name,
        config_path=Path(args.config),
        output_dir=Path(args.output),
        priority=args.priority,
        requirements=requirements
    )

    print(f"Job submitted: {job_id}")
    print(f"Name: {args.name}")
    print(f"Config: {args.config}")
    print(f"Priority: {args.priority}")


def cmd_list(args):
    """List jobs"""
    orchestrator = BatchTrainingOrchestrator()

    # Filter by state if specified
    state = JobState(args.state) if args.state else None

    jobs = orchestrator.list_jobs(state=state)

    if not jobs:
        print("No jobs found")
        return

    # Print header
    print(f"{'Job ID':<20} {'Name':<30} {'State':<12} {'Priority':<10} {'Duration':<15}")
    print("-" * 90)

    # Print jobs
    for job in jobs:
        duration_str = format_duration(job["duration"]) if job["duration"] else "N/A"
        print(f"{job['job_id']:<20} {job['name']:<30} {job['state']:<12} {job['priority']:<10} {duration_str:<15}")


def cmd_status(args):
    """Show job status"""
    orchestrator = BatchTrainingOrchestrator()

    status = orchestrator.get_job_status(args.job_id)

    if not status:
        print(f"Job not found: {args.job_id}")
        return

    # Print status
    print(f"Job ID: {status['job_id']}")
    print(f"Name: {status['name']}")
    print(f"State: {status['state']}")
    print(f"Priority: {status['priority']}")
    print(f"Allocated GPUs: {status['allocated_gpus']}")

    if status['started_at']:
        print(f"Started: {status['started_at']}")

    if status['duration']:
        print(f"Duration: {format_duration(status['duration'])}")

    # Print progress if available
    progress = status.get('progress', {})
    if progress:
        print("\nProgress:")
        if progress.get('current_epoch'):
            print(f"  Epoch: {progress['current_epoch']}/{progress.get('total_epochs', '?')}")
        if progress.get('loss'):
            print(f"  Loss: {progress['loss']:.6f}")
        if progress.get('progress_percent'):
            print(f"  Progress: {progress['progress_percent']:.1f}%")

    # Print error if failed
    if status['error_message']:
        print(f"\nError: {status['error_message']}")


def cmd_cancel(args):
    """Cancel job"""
    orchestrator = BatchTrainingOrchestrator()

    success = orchestrator.cancel_job(args.job_id)

    if success:
        print(f"Job cancelled: {args.job_id}")
    else:
        print(f"Failed to cancel job: {args.job_id}")


def cmd_stats(args):
    """Show statistics"""
    orchestrator = BatchTrainingOrchestrator()

    stats = orchestrator.get_statistics()

    # Print job statistics
    print("Job Statistics:")
    print(f"  Total jobs: {stats['jobs']['total_jobs']}")
    print(f"  Pending: {stats['jobs']['pending_jobs']}")
    print(f"  Queued: {stats['jobs']['queued_jobs']}")
    print(f"  Running: {stats['jobs']['running_jobs']}")
    print(f"  Completed: {stats['jobs']['completed_jobs']}")
    print(f"  Failed: {stats['jobs']['failed_jobs']}")
    print(f"  Success rate: {stats['jobs']['success_rate']*100:.1f}%")

    # Print resource statistics
    print("\nResource Statistics:")
    print(f"  Total GPUs: {stats['resources']['total_gpus']}")
    print(f"  Available GPUs: {stats['resources']['available_gpus']}")
    print(f"  Total Memory: {format_memory(stats['resources']['total_memory_mb'])}")
    print(f"  Available Memory: {format_memory(stats['resources']['available_memory_mb'])}")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Batch Training Orchestrator - Distributed LoRA Training Management",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Submit command
    submit_parser = subparsers.add_parser("submit", help="Submit training job")
    submit_parser.add_argument("--config", required=True, help="Path to training config (TOML)")
    submit_parser.add_argument("--name", required=True, help="Job name")
    submit_parser.add_argument("--output", required=True, help="Output directory for checkpoints")
    submit_parser.add_argument("--priority", type=int, default=5, help="Job priority (0-20, default: 5)")
    submit_parser.add_argument("--gpu-count", type=int, default=1, help="Number of GPUs (default: 1)")
    submit_parser.add_argument("--gpu-memory", type=int, default=16384, help="GPU memory (MB, default: 16384)")
    submit_parser.add_argument("--cpu-cores", type=int, default=4, help="CPU cores (default: 4)")
    submit_parser.add_argument("--system-memory", type=int, default=32768, help="System memory (MB, default: 32768)")
    submit_parser.set_defaults(func=cmd_submit)

    # List command
    list_parser = subparsers.add_parser("list", help="List jobs")
    list_parser.add_argument("--state", help="Filter by state (pending, queued, running, completed, failed)")
    list_parser.set_defaults(func=cmd_list)

    # Status command
    status_parser = subparsers.add_parser("status", help="Show job status")
    status_parser.add_argument("job_id", help="Job ID")
    status_parser.set_defaults(func=cmd_status)

    # Cancel command
    cancel_parser = subparsers.add_parser("cancel", help="Cancel job")
    cancel_parser.add_argument("job_id", help="Job ID")
    cancel_parser.set_defaults(func=cmd_cancel)

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show statistics")
    stats_parser.set_defaults(func=cmd_stats)

    # Parse arguments
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Execute command
    try:
        args.func(args)
    except Exception as e:
        logger.error(f"Command failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

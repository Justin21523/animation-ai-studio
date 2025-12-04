"""
Data Pipeline Automation - CLI Interface

Command-line interface for pipeline operations.

Author: Animation AI Studio
Date: 2025-12-03
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from .common import OrchestratorConfig, PipelineState
from .orchestrator import PipelineOrchestrator
from .pipeline_builder import PipelineBuilder

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_orchestrator(
    checkpoint_dir: Path,
    max_parallel: int = 4,
    enable_checkpoints: bool = True
) -> PipelineOrchestrator:
    """
    Create pipeline orchestrator

    Args:
        checkpoint_dir: Checkpoint directory
        max_parallel: Maximum parallel stages
        enable_checkpoints: Enable checkpoint saving

    Returns:
        Configured orchestrator
    """
    config = OrchestratorConfig(
        max_parallel_stages=max_parallel,
        checkpoint_dir=checkpoint_dir,
        enable_checkpoints=enable_checkpoints,
        retry_failed_stages=False,
        log_dir=Path("logs"),
    )

    orchestrator = PipelineOrchestrator(config)
    return orchestrator


def cmd_run(args):
    """
    Run pipeline from YAML configuration

    Args:
        args: Command arguments
    """
    logger.info(f"Running pipeline from: {args.config}")

    # Create builder
    builder = PipelineBuilder()

    # Load pipeline
    try:
        pipeline = builder.load_from_yaml(Path(args.config))
        logger.info(f"Loaded pipeline: '{pipeline.name}' (ID: {pipeline.id})")

    except Exception as e:
        logger.error(f"Failed to load pipeline: {e}")
        sys.exit(1)

    # Validate
    try:
        builder.validate_pipeline(pipeline)
        logger.info("Pipeline validation successful")

    except Exception as e:
        logger.error(f"Pipeline validation failed: {e}")
        sys.exit(1)

    # Build execution plan
    try:
        execution_plan = builder.build_execution_plan(pipeline)
        logger.info(f"Execution plan: {len(execution_plan)} stage groups")

        for i, group in enumerate(execution_plan):
            logger.info(f"  Group {i + 1}: {group}")

    except Exception as e:
        logger.error(f"Failed to build execution plan: {e}")
        sys.exit(1)

    # Create orchestrator
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else Path("checkpoints")
    orchestrator = create_orchestrator(
        checkpoint_dir=checkpoint_dir,
        max_parallel=args.max_parallel,
        enable_checkpoints=not args.no_checkpoint
    )

    # Register stage executors
    from .stage_executors.frame_extraction import FrameExtractionExecutor
    from .stage_executors.segmentation import SegmentationExecutor
    from .stage_executors.clustering import ClusteringExecutor
    from .stage_executors.training_data_prep import TrainingDataPrepExecutor

    orchestrator.register_executor("frame_extraction", FrameExtractionExecutor)
    orchestrator.register_executor("segmentation", SegmentationExecutor)
    orchestrator.register_executor("clustering", ClusteringExecutor)
    orchestrator.register_executor("training_data_prep", TrainingDataPrepExecutor)

    logger.info("Registered 4 stage executors (frame_extraction, segmentation, clustering, training_data_prep)")

    # Dry run mode
    if args.dry_run:
        logger.info("DRY RUN - Pipeline would execute with configuration:")
        logger.info(f"  - Stages: {len(pipeline.stages)}")
        logger.info(f"  - Parallel groups: {len(execution_plan)}")
        logger.info(f"  - Max parallel: {args.max_parallel}")
        logger.info(f"  - Checkpoints: {not args.no_checkpoint}")
        return

    # Execute pipeline
    logger.info("Executing pipeline...")
    result = orchestrator.execute_pipeline(pipeline, resume=args.resume)

    if result.status == PipelineState.COMPLETED:
        logger.info(f"✓ Pipeline completed successfully in {result.total_duration:.2f}s")
        logger.info(f"  - Total stages: {len(pipeline.stages)}")
        logger.info(f"  - Successful: {len([s for s in pipeline.stages if s.is_completed])}")
        logger.info(f"  - Failed: {len([s for s in pipeline.stages if s.is_failed])}")
        logger.info(f"  - Final outputs: {result.final_outputs}")
    else:
        logger.error(f"✗ Pipeline failed: {result.error_message}")
        sys.exit(1)


def cmd_status(args):
    """
    Show pipeline status

    Args:
        args: Command arguments
    """
    logger.info(f"Checking status for pipeline: {args.pipeline_id}")

    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else Path("checkpoints")
    orchestrator = create_orchestrator(checkpoint_dir=checkpoint_dir)

    # Get status
    status = orchestrator.get_pipeline_status(args.pipeline_id)

    if not status:
        logger.warning(f"Pipeline not found or not running: {args.pipeline_id}")
        return

    # Display status
    print("\n" + "=" * 60)
    print(f"Pipeline: {status.get('pipeline_name', 'Unknown')}")
    print(f"ID: {status.get('pipeline_id', 'Unknown')}")
    print("=" * 60)
    print(f"State: {status.get('state', 'Unknown')}")
    print(f"Progress: {status.get('progress_percent', 0):.1f}%")
    print(f"Completed: {status.get('completed_stages', 0)}/{status.get('total_stages', 0)} stages")
    print(f"Elapsed: {status.get('elapsed_seconds', 0):.1f}s")
    print(f"ETA: {status.get('eta_seconds', 0):.1f}s")
    print("=" * 60 + "\n")


def cmd_list(args):
    """
    List pipelines

    Args:
        args: Command arguments
    """
    logger.info("Listing pipelines...")

    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else Path("checkpoints")

    if not checkpoint_dir.exists():
        logger.info("No checkpoint directory found")
        return

    # Find all checkpoints
    checkpoints = list(checkpoint_dir.glob("*_checkpoint_*.json"))

    if not checkpoints:
        logger.info("No pipelines found")
        return

    # Group by pipeline ID
    pipelines = {}
    for checkpoint in checkpoints:
        parts = checkpoint.stem.split('_checkpoint_')
        if len(parts) == 2:
            pipeline_id = parts[0]
            if pipeline_id not in pipelines:
                pipelines[pipeline_id] = []
            pipelines[pipeline_id].append(checkpoint)

    # Display
    print("\n" + "=" * 60)
    print(f"Found {len(pipelines)} pipeline(s)")
    print("=" * 60)

    for pipeline_id, checkpoints in pipelines.items():
        print(f"\nPipeline ID: {pipeline_id}")
        print(f"  Checkpoints: {len(checkpoints)}")
        latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
        print(f"  Latest: {latest.name}")

    print("\n")


def cmd_validate(args):
    """
    Validate pipeline configuration

    Args:
        args: Command arguments
    """
    logger.info(f"Validating pipeline: {args.config}")

    # Create builder
    builder = PipelineBuilder()

    # Load pipeline
    try:
        pipeline = builder.load_from_yaml(Path(args.config))
        logger.info(f"Loaded pipeline: '{pipeline.name}'")

    except Exception as e:
        logger.error(f"Failed to load pipeline: {e}")
        sys.exit(1)

    # Validate
    try:
        builder.validate_pipeline(pipeline)
        logger.info("✅ Pipeline validation successful")

    except Exception as e:
        logger.error(f"❌ Pipeline validation failed: {e}")
        sys.exit(1)

    # Build execution plan
    try:
        execution_plan = builder.build_execution_plan(pipeline)
        logger.info(f"✅ Execution plan: {len(execution_plan)} stage groups")

        print("\nExecution Plan:")
        print("=" * 60)
        for i, group in enumerate(execution_plan):
            print(f"Group {i + 1}: {', '.join(group)}")
        print("=" * 60)

    except Exception as e:
        logger.error(f"❌ Failed to build execution plan: {e}")
        sys.exit(1)

    # Show summary
    print("\nPipeline Summary:")
    print("=" * 60)
    print(f"Name: {pipeline.name}")
    print(f"Version: {pipeline.version}")
    print(f"Total Stages: {len(pipeline.stages)}")
    print(f"Parallel Groups: {len(execution_plan)}")
    print("\nStages:")
    for stage in pipeline.stages:
        deps = ", ".join(stage.depends_on) if stage.depends_on else "none"
        print(f"  - {stage.id} (type: {stage.type.value}, depends_on: {deps})")
    print("=" * 60 + "\n")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Data Pipeline Automation CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate pipeline configuration
  python -m scripts.scenarios.data_pipeline_automation validate --config pipeline.yaml

  # Run pipeline
  python -m scripts.scenarios.data_pipeline_automation run --config pipeline.yaml

  # Show pipeline status
  python -m scripts.scenarios.data_pipeline_automation status --pipeline-id PIPELINE_ID

  # List all pipelines
  python -m scripts.scenarios.data_pipeline_automation list
"""
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Run command
    parser_run = subparsers.add_parser("run", help="Run pipeline")
    parser_run.add_argument("--config", required=True, help="Pipeline YAML configuration")
    parser_run.add_argument("--checkpoint-dir", help="Checkpoint directory (default: checkpoints)")
    parser_run.add_argument("--max-parallel", type=int, default=4, help="Max parallel stages")
    parser_run.add_argument("--no-checkpoint", action="store_true", help="Disable checkpoints")
    parser_run.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser_run.add_argument("--dry-run", action="store_true", help="Validate pipeline without executing")
    parser_run.set_defaults(func=cmd_run)

    # Status command
    parser_status = subparsers.add_parser("status", help="Show pipeline status")
    parser_status.add_argument("--pipeline-id", required=True, help="Pipeline ID")
    parser_status.add_argument("--checkpoint-dir", help="Checkpoint directory")
    parser_status.set_defaults(func=cmd_status)

    # List command
    parser_list = subparsers.add_parser("list", help="List pipelines")
    parser_list.add_argument("--checkpoint-dir", help="Checkpoint directory")
    parser_list.set_defaults(func=cmd_list)

    # Validate command
    parser_validate = subparsers.add_parser("validate", help="Validate pipeline configuration")
    parser_validate.add_argument("--config", required=True, help="Pipeline YAML configuration")
    parser_validate.set_defaults(func=cmd_validate)

    # Parse arguments
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Execute command
    args.func(args)


if __name__ == "__main__":
    main()

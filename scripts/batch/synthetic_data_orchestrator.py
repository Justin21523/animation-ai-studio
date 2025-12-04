#!/usr/bin/env python3
"""
Synthetic Data Generation Pipeline Orchestrator

This Python script coordinates the entire synthetic data generation pipeline,
reading configuration from YAML and executing the workflow phases.

Author: Claude Code
Date: 2025-11-30
"""

import argparse
import json
import logging
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional

import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class PipelineConfig:
    """Pipeline configuration loaded from YAML"""
    workspace_root: Path
    base_model: Path
    identity_loras_dir: Path
    characters: List[str]
    lora_types: List[str]
    character_descriptions: Dict[str, str]

    # Vocabulary settings
    num_prompts_per_type: int
    use_templates: bool
    template_variations: int
    vocab_seed: Optional[int]

    # Image generation settings
    num_images_per_prompt: int
    num_inference_steps: int
    guidance_scale: float
    height: int
    width: int
    use_random_seeds: bool
    negative_prompt: str
    lora_scale: float
    device: str

    # Resilience settings
    max_retries: int
    retry_delay_seconds: int
    gpu_recovery_delay_seconds: int
    enable_checkpointing: bool
    checkpoint_filename: str

    # Logging
    log_level: str
    conda_env: str

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "PipelineConfig":
        """Load configuration from YAML file"""
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)

        workspace = config['workspace']
        models = config['models']
        vocab = config['vocabulary_generation']
        image_gen = config['image_generation']
        resilience = config['resilience']
        logging_cfg = config['logging']
        conda = config['conda']

        return cls(
            workspace_root=Path(workspace['root']),
            base_model=Path(models['base_model']),
            identity_loras_dir=Path(models['identity_loras_dir']),
            characters=config['characters'],
            lora_types=config['lora_types'],
            character_descriptions=config.get('character_descriptions', {}),

            num_prompts_per_type=vocab['num_prompts_per_type'],
            use_templates=vocab['use_templates'],
            template_variations=vocab['template_variations'],
            vocab_seed=vocab.get('seed'),

            num_images_per_prompt=image_gen['num_images_per_prompt'],
            num_inference_steps=image_gen['num_inference_steps'],
            guidance_scale=image_gen['guidance_scale'],
            height=image_gen['height'],
            width=image_gen['width'],
            use_random_seeds=image_gen['use_random_seeds'],
            negative_prompt=image_gen.get('negative_prompt', ''),
            lora_scale=image_gen['lora_scale'],
            device=image_gen['device'],

            max_retries=resilience['max_retries'],
            retry_delay_seconds=resilience['retry_delay_seconds'],
            gpu_recovery_delay_seconds=resilience['gpu_recovery_delay_seconds'],
            enable_checkpointing=resilience['enable_checkpointing'],
            checkpoint_filename=resilience['checkpoint_filename'],

            log_level=logging_cfg['log_level'],
            conda_env=conda['env_name']
        )


class CheckpointManager:
    """Manages pipeline checkpoint for resume capability"""

    def __init__(self, checkpoint_path: Path):
        self.checkpoint_path = checkpoint_path
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> Dict[str, Any]:
        """Load checkpoint data"""
        if self.checkpoint_path.exists():
            with open(self.checkpoint_path, 'r') as f:
                return json.load(f)
        return {}

    def is_completed(self, task_key: str) -> bool:
        """Check if a task is marked as completed"""
        data = self.load()
        return data.get(task_key) == "completed"

    def mark_completed(self, task_key: str):
        """Mark a task as completed"""
        data = self.load()
        data[task_key] = "completed"
        data[f"{task_key}_timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        with open(self.checkpoint_path, 'w') as f:
            json.dump(data, f, indent=2)

        logging.info(f"✓ Marked completed: {task_key}")


class SyntheticDataOrchestrator:
    """Main orchestrator for synthetic data generation pipeline"""

    def __init__(self, config: PipelineConfig, resume: bool = True, dry_run: bool = False):
        self.config = config
        self.resume = resume
        self.dry_run = dry_run

        # Setup directories
        self.config.workspace_root.mkdir(parents=True, exist_ok=True)
        (self.config.workspace_root / "logs").mkdir(exist_ok=True)
        (self.config.workspace_root / "checkpoints").mkdir(exist_ok=True)
        (self.config.workspace_root / "generated_data").mkdir(exist_ok=True)

        # Setup checkpoint manager
        checkpoint_path = self.config.workspace_root / "checkpoints" / self.config.checkpoint_filename
        self.checkpoint_mgr = CheckpointManager(checkpoint_path)

        # Setup logging
        log_format = '%(asctime)s - %(levelname)s - %(message)s'
        log_level = getattr(logging, self.config.log_level.upper())
        logging.basicConfig(level=log_level, format=log_format)

        self.stats = {
            "total_vocabularies": 0,
            "total_images": 0,
            "failed_tasks": []
        }

    def run_command(self, cmd: List[str], task_name: str, retries: int = 0) -> bool:
        """Execute a command with optional retry logic"""
        max_attempts = retries + 1

        # Set PYTHONPATH to project root for module imports
        import os
        env = os.environ.copy()
        env['PYTHONPATH'] = str(PROJECT_ROOT)

        for attempt in range(1, max_attempts + 1):
            logging.info(f"[{attempt}/{max_attempts}] Executing: {task_name}")
            logging.debug(f"Command: {' '.join(cmd)}")

            if self.dry_run:
                logging.info("[DRY RUN] Command prepared but not executed")
                return True

            try:
                result = subprocess.run(
                    cmd,
                    check=True,
                    capture_output=True,
                    text=True,
                    env=env
                )
                logging.info(f"✓ Success: {task_name}")
                if result.stdout:
                    logging.debug(f"Output: {result.stdout[:500]}")
                return True

            except subprocess.CalledProcessError as e:
                logging.error(f"✗ Failed: {task_name} (exit code: {e.returncode})")
                if e.stderr:
                    logging.error(f"Error output: {e.stderr[:500]}")

                if attempt < max_attempts:
                    delay = self.config.retry_delay_seconds
                    logging.info(f"Retrying in {delay}s... ({max_attempts - attempt} attempts left)")
                    time.sleep(delay)
                else:
                    logging.error(f"All {max_attempts} attempts exhausted for: {task_name}")
                    self.stats["failed_tasks"].append(task_name)
                    return False

        return False

    def phase_1_vocabulary_generation(self, characters: Optional[List[str]] = None,
                                     lora_types: Optional[List[str]] = None):
        """Phase 1: Generate prompt vocabularies"""
        logging.info("=" * 72)
        logging.info("PHASE 1: VOCABULARY GENERATION")
        logging.info("=" * 72)

        characters = characters or self.config.characters
        lora_types = lora_types or self.config.lora_types

        total_tasks = len(characters) * len(lora_types)
        completed = 0

        for char in characters:
            for lora_type in lora_types:
                task_key = f"vocab_{char}_{lora_type}"

                if self.resume and self.checkpoint_mgr.is_completed(task_key):
                    logging.info(f"⏭️  Skipping completed: {task_key}")
                    completed += 1
                    self.stats["total_vocabularies"] += 1
                    continue

                # Get character description
                char_desc = self.config.character_descriptions.get(
                    char,
                    self.config.character_descriptions.get("default", "A 3D animated character")
                )

                # Build output path
                output_file = self.config.workspace_root / "generated_data" / char / lora_type / "prompts.json"
                output_file.parent.mkdir(parents=True, exist_ok=True)

                # Build command
                cmd = [
                    "conda", "run", "-n", self.config.conda_env, "python",
                    str(PROJECT_ROOT / "scripts/generic/training/orchestration/vocabulary_generator.py"),
                    "--character-name", char,
                    "--character-description", char_desc,
                    "--lora-type", lora_type,
                    "--num-prompts", str(self.config.num_prompts_per_type),
                    "--output-file", str(output_file)
                ]

                if self.config.use_templates:
                    cmd.append("--use-templates")
                    cmd.extend(["--template-variations", str(self.config.template_variations)])

                if self.config.vocab_seed is not None:
                    cmd.extend(["--seed", str(self.config.vocab_seed)])

                # Execute
                success = self.run_command(cmd, task_key, retries=self.config.max_retries)

                if success:
                    self.checkpoint_mgr.mark_completed(task_key)
                    completed += 1
                    self.stats["total_vocabularies"] += 1

        logging.info(f"Phase 1 complete: {completed}/{total_tasks} vocabularies generated")

    def phase_2_image_generation(self, characters: Optional[List[str]] = None,
                                lora_types: Optional[List[str]] = None):
        """Phase 2: Generate synthetic images"""
        logging.info("=" * 72)
        logging.info("PHASE 2: IMAGE GENERATION")
        logging.info("=" * 72)

        characters = characters or self.config.characters
        lora_types = lora_types or self.config.lora_types

        total_tasks = len(characters) * len(lora_types)
        completed = 0

        for char in characters:
            # Find identity LoRA - use exact filename to avoid matching wrong characters
            # (e.g. "alberto" should not match "alberto_seamonster")
            lora_filename = f"BEST_{char}_lora_sdxl.safetensors"
            lora_path = self.config.identity_loras_dir / lora_filename

            if not lora_path.exists():
                logging.warning(f"⚠️  Identity LoRA not found: {lora_path}, skipping {char}")
                continue

            logging.info(f"Using identity LoRA: {lora_path.name}")

            for lora_type in lora_types:
                task_key = f"generation_{char}_{lora_type}"

                if self.resume and self.checkpoint_mgr.is_completed(task_key):
                    logging.info(f"⏭️  Skipping completed: {task_key}")
                    completed += 1
                    continue

                # Build paths
                prompts_file = self.config.workspace_root / "generated_data" / char / lora_type / "prompts.json"
                prompts_converted = self.config.workspace_root / "generated_data" / char / lora_type / "prompts_converted.json"
                output_dir = self.config.workspace_root / "generated_data" / char / lora_type / "generated"

                if not prompts_file.exists():
                    logging.error(f"✗ Prompts file not found: {prompts_file}")
                    continue

                # Convert prompts format
                self._convert_prompts_format(prompts_file, prompts_converted)

                # Build command
                output_dir.mkdir(parents=True, exist_ok=True)

                cmd = [
                    "conda", "run", "-n", self.config.conda_env, "python",
                    str(PROJECT_ROOT / "scripts/generic/training/batch_image_generator.py"),
                    "--prompts-file", str(prompts_converted),
                    "--base-model", str(self.config.base_model),
                    "--lora-paths", str(lora_path),
                    "--output-dir", str(output_dir),
                    "--num-images-per-prompt", str(self.config.num_images_per_prompt),
                    "--steps", str(self.config.num_inference_steps),
                    "--guidance-scale", str(self.config.guidance_scale),
                    "--height", str(self.config.height),
                    "--width", str(self.config.width),
                    "--device", self.config.device
                ]

                # Add lora_scales only if configured
                if self.config.lora_scale:
                    cmd.extend(["--lora-scales", str(self.config.lora_scale)])

                if self.config.use_random_seeds:
                    cmd.append("--use-random-seeds")

                if self.config.negative_prompt:
                    cmd.extend(["--negative-prompt", self.config.negative_prompt])

                # Execute
                success = self.run_command(cmd, task_key, retries=self.config.max_retries)

                if success:
                    self.checkpoint_mgr.mark_completed(task_key)
                    completed += 1

                    # Count generated images
                    image_count = len(list(output_dir.glob("*.png")))
                    self.stats["total_images"] += image_count
                    logging.info(f"Generated {image_count} images for {char}/{lora_type}")

        logging.info(f"Phase 2 complete: {completed}/{total_tasks} generation tasks completed")
        logging.info(f"Total images generated: {self.stats['total_images']}")

    def _convert_prompts_format(self, input_file: Path, output_file: Path):
        """Convert vocabulary JSON format to batch_image_generator format"""
        with open(input_file, 'r') as f:
            data = json.load(f)

        prompts = [p["prompt"] for p in data["prompts"]]

        # Include negative_prompt in the converted file
        output = {
            "prompts": prompts,
            "negative_prompt": self.config.negative_prompt
        }

        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)

        logging.debug(f"Converted {len(prompts)} prompts: {input_file.name} → {output_file.name}")


def main():
    parser = argparse.ArgumentParser(description="Synthetic Data Generation Pipeline Orchestrator")

    parser.add_argument("--config", type=str, required=True,
                       help="Path to YAML configuration file")
    parser.add_argument("--phase", type=str, choices=["1", "2", "all"], default="all",
                       help="Which phase to run (1=vocab, 2=generation, all=both)")
    parser.add_argument("--resume", action="store_true", default=True,
                       help="Resume from checkpoint")
    parser.add_argument("--no-resume", action="store_false", dest="resume",
                       help="Start fresh, ignoring checkpoints")
    parser.add_argument("--characters", type=str,
                       help="Comma-separated list of characters to process (overrides config)")
    parser.add_argument("--lora-types", type=str,
                       help="Comma-separated list of LoRA types (overrides config)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be done without executing")

    args = parser.parse_args()

    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}")
        sys.exit(1)

    config = PipelineConfig.from_yaml(config_path)

    # Override characters/lora_types if specified
    characters = args.characters.split(',') if args.characters else None
    lora_types = args.lora_types.split(',') if args.lora_types else None

    # Create orchestrator
    orchestrator = SyntheticDataOrchestrator(config, resume=args.resume, dry_run=args.dry_run)

    # Run phases
    start_time = time.time()

    if args.phase in ["1", "all"]:
        orchestrator.phase_1_vocabulary_generation(characters, lora_types)

    if args.phase in ["2", "all"]:
        orchestrator.phase_2_image_generation(characters, lora_types)

    # Report
    duration = time.time() - start_time

    logging.info("=" * 72)
    logging.info("PIPELINE COMPLETE")
    logging.info("=" * 72)
    logging.info(f"Duration: {duration:.1f}s ({duration/60:.1f}min)")
    logging.info(f"Vocabularies generated: {orchestrator.stats['total_vocabularies']}")
    logging.info(f"Images generated: {orchestrator.stats['total_images']}")

    if orchestrator.stats["failed_tasks"]:
        logging.warning(f"Failed tasks: {len(orchestrator.stats['failed_tasks'])}")
        for task in orchestrator.stats["failed_tasks"]:
            logging.warning(f"  - {task}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
GPT-SoVITS Training Wrapper

This script prepares voice samples for GPT-SoVITS training and manages the training process.

GPT-SoVITS Training Workflow:
1. Data preparation: Convert voice samples to GPT-SoVITS format
2. Stage 1: Train GPT model (semantic tokens)
3. Stage 2: Train SoVITS model (acoustic features)

Usage:
    python scripts/synthesis/tts/gpt_sovits_trainer.py \\
        --character Luca \\
        --samples data/films/luca/voice_samples_auto/by_character/Luca \\
        --output models/voices/luca/gpt_sovits \\
        --pretrained-gpt /path/to/pretrained/gpt.ckpt \\
        --pretrained-sovits /path/to/pretrained/sovits.pth \\
        --epochs-s1 15 \\
        --epochs-s2 10 \\
        --device cuda
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GPTSoVITSTrainer:
    """
    Wrapper class for GPT-SoVITS training
    """

    def __init__(
        self,
        character_name: str,
        samples_dir: Path,
        output_dir: Path,
        gpt_sovits_root: Path,
        pretrained_gpt: Path,
        pretrained_sovits: Path,
        language: str = "en",
        device: str = "cuda"
    ):
        """
        Initialize GPT-SoVITS trainer

        Args:
            character_name: Character name (e.g., "Luca")
            samples_dir: Directory containing voice samples
            output_dir: Output directory for trained models
            gpt_sovits_root: Path to GPT-SoVITS project root
            pretrained_gpt: Path to pretrained GPT model
            pretrained_sovits: Path to pretrained SoVITS model
            language: Language code (en, zh, ja, etc.)
            device: Device to use (cuda, cpu)
        """
        self.character_name = character_name
        self.samples_dir = Path(samples_dir)
        self.output_dir = Path(output_dir)
        self.gpt_sovits_root = Path(gpt_sovits_root)
        self.pretrained_gpt = Path(pretrained_gpt)
        self.pretrained_sovits = Path(pretrained_sovits)
        self.language = language
        self.device = device

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir = self.output_dir / "data"
        self.data_dir.mkdir(exist_ok=True)

        # Paths within GPT-SoVITS project
        self.exp_dir = self.gpt_sovits_root / "logs" / character_name
        self.exp_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized GPT-SoVITS trainer for {character_name}")
        logger.info(f"Samples dir: {self.samples_dir}")
        logger.info(f"Output dir: {self.output_dir}")
        logger.info(f"GPT-SoVITS root: {self.gpt_sovits_root}")

    def prepare_data(self) -> bool:
        """
        Prepare voice samples for GPT-SoVITS training

        Converts our training_filelist.json format to GPT-SoVITS's expected format:
        - Create text list files (audio_path|speaker|text)
        - Copy audio files to GPT-SoVITS data directory

        Returns:
            True if successful, False otherwise
        """
        logger.info("Preparing data for GPT-SoVITS training...")

        # Load training filelist
        filelist_path = self.samples_dir / "training_filelist.json"
        if not filelist_path.exists():
            logger.error(f"Training filelist not found: {filelist_path}")
            return False

        with open(filelist_path, 'r', encoding='utf-8') as f:
            training_data = json.load(f)

        logger.info(f"Loaded {len(training_data)} training samples")

        # Create audio directory in GPT-SoVITS format
        audio_dir = self.exp_dir / "0-audio"
        audio_dir.mkdir(parents=True, exist_ok=True)

        # Split into train/val (90/10 split)
        train_size = int(len(training_data) * 0.9)
        train_data = training_data[:train_size]
        val_data = training_data[train_size:]

        logger.info(f"Train samples: {len(train_data)}, Val samples: {len(val_data)}")

        # Prepare training and validation lists
        train_list = []
        val_list = []

        for i, sample in enumerate(training_data):
            # Get audio path
            audio_rel_path = sample['audio_path']
            audio_src = self.samples_dir.parent.parent / audio_rel_path

            if not audio_src.exists():
                logger.warning(f"Audio file not found: {audio_src}, skipping")
                continue

            # Copy audio to GPT-SoVITS audio directory
            audio_filename = f"{self.character_name}_{i:04d}.wav"
            audio_dst = audio_dir / audio_filename

            if not audio_dst.exists():
                shutil.copy2(audio_src, audio_dst)

            # Format: audio_path|speaker|text|language
            # GPT-SoVITS expects relative path from project root
            audio_rel = audio_dst.relative_to(self.gpt_sovits_root)
            text = sample['text'].strip()
            entry = f"{audio_rel}|{self.character_name}|{text}|{self.language}"

            if i < train_size:
                train_list.append(entry)
            else:
                val_list.append(entry)

        # Write training list files
        train_list_path = self.exp_dir / "train.list"
        val_list_path = self.exp_dir / "val.list"

        with open(train_list_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(train_list))

        with open(val_list_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(val_list))

        logger.info(f"Created train list: {train_list_path} ({len(train_list)} samples)")
        logger.info(f"Created val list: {val_list_path} ({len(val_list)} samples)")

        # Save metadata
        metadata = {
            "character": self.character_name,
            "language": self.language,
            "train_samples": len(train_list),
            "val_samples": len(val_list),
            "total_samples": len(training_data),
            "audio_dir": str(audio_dir),
            "train_list": str(train_list_path),
            "val_list": str(val_list_path)
        }

        metadata_path = self.output_dir / "training_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved training metadata: {metadata_path}")
        logger.info("✅ Data preparation complete!")

        return True

    def create_s1_config(self, epochs: int = 15, batch_size: int = 8) -> Path:
        """
        Create Stage 1 (GPT) training config

        Args:
            epochs: Number of training epochs
            batch_size: Batch size

        Returns:
            Path to created config file
        """
        config_template = {
            "train": {
                "seed": 1234,
                "epochs": epochs,
                "batch_size": batch_size,
                "save_every_n_epoch": 2,
                "precision": "16-mixed",
                "gradient_clip": 1.0
            },
            "optimizer": {
                "lr": 0.01,
                "lr_init": 0.00001,
                "lr_end": 0.0001,
                "warmup_steps": 2000,
                "decay_steps": 40000
            },
            "data": {
                "max_eval_sample": 8,
                "max_sec": 54,
                "num_workers": 4,
                "pad_val": 1024
            },
            "model": {
                "vocab_size": 1025,
                "phoneme_vocab_size": 512,
                "embedding_dim": 512,
                "hidden_dim": 512,
                "head": 16,
                "linear_units": 2048,
                "n_layer": 24,
                "dropout": 0,
                "EOS": 1024,
                "random_bert": 0
            },
            "inference": {
                "top_k": 5
            }
        }

        # Save config
        config_path = self.exp_dir / "s1_config.yaml"

        import yaml
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_template, f, default_flow_style=False, allow_unicode=True)

        logger.info(f"Created S1 config: {config_path}")
        return config_path

    def create_s2_config(self, epochs: int = 10, batch_size: int = 8) -> Path:
        """
        Create Stage 2 (SoVITS) training config

        Args:
            epochs: Number of training epochs
            batch_size: Batch size

        Returns:
            Path to created config file
        """
        config_template = {
            "train": {
                "seed": 1234,
                "epochs": epochs,
                "batch_size": batch_size,
                "save_every_epoch": 2,
                "learning_rate": 0.0001,
                "precision": "16-mixed",
                "gradient_clip": 1.0
            },
            "data": {
                "sampling_rate": 32000,
                "filter_length": 2048,
                "hop_length": 640,
                "win_length": 2048,
                "n_mel_channels": 128,
                "mel_fmin": 0,
                "mel_fmax": None,
                "max_wav_value": 32768.0,
                "num_workers": 4
            },
            "model": {
                "inter_channels": 192,
                "hidden_channels": 192,
                "filter_channels": 768,
                "n_heads": 2,
                "n_layers": 6,
                "kernel_size": 3,
                "p_dropout": 0.1,
                "resblock": "1",
                "resblock_kernel_sizes": [3, 7, 11],
                "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                "upsample_rates": [10, 8, 2, 2, 2],
                "upsample_initial_channel": 512,
                "upsample_kernel_sizes": [16, 16, 8, 2, 2]
            }
        }

        # Save config
        config_path = self.exp_dir / "s2_config.json"

        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_template, f, indent=2)

        logger.info(f"Created S2 config: {config_path}")
        return config_path

    def train_stage1(self, epochs: int = 15, batch_size: int = 8) -> bool:
        """
        Train Stage 1 (GPT model)

        Args:
            epochs: Number of training epochs
            batch_size: Batch size

        Returns:
            True if successful, False otherwise
        """
        logger.info("=" * 60)
        logger.info("STAGE 1: Training GPT Model")
        logger.info("=" * 60)

        # Create config
        config_path = self.create_s1_config(epochs=epochs, batch_size=batch_size)

        # Training command
        train_script = self.gpt_sovits_root / "GPT_SoVITS" / "s1_train.py"

        cmd = [
            "python",
            str(train_script),
            "--config_file", str(config_path),
            "--train_list", str(self.exp_dir / "train.list"),
            "--val_list", str(self.exp_dir / "val.list"),
            "--pretrained_model", str(self.pretrained_gpt),
            "--output_dir", str(self.exp_dir / "s1_ckpt"),
            "--exp_name", self.character_name
        ]

        logger.info(f"Running command: {' '.join(cmd)}")

        try:
            result = subprocess.run(cmd, check=True, cwd=str(self.gpt_sovits_root))
            logger.info("✅ Stage 1 training complete!")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Stage 1 training failed: {e}")
            return False

    def train_stage2(self, epochs: int = 10, batch_size: int = 8) -> bool:
        """
        Train Stage 2 (SoVITS model)

        Args:
            epochs: Number of training epochs
            batch_size: Batch size

        Returns:
            True if successful, False otherwise
        """
        logger.info("=" * 60)
        logger.info("STAGE 2: Training SoVITS Model")
        logger.info("=" * 60)

        # Create config
        config_path = self.create_s2_config(epochs=epochs, batch_size=batch_size)

        # Training command
        train_script = self.gpt_sovits_root / "GPT_SoVITS" / "s2_train.py"

        cmd = [
            "python",
            str(train_script),
            "--config_file", str(config_path),
            "--train_list", str(self.exp_dir / "train.list"),
            "--val_list", str(self.exp_dir / "val.list"),
            "--pretrained_model", str(self.pretrained_sovits),
            "--output_dir", str(self.exp_dir / "s2_ckpt"),
            "--exp_name", self.character_name
        ]

        logger.info(f"Running command: {' '.join(cmd)}")

        try:
            result = subprocess.run(cmd, check=True, cwd=str(self.gpt_sovits_root))
            logger.info("✅ Stage 2 training complete!")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Stage 2 training failed: {e}")
            return False

    def train_full(
        self,
        s1_epochs: int = 15,
        s2_epochs: int = 10,
        s1_batch_size: int = 8,
        s2_batch_size: int = 8
    ) -> bool:
        """
        Run full training pipeline (both stages)

        Args:
            s1_epochs: Stage 1 epochs
            s2_epochs: Stage 2 epochs
            s1_batch_size: Stage 1 batch size
            s2_batch_size: Stage 2 batch size

        Returns:
            True if successful, False otherwise
        """
        logger.info("🚀 Starting full GPT-SoVITS training pipeline")

        # Step 1: Prepare data
        if not self.prepare_data():
            logger.error("Data preparation failed")
            return False

        # Step 2: Train Stage 1
        if not self.train_stage1(epochs=s1_epochs, batch_size=s1_batch_size):
            logger.error("Stage 1 training failed")
            return False

        # Step 3: Train Stage 2
        if not self.train_stage2(epochs=s2_epochs, batch_size=s2_batch_size):
            logger.error("Stage 2 training failed")
            return False

        logger.info("=" * 60)
        logger.info("🎉 Full training pipeline complete!")
        logger.info("=" * 60)
        logger.info(f"Trained models saved in: {self.exp_dir}")

        return True


def main():
    parser = argparse.ArgumentParser(description="GPT-SoVITS Training Wrapper")

    # Required arguments
    parser.add_argument(
        "--character",
        type=str,
        required=True,
        help="Character name (e.g., Luca, Alberto)"
    )
    parser.add_argument(
        "--samples",
        type=str,
        required=True,
        help="Path to voice samples directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for trained models"
    )

    # GPT-SoVITS paths
    parser.add_argument(
        "--gpt-sovits-root",
        type=str,
        default="/mnt/c/AI_LLM_projects/GPT-SoVITS",
        help="Path to GPT-SoVITS project root"
    )
    parser.add_argument(
        "--pretrained-gpt",
        type=str,
        default="/mnt/c/AI_LLM_projects/ai_warehouse/models/audio/gpt_sovits/pretrained/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt",
        help="Path to pretrained GPT model"
    )
    parser.add_argument(
        "--pretrained-sovits",
        type=str,
        default="/mnt/c/AI_LLM_projects/ai_warehouse/models/audio/gpt_sovits/pretrained/s2G488k.pth",
        help="Path to pretrained SoVITS model"
    )

    # Training parameters
    parser.add_argument(
        "--s1-epochs",
        type=int,
        default=15,
        help="Number of epochs for Stage 1 (GPT) training"
    )
    parser.add_argument(
        "--s2-epochs",
        type=int,
        default=10,
        help="Number of epochs for Stage 2 (SoVITS) training"
    )
    parser.add_argument(
        "--s1-batch-size",
        type=int,
        default=8,
        help="Batch size for Stage 1 training"
    )
    parser.add_argument(
        "--s2-batch-size",
        type=int,
        default=8,
        help="Batch size for Stage 2 training"
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        help="Language code (en, zh, ja, etc.)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda, cpu)"
    )

    # Mode
    parser.add_argument(
        "--mode",
        type=str,
        choices=["prepare", "s1", "s2", "full"],
        default="full",
        help="Training mode: prepare (data only), s1 (stage 1 only), s2 (stage 2 only), full (complete pipeline)"
    )

    # Logging
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        force=True
    )

    # Create trainer
    trainer = GPTSoVITSTrainer(
        character_name=args.character,
        samples_dir=Path(args.samples),
        output_dir=Path(args.output),
        gpt_sovits_root=Path(args.gpt_sovits_root),
        pretrained_gpt=Path(args.pretrained_gpt),
        pretrained_sovits=Path(args.pretrained_sovits),
        language=args.language,
        device=args.device
    )

    # Run training based on mode
    success = False

    if args.mode == "prepare":
        success = trainer.prepare_data()
    elif args.mode == "s1":
        if trainer.prepare_data():
            success = trainer.train_stage1(
                epochs=args.s1_epochs,
                batch_size=args.s1_batch_size
            )
    elif args.mode == "s2":
        success = trainer.train_stage2(
            epochs=args.s2_epochs,
            batch_size=args.s2_batch_size
        )
    elif args.mode == "full":
        success = trainer.train_full(
            s1_epochs=args.s1_epochs,
            s2_epochs=args.s2_epochs,
            s1_batch_size=args.s1_batch_size,
            s2_batch_size=args.s2_batch_size
        )

    if success:
        logger.info("🎉 Training completed successfully!")
        sys.exit(0)
    else:
        logger.error("❌ Training failed")
        sys.exit(1)


if __name__ == "__main__":
    main()

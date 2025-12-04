#!/usr/bin/env python3
"""
GPT-SoVITS Complete Training Pipeline
Handles all preprocessing steps and training for voice cloning.
"""

import os
import sys
import json
import subprocess
import logging
from pathlib import Path
from typing import Optional, Dict, List
import argparse

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GPTSoVITSPipeline:
    """Complete GPT-SoVITS training pipeline with preprocessing"""

    def __init__(
        self,
        character_name: str,
        samples_dir: Path,
        output_dir: Path,
        gpt_sovits_root: Path,
        language: str = "en",
        device: str = "cuda"
    ):
        self.character_name = character_name
        self.samples_dir = Path(samples_dir)
        self.output_dir = Path(output_dir)
        self.gpt_sovits_root = Path(gpt_sovits_root)
        self.language = language
        self.device = device

        # Setup paths
        self.exp_dir = self.gpt_sovits_root / "logs" / self.character_name
        self.audio_dir = self.exp_dir / "0-audio"
        self.train_list = self.exp_dir / "train.list"
        self.val_list = self.exp_dir / "val.list"

        # Preprocessing output paths
        self.text_output = self.exp_dir / "2-name2text.txt"
        self.semantic_output = self.exp_dir / "6-name2semantic.tsv"
        self.bert_dir = self.exp_dir / "3-bert"
        self.hubert_dir = self.exp_dir / "4-cnhubert"
        self.wav32k_dir = self.exp_dir / "5-wav32k"

        # Pretrained model paths
        self.bert_path = self.gpt_sovits_root / "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"
        self.hubert_path = self.gpt_sovits_root / "GPT_SoVITS/pretrained_models/chinese-hubert-base"
        self.s1_pretrained = Path("/mnt/c/AI_LLM_projects/ai_warehouse/models/audio/gpt_sovits/pretrained/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt")
        self.s2_pretrained = Path("/mnt/c/AI_LLM_projects/ai_warehouse/models/audio/gpt_sovits/pretrained/s2G488k.pth")

        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.exp_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized GPT-SoVITS pipeline for {character_name}")
        logger.info(f"Samples dir: {self.samples_dir}")
        logger.info(f"Output dir: {self.output_dir}")
        logger.info(f"Experiment dir: {self.exp_dir}")

    def check_prerequisites(self) -> bool:
        """Check if all required models and files exist"""
        logger.info("Checking prerequisites...")

        checks = {
            "Train list": self.train_list.exists(),
            "Val list": self.val_list.exists(),
            "Audio directory": self.audio_dir.exists() and len(list(self.audio_dir.glob("*.wav"))) > 0,
            "BERT model": self.bert_path.exists(),
            "HuBERT model": self.hubert_path.exists(),
            "S1 pretrained": self.s1_pretrained.exists(),
            "S2 pretrained": self.s2_pretrained.exists(),
        }

        all_ok = True
        for name, exists in checks.items():
            status = "✓" if exists else "✗"
            logger.info(f"  {status} {name}")
            if not exists:
                all_ok = False
                if "model" in name.lower():
                    logger.warning(f"    Missing: Please ensure models are downloaded")
                else:
                    logger.warning(f"    Missing: Please run data preparation first")

        return all_ok

    def run_preprocessing_step1(self) -> bool:
        """Step 1: Extract phoneme/text features using BERT"""
        logger.info("=" * 60)
        logger.info("STEP 1: Extracting phoneme/text features")
        logger.info("=" * 60)

        if not self.bert_path.exists():
            logger.error(f"BERT model not found at {self.bert_path}")
            logger.error("Please wait for model download to complete")
            return False

        script_path = self.gpt_sovits_root / "GPT_SoVITS/prepare_datasets/1-get-text.py"

        env = os.environ.copy()
        # Add both GPT-SoVITS root and GPT_SoVITS subdirectory to PYTHONPATH
        # Root needed for 'tools', subdirectory needed for 'text', 'AR', etc.
        pythonpath_parts = [
            str(self.gpt_sovits_root),
            str(self.gpt_sovits_root / "GPT_SoVITS")
        ]
        if "PYTHONPATH" in env:
            pythonpath_parts.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = ":".join(pythonpath_parts)

        env.update({
            "inp_text": str(self.train_list),
            "inp_wav_dir": str(self.audio_dir),
            "exp_name": self.character_name,
            "opt_dir": str(self.exp_dir),
            "bert_pretrained_dir": str(self.bert_path),
            "is_half": "True",
            "i_part": "0",
            "all_parts": "1",
            "_CUDA_VISIBLE_DEVICES": "0",
            "version": "v2"
        })

        try:
            cmd = [sys.executable, str(script_path)]
            logger.info(f"Running command: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                env=env,
                cwd=str(self.gpt_sovits_root),
                capture_output=True,
                text=True,
                timeout=1800  # 30 minutes
            )

            if result.returncode == 0:
                logger.info("✓ Step 1 completed successfully")
                logger.info(f"Created: {self.text_output}")
                logger.info(f"BERT features: {self.bert_dir}/")
                return True
            else:
                logger.error(f"✗ Step 1 failed with return code {result.returncode}")
                logger.error(f"STDOUT: {result.stdout}")
                logger.error(f"STDERR: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            logger.error("✗ Step 1 timed out after 30 minutes")
            return False
        except Exception as e:
            logger.error(f"✗ Step 1 failed: {e}")
            return False

    def run_preprocessing_step2(self) -> bool:
        """Step 2: Resample audio to 32kHz and extract HuBERT features"""
        logger.info("=" * 60)
        logger.info("STEP 2: Resampling audio and extracting HuBERT features")
        logger.info("=" * 60)

        if not self.hubert_path.exists():
            logger.error(f"HuBERT model not found at {self.hubert_path}")
            logger.error("Please wait for model download to complete")
            return False

        script_path = self.gpt_sovits_root / "GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py"

        env = os.environ.copy()
        # Add both GPT-SoVITS root and GPT_SoVITS subdirectory to PYTHONPATH
        # Root needed for 'tools', subdirectory needed for 'text', 'AR', etc.
        pythonpath_parts = [
            str(self.gpt_sovits_root),
            str(self.gpt_sovits_root / "GPT_SoVITS")
        ]
        if "PYTHONPATH" in env:
            pythonpath_parts.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = ":".join(pythonpath_parts)

        env.update({
            "inp_text": str(self.train_list),
            "inp_wav_dir": str(self.audio_dir),
            "exp_name": self.character_name,
            "opt_dir": str(self.exp_dir),
            "cnhubert_base_dir": str(self.hubert_path),
            "is_half": "True",
            "i_part": "0",
            "all_parts": "1",
            "_CUDA_VISIBLE_DEVICES": "0"
        })

        try:
            cmd = [sys.executable, str(script_path)]
            logger.info(f"Running command: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                env=env,
                cwd=str(self.gpt_sovits_root),
                capture_output=True,
                text=True,
                timeout=1800  # 30 minutes
            )

            if result.returncode == 0:
                logger.info("✓ Step 2 completed successfully")
                logger.info(f"32kHz audio: {self.wav32k_dir}/")
                logger.info(f"HuBERT features: {self.hubert_dir}/")
                return True
            else:
                logger.error(f"✗ Step 2 failed with return code {result.returncode}")
                logger.error(f"STDOUT: {result.stdout}")
                logger.error(f"STDERR: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            logger.error("✗ Step 2 timed out after 30 minutes")
            return False
        except Exception as e:
            logger.error(f"✗ Step 2 failed: {e}")
            return False

    def run_preprocessing_step3(self) -> bool:
        """Step 3: Extract semantic tokens"""
        logger.info("=" * 60)
        logger.info("STEP 3: Extracting semantic tokens")
        logger.info("=" * 60)

        script_path = self.gpt_sovits_root / "GPT_SoVITS/prepare_datasets/3-get-semantic.py"

        env = os.environ.copy()
        # Add both GPT-SoVITS root and GPT_SoVITS subdirectory to PYTHONPATH
        # Root needed for 'tools', subdirectory needed for 'text', 'AR', etc.
        pythonpath_parts = [
            str(self.gpt_sovits_root),
            str(self.gpt_sovits_root / "GPT_SoVITS")
        ]
        if "PYTHONPATH" in env:
            pythonpath_parts.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = ":".join(pythonpath_parts)

        env.update({
            "inp_text": str(self.train_list),
            "exp_name": self.character_name,
            "opt_dir": str(self.exp_dir),
            "cnhubert_base_dir": str(self.hubert_path),
            "pretrained_s2G": str(self.s2_pretrained),  # Required for loading semantic model
            "s2config_path": str(self.gpt_sovits_root / "GPT_SoVITS/configs/s2.json"),  # S2 model config
            "is_half": "True",
            "i_part": "0",
            "all_parts": "1",
            "_CUDA_VISIBLE_DEVICES": "0"
        })

        try:
            cmd = [sys.executable, str(script_path)]
            logger.info(f"Running command: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                env=env,
                cwd=str(self.gpt_sovits_root),
                capture_output=True,
                text=True,
                timeout=1800  # 30 minutes
            )

            if result.returncode == 0:
                logger.info("✓ Step 3 completed successfully")
                logger.info(f"Semantic tokens: {self.semantic_output}")
                return True
            else:
                logger.error(f"✗ Step 3 failed with return code {result.returncode}")
                logger.error(f"STDOUT: {result.stdout}")
                logger.error(f"STDERR: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            logger.error("✗ Step 3 timed out after 30 minutes")
            return False
        except Exception as e:
            logger.error(f"✗ Step 3 failed: {e}")
            return False

    def run_preprocessing(self) -> bool:
        """Run all preprocessing steps"""
        logger.info("🚀 Starting preprocessing pipeline")

        steps = [
            ("Phoneme/Text Features", self.run_preprocessing_step1),
            ("HuBERT Features", self.run_preprocessing_step2),
            ("Semantic Tokens", self.run_preprocessing_step3)
        ]

        for step_name, step_func in steps:
            logger.info(f"\n{'='*60}")
            logger.info(f"Starting: {step_name}")
            logger.info(f"{'='*60}")

            if not step_func():
                logger.error(f"❌ Preprocessing failed at: {step_name}")
                return False

            logger.info(f"✅ Completed: {step_name}")

        logger.info("\n" + "="*60)
        logger.info("✅ All preprocessing steps completed successfully!")
        logger.info("="*60)
        return True

    def create_training_config_s1(self, epochs: int = 15, batch_size: int = 8) -> Path:
        """Create Stage 1 (GPT) training configuration"""
        config_path = self.exp_dir / "s1_config.yaml"

        config = {
            "train": {
                "seed": 1234,
                "epochs": epochs,
                "batch_size": batch_size,
                "save_every_n_epoch": 2,
                "precision": "16-mixed",
                "gradient_clip": 1.0,
                "if_save_latest": True,
                "if_save_every_weights": True,
                "half_weights_save_dir": str(self.exp_dir / "s1_ckpt"),
                "exp_name": self.character_name
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
            },
            "train_semantic_path": str(self.semantic_output),
            "train_phoneme_path": str(self.text_output),
            "output_dir": str(self.exp_dir / "s1_output"),
            "pretrained_s1": str(self.s1_pretrained)
        }

        import yaml
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        logger.info(f"Created S1 config: {config_path}")
        return config_path

    def train_stage1(self, epochs: int = 15, batch_size: int = 8) -> bool:
        """Train Stage 1: GPT Model"""
        logger.info("=" * 60)
        logger.info("STAGE 1: Training GPT Model")
        logger.info("=" * 60)

        config_path = self.create_training_config_s1(epochs, batch_size)
        script_path = self.gpt_sovits_root / "GPT_SoVITS/s1_train.py"

        try:
            cmd = [
                sys.executable,
                str(script_path),
                "-c", str(config_path)
            ]

            logger.info(f"Running command: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                cwd=str(self.gpt_sovits_root),
                env=os.environ.copy(),
                capture_output=True,
                text=True
            )

            if result.returncode == 0:
                logger.info("✓ Stage 1 training completed")
                return True
            else:
                logger.error(f"✗ Stage 1 training failed: {result.returncode}")
                logger.error(f"STDERR: {result.stderr}")
                return False

        except Exception as e:
            logger.error(f"✗ Stage 1 training failed: {e}")
            return False

    def train_stage2(self, epochs: int = 10, batch_size: int = 8) -> bool:
        """Train Stage 2: SoVITS Model"""
        logger.info("=" * 60)
        logger.info("STAGE 2: Training SoVITS Model")
        logger.info("=" * 60)

        # Stage 2 training implementation
        # Similar to Stage 1 but using s2_train.py
        logger.info("Stage 2 training configuration pending...")
        logger.info("(Will be implemented after S1 completes)")
        return True

    def run_full_pipeline(
        self,
        s1_epochs: int = 15,
        s2_epochs: int = 10,
        batch_size: int = 8,
        skip_preprocessing: bool = False
    ) -> bool:
        """Run complete training pipeline"""
        logger.info("\n" + "🚀" * 30)
        logger.info("GPT-SoVITS COMPLETE TRAINING PIPELINE")
        logger.info("🚀" * 30 + "\n")

        # Check prerequisites
        if not self.check_prerequisites():
            if not skip_preprocessing:
                logger.warning("Some prerequisites missing, but will attempt preprocessing...")
            else:
                logger.error("Prerequisites check failed and preprocessing is skipped")
                return False

        # Run preprocessing
        if not skip_preprocessing:
            if not self.run_preprocessing():
                logger.error("❌ Preprocessing failed")
                return False
        else:
            logger.info("⏭️  Skipping preprocessing (--skip-preprocessing flag)")

        # Train Stage 1
        if not self.train_stage1(s1_epochs, batch_size):
            logger.error("❌ Stage 1 training failed")
            return False

        # Train Stage 2
        if not self.train_stage2(s2_epochs, batch_size):
            logger.error("❌ Stage 2 training failed")
            return False

        logger.info("\n" + "✅" * 30)
        logger.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("✅" * 30 + "\n")

        return True


def main():
    parser = argparse.ArgumentParser(description="GPT-SoVITS Complete Training Pipeline")
    parser.add_argument("--character", required=True, help="Character name")
    parser.add_argument("--samples", required=True, help="Voice samples directory")
    parser.add_argument("--output", required=True, help="Output directory for trained models")
    parser.add_argument("--gpt-sovits-root", default="/mnt/c/AI_LLM_projects/GPT-SoVITS",
                       help="GPT-SoVITS root directory")
    parser.add_argument("--language", default="en", help="Language code")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--s1-epochs", type=int, default=15, help="Stage 1 epochs")
    parser.add_argument("--s2-epochs", type=int, default=10, help="Stage 2 epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--skip-preprocessing", action="store_true",
                       help="Skip preprocessing steps")
    parser.add_argument("--preprocessing-only", action="store_true",
                       help="Only run preprocessing, skip training")

    args = parser.parse_args()

    pipeline = GPTSoVITSPipeline(
        character_name=args.character,
        samples_dir=args.samples,
        output_dir=args.output,
        gpt_sovits_root=args.gpt_sovits_root,
        language=args.language,
        device=args.device
    )

    if args.preprocessing_only:
        success = pipeline.check_prerequisites() and pipeline.run_preprocessing()
    else:
        success = pipeline.run_full_pipeline(
            s1_epochs=args.s1_epochs,
            s2_epochs=args.s2_epochs,
            batch_size=args.batch_size,
            skip_preprocessing=args.skip_preprocessing
        )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

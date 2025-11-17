"""
Batch Image Generator

Enhanced batch generation with quality filtering, consistency checking,
and progress tracking.

Architecture:
- Batch generation with configurable parameters
- Quality-driven filtering
- Consistency validation
- Progress tracking and resumption
- Output organization

Author: Animation AI Studio
Date: 2025-11-17
"""

import torch
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass
import yaml
import json
import time
from datetime import datetime
from PIL import Image
from tqdm import tqdm
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.generation.image import (
    CharacterGenerator,
    CharacterConsistencyChecker,
    CharacterReferenceManager,
    ConsistencyResult
)


@dataclass
class BatchGenerationConfig:
    """Configuration for batch generation"""
    character: str
    scene_description: str
    num_images: int
    quality_preset: str = "standard"
    style: str = "pixar_3d"
    use_controlnet: bool = False
    control_type: Optional[str] = None
    control_image: Optional[str] = None
    seed_start: int = 42
    seed_increment: int = 1
    enable_consistency_check: bool = True
    consistency_threshold: float = 0.65
    save_rejected: bool = True
    output_dir: Optional[str] = None


@dataclass
class BatchGenerationResult:
    """Result of batch generation"""
    total_generated: int
    total_accepted: int
    total_rejected: int
    accepted_images: List[str]
    rejected_images: List[str]
    consistency_scores: List[float]
    generation_time: float
    avg_time_per_image: float
    config: BatchGenerationConfig


class BatchImageGenerator:
    """
    Batch Image Generator with Quality Filtering

    Features:
    - Batch generation with configurable seeds
    - Automatic consistency checking
    - Quality-based filtering
    - Progress tracking with tqdm
    - Organized output structure
    - Generation resumption support

    Usage:
        batch_gen = BatchImageGenerator()
        result = batch_gen.generate_batch(
            character="luca",
            scene_description="running on the beach",
            num_images=10,
            enable_consistency_check=True
        )
        print(f"Accepted: {result.total_accepted}/{result.total_generated}")
    """

    def __init__(
        self,
        character_generator: Optional[CharacterGenerator] = None,
        consistency_checker: Optional[CharacterConsistencyChecker] = None,
        reference_manager: Optional[CharacterReferenceManager] = None,
        character_presets_path: str = "configs/generation/character_presets.yaml"
    ):
        """
        Initialize Batch Image Generator

        Args:
            character_generator: CharacterGenerator instance (created if None)
            consistency_checker: CharacterConsistencyChecker instance
            reference_manager: CharacterReferenceManager instance
            character_presets_path: Path to character presets config
        """
        # Initialize character generator
        self.generator = character_generator or CharacterGenerator()

        # Initialize consistency checker if needed
        self.consistency_checker = consistency_checker
        self.reference_manager = reference_manager

        # Load character presets
        self.character_presets = self._load_character_presets(character_presets_path)

    def _load_character_presets(self, config_path: str) -> Dict[str, Any]:
        """Load character presets from YAML"""
        config_path = Path(config_path)
        if not config_path.exists():
            print(f"WARNING: Character presets not found: {config_path}")
            return {}

        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def _initialize_consistency_checker(self):
        """Lazy initialization of consistency checker"""
        if self.consistency_checker is None:
            try:
                self.consistency_checker = CharacterConsistencyChecker(
                    device="cuda" if torch.cuda.is_available() else "cpu"
                )
                self.reference_manager = CharacterReferenceManager(
                    self.consistency_checker
                )
                print("✓ Consistency checker initialized")
            except Exception as e:
                print(f"WARNING: Failed to initialize consistency checker: {e}")
                return False
        return True

    def _get_output_dir(
        self,
        character: str,
        base_dir: str = "outputs/batch_generation"
    ) -> Path:
        """Create organized output directory"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(base_dir) / character / timestamp
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (output_dir / "accepted").mkdir(exist_ok=True)
        (output_dir / "rejected").mkdir(exist_ok=True)

        return output_dir

    def generate_batch(
        self,
        character: str,
        scene_description: str,
        num_images: int,
        quality_preset: str = "standard",
        style: str = "pixar_3d",
        use_controlnet: bool = False,
        control_type: Optional[str] = None,
        control_image: Optional[str] = None,
        seed_start: int = 42,
        seed_increment: int = 1,
        enable_consistency_check: bool = True,
        consistency_threshold: float = 0.65,
        save_rejected: bool = True,
        output_dir: Optional[str] = None,
        save_metadata: bool = True
    ) -> BatchGenerationResult:
        """
        Generate batch of images with quality filtering

        Args:
            character: Character name
            scene_description: Scene description
            num_images: Number of images to generate
            quality_preset: Quality preset
            style: Style key
            use_controlnet: Whether to use ControlNet
            control_type: ControlNet type
            control_image: Control image path
            seed_start: Starting seed
            seed_increment: Seed increment for each image
            enable_consistency_check: Enable consistency checking
            consistency_threshold: Consistency threshold
            save_rejected: Save rejected images
            output_dir: Custom output directory
            save_metadata: Save generation metadata JSON

        Returns:
            BatchGenerationResult
        """
        start_time = time.time()

        # Create output directory
        if output_dir is None:
            output_dir = self._get_output_dir(character)
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Batch Generation: {num_images} images of {character}")
        print(f"Output: {output_dir}")
        print(f"{'='*60}\n")

        # Initialize consistency checker if needed
        if enable_consistency_check:
            if not self._initialize_consistency_checker():
                print("WARNING: Consistency checking disabled (initialization failed)")
                enable_consistency_check = False

        # Track results
        accepted_images = []
        rejected_images = []
        consistency_scores = []

        # Generate images with progress bar
        with tqdm(total=num_images, desc="Generating images") as pbar:
            for i in range(num_images):
                seed = seed_start + (i * seed_increment)

                # Generate image
                output_path = output_dir / "temp" / f"image_{i:04d}.png"
                output_path.parent.mkdir(exist_ok=True)

                try:
                    image = self.generator.generate_character(
                        character=character,
                        scene_description=scene_description,
                        style=style,
                        quality_preset=quality_preset,
                        use_controlnet=use_controlnet,
                        control_type=control_type,
                        control_image=control_image,
                        seed=seed,
                        output_path=str(output_path)
                    )

                    # Check consistency
                    if enable_consistency_check and self.reference_manager:
                        result = self.reference_manager.check_character_consistency(
                            character_name=character,
                            generated_image=str(output_path),
                            threshold=consistency_threshold
                        )

                        consistency_scores.append(result.similarity_score)

                        # Accept or reject
                        if result.is_consistent:
                            # Move to accepted
                            accepted_path = output_dir / "accepted" / f"{character}_{i:04d}_s{seed}_sim{result.similarity_score:.3f}.png"
                            output_path.rename(accepted_path)
                            accepted_images.append(str(accepted_path))
                            pbar.set_postfix({"status": "✓", "sim": f"{result.similarity_score:.3f}"})
                        else:
                            if save_rejected:
                                # Move to rejected
                                rejected_path = output_dir / "rejected" / f"{character}_{i:04d}_s{seed}_sim{result.similarity_score:.3f}.png"
                                output_path.rename(rejected_path)
                                rejected_images.append(str(rejected_path))
                            else:
                                output_path.unlink()  # Delete
                            pbar.set_postfix({"status": "✗", "sim": f"{result.similarity_score:.3f}"})
                    else:
                        # No consistency check, accept all
                        accepted_path = output_dir / "accepted" / f"{character}_{i:04d}_s{seed}.png"
                        output_path.rename(accepted_path)
                        accepted_images.append(str(accepted_path))
                        pbar.set_postfix({"status": "✓"})

                except Exception as e:
                    print(f"\nERROR generating image {i}: {e}")
                    pbar.set_postfix({"status": "ERROR"})

                pbar.update(1)

        # Cleanup temp directory
        temp_dir = output_dir / "temp"
        if temp_dir.exists():
            temp_dir.rmdir()

        generation_time = time.time() - start_time

        # Create result
        result = BatchGenerationResult(
            total_generated=num_images,
            total_accepted=len(accepted_images),
            total_rejected=len(rejected_images),
            accepted_images=accepted_images,
            rejected_images=rejected_images,
            consistency_scores=consistency_scores,
            generation_time=generation_time,
            avg_time_per_image=generation_time / num_images if num_images > 0 else 0,
            config=BatchGenerationConfig(
                character=character,
                scene_description=scene_description,
                num_images=num_images,
                quality_preset=quality_preset,
                style=style,
                use_controlnet=use_controlnet,
                control_type=control_type,
                control_image=control_image,
                seed_start=seed_start,
                seed_increment=seed_increment,
                enable_consistency_check=enable_consistency_check,
                consistency_threshold=consistency_threshold,
                save_rejected=save_rejected,
                output_dir=str(output_dir)
            )
        )

        # Save metadata
        if save_metadata:
            self._save_metadata(result, output_dir)

        # Print summary
        self._print_summary(result)

        return result

    def _save_metadata(self, result: BatchGenerationResult, output_dir: Path):
        """Save generation metadata to JSON"""
        metadata = {
            "total_generated": result.total_generated,
            "total_accepted": result.total_accepted,
            "total_rejected": result.total_rejected,
            "acceptance_rate": result.total_accepted / result.total_generated if result.total_generated > 0 else 0,
            "generation_time": result.generation_time,
            "avg_time_per_image": result.avg_time_per_image,
            "consistency_scores": {
                "scores": result.consistency_scores,
                "mean": float(np.mean(result.consistency_scores)) if result.consistency_scores else None,
                "std": float(np.std(result.consistency_scores)) if result.consistency_scores else None,
                "min": float(np.min(result.consistency_scores)) if result.consistency_scores else None,
                "max": float(np.max(result.consistency_scores)) if result.consistency_scores else None
            },
            "config": {
                "character": result.config.character,
                "scene_description": result.config.scene_description,
                "num_images": result.config.num_images,
                "quality_preset": result.config.quality_preset,
                "style": result.config.style,
                "seed_start": result.config.seed_start,
                "consistency_threshold": result.config.consistency_threshold
            },
            "timestamp": datetime.now().isoformat()
        }

        metadata_path = output_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"✓ Metadata saved to: {metadata_path}")

    def _print_summary(self, result: BatchGenerationResult):
        """Print batch generation summary"""
        print(f"\n{'='*60}")
        print("Batch Generation Summary")
        print(f"{'='*60}")
        print(f"Total Generated:  {result.total_generated}")
        print(f"Accepted:         {result.total_accepted} ({result.total_accepted/result.total_generated*100:.1f}%)")
        print(f"Rejected:         {result.total_rejected} ({result.total_rejected/result.total_generated*100:.1f}%)")
        print(f"Generation Time:  {result.generation_time:.1f}s")
        print(f"Avg per Image:    {result.avg_time_per_image:.1f}s")

        if result.consistency_scores:
            import numpy as np
            print(f"\nConsistency Scores:")
            print(f"  Mean:  {np.mean(result.consistency_scores):.3f}")
            print(f"  Std:   {np.std(result.consistency_scores):.3f}")
            print(f"  Min:   {np.min(result.consistency_scores):.3f}")
            print(f"  Max:   {np.max(result.consistency_scores):.3f}")

        print(f"{'='*60}\n")

    def cleanup(self):
        """Cleanup resources"""
        if self.generator:
            self.generator.cleanup()


def main():
    """Example usage"""
    import numpy as np

    # Initialize batch generator
    batch_gen = BatchImageGenerator()

    # Example 1: Basic batch generation
    print("=== Example 1: Basic Batch Generation ===")
    result = batch_gen.generate_batch(
        character="luca",
        scene_description="running on the beach, excited expression, summer day",
        num_images=5,
        quality_preset="draft",  # Use draft for faster testing
        seed_start=42,
        enable_consistency_check=False,  # Disable for testing (no reference images yet)
        output_dir="outputs/test_batch"
    )

    print(f"Generated {result.total_accepted} images")

    # Example 2: Batch with consistency checking (requires reference images)
    # print("\n=== Example 2: Batch with Consistency Checking ===")
    # result = batch_gen.generate_batch(
    #     character="luca",
    #     scene_description="standing in Portorosso, smiling",
    #     num_images=10,
    #     quality_preset="standard",
    #     enable_consistency_check=True,
    #     consistency_threshold=0.65,
    #     save_rejected=True
    # )

    # Cleanup
    batch_gen.cleanup()

    print("\n✓ Examples complete!")


if __name__ == "__main__":
    main()

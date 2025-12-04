"""
Data Pipeline Automation - Training Data Preparation Executor

Custom implementation to organize clustered instances into LoRA training format.

Author: Animation AI Studio
Date: 2025-12-04
"""

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import logging

import cv2
import numpy as np
from PIL import Image

from .base_executor import StageExecutor
from ..common import StageResult, ExecutionStatus

logger = logging.getLogger(__name__)


class TrainingDataPrepExecutor(StageExecutor):
    """
    Training data preparation stage executor

    Custom implementation to organize clustered character instances
    into LoRA-compatible training format.

    Process:
        1. Read each cluster directory
        2. Apply quality filtering (size, blur detection)
        3. Copy and resize images to output_dir/character_N/img/
        4. (Optional) Generate captions
        5. Create dataset_info.json

    Required config keys:
        - cluster_dirs: List of cluster directory paths (from clustering stage)
        - output_dir: Output directory for training data

    Optional config keys:
        - target_size: Target image size (default: 512)
            Images will be resized to target_size x target_size
        - min_image_size: Minimum image dimension (default: 256)
            Images smaller than this will be filtered out
        - blur_threshold: Blur detection threshold (default: 100.0)
            Lower value = stricter (higher values detected as blurry)
        - max_images_per_char: Maximum images per character (default: 200)
            Limits dataset size per character
        - generate_captions: Generate caption files (default: false)
        - caption_template: Caption template string (default: "character")
        - jpeg_quality: Output JPEG quality (default: 95)

    Outputs:
        - output_dir: Training data root directory
        - dataset_stats: Dictionary with character stats
        - image_count: Total images prepared
        - character_count: Number of characters
        - filtered_count: Number of images filtered out

    Metrics:
        - images_prepared: Total images
        - characters_prepared: Number of characters
        - images_filtered: Filtered count
        - avg_images_per_char: Average images per character
        - preparation_time: Execution time
    """

    # No script_path - custom implementation
    required_config_keys = ["cluster_dirs", "output_dir"]
    output_keys = ["output_dir", "dataset_stats", "image_count", "character_count", "filtered_count"]

    def validate_config(self) -> bool:
        """
        Validate training data preparation configuration

        Returns:
            True if configuration is valid

        Raises:
            ValueError: If configuration is invalid
        """
        # Validate cluster_dirs
        cluster_dirs = self._get_config_value("cluster_dirs", required=True)
        if not isinstance(cluster_dirs, list):
            raise ValueError(f"cluster_dirs must be a list, got {type(cluster_dirs)}")

        if len(cluster_dirs) == 0:
            raise ValueError("cluster_dirs list is empty")

        # Validate each cluster directory exists
        for cluster_dir in cluster_dirs:
            cluster_path = Path(cluster_dir)
            if not cluster_path.exists():
                raise ValueError(f"Cluster directory does not exist: {cluster_path}")
            if not cluster_path.is_dir():
                raise ValueError(f"Cluster path is not a directory: {cluster_path}")

        # Validate output_dir
        output_dir = self._get_config_value("output_dir", required=True)
        # Output dir will be created, so just validate it's a valid path
        try:
            Path(output_dir)
        except Exception as e:
            raise ValueError(f"Invalid output_dir path: {e}")

        # Validate numeric parameters
        target_size = self._get_config_value("target_size", default=512)
        if target_size < 64 or target_size > 2048:
            raise ValueError(f"target_size must be between 64 and 2048, got {target_size}")

        min_image_size = self._get_config_value("min_image_size", default=256)
        if min_image_size < 64:
            raise ValueError(f"min_image_size must be >= 64, got {min_image_size}")

        blur_threshold = self._get_config_value("blur_threshold", default=100.0)
        if blur_threshold <= 0:
            raise ValueError(f"blur_threshold must be > 0, got {blur_threshold}")

        max_images_per_char = self._get_config_value("max_images_per_char", default=200)
        if max_images_per_char < 1:
            raise ValueError(f"max_images_per_char must be >= 1, got {max_images_per_char}")

        jpeg_quality = self._get_config_value("jpeg_quality", default=95)
        if jpeg_quality < 1 or jpeg_quality > 100:
            raise ValueError(f"jpeg_quality must be between 1 and 100, got {jpeg_quality}")

        return True

    def execute(self, inputs: Dict[str, Any]) -> StageResult:
        """
        Execute training data preparation

        Args:
            inputs: Input data from previous stages
                   Can contain 'cluster_dirs' from clustering stage

        Returns:
            Stage execution result
        """
        self._mark_started()

        try:
            # Get configuration
            cluster_dirs = self._get_config_value("cluster_dirs", required=True)

            # Handle template strings from previous stages
            if isinstance(cluster_dirs, str) and "{" in cluster_dirs:
                from ..common import parse_stage_outputs
                cluster_dirs = parse_stage_outputs(cluster_dirs, inputs)
                # If still a string, try to parse as list
                if isinstance(cluster_dirs, str):
                    import ast
                    cluster_dirs = ast.literal_eval(cluster_dirs)

            output_dir = Path(self._get_config_value("output_dir", required=True))
            target_size = self._get_config_value("target_size", default=512)
            min_image_size = self._get_config_value("min_image_size", default=256)
            blur_threshold = self._get_config_value("blur_threshold", default=100.0)
            max_images_per_char = self._get_config_value("max_images_per_char", default=200)
            generate_captions = self._get_config_value("generate_captions", default=False)
            caption_template = self._get_config_value("caption_template", default="character")
            jpeg_quality = self._get_config_value("jpeg_quality", default=95)

            # Create output directory
            output_dir.mkdir(parents=True, exist_ok=True)

            # Process each cluster
            dataset_stats = {}
            total_images = 0
            total_filtered = 0
            character_count = 0

            logger.info(f"Processing {len(cluster_dirs)} character clusters...")

            for cluster_idx, cluster_dir in enumerate(cluster_dirs):
                cluster_path = Path(cluster_dir)

                # Create character directory
                char_name = f"character_{cluster_idx:02d}"
                char_dir = output_dir / char_name / "img"
                char_dir.mkdir(parents=True, exist_ok=True)

                # Find all images in cluster
                image_files = sorted(
                    list(cluster_path.glob("*.png")) +
                    list(cluster_path.glob("*.jpg")) +
                    list(cluster_path.glob("*.jpeg"))
                )

                if len(image_files) == 0:
                    logger.warning(f"No images found in cluster: {cluster_path}")
                    continue

                # Process images with quality filtering
                processed_count = 0
                filtered_count = 0

                for img_path in image_files:
                    # Stop if we hit the max images per character
                    if processed_count >= max_images_per_char:
                        logger.info(f"Reached max images ({max_images_per_char}) for {char_name}")
                        break

                    try:
                        # Load image
                        img = Image.open(img_path).convert("RGB")
                        width, height = img.size

                        # Quality filter: minimum size
                        if min(width, height) < min_image_size:
                            filtered_count += 1
                            continue

                        # Quality filter: blur detection
                        if self._is_blurry(img, blur_threshold):
                            filtered_count += 1
                            continue

                        # Resize image (preserve aspect ratio, then crop)
                        resized_img = self._resize_and_crop(img, target_size)

                        # Generate output filename
                        output_filename = f"{char_name}_{processed_count:04d}.jpg"
                        output_path = char_dir / output_filename

                        # Save image
                        resized_img.save(
                            output_path,
                            "JPEG",
                            quality=jpeg_quality,
                            optimize=True
                        )

                        # Generate caption if requested
                        if generate_captions:
                            caption_path = output_path.with_suffix('.txt')
                            with open(caption_path, 'w') as f:
                                f.write(caption_template)

                        processed_count += 1

                    except Exception as e:
                        logger.warning(f"Failed to process {img_path.name}: {e}")
                        filtered_count += 1
                        continue

                # Update stats
                if processed_count > 0:
                    dataset_stats[char_name] = {
                        "source_cluster": str(cluster_path),
                        "image_count": processed_count,
                        "filtered_count": filtered_count,
                        "output_dir": str(char_dir)
                    }
                    total_images += processed_count
                    total_filtered += filtered_count
                    character_count += 1

                    logger.info(f"Character {cluster_idx}: {processed_count} images prepared, {filtered_count} filtered")

            # Create dataset_info.json
            dataset_info = {
                "total_images": total_images,
                "total_filtered": total_filtered,
                "character_count": character_count,
                "characters": dataset_stats,
                "config": {
                    "target_size": target_size,
                    "min_image_size": min_image_size,
                    "blur_threshold": blur_threshold,
                    "max_images_per_char": max_images_per_char,
                    "jpeg_quality": jpeg_quality
                }
            }

            dataset_info_path = output_dir / "dataset_info.json"
            with open(dataset_info_path, 'w') as f:
                json.dump(dataset_info, f, indent=2)

            # Prepare outputs
            outputs = {
                "output_dir": str(output_dir),
                "dataset_stats": dataset_stats,
                "image_count": total_images,
                "character_count": character_count,
                "filtered_count": total_filtered
            }

            # Extract metrics
            metrics = {
                "images_prepared": float(total_images),
                "characters_prepared": float(character_count),
                "images_filtered": float(total_filtered),
                "avg_images_per_char": float(total_images / character_count) if character_count > 0 else 0.0
            }

            self._mark_completed()
            return self._create_success_result(outputs, metrics)

        except Exception as e:
            logger.error(f"Training data preparation failed: {e}")
            self._mark_completed()
            return self._create_failure_result(str(e))

    def _is_blurry(self, img: Image.Image, threshold: float) -> bool:
        """
        Detect if image is blurry using Laplacian variance

        Args:
            img: PIL Image
            threshold: Blur threshold (lower = stricter)

        Returns:
            True if image is blurry
        """
        try:
            # Convert to grayscale numpy array
            gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)

            # Calculate Laplacian variance
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

            # Lower variance = more blurry
            return laplacian_var < threshold

        except Exception as e:
            logger.warning(f"Blur detection failed: {e}")
            return False  # Don't filter on error

    def _resize_and_crop(self, img: Image.Image, target_size: int) -> Image.Image:
        """
        Resize image to target size, preserving aspect ratio, then center crop

        Args:
            img: PIL Image
            target_size: Target size (square)

        Returns:
            Resized and cropped image
        """
        width, height = img.size

        # Calculate scaling to fit target_size
        if width < height:
            # Portrait: scale width to target_size
            new_width = target_size
            new_height = int(height * (target_size / width))
        else:
            # Landscape: scale height to target_size
            new_height = target_size
            new_width = int(width * (target_size / height))

        # Resize
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Center crop to square
        left = (new_width - target_size) // 2
        top = (new_height - target_size) // 2
        right = left + target_size
        bottom = top + target_size

        img = img.crop((left, top, right, bottom))

        return img

    def estimate_duration(self) -> float:
        """
        Estimate training data preparation duration

        Returns:
            Estimated duration in seconds

        Notes:
            Image processing is relatively fast:
            - Loading: ~0.01 sec/image
            - Resize: ~0.02 sec/image
            - Blur detection: ~0.03 sec/image
            - Save: ~0.02 sec/image
            Total: ~0.08 sec/image
        """
        time_per_image = 0.08  # seconds

        try:
            # Estimate total images from cluster_dirs
            cluster_dirs = self._get_config_value("cluster_dirs", required=True)

            if isinstance(cluster_dirs, list):
                total_images = 0
                for cluster_dir in cluster_dirs:
                    cluster_path = Path(cluster_dir)
                    if cluster_path.exists():
                        image_count = (
                            len(list(cluster_path.glob("*.png"))) +
                            len(list(cluster_path.glob("*.jpg"))) +
                            len(list(cluster_path.glob("*.jpeg")))
                        )
                        total_images += image_count

                if total_images > 0:
                    # Add 20% overhead
                    return total_images * time_per_image * 1.2

        except Exception:
            pass

        # Default: 5 minutes
        return 300.0

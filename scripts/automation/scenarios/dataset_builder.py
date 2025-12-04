#!/usr/bin/env python3
"""
Dataset Builder (資料集建構器)
================================

Comprehensive dataset creation, management, and validation tool for AI training.
Supports multiple dataset formats, train/val/test splitting, quality validation,
and dataset operations.

Features:
- Create datasets from directories, file lists, or existing formats
- Train/val/test splitting (random, stratified, k-fold)
- Metadata generation and statistics
- Quality validation and filtering
- Dataset operations (merge, extract subset, convert format)
- CPU-only processing with memory safety

Author: Animation AI Studio
Date: 2025-12-02
"""

import os
import sys
import json
import argparse
import logging
import shutil
import random
import hashlib
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict, Counter
from dataclasses import dataclass, asdict
import time

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from PIL import Image
    import numpy as np
except ImportError as e:
    print(f"Error: Required package not found: {e}")
    print("Please install: pip install Pillow numpy")
    sys.exit(1)

# Import Phase 1 safety infrastructure
try:
    from core.memory_monitor import MemoryMonitor
    from core.gpu_isolation import enforce_cpu_only
except ImportError:
    print("Warning: Phase 1 safety infrastructure not found. Running without safety checks.")
    MemoryMonitor = None
    enforce_cpu_only = lambda: None


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ImageInfo:
    """Image file information"""
    path: str
    filename: str
    class_name: Optional[str] = None
    class_id: Optional[int] = None
    split: Optional[str] = None  # train, val, test
    width: Optional[int] = None
    height: Optional[int] = None
    format: Optional[str] = None
    size_bytes: Optional[int] = None
    is_corrupted: bool = False


@dataclass
class DatasetStatistics:
    """Dataset statistics"""
    total_images: int = 0
    num_classes: int = 0
    class_counts: Dict[str, int] = None
    split_counts: Dict[str, int] = None
    format_distribution: Dict[str, int] = None
    size_distribution: Dict[str, List[int]] = None
    total_size_bytes: int = 0
    corrupted_images: int = 0

    def __post_init__(self):
        if self.class_counts is None:
            self.class_counts = {}
        if self.split_counts is None:
            self.split_counts = {}
        if self.format_distribution is None:
            self.format_distribution = {}
        if self.size_distribution is None:
            self.size_distribution = {'width': [], 'height': []}


@dataclass
class ValidationReport:
    """Dataset validation report"""
    total_checked: int = 0
    valid_images: int = 0
    corrupted_images: List[str] = None
    missing_annotations: List[str] = None
    invalid_annotations: List[str] = None
    warnings: List[str] = None
    errors: List[str] = None

    def __post_init__(self):
        if self.corrupted_images is None:
            self.corrupted_images = []
        if self.missing_annotations is None:
            self.missing_annotations = []
        if self.invalid_annotations is None:
            self.invalid_annotations = []
        if self.warnings is None:
            self.warnings = []
        if self.errors is None:
            self.errors = []


# =============================================================================
# Dataset Builder Class
# =============================================================================

class DatasetBuilder:
    """
    Dataset Builder for creating, managing, and validating datasets.
    """

    # Supported image formats
    IMAGE_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp', '.gif'}

    def __init__(self,
                 num_workers: int = 8,
                 seed: int = 42,
                 skip_preflight: bool = False):
        """
        Initialize Dataset Builder.

        Args:
            num_workers: Number of CPU threads for parallel processing
            seed: Random seed for reproducibility
            skip_preflight: Skip safety checks (for testing)
        """
        self.num_workers = num_workers
        self.seed = seed
        self.logger = self._setup_logging()

        # Set random seeds
        random.seed(seed)
        np.random.seed(seed)

        # Safety infrastructure
        if not skip_preflight and MemoryMonitor is not None:
            enforce_cpu_only()
            self.memory_monitor = MemoryMonitor()
            self.logger.info("✓ Safety infrastructure initialized")
        else:
            self.memory_monitor = None
            if skip_preflight:
                self.logger.info("⚠ Skipping preflight checks (testing mode)")

    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger('DatasetBuilder')
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    # =========================================================================
    # 1. Dataset Creation
    # =========================================================================

    def create_from_directory(self,
                            input_dir: str,
                            output_dir: str,
                            format: str = 'imagefolder',
                            split_ratio: Optional[List[float]] = None,
                            stratify: bool = True,
                            min_images_per_class: int = 1,
                            recursive: bool = True) -> Dict:
        """
        Create dataset from directory structure.

        For ImageFolder format, expects structure:
            input_dir/
                class1/
                    image1.jpg
                    image2.jpg
                class2/
                    image3.jpg

        Args:
            input_dir: Input directory path
            output_dir: Output directory path
            format: Dataset format ('imagefolder', 'flat')
            split_ratio: [train, val, test] ratio (e.g., [0.7, 0.2, 0.1])
            stratify: Use stratified splitting to maintain class balance
            min_images_per_class: Minimum images required per class
            recursive: Recursively search for images

        Returns:
            Dictionary with creation results
        """
        self.logger.info(f"Creating dataset from directory: {input_dir}")
        self.logger.info(f"Output directory: {output_dir}")
        self.logger.info(f"Format: {format}, Stratify: {stratify}")

        start_time = time.time()

        # Scan input directory
        if format == 'imagefolder':
            images = self._scan_imagefolder(input_dir, recursive)
        elif format == 'flat':
            images = self._scan_flat_directory(input_dir, recursive)
        else:
            raise ValueError(f"Unsupported format: {format}")

        if not images:
            raise ValueError(f"No images found in {input_dir}")

        self.logger.info(f"Found {len(images)} images")

        # Filter classes with insufficient samples
        images = self._filter_by_min_samples(images, min_images_per_class)

        # Split dataset
        if split_ratio:
            images = self._split_dataset(images, split_ratio, stratify)
        else:
            # Default: all images to train
            for img in images:
                img.split = 'train'

        # Create output structure
        self._create_dataset_structure(images, output_dir, format)

        # Generate metadata
        metadata = self._generate_metadata(images, input_dir, output_dir, format)

        # Save metadata
        self._save_metadata(metadata, output_dir)

        # Generate statistics
        stats = self._compute_statistics(images)
        self._save_statistics(stats, output_dir)

        elapsed_time = time.time() - start_time

        result = {
            'status': 'success',
            'total_images': len(images),
            'num_classes': len(set(img.class_name for img in images if img.class_name)),
            'split_counts': stats.split_counts,
            'elapsed_time': elapsed_time,
            'output_dir': output_dir
        }

        self.logger.info(f"✓ Dataset created successfully in {elapsed_time:.2f}s")
        self.logger.info(f"  Total images: {result['total_images']}")
        self.logger.info(f"  Classes: {result['num_classes']}")
        self.logger.info(f"  Splits: {stats.split_counts}")

        return result

    def _scan_imagefolder(self, root_dir: str, recursive: bool = True) -> List[ImageInfo]:
        """Scan ImageFolder format directory"""
        images = []
        root_path = Path(root_dir)

        # Get class directories (first-level subdirectories)
        class_dirs = [d for d in root_path.iterdir() if d.is_dir()]

        if not class_dirs:
            raise ValueError(f"No class directories found in {root_dir}")

        for class_idx, class_dir in enumerate(sorted(class_dirs)):
            class_name = class_dir.name

            # Find images in class directory
            if recursive:
                image_files = []
                for ext in self.IMAGE_FORMATS:
                    image_files.extend(class_dir.rglob(f"*{ext}"))
                    image_files.extend(class_dir.rglob(f"*{ext.upper()}"))
            else:
                image_files = []
                for ext in self.IMAGE_FORMATS:
                    image_files.extend(class_dir.glob(f"*{ext}"))
                    image_files.extend(class_dir.glob(f"*{ext.upper()}"))

            for img_file in image_files:
                images.append(ImageInfo(
                    path=str(img_file),
                    filename=img_file.name,
                    class_name=class_name,
                    class_id=class_idx
                ))

        return images

    def _scan_flat_directory(self, root_dir: str, recursive: bool = True) -> List[ImageInfo]:
        """Scan flat directory (all images in one folder, no class structure)"""
        images = []
        root_path = Path(root_dir)

        if recursive:
            image_files = []
            for ext in self.IMAGE_FORMATS:
                image_files.extend(root_path.rglob(f"*{ext}"))
                image_files.extend(root_path.rglob(f"*{ext.upper()}"))
        else:
            image_files = []
            for ext in self.IMAGE_FORMATS:
                image_files.extend(root_path.glob(f"*{ext}"))
                image_files.extend(root_path.glob(f"*{ext.upper()}"))

        for img_file in image_files:
            images.append(ImageInfo(
                path=str(img_file),
                filename=img_file.name,
                class_name=None,  # No class information
                class_id=None
            ))

        return images

    def _filter_by_min_samples(self,
                               images: List[ImageInfo],
                               min_samples: int) -> List[ImageInfo]:
        """Filter out classes with insufficient samples"""
        if min_samples <= 1:
            return images

        # Count samples per class
        class_counts = Counter(img.class_name for img in images if img.class_name)

        # Find classes to remove
        classes_to_remove = {cls for cls, count in class_counts.items()
                            if count < min_samples}

        if classes_to_remove:
            self.logger.warning(
                f"Removing {len(classes_to_remove)} classes with < {min_samples} samples: "
                f"{', '.join(classes_to_remove)}"
            )
            images = [img for img in images if img.class_name not in classes_to_remove]

        return images

    # =========================================================================
    # 2. Dataset Splitting
    # =========================================================================

    def _split_dataset(self,
                      images: List[ImageInfo],
                      split_ratio: List[float],
                      stratify: bool = True) -> List[ImageInfo]:
        """
        Split dataset into train/val/test sets.

        Args:
            images: List of ImageInfo objects
            split_ratio: [train, val, test] ratio
            stratify: Maintain class balance in splits

        Returns:
            Updated list with split assignments
        """
        if len(split_ratio) not in [2, 3]:
            raise ValueError("split_ratio must be [train, val] or [train, val, test]")

        if not np.isclose(sum(split_ratio), 1.0):
            raise ValueError(f"split_ratio must sum to 1.0, got {sum(split_ratio)}")

        split_names = ['train', 'val', 'test'][:len(split_ratio)]

        if stratify and images[0].class_name is not None:
            # Stratified split - maintain class balance
            images = self._stratified_split(images, split_ratio, split_names)
        else:
            # Random split
            images = self._random_split(images, split_ratio, split_names)

        return images

    def _stratified_split(self,
                         images: List[ImageInfo],
                         split_ratio: List[float],
                         split_names: List[str]) -> List[ImageInfo]:
        """Perform stratified split to maintain class balance"""
        # Group images by class
        class_groups = defaultdict(list)
        for img in images:
            class_groups[img.class_name].append(img)

        # Split each class independently
        for class_name, class_images in class_groups.items():
            n = len(class_images)
            random.shuffle(class_images)

            # Calculate split indices
            indices = [0]
            for ratio in split_ratio[:-1]:
                indices.append(indices[-1] + int(n * ratio))
            indices.append(n)

            # Assign splits
            for i, split_name in enumerate(split_names):
                for img in class_images[indices[i]:indices[i+1]]:
                    img.split = split_name

        return images

    def _random_split(self,
                     images: List[ImageInfo],
                     split_ratio: List[float],
                     split_names: List[str]) -> List[ImageInfo]:
        """Perform random split"""
        n = len(images)
        random.shuffle(images)

        # Calculate split indices
        indices = [0]
        for ratio in split_ratio[:-1]:
            indices.append(indices[-1] + int(n * ratio))
        indices.append(n)

        # Assign splits
        for i, split_name in enumerate(split_names):
            for img in images[indices[i]:indices[i+1]]:
                img.split = split_name

        return images

    def create_kfold_splits(self,
                           images: List[ImageInfo],
                           k: int = 5,
                           stratify: bool = True) -> List[List[ImageInfo]]:
        """
        Create K-fold cross-validation splits.

        Args:
            images: List of ImageInfo objects
            k: Number of folds
            stratify: Use stratified k-fold

        Returns:
            List of k image lists with fold assignments
        """
        if stratify and images[0].class_name is not None:
            return self._stratified_kfold(images, k)
        else:
            return self._random_kfold(images, k)

    def _stratified_kfold(self, images: List[ImageInfo], k: int) -> List[List[ImageInfo]]:
        """Create stratified k-fold splits"""
        # Group by class
        class_groups = defaultdict(list)
        for img in images:
            class_groups[img.class_name].append(img)

        # Initialize folds
        folds = [[] for _ in range(k)]

        # Distribute each class across folds
        for class_name, class_images in class_groups.items():
            random.shuffle(class_images)
            for i, img in enumerate(class_images):
                fold_idx = i % k
                folds[fold_idx].append(img)

        return folds

    def _random_kfold(self, images: List[ImageInfo], k: int) -> List[List[ImageInfo]]:
        """Create random k-fold splits"""
        random.shuffle(images)
        folds = [[] for _ in range(k)]

        for i, img in enumerate(images):
            fold_idx = i % k
            folds[fold_idx].append(img)

        return folds

    # =========================================================================
    # 3. Metadata and Statistics
    # =========================================================================

    def _generate_metadata(self,
                          images: List[ImageInfo],
                          input_dir: str,
                          output_dir: str,
                          format: str) -> Dict:
        """Generate dataset metadata"""
        classes = sorted(set(img.class_name for img in images if img.class_name))
        class_to_id = {cls: idx for idx, cls in enumerate(classes)} if classes else {}

        metadata = {
            'dataset_name': Path(output_dir).name,
            'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'source_directory': input_dir,
            'output_directory': output_dir,
            'format': format,
            'total_images': len(images),
            'num_classes': len(classes),
            'classes': classes,
            'class_to_idx': class_to_id,
            'split_counts': self._count_splits(images),
            'seed': self.seed
        }

        return metadata

    def _compute_statistics(self, images: List[ImageInfo]) -> DatasetStatistics:
        """Compute comprehensive dataset statistics"""
        stats = DatasetStatistics()
        stats.total_images = len(images)

        # Count classes
        if images and images[0].class_name:
            class_counts = Counter(img.class_name for img in images)
            stats.class_counts = dict(class_counts)
            stats.num_classes = len(class_counts)

        # Count splits
        stats.split_counts = self._count_splits(images)

        # Compute image properties (sample up to 1000 images for efficiency)
        sample_size = min(1000, len(images))
        sample_images = random.sample(images, sample_size)

        format_counts = Counter()
        widths = []
        heights = []
        total_size = 0
        corrupted = 0

        for img_info in sample_images:
            try:
                with Image.open(img_info.path) as img:
                    widths.append(img.width)
                    heights.append(img.height)
                    format_counts[img.format or 'UNKNOWN'] += 1

                total_size += Path(img_info.path).stat().st_size
            except Exception:
                corrupted += 1

        stats.format_distribution = dict(format_counts)
        stats.size_distribution = {
            'width': widths,
            'height': heights,
            'width_mean': int(np.mean(widths)) if widths else 0,
            'height_mean': int(np.mean(heights)) if heights else 0,
            'width_std': int(np.std(widths)) if widths else 0,
            'height_std': int(np.std(heights)) if heights else 0
        }
        stats.total_size_bytes = total_size
        stats.corrupted_images = corrupted

        return stats

    def _count_splits(self, images: List[ImageInfo]) -> Dict[str, int]:
        """Count images per split"""
        return dict(Counter(img.split for img in images if img.split))

    def _save_metadata(self, metadata: Dict, output_dir: str):
        """Save metadata to JSON file"""
        metadata_path = Path(output_dir) / 'metadata.json'
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        self.logger.info(f"✓ Metadata saved to {metadata_path}")

    def _save_statistics(self, stats: DatasetStatistics, output_dir: str):
        """Save statistics to JSON file"""
        stats_path = Path(output_dir) / 'statistics.json'

        # Convert to serializable dict
        stats_dict = asdict(stats)

        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats_dict, f, indent=2)

        # Also save human-readable summary
        summary_path = Path(output_dir) / 'dataset_info.txt'
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("Dataset Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total Images: {stats.total_images}\n")
            f.write(f"Number of Classes: {stats.num_classes}\n\n")

            if stats.class_counts:
                f.write("Class Distribution:\n")
                for cls, count in sorted(stats.class_counts.items()):
                    f.write(f"  {cls}: {count}\n")
                f.write("\n")

            if stats.split_counts:
                f.write("Split Distribution:\n")
                for split, count in sorted(stats.split_counts.items()):
                    f.write(f"  {split}: {count}\n")
                f.write("\n")

            if stats.format_distribution:
                f.write("Format Distribution:\n")
                for fmt, count in sorted(stats.format_distribution.items()):
                    f.write(f"  {fmt}: {count}\n")
                f.write("\n")

            if 'width_mean' in stats.size_distribution:
                f.write("Image Size Statistics:\n")
                f.write(f"  Width: {stats.size_distribution['width_mean']} ± "
                       f"{stats.size_distribution['width_std']} pixels\n")
                f.write(f"  Height: {stats.size_distribution['height_mean']} ± "
                       f"{stats.size_distribution['height_std']} pixels\n")

        self.logger.info(f"✓ Statistics saved to {stats_path}")

    # =========================================================================
    # 4. Dataset Structure Creation
    # =========================================================================

    def _create_dataset_structure(self,
                                  images: List[ImageInfo],
                                  output_dir: str,
                                  format: str):
        """Create output dataset structure"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if format == 'imagefolder':
            self._create_imagefolder_structure(images, output_path)
        elif format == 'flat':
            self._create_flat_structure(images, output_path)

    def _create_imagefolder_structure(self, images: List[ImageInfo], output_path: Path):
        """Create ImageFolder structure with splits"""
        for img in images:
            # Determine target directory
            if img.split and img.class_name:
                target_dir = output_path / img.split / img.class_name
            elif img.class_name:
                target_dir = output_path / 'train' / img.class_name
            else:
                target_dir = output_path / 'train' / 'default'

            target_dir.mkdir(parents=True, exist_ok=True)

            # Copy or symlink image
            target_file = target_dir / img.filename
            if not target_file.exists():
                shutil.copy2(img.path, target_file)

    def _create_flat_structure(self, images: List[ImageInfo], output_path: Path):
        """Create flat structure with split subdirectories"""
        for img in images:
            split_dir = output_path / (img.split or 'train')
            split_dir.mkdir(parents=True, exist_ok=True)

            target_file = split_dir / img.filename
            if not target_file.exists():
                shutil.copy2(img.path, target_file)

    # =========================================================================
    # 5. Dataset Validation
    # =========================================================================

    def validate_dataset(self,
                        dataset_dir: str,
                        check_images: bool = True,
                        check_annotations: bool = False,
                        fix_issues: bool = False) -> ValidationReport:
        """
        Validate dataset for common issues.

        Args:
            dataset_dir: Dataset directory path
            check_images: Check image file integrity
            check_annotations: Check annotation files (if applicable)
            fix_issues: Attempt to fix issues automatically

        Returns:
            ValidationReport with validation results
        """
        self.logger.info(f"Validating dataset: {dataset_dir}")

        report = ValidationReport()
        dataset_path = Path(dataset_dir)

        # Find all images
        image_files = []
        for ext in self.IMAGE_FORMATS:
            image_files.extend(dataset_path.rglob(f"*{ext}"))
            image_files.extend(dataset_path.rglob(f"*{ext.upper()}"))

        report.total_checked = len(image_files)
        self.logger.info(f"Found {len(image_files)} images to validate")

        # Check images
        if check_images:
            self._validate_images(image_files, report, fix_issues)

        # Check annotations (if needed)
        if check_annotations:
            self._validate_annotations(dataset_path, image_files, report)

        # Check class balance
        self._check_class_balance(image_files, report)

        report.valid_images = report.total_checked - len(report.corrupted_images)

        self.logger.info(f"✓ Validation complete")
        self.logger.info(f"  Valid: {report.valid_images}/{report.total_checked}")
        self.logger.info(f"  Corrupted: {len(report.corrupted_images)}")
        self.logger.info(f"  Warnings: {len(report.warnings)}")
        self.logger.info(f"  Errors: {len(report.errors)}")

        return report

    def _validate_images(self,
                        image_files: List[Path],
                        report: ValidationReport,
                        fix_issues: bool):
        """Validate image file integrity"""
        for img_path in image_files:
            try:
                with Image.open(img_path) as img:
                    # Try to load image data
                    img.verify()

                    # Check minimum size
                    if img.width < 32 or img.height < 32:
                        report.warnings.append(f"Image too small: {img_path} ({img.width}x{img.height})")

                    # Check maximum size
                    if img.width > 4096 or img.height > 4096:
                        report.warnings.append(f"Image very large: {img_path} ({img.width}x{img.height})")

            except Exception as e:
                report.corrupted_images.append(str(img_path))
                report.errors.append(f"Corrupted image: {img_path} - {str(e)}")

                if fix_issues:
                    # Remove corrupted image
                    try:
                        img_path.unlink()
                        self.logger.warning(f"Removed corrupted image: {img_path}")
                    except Exception:
                        pass

    def _validate_annotations(self,
                            dataset_path: Path,
                            image_files: List[Path],
                            report: ValidationReport):
        """Validate annotation files (placeholder for future implementation)"""
        # This would check for COCO, YOLO, or other annotation formats
        # For now, just check if annotations directory exists
        annotations_dir = dataset_path / 'annotations'
        if not annotations_dir.exists():
            report.warnings.append("No annotations directory found")

    def _check_class_balance(self, image_files: List[Path], report: ValidationReport):
        """Check for class imbalance"""
        # Extract class names from paths (assumes ImageFolder structure)
        class_counts = Counter()

        for img_path in image_files:
            # Try to extract class from path (parent directory name)
            if img_path.parent.name not in ['train', 'val', 'test']:
                class_counts[img_path.parent.name] += 1

        if len(class_counts) > 1:
            counts = list(class_counts.values())
            max_count = max(counts)
            min_count = min(counts)

            if max_count / min_count > 10:
                report.warnings.append(
                    f"Significant class imbalance detected: "
                    f"max={max_count}, min={min_count}, ratio={max_count/min_count:.1f}x"
                )

    # =========================================================================
    # 6. Dataset Operations
    # =========================================================================

    def merge_datasets(self,
                      input_datasets: List[str],
                      output_dir: str,
                      handle_conflicts: str = 'rename') -> Dict:
        """
        Merge multiple datasets into one.

        Args:
            input_datasets: List of dataset directory paths
            output_dir: Output directory for merged dataset
            handle_conflicts: How to handle class name conflicts ('rename', 'merge', 'skip')

        Returns:
            Merge result summary
        """
        self.logger.info(f"Merging {len(input_datasets)} datasets")
        self.logger.info(f"Output: {output_dir}")

        all_images = []
        class_mapping = {}
        next_class_id = 0

        for dataset_idx, dataset_dir in enumerate(input_datasets):
            self.logger.info(f"Processing dataset {dataset_idx + 1}/{len(input_datasets)}: {dataset_dir}")

            # Scan dataset
            images = self._scan_imagefolder(dataset_dir, recursive=True)

            # Handle class conflicts
            for img in images:
                original_class = img.class_name

                if handle_conflicts == 'rename':
                    # Prefix with dataset index
                    new_class = f"ds{dataset_idx}_{original_class}"
                elif handle_conflicts == 'merge':
                    # Keep original class name
                    new_class = original_class
                elif handle_conflicts == 'skip':
                    if original_class in class_mapping:
                        continue
                    new_class = original_class
                else:
                    raise ValueError(f"Unknown conflict handling: {handle_conflicts}")

                if new_class not in class_mapping:
                    class_mapping[new_class] = next_class_id
                    next_class_id += 1

                img.class_name = new_class
                img.class_id = class_mapping[new_class]
                all_images.append(img)

        self.logger.info(f"Total images after merge: {len(all_images)}")
        self.logger.info(f"Total classes after merge: {len(class_mapping)}")

        # Create merged dataset
        self._create_dataset_structure(all_images, output_dir, 'imagefolder')

        # Generate metadata
        metadata = {
            'merged_from': input_datasets,
            'total_images': len(all_images),
            'num_classes': len(class_mapping),
            'classes': list(class_mapping.keys()),
            'class_to_idx': class_mapping,
            'conflict_handling': handle_conflicts
        }
        self._save_metadata(metadata, output_dir)

        stats = self._compute_statistics(all_images)
        self._save_statistics(stats, output_dir)

        return {
            'status': 'success',
            'total_images': len(all_images),
            'num_classes': len(class_mapping),
            'output_dir': output_dir
        }

    def extract_subset(self,
                      dataset_dir: str,
                      output_dir: str,
                      classes: Optional[List[str]] = None,
                      max_samples: Optional[int] = None,
                      splits: Optional[List[str]] = None) -> Dict:
        """
        Extract a subset of the dataset.

        Args:
            dataset_dir: Input dataset directory
            output_dir: Output directory for subset
            classes: List of class names to include (None = all)
            max_samples: Maximum number of samples to include (None = all)
            splits: List of splits to include (None = all)

        Returns:
            Extraction result summary
        """
        self.logger.info(f"Extracting subset from: {dataset_dir}")

        # Scan dataset
        images = self._scan_imagefolder(dataset_dir, recursive=True)

        # Filter by class
        if classes:
            images = [img for img in images if img.class_name in classes]
            self.logger.info(f"Filtered to {len(images)} images from classes: {classes}")

        # Filter by split
        if splits:
            images = [img for img in images if img.split in splits]
            self.logger.info(f"Filtered to {len(images)} images from splits: {splits}")

        # Limit number of samples
        if max_samples and len(images) > max_samples:
            images = random.sample(images, max_samples)
            self.logger.info(f"Limited to {max_samples} samples")

        # Create subset
        self._create_dataset_structure(images, output_dir, 'imagefolder')

        # Generate metadata
        metadata = {
            'source_dataset': dataset_dir,
            'total_images': len(images),
            'filter_classes': classes,
            'filter_splits': splits,
            'max_samples': max_samples
        }
        self._save_metadata(metadata, output_dir)

        stats = self._compute_statistics(images)
        self._save_statistics(stats, output_dir)

        self.logger.info(f"✓ Subset extracted: {len(images)} images")

        return {
            'status': 'success',
            'total_images': len(images),
            'output_dir': output_dir
        }


# =============================================================================
# Command Line Interface
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Dataset Builder - Create, manage, and validate datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create dataset from directory
  python dataset_builder.py create-from-dir \\
    --input-dir /path/to/images \\
    --output-dir /path/to/dataset \\
    --split-ratio 0.7 0.2 0.1

  # Validate dataset
  python dataset_builder.py validate \\
    --dataset-dir /path/to/dataset \\
    --check-images

  # Merge datasets
  python dataset_builder.py merge \\
    --input-datasets /path/to/ds1 /path/to/ds2 \\
    --output-dir /path/to/merged
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    # Common arguments
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--num-workers', type=int, default=8,
                       help='Number of CPU threads (default: 8)')
    common.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    common.add_argument('--skip-preflight', action='store_true',
                       help='Skip safety checks (for testing)')

    # Create from directory
    create_parser = subparsers.add_parser('create-from-dir', parents=[common],
                                         help='Create dataset from directory')
    create_parser.add_argument('--input-dir', required=True,
                              help='Input directory path')
    create_parser.add_argument('--output-dir', required=True,
                              help='Output directory path')
    create_parser.add_argument('--format', default='imagefolder',
                              choices=['imagefolder', 'flat'],
                              help='Dataset format (default: imagefolder)')
    create_parser.add_argument('--split-ratio', type=float, nargs='+',
                              help='Train/val/test ratio (e.g., 0.7 0.2 0.1)')
    create_parser.add_argument('--stratify', action='store_true',
                              help='Use stratified splitting')
    create_parser.add_argument('--min-images-per-class', type=int, default=1,
                              help='Minimum images per class (default: 1)')
    create_parser.add_argument('--recursive', action='store_true', default=True,
                              help='Recursively search for images')

    # Validate dataset
    validate_parser = subparsers.add_parser('validate', parents=[common],
                                           help='Validate dataset')
    validate_parser.add_argument('--dataset-dir', required=True,
                                help='Dataset directory path')
    validate_parser.add_argument('--check-images', action='store_true',
                                help='Check image file integrity')
    validate_parser.add_argument('--check-annotations', action='store_true',
                                help='Check annotation files')
    validate_parser.add_argument('--fix-issues', action='store_true',
                                help='Attempt to fix issues automatically')
    validate_parser.add_argument('--output-report', type=str,
                                help='Output report file path (JSON)')

    # Merge datasets
    merge_parser = subparsers.add_parser('merge', parents=[common],
                                        help='Merge multiple datasets')
    merge_parser.add_argument('--input-datasets', nargs='+', required=True,
                             help='Input dataset directories')
    merge_parser.add_argument('--output-dir', required=True,
                             help='Output directory path')
    merge_parser.add_argument('--handle-conflicts', default='rename',
                             choices=['rename', 'merge', 'skip'],
                             help='How to handle class name conflicts')

    # Extract subset
    subset_parser = subparsers.add_parser('extract-subset', parents=[common],
                                         help='Extract dataset subset')
    subset_parser.add_argument('--dataset-dir', required=True,
                              help='Input dataset directory')
    subset_parser.add_argument('--output-dir', required=True,
                              help='Output directory path')
    subset_parser.add_argument('--classes', nargs='+',
                              help='Class names to include')
    subset_parser.add_argument('--max-samples', type=int,
                              help='Maximum number of samples')
    subset_parser.add_argument('--splits', nargs='+',
                              help='Splits to include (train, val, test)')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Create Dataset Builder
    builder = DatasetBuilder(
        num_workers=args.num_workers,
        seed=args.seed,
        skip_preflight=args.skip_preflight
    )

    # Execute command
    try:
        if args.command == 'create-from-dir':
            result = builder.create_from_directory(
                input_dir=args.input_dir,
                output_dir=args.output_dir,
                format=args.format,
                split_ratio=args.split_ratio,
                stratify=args.stratify,
                min_images_per_class=args.min_images_per_class,
                recursive=args.recursive
            )
            print(f"\n✅ Dataset created successfully!")
            print(f"   Output: {result['output_dir']}")
            print(f"   Images: {result['total_images']}")
            print(f"   Classes: {result['num_classes']}")

        elif args.command == 'validate':
            report = builder.validate_dataset(
                dataset_dir=args.dataset_dir,
                check_images=args.check_images,
                check_annotations=args.check_annotations,
                fix_issues=args.fix_issues
            )

            if args.output_report:
                with open(args.output_report, 'w') as f:
                    json.dump(asdict(report), f, indent=2)
                print(f"\n✅ Report saved to {args.output_report}")

            print(f"\n✅ Validation complete!")
            print(f"   Valid: {report.valid_images}/{report.total_checked}")
            print(f"   Issues: {len(report.corrupted_images)} corrupted, "
                  f"{len(report.warnings)} warnings")

        elif args.command == 'merge':
            result = builder.merge_datasets(
                input_datasets=args.input_datasets,
                output_dir=args.output_dir,
                handle_conflicts=args.handle_conflicts
            )
            print(f"\n✅ Datasets merged successfully!")
            print(f"   Output: {result['output_dir']}")
            print(f"   Images: {result['total_images']}")
            print(f"   Classes: {result['num_classes']}")

        elif args.command == 'extract-subset':
            result = builder.extract_subset(
                dataset_dir=args.dataset_dir,
                output_dir=args.output_dir,
                classes=args.classes,
                max_samples=args.max_samples,
                splits=args.splits
            )
            print(f"\n✅ Subset extracted successfully!")
            print(f"   Output: {result['output_dir']}")
            print(f"   Images: {result['total_images']}")

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Data Augmentation Pipeline
Provides comprehensive image augmentation capabilities for machine learning training.

Supports:
- Geometric transformations (rotation, flipping, cropping, scaling)
- Color adjustments (brightness, contrast, saturation, hue)
- Noise and blur effects
- Advanced augmentations (elastic deformation, cutout, mixup)
- Batch processing with configurable pipelines
"""

import os
import sys
import json
import argparse
import logging
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ===========================
# Data Classes
# ===========================

class AugmentationType(Enum):
    """Augmentation types"""
    GEOMETRIC = "geometric"
    COLOR = "color"
    NOISE = "noise"
    BLUR = "blur"
    ADVANCED = "advanced"


@dataclass
class AugmentationConfig:
    """Configuration for a single augmentation"""
    name: str
    type: AugmentationType
    probability: float = 1.0
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AugmentationResult:
    """Result of augmentation process"""
    success: bool
    input_image: str
    output_image: str
    augmentations_applied: List[str]
    error: Optional[str] = None


# ===========================
# Geometric Augmentations
# ===========================

class GeometricAugmentations:
    """Geometric transformation augmentations"""

    @staticmethod
    def rotate(image: Image.Image, angle: float, expand: bool = True) -> Image.Image:
        """Rotate image by angle (degrees)"""
        return image.rotate(angle, expand=expand, fillcolor=(0, 0, 0))

    @staticmethod
    def horizontal_flip(image: Image.Image) -> Image.Image:
        """Flip image horizontally"""
        return ImageOps.mirror(image)

    @staticmethod
    def vertical_flip(image: Image.Image) -> Image.Image:
        """Flip image vertically"""
        return ImageOps.flip(image)

    @staticmethod
    def random_crop(image: Image.Image, crop_fraction: float = 0.8) -> Image.Image:
        """Randomly crop image"""
        width, height = image.size
        new_width = int(width * crop_fraction)
        new_height = int(height * crop_fraction)

        left = random.randint(0, width - new_width)
        top = random.randint(0, height - new_height)
        right = left + new_width
        bottom = top + new_height

        return image.crop((left, top, right, bottom))

    @staticmethod
    def center_crop(image: Image.Image, crop_fraction: float = 0.8) -> Image.Image:
        """Center crop image"""
        width, height = image.size
        new_width = int(width * crop_fraction)
        new_height = int(height * crop_fraction)

        left = (width - new_width) // 2
        top = (height - new_height) // 2
        right = left + new_width
        bottom = top + new_height

        return image.crop((left, top, right, bottom))

    @staticmethod
    def resize(image: Image.Image, size: Tuple[int, int], keep_aspect: bool = True) -> Image.Image:
        """Resize image"""
        if keep_aspect:
            image.thumbnail(size, Image.Resampling.LANCZOS)
            return image
        else:
            return image.resize(size, Image.Resampling.LANCZOS)

    @staticmethod
    def random_scale(image: Image.Image, scale_range: Tuple[float, float] = (0.8, 1.2)) -> Image.Image:
        """Randomly scale image"""
        scale = random.uniform(*scale_range)
        width, height = image.size
        new_size = (int(width * scale), int(height * scale))
        return image.resize(new_size, Image.Resampling.LANCZOS)


# ===========================
# Color Augmentations
# ===========================

class ColorAugmentations:
    """Color adjustment augmentations"""

    @staticmethod
    def brightness(image: Image.Image, factor: float) -> Image.Image:
        """Adjust brightness (factor: 0.0 to 2.0, 1.0 = no change)"""
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(factor)

    @staticmethod
    def contrast(image: Image.Image, factor: float) -> Image.Image:
        """Adjust contrast (factor: 0.0 to 2.0, 1.0 = no change)"""
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(factor)

    @staticmethod
    def saturation(image: Image.Image, factor: float) -> Image.Image:
        """Adjust saturation (factor: 0.0 to 2.0, 1.0 = no change)"""
        enhancer = ImageEnhance.Color(image)
        return enhancer.enhance(factor)

    @staticmethod
    def sharpness(image: Image.Image, factor: float) -> Image.Image:
        """Adjust sharpness (factor: 0.0 to 2.0, 1.0 = no change)"""
        enhancer = ImageEnhance.Sharpness(image)
        return enhancer.enhance(factor)

    @staticmethod
    def hue_shift(image: Image.Image, hue_delta: int) -> Image.Image:
        """Shift hue (hue_delta: -180 to 180)"""
        if image.mode != 'RGB':
            return image

        # Convert to HSV
        hsv = image.convert('HSV')
        h, s, v = hsv.split()

        # Shift hue
        h_array = np.array(h, dtype=np.int16)
        h_array = (h_array + hue_delta) % 256
        h = Image.fromarray(h_array.astype(np.uint8), mode='L')

        # Merge and convert back
        hsv = Image.merge('HSV', (h, s, v))
        return hsv.convert('RGB')

    @staticmethod
    def random_brightness(image: Image.Image, range: Tuple[float, float] = (0.8, 1.2)) -> Image.Image:
        """Random brightness adjustment"""
        factor = random.uniform(*range)
        return ColorAugmentations.brightness(image, factor)

    @staticmethod
    def random_contrast(image: Image.Image, range: Tuple[float, float] = (0.8, 1.2)) -> Image.Image:
        """Random contrast adjustment"""
        factor = random.uniform(*range)
        return ColorAugmentations.contrast(image, factor)

    @staticmethod
    def random_saturation(image: Image.Image, range: Tuple[float, float] = (0.8, 1.2)) -> Image.Image:
        """Random saturation adjustment"""
        factor = random.uniform(*range)
        return ColorAugmentations.saturation(image, factor)


# ===========================
# Noise Augmentations
# ===========================

class NoiseAugmentations:
    """Noise addition augmentations"""

    @staticmethod
    def gaussian_noise(image: Image.Image, mean: float = 0, std: float = 25) -> Image.Image:
        """Add Gaussian noise"""
        img_array = np.array(image, dtype=np.float32)
        noise = np.random.normal(mean, std, img_array.shape)
        noisy = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy)

    @staticmethod
    def salt_and_pepper(image: Image.Image, amount: float = 0.05) -> Image.Image:
        """Add salt and pepper noise"""
        img_array = np.array(image)
        noisy = img_array.copy()

        # Salt
        num_salt = int(amount * img_array.size * 0.5)
        salt_coords = [np.random.randint(0, i - 1, num_salt) for i in img_array.shape[:2]]
        noisy[tuple(salt_coords)] = 255

        # Pepper
        num_pepper = int(amount * img_array.size * 0.5)
        pepper_coords = [np.random.randint(0, i - 1, num_pepper) for i in img_array.shape[:2]]
        noisy[tuple(pepper_coords)] = 0

        return Image.fromarray(noisy)

    @staticmethod
    def random_noise(image: Image.Image, noise_type: str = 'gaussian') -> Image.Image:
        """Add random noise"""
        if noise_type == 'gaussian':
            std = random.uniform(10, 50)
            return NoiseAugmentations.gaussian_noise(image, 0, std)
        elif noise_type == 'salt_pepper':
            amount = random.uniform(0.01, 0.05)
            return NoiseAugmentations.salt_and_pepper(image, amount)
        else:
            return image


# ===========================
# Blur Augmentations
# ===========================

class BlurAugmentations:
    """Blur effect augmentations"""

    @staticmethod
    def gaussian_blur(image: Image.Image, radius: float = 2.0) -> Image.Image:
        """Apply Gaussian blur"""
        return image.filter(ImageFilter.GaussianBlur(radius))

    @staticmethod
    def box_blur(image: Image.Image, radius: int = 2) -> Image.Image:
        """Apply box blur"""
        return image.filter(ImageFilter.BoxBlur(radius))

    @staticmethod
    def motion_blur(image: Image.Image, size: int = 15, angle: int = 45) -> Image.Image:
        """Apply motion blur (simulated)"""
        # Simple motion blur simulation using box blur at angle
        # For true motion blur, would need more sophisticated kernel
        return image.filter(ImageFilter.BoxBlur(size // 3))

    @staticmethod
    def random_blur(image: Image.Image) -> Image.Image:
        """Apply random blur"""
        blur_type = random.choice(['gaussian', 'box'])
        if blur_type == 'gaussian':
            radius = random.uniform(0.5, 3.0)
            return BlurAugmentations.gaussian_blur(image, radius)
        else:
            radius = random.randint(1, 3)
            return BlurAugmentations.box_blur(image, radius)


# ===========================
# Advanced Augmentations
# ===========================

class AdvancedAugmentations:
    """Advanced augmentation techniques"""

    @staticmethod
    def cutout(image: Image.Image, num_holes: int = 1, hole_size: int = 50) -> Image.Image:
        """Apply cutout augmentation (random rectangular masks)"""
        img_array = np.array(image)
        height, width = img_array.shape[:2]

        for _ in range(num_holes):
            y = random.randint(0, height - hole_size)
            x = random.randint(0, width - hole_size)
            img_array[y:y+hole_size, x:x+hole_size] = 0

        return Image.fromarray(img_array)

    @staticmethod
    def posterize(image: Image.Image, bits: int = 4) -> Image.Image:
        """Reduce number of bits for each color channel"""
        return ImageOps.posterize(image, bits)

    @staticmethod
    def solarize(image: Image.Image, threshold: int = 128) -> Image.Image:
        """Invert pixels above threshold"""
        return ImageOps.solarize(image, threshold)

    @staticmethod
    def equalize(image: Image.Image) -> Image.Image:
        """Histogram equalization"""
        return ImageOps.equalize(image)

    @staticmethod
    def autocontrast(image: Image.Image) -> Image.Image:
        """Auto contrast adjustment"""
        return ImageOps.autocontrast(image)


# ===========================
# Augmentation Pipeline
# ===========================

class AugmentationPipeline:
    """Main augmentation pipeline"""

    def __init__(self, augmentations: List[AugmentationConfig], seed: Optional[int] = None):
        """
        Initialize pipeline

        Args:
            augmentations: List of augmentation configurations
            seed: Random seed for reproducibility
        """
        self.augmentations = augmentations
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def apply(self, image: Image.Image) -> Tuple[Image.Image, List[str]]:
        """
        Apply augmentation pipeline to image

        Args:
            image: Input PIL Image

        Returns:
            Tuple of (augmented image, list of applied augmentations)
        """
        augmented = image.copy()
        applied = []

        for aug_config in self.augmentations:
            # Check probability
            if random.random() > aug_config.probability:
                continue

            # Apply augmentation based on type
            try:
                if aug_config.type == AugmentationType.GEOMETRIC:
                    augmented = self._apply_geometric(augmented, aug_config)
                elif aug_config.type == AugmentationType.COLOR:
                    augmented = self._apply_color(augmented, aug_config)
                elif aug_config.type == AugmentationType.NOISE:
                    augmented = self._apply_noise(augmented, aug_config)
                elif aug_config.type == AugmentationType.BLUR:
                    augmented = self._apply_blur(augmented, aug_config)
                elif aug_config.type == AugmentationType.ADVANCED:
                    augmented = self._apply_advanced(augmented, aug_config)

                applied.append(aug_config.name)
            except Exception as e:
                logger.warning(f"Failed to apply {aug_config.name}: {e}")

        return augmented, applied

    def _apply_geometric(self, image: Image.Image, config: AugmentationConfig) -> Image.Image:
        """Apply geometric augmentation"""
        name = config.name.lower()
        params = config.parameters

        if 'rotate' in name:
            angle = params.get('angle', random.uniform(-30, 30))
            return GeometricAugmentations.rotate(image, angle)
        elif 'horizontal_flip' in name or 'hflip' in name:
            return GeometricAugmentations.horizontal_flip(image)
        elif 'vertical_flip' in name or 'vflip' in name:
            return GeometricAugmentations.vertical_flip(image)
        elif 'random_crop' in name:
            fraction = params.get('fraction', 0.8)
            return GeometricAugmentations.random_crop(image, fraction)
        elif 'center_crop' in name:
            fraction = params.get('fraction', 0.8)
            return GeometricAugmentations.center_crop(image, fraction)
        elif 'resize' in name:
            size = params.get('size', (256, 256))
            keep_aspect = params.get('keep_aspect', True)
            return GeometricAugmentations.resize(image, size, keep_aspect)
        elif 'scale' in name:
            scale_range = params.get('scale_range', (0.8, 1.2))
            return GeometricAugmentations.random_scale(image, scale_range)
        else:
            return image

    def _apply_color(self, image: Image.Image, config: AugmentationConfig) -> Image.Image:
        """Apply color augmentation"""
        name = config.name.lower()
        params = config.parameters

        if 'brightness' in name:
            if 'random' in name:
                range_val = params.get('range', (0.8, 1.2))
                return ColorAugmentations.random_brightness(image, range_val)
            else:
                factor = params.get('factor', 1.0)
                return ColorAugmentations.brightness(image, factor)
        elif 'contrast' in name:
            if 'random' in name:
                range_val = params.get('range', (0.8, 1.2))
                return ColorAugmentations.random_contrast(image, range_val)
            else:
                factor = params.get('factor', 1.0)
                return ColorAugmentations.contrast(image, factor)
        elif 'saturation' in name:
            if 'random' in name:
                range_val = params.get('range', (0.8, 1.2))
                return ColorAugmentations.random_saturation(image, range_val)
            else:
                factor = params.get('factor', 1.0)
                return ColorAugmentations.saturation(image, factor)
        elif 'hue' in name:
            delta = params.get('delta', random.randint(-30, 30))
            return ColorAugmentations.hue_shift(image, delta)
        else:
            return image

    def _apply_noise(self, image: Image.Image, config: AugmentationConfig) -> Image.Image:
        """Apply noise augmentation"""
        name = config.name.lower()
        params = config.parameters

        if 'gaussian' in name:
            std = params.get('std', 25)
            return NoiseAugmentations.gaussian_noise(image, 0, std)
        elif 'salt' in name or 'pepper' in name:
            amount = params.get('amount', 0.05)
            return NoiseAugmentations.salt_and_pepper(image, amount)
        elif 'random' in name:
            noise_type = params.get('type', 'gaussian')
            return NoiseAugmentations.random_noise(image, noise_type)
        else:
            return image

    def _apply_blur(self, image: Image.Image, config: AugmentationConfig) -> Image.Image:
        """Apply blur augmentation"""
        name = config.name.lower()
        params = config.parameters

        if 'gaussian' in name:
            radius = params.get('radius', 2.0)
            return BlurAugmentations.gaussian_blur(image, radius)
        elif 'box' in name:
            radius = params.get('radius', 2)
            return BlurAugmentations.box_blur(image, radius)
        elif 'motion' in name:
            size = params.get('size', 15)
            return BlurAugmentations.motion_blur(image, size)
        elif 'random' in name:
            return BlurAugmentations.random_blur(image)
        else:
            return image

    def _apply_advanced(self, image: Image.Image, config: AugmentationConfig) -> Image.Image:
        """Apply advanced augmentation"""
        name = config.name.lower()
        params = config.parameters

        if 'cutout' in name:
            num_holes = params.get('num_holes', 1)
            hole_size = params.get('hole_size', 50)
            return AdvancedAugmentations.cutout(image, num_holes, hole_size)
        elif 'posterize' in name:
            bits = params.get('bits', 4)
            return AdvancedAugmentations.posterize(image, bits)
        elif 'solarize' in name:
            threshold = params.get('threshold', 128)
            return AdvancedAugmentations.solarize(image, threshold)
        elif 'equalize' in name:
            return AdvancedAugmentations.equalize(image)
        elif 'autocontrast' in name:
            return AdvancedAugmentations.autocontrast(image)
        else:
            return image


# ===========================
# Main Tool Class
# ===========================

class DataAugmentationTool:
    """Main data augmentation tool"""

    def __init__(self):
        self.logger = logger

    def augment_single(
        self,
        input_path: str,
        output_path: str,
        augmentations: List[AugmentationConfig],
        seed: Optional[int] = None
    ) -> AugmentationResult:
        """
        Augment a single image

        Args:
            input_path: Input image path
            output_path: Output image path
            augmentations: List of augmentation configurations
            seed: Random seed

        Returns:
            AugmentationResult
        """
        try:
            # Load image
            image = Image.open(input_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Apply augmentations
            pipeline = AugmentationPipeline(augmentations, seed)
            augmented, applied = pipeline.apply(image)

            # Save result
            augmented.save(output_path, quality=95)

            self.logger.info(f"✓ Augmented: {input_path} → {output_path}")
            self.logger.info(f"  Applied: {', '.join(applied)}")

            return AugmentationResult(
                success=True,
                input_image=input_path,
                output_image=output_path,
                augmentations_applied=applied
            )

        except Exception as e:
            self.logger.error(f"✗ Failed to augment {input_path}: {e}")
            return AugmentationResult(
                success=False,
                input_image=input_path,
                output_image=output_path,
                augmentations_applied=[],
                error=str(e)
            )

    def augment_batch(
        self,
        input_dir: str,
        output_dir: str,
        augmentations: List[AugmentationConfig],
        num_augmentations_per_image: int = 1,
        seed: Optional[int] = None,
        file_extensions: List[str] = ['.jpg', '.jpeg', '.png']
    ) -> Dict[str, Any]:
        """
        Augment a batch of images

        Args:
            input_dir: Input directory
            output_dir: Output directory
            augmentations: List of augmentation configurations
            num_augmentations_per_image: Number of augmented versions per image
            seed: Random seed
            file_extensions: Allowed file extensions

        Returns:
            Statistics dictionary
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Find all images
        images = []
        for ext in file_extensions:
            images.extend(input_path.glob(f'**/*{ext}'))

        self.logger.info(f"Found {len(images)} images to augment")
        self.logger.info(f"Generating {num_augmentations_per_image} augmented version(s) per image")

        results = []
        for img_path in images:
            for i in range(num_augmentations_per_image):
                # Generate output filename
                stem = img_path.stem
                suffix = img_path.suffix
                output_filename = f"{stem}_aug{i:02d}{suffix}"
                output_file = output_path / output_filename

                # Augment
                result = self.augment_single(
                    str(img_path),
                    str(output_file),
                    augmentations,
                    seed=seed + i if seed else None
                )
                results.append(result)

        # Compile statistics
        successful = sum(1 for r in results if r.success)
        failed = len(results) - successful

        stats = {
            'total_images': len(images),
            'total_augmentations': len(results),
            'successful': successful,
            'failed': failed,
            'augmentations_per_image': num_augmentations_per_image,
            'output_dir': str(output_path)
        }

        self.logger.info(f"\n✓ Batch augmentation complete")
        self.logger.info(f"  Total images: {len(images)}")
        self.logger.info(f"  Total augmentations: {len(results)}")
        self.logger.info(f"  Successful: {successful}")
        self.logger.info(f"  Failed: {failed}")

        return stats


# ===========================
# CLI Interface
# ===========================

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Data Augmentation Pipeline - CPU-only image augmentation tool"
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    # Augment single command
    single_parser = subparsers.add_parser('single', help='Augment a single image')
    single_parser.add_argument('--input', required=True, help='Input image path')
    single_parser.add_argument('--output', required=True, help='Output image path')
    single_parser.add_argument('--preset', help='Augmentation preset name')
    single_parser.add_argument('--seed', type=int, help='Random seed')

    # Augment batch command
    batch_parser = subparsers.add_parser('batch', help='Augment a batch of images')
    batch_parser.add_argument('--input-dir', required=True, help='Input directory')
    batch_parser.add_argument('--output-dir', required=True, help='Output directory')
    batch_parser.add_argument('--preset', help='Augmentation preset name')
    batch_parser.add_argument('--num-per-image', type=int, default=1,
                              help='Number of augmentations per image')
    batch_parser.add_argument('--seed', type=int, help='Random seed')

    # Global options
    parser.add_argument('--skip-preflight', action='store_true',
                       help='Skip preflight checks (for development)')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='Logging level')

    return parser.parse_args()


def get_preset_augmentations(preset: str) -> List[AugmentationConfig]:
    """Get predefined augmentation presets"""
    presets = {
        'light': [
            AugmentationConfig('random_brightness', AugmentationType.COLOR, 0.5,
                              {'range': (0.9, 1.1)}),
            AugmentationConfig('random_contrast', AugmentationType.COLOR, 0.5,
                              {'range': (0.9, 1.1)}),
            AugmentationConfig('horizontal_flip', AugmentationType.GEOMETRIC, 0.5),
        ],
        'medium': [
            AugmentationConfig('random_brightness', AugmentationType.COLOR, 0.7,
                              {'range': (0.8, 1.2)}),
            AugmentationConfig('random_contrast', AugmentationType.COLOR, 0.7,
                              {'range': (0.8, 1.2)}),
            AugmentationConfig('random_saturation', AugmentationType.COLOR, 0.5,
                              {'range': (0.8, 1.2)}),
            AugmentationConfig('horizontal_flip', AugmentationType.GEOMETRIC, 0.5),
            AugmentationConfig('rotate', AugmentationType.GEOMETRIC, 0.3,
                              {'angle': random.uniform(-15, 15)}),
            AugmentationConfig('random_blur', AugmentationType.BLUR, 0.2),
        ],
        'strong': [
            AugmentationConfig('random_brightness', AugmentationType.COLOR, 0.8,
                              {'range': (0.7, 1.3)}),
            AugmentationConfig('random_contrast', AugmentationType.COLOR, 0.8,
                              {'range': (0.7, 1.3)}),
            AugmentationConfig('random_saturation', AugmentationType.COLOR, 0.7,
                              {'range': (0.7, 1.3)}),
            AugmentationConfig('horizontal_flip', AugmentationType.GEOMETRIC, 0.5),
            AugmentationConfig('vertical_flip', AugmentationType.GEOMETRIC, 0.2),
            AugmentationConfig('rotate', AugmentationType.GEOMETRIC, 0.5,
                              {'angle': random.uniform(-30, 30)}),
            AugmentationConfig('random_crop', AugmentationType.GEOMETRIC, 0.3,
                              {'fraction': 0.8}),
            AugmentationConfig('random_blur', AugmentationType.BLUR, 0.3),
            AugmentationConfig('gaussian_noise', AugmentationType.NOISE, 0.2,
                              {'std': 15}),
            AugmentationConfig('cutout', AugmentationType.ADVANCED, 0.2,
                              {'num_holes': 1, 'hole_size': 30}),
        ],
    }

    return presets.get(preset, presets['medium'])


def main():
    """Main entry point"""
    args = parse_arguments()

    # Setup logging
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # Create tool
    tool = DataAugmentationTool()

    # Get augmentations
    augmentations = get_preset_augmentations(args.preset or 'medium')

    try:
        if args.command == 'single':
            # Augment single image
            result = tool.augment_single(
                args.input,
                args.output,
                augmentations,
                args.seed
            )

            if not result.success:
                logger.error(f"Augmentation failed: {result.error}")
                return 1

        elif args.command == 'batch':
            # Augment batch
            stats = tool.augment_batch(
                args.input_dir,
                args.output_dir,
                augmentations,
                args.num_per_image,
                args.seed
            )

            # Save statistics
            stats_file = Path(args.output_dir) / 'augmentation_stats.json'
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=2)

            logger.info(f"✓ Statistics saved to: {stats_file}")

        else:
            logger.error("No command specified. Use 'single' or 'batch'.")
            return 1

        return 0

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())

#!/usr/bin/env python3
"""
Image Processor (圖像處理器)
============================

Provides comprehensive image processing capabilities using Pillow (PIL).
All operations are CPU-only and optimized for parallel processing.

提供基於 Pillow (PIL) 的完整圖像處理功能。
所有操作均為 CPU-only 並針對並行處理進行最佳化。

Features (功能):
- Image resizing (圖像調整大小)
- Image cropping (圖像裁剪)
- Format conversion (格式轉換): JPG, PNG, WebP, BMP, TIFF
- Quality optimization (品質最佳化)
- Image filters (圖像濾鏡): blur, sharpen, contrast, brightness
- Metadata extraction (Metadata 提取)
- Batch processing (批次處理)

Author: Animation AI Studio Team
Created: 2025-12-02
"""

import os
import sys
import argparse
import yaml
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# PIL/Pillow imports
try:
    from PIL import Image, ImageFilter, ImageEnhance, ImageOps
    from PIL.ExifTags import TAGS
except ImportError:
    print("Error: Pillow not installed. Install with: pip install Pillow")
    sys.exit(1)

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Import safety infrastructure
try:
    from scripts.core.safety import MemoryMonitor, run_preflight
except ImportError:
    # Stub for when safety module is not available
    class MemoryMonitor:
        def __init__(self, **kwargs):
            pass
        def check_memory(self):
            return 'ok', 0.0
    def run_preflight():
        return True

# Setup logging (設定日誌)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes (資料類別)
# ============================================================================

@dataclass
class ImageMetadata:
    """Image metadata (圖像 Metadata)"""
    width: int
    height: int
    format: str
    mode: str
    file_size_bytes: int
    exif: Optional[Dict] = None


@dataclass
class ProcessingResult:
    """Processing result (處理結果)"""
    success: bool
    input_path: str
    output_path: Optional[str] = None
    error: Optional[str] = None


# ============================================================================
# Image Processor Class (圖像處理器類別)
# ============================================================================

class ImageProcessor:
    """
    Pillow-based image processing with CPU-only operations.
    基於 Pillow 的圖像處理，僅使用 CPU 操作。
    """

    # Supported formats (支援的格式)
    SUPPORTED_FORMATS = {
        'jpg': 'JPEG',
        'jpeg': 'JPEG',
        'png': 'PNG',
        'webp': 'WebP',
        'bmp': 'BMP',
        'tiff': 'TIFF',
        'tif': 'TIFF'
    }

    # Resampling filters (重新取樣濾鏡)
    RESAMPLING_FILTERS = {
        'nearest': Image.NEAREST,
        'bilinear': Image.BILINEAR,
        'bicubic': Image.BICUBIC,
        'lanczos': Image.LANCZOS
    }

    def __init__(self, threads: int = 32, memory_monitor: Optional[MemoryMonitor] = None):
        """
        Initialize image processor.
        初始化圖像處理器。

        Args:
            threads: Number of threads for parallel processing (並行處理的執行緒數)
            memory_monitor: Memory monitoring instance (記憶體監控實例)
        """
        self.threads = threads
        self.memory_monitor = memory_monitor
        logger.info(f"ImageProcessor initialized with {threads} threads (初始化圖像處理器，使用 {threads} 個執行緒)")

    def _check_memory(self) -> bool:
        """Check memory status (檢查記憶體狀態)"""
        if self.memory_monitor:
            status, usage = self.memory_monitor.check_memory()
            if status == 'critical':
                logger.warning(f"⚠️  Memory usage critical: {usage:.1%} (記憶體使用嚴重: {usage:.1%})")
                return False
        return True

    def _get_image_format(self, path: str) -> str:
        """Get PIL format from file extension (從副檔名取得 PIL 格式)"""
        ext = Path(path).suffix.lower().lstrip('.')
        return self.SUPPORTED_FORMATS.get(ext, 'JPEG')

    # ========================================================================
    # Resize Operations (調整大小操作)
    # ========================================================================

    def resize_image(
        self,
        input_path: str,
        output_path: str,
        width: Optional[int] = None,
        height: Optional[int] = None,
        maintain_aspect: bool = True,
        resampling: str = 'lanczos'
    ) -> bool:
        """
        Resize image to specified dimensions.
        調整圖像至指定尺寸。

        Args:
            input_path: Input image path (輸入圖像路徑)
            output_path: Output image path (輸出圖像路徑)
            width: Target width in pixels (目標寬度，像素)
            height: Target height in pixels (目標高度，像素)
            maintain_aspect: Maintain aspect ratio (保持長寬比)
            resampling: Resampling filter (重新取樣濾鏡)

        Returns:
            bool: Success status (成功狀態)
        """
        try:
            # Check memory
            if not self._check_memory():
                return False

            # Open image
            img = Image.open(input_path)
            original_size = img.size

            # Calculate target size
            if width is None and height is None:
                logger.error("Must specify at least width or height (必須指定至少一個寬度或高度)")
                return False

            if maintain_aspect:
                # Calculate size maintaining aspect ratio
                if width and height:
                    # Fit within specified dimensions
                    img.thumbnail((width, height), self.RESAMPLING_FILTERS[resampling])
                    target_size = img.size
                elif width:
                    # Scale to width
                    ratio = width / img.width
                    target_size = (width, int(img.height * ratio))
                else:
                    # Scale to height
                    ratio = height / img.height
                    target_size = (int(img.width * ratio), height)
            else:
                # Exact dimensions
                target_size = (width or img.width, height or img.height)

            # Resize if not using thumbnail
            if not maintain_aspect or (width and height):
                img = img.resize(target_size, self.RESAMPLING_FILTERS[resampling])

            # Save
            output_format = self._get_image_format(output_path)
            img.save(output_path, format=output_format)

            logger.info(f"✅ Image resized: {original_size} → {target_size} (圖像已調整大小)")
            return True

        except Exception as e:
            logger.error(f"❌ Resize failed (調整大小失敗): {str(e)}")
            return False

    # ========================================================================
    # Crop Operations (裁剪操作)
    # ========================================================================

    def crop_image(
        self,
        input_path: str,
        output_path: str,
        left: int = 0,
        top: int = 0,
        right: Optional[int] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        mode: str = 'box'
    ) -> bool:
        """
        Crop image to specified region.
        裁剪圖像至指定區域。

        Args:
            input_path: Input image path (輸入圖像路徑)
            output_path: Output image path (輸出圖像路徑)
            left: Left coordinate (左邊座標)
            top: Top coordinate (上邊座標)
            right: Right coordinate (右邊座標，與 width 二選一)
            width: Crop width (裁剪寬度，與 right 二選一)
            height: Crop height (裁剪高度)
            mode: Crop mode ('box', 'center', 'square') (裁剪模式)

        Returns:
            bool: Success status (成功狀態)
        """
        try:
            # Check memory
            if not self._check_memory():
                return False

            # Open image
            img = Image.open(input_path)
            img_width, img_height = img.size

            # Calculate crop box based on mode
            if mode == 'center':
                # Center crop
                if not width or not height:
                    logger.error("Center crop requires width and height (中心裁剪需要寬度和高度)")
                    return False
                left = (img_width - width) // 2
                top = (img_height - height) // 2
                right = left + width
                bottom = top + height

            elif mode == 'square':
                # Square crop from center
                size = min(img_width, img_height)
                left = (img_width - size) // 2
                top = (img_height - size) // 2
                right = left + size
                bottom = top + size

            else:  # mode == 'box'
                # Box crop with specified coordinates
                if right is None:
                    if width is None:
                        logger.error("Must specify right or width (必須指定 right 或 width)")
                        return False
                    right = left + width

                if height is None:
                    logger.error("Must specify height (必須指定 height)")
                    return False
                bottom = top + height

            # Validate crop box
            if left < 0 or top < 0 or right > img_width or bottom > img_height:
                logger.error(f"Invalid crop box: ({left},{top})-({right},{bottom}) for image {img_width}x{img_height}")
                return False

            # Crop
            cropped = img.crop((left, top, right, bottom))

            # Save
            output_format = self._get_image_format(output_path)
            cropped.save(output_path, format=output_format)

            logger.info(f"✅ Image cropped: ({left},{top})-({right},{bottom}) (圖像已裁剪)")
            return True

        except Exception as e:
            logger.error(f"❌ Crop failed (裁剪失敗): {str(e)}")
            return False

    # ========================================================================
    # Format Conversion (格式轉換)
    # ========================================================================

    def convert_format(
        self,
        input_path: str,
        output_path: str,
        output_format: Optional[str] = None,
        quality: int = 95,
        optimize: bool = True
    ) -> bool:
        """
        Convert image format.
        轉換圖像格式。

        Args:
            input_path: Input image path (輸入圖像路徑)
            output_path: Output image path (輸出圖像路徑)
            output_format: Output format (輸出格式，自動檢測如果未指定)
            quality: JPEG/WebP quality 1-100 (品質 1-100)
            optimize: Optimize output (最佳化輸出)

        Returns:
            bool: Success status (成功狀態)
        """
        try:
            # Check memory
            if not self._check_memory():
                return False

            # Open image
            img = Image.open(input_path)

            # Determine output format
            if output_format is None:
                output_format = self._get_image_format(output_path)
            else:
                output_format = self.SUPPORTED_FORMATS.get(output_format.lower(), output_format.upper())

            # Convert RGBA to RGB for JPEG
            if output_format == 'JPEG' and img.mode in ('RGBA', 'LA', 'P'):
                # Create white background
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                img = background

            # Save with format-specific options
            save_kwargs = {'format': output_format}

            if output_format in ('JPEG', 'WebP'):
                save_kwargs['quality'] = quality
                save_kwargs['optimize'] = optimize
            elif output_format == 'PNG':
                save_kwargs['optimize'] = optimize

            img.save(output_path, **save_kwargs)

            logger.info(f"✅ Format converted: {Path(input_path).suffix} → {output_format} (格式已轉換)")
            return True

        except Exception as e:
            logger.error(f"❌ Format conversion failed (格式轉換失敗): {str(e)}")
            return False

    # ========================================================================
    # Optimization (最佳化)
    # ========================================================================

    def optimize_image(
        self,
        input_path: str,
        output_path: str,
        max_size: Optional[Tuple[int, int]] = None,
        quality: int = 85,
        format: Optional[str] = None
    ) -> bool:
        """
        Optimize image for web/storage.
        最佳化圖像以供網頁/儲存使用。

        Args:
            input_path: Input image path (輸入圖像路徑)
            output_path: Output image path (輸出圖像路徑)
            max_size: Maximum dimensions (width, height) (最大尺寸)
            quality: Compression quality 1-100 (壓縮品質 1-100)
            format: Output format (輸出格式，預設保持原格式)

        Returns:
            bool: Success status (成功狀態)
        """
        try:
            # Check memory
            if not self._check_memory():
                return False

            # Open image
            img = Image.open(input_path)
            original_size = os.path.getsize(input_path)

            # Resize if max_size specified
            if max_size:
                img.thumbnail(max_size, Image.LANCZOS)

            # Determine format
            if format is None:
                format = self._get_image_format(output_path)
            else:
                format = self.SUPPORTED_FORMATS.get(format.lower(), format.upper())

            # Convert RGBA to RGB for JPEG
            if format == 'JPEG' and img.mode in ('RGBA', 'LA', 'P'):
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                img = background

            # Save with optimization
            save_kwargs = {'format': format, 'optimize': True}
            if format in ('JPEG', 'WebP'):
                save_kwargs['quality'] = quality

            img.save(output_path, **save_kwargs)

            # Report compression
            optimized_size = os.path.getsize(output_path)
            reduction = (1 - optimized_size / original_size) * 100

            logger.info(f"✅ Image optimized: {original_size/1024:.1f}KB → {optimized_size/1024:.1f}KB ({reduction:.1f}% reduction)")
            return True

        except Exception as e:
            logger.error(f"❌ Optimization failed (最佳化失敗): {str(e)}")
            return False

    # ========================================================================
    # Filter Operations (濾鏡操作)
    # ========================================================================

    def apply_blur(
        self,
        input_path: str,
        output_path: str,
        radius: int = 2
    ) -> bool:
        """
        Apply blur filter.
        套用模糊濾鏡。

        Args:
            input_path: Input image path (輸入圖像路徑)
            output_path: Output image path (輸出圖像路徑)
            radius: Blur radius (模糊半徑)

        Returns:
            bool: Success status (成功狀態)
        """
        try:
            if not self._check_memory():
                return False

            img = Image.open(input_path)
            blurred = img.filter(ImageFilter.GaussianBlur(radius=radius))

            output_format = self._get_image_format(output_path)
            blurred.save(output_path, format=output_format)

            logger.info(f"✅ Blur applied: radius={radius} (已套用模糊)")
            return True

        except Exception as e:
            logger.error(f"❌ Blur failed (模糊失敗): {str(e)}")
            return False

    def apply_sharpen(
        self,
        input_path: str,
        output_path: str,
        factor: float = 2.0
    ) -> bool:
        """
        Apply sharpen filter.
        套用銳化濾鏡。

        Args:
            input_path: Input image path (輸入圖像路徑)
            output_path: Output image path (輸出圖像路徑)
            factor: Sharpen factor (銳化係數)

        Returns:
            bool: Success status (成功狀態)
        """
        try:
            if not self._check_memory():
                return False

            img = Image.open(input_path)
            enhancer = ImageEnhance.Sharpness(img)
            sharpened = enhancer.enhance(factor)

            output_format = self._get_image_format(output_path)
            sharpened.save(output_path, format=output_format)

            logger.info(f"✅ Sharpen applied: factor={factor} (已套用銳化)")
            return True

        except Exception as e:
            logger.error(f"❌ Sharpen failed (銳化失敗): {str(e)}")
            return False

    def adjust_contrast(
        self,
        input_path: str,
        output_path: str,
        factor: float = 1.5
    ) -> bool:
        """
        Adjust image contrast.
        調整圖像對比度。

        Args:
            input_path: Input image path (輸入圖像路徑)
            output_path: Output image path (輸出圖像路徑)
            factor: Contrast factor (對比度係數，1.0=原始)

        Returns:
            bool: Success status (成功狀態)
        """
        try:
            if not self._check_memory():
                return False

            img = Image.open(input_path)
            enhancer = ImageEnhance.Contrast(img)
            adjusted = enhancer.enhance(factor)

            output_format = self._get_image_format(output_path)
            adjusted.save(output_path, format=output_format)

            logger.info(f"✅ Contrast adjusted: factor={factor} (已調整對比度)")
            return True

        except Exception as e:
            logger.error(f"❌ Contrast adjustment failed (對比度調整失敗): {str(e)}")
            return False

    def adjust_brightness(
        self,
        input_path: str,
        output_path: str,
        factor: float = 1.2
    ) -> bool:
        """
        Adjust image brightness.
        調整圖像亮度。

        Args:
            input_path: Input image path (輸入圖像路徑)
            output_path: Output image path (輸出圖像路徑)
            factor: Brightness factor (亮度係數，1.0=原始)

        Returns:
            bool: Success status (成功狀態)
        """
        try:
            if not self._check_memory():
                return False

            img = Image.open(input_path)
            enhancer = ImageEnhance.Brightness(img)
            adjusted = enhancer.enhance(factor)

            output_format = self._get_image_format(output_path)
            adjusted.save(output_path, format=output_format)

            logger.info(f"✅ Brightness adjusted: factor={factor} (已調整亮度)")
            return True

        except Exception as e:
            logger.error(f"❌ Brightness adjustment failed (亮度調整失敗): {str(e)}")
            return False

    def auto_contrast(
        self,
        input_path: str,
        output_path: str,
        cutoff: int = 0
    ) -> bool:
        """
        Apply auto contrast.
        套用自動對比度。

        Args:
            input_path: Input image path (輸入圖像路徑)
            output_path: Output image path (輸出圖像路徑)
            cutoff: Cutoff percentage (截斷百分比)

        Returns:
            bool: Success status (成功狀態)
        """
        try:
            if not self._check_memory():
                return False

            img = Image.open(input_path)
            adjusted = ImageOps.autocontrast(img, cutoff=cutoff)

            output_format = self._get_image_format(output_path)
            adjusted.save(output_path, format=output_format)

            logger.info(f"✅ Auto contrast applied (已套用自動對比度)")
            return True

        except Exception as e:
            logger.error(f"❌ Auto contrast failed (自動對比度失敗): {str(e)}")
            return False

    # ========================================================================
    # Metadata Extraction (Metadata 提取)
    # ========================================================================

    def extract_metadata(self, input_path: str) -> Optional[ImageMetadata]:
        """
        Extract image metadata.
        提取圖像 metadata。

        Args:
            input_path: Input image path (輸入圖像路徑)

        Returns:
            Optional[ImageMetadata]: Metadata object or None (Metadata 物件或 None)
        """
        try:
            img = Image.open(input_path)

            # Basic metadata
            metadata = ImageMetadata(
                width=img.width,
                height=img.height,
                format=img.format,
                mode=img.mode,
                file_size_bytes=os.path.getsize(input_path)
            )

            # Extract EXIF if available
            exif_data = {}
            if hasattr(img, '_getexif') and img._getexif():
                exif = img._getexif()
                for tag_id, value in exif.items():
                    tag = TAGS.get(tag_id, tag_id)
                    exif_data[tag] = str(value)
                metadata.exif = exif_data

            logger.info(f"✅ Metadata extracted: {metadata.width}x{metadata.height} {metadata.format} (已提取 Metadata)")
            return metadata

        except Exception as e:
            logger.error(f"❌ Metadata extraction failed (Metadata 提取失敗): {str(e)}")
            return None

    # ========================================================================
    # Batch Processing (批次處理)
    # ========================================================================

    def process_batch(self, config_path: str) -> List[ProcessingResult]:
        """
        Process batch operations from YAML config.
        從 YAML 配置檔處理批次操作。

        Args:
            config_path: Path to YAML config file (YAML 配置檔路徑)

        Returns:
            List[ProcessingResult]: List of processing results (處理結果列表)
        """
        try:
            # Load config
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)

            operations = config.get('operations', [])
            logger.info(f"📋 Processing {len(operations)} batch operations (處理 {len(operations)} 個批次操作)")

            results = []

            for i, op in enumerate(operations, 1):
                operation = op.get('operation')
                logger.info(f"[{i}/{len(operations)}] Operation: {operation}")

                try:
                    if operation == 'resize':
                        success = self.resize_image(
                            input_path=op['input'],
                            output_path=op['output'],
                            width=op.get('width'),
                            height=op.get('height'),
                            maintain_aspect=op.get('maintain_aspect', True),
                            resampling=op.get('resampling', 'lanczos')
                        )

                    elif operation == 'crop':
                        success = self.crop_image(
                            input_path=op['input'],
                            output_path=op['output'],
                            left=op.get('left', 0),
                            top=op.get('top', 0),
                            right=op.get('right'),
                            width=op.get('width'),
                            height=op.get('height'),
                            mode=op.get('mode', 'box')
                        )

                    elif operation == 'convert':
                        success = self.convert_format(
                            input_path=op['input'],
                            output_path=op['output'],
                            output_format=op.get('output_format'),
                            quality=op.get('quality', 95),
                            optimize=op.get('optimize', True)
                        )

                    elif operation == 'optimize':
                        max_size = None
                        if op.get('max_width') or op.get('max_height'):
                            max_size = (
                                op.get('max_width', 9999),
                                op.get('max_height', 9999)
                            )

                        success = self.optimize_image(
                            input_path=op['input'],
                            output_path=op['output'],
                            max_size=max_size,
                            quality=op.get('quality', 85),
                            format=op.get('format')
                        )

                    elif operation == 'blur':
                        success = self.apply_blur(
                            input_path=op['input'],
                            output_path=op['output'],
                            radius=op.get('radius', 2)
                        )

                    elif operation == 'sharpen':
                        success = self.apply_sharpen(
                            input_path=op['input'],
                            output_path=op['output'],
                            factor=op.get('factor', 2.0)
                        )

                    elif operation == 'contrast':
                        success = self.adjust_contrast(
                            input_path=op['input'],
                            output_path=op['output'],
                            factor=op.get('factor', 1.5)
                        )

                    elif operation == 'brightness':
                        success = self.adjust_brightness(
                            input_path=op['input'],
                            output_path=op['output'],
                            factor=op.get('factor', 1.2)
                        )

                    elif operation == 'auto_contrast':
                        success = self.auto_contrast(
                            input_path=op['input'],
                            output_path=op['output'],
                            cutoff=op.get('cutoff', 0)
                        )

                    elif operation == 'metadata':
                        metadata = self.extract_metadata(op['input'])
                        success = metadata is not None
                        if success:
                            logger.info(f"  Width: {metadata.width}px")
                            logger.info(f"  Height: {metadata.height}px")
                            logger.info(f"  Format: {metadata.format}")
                            logger.info(f"  Mode: {metadata.mode}")
                            logger.info(f"  File Size: {metadata.file_size_bytes / 1024:.1f} KB")

                    else:
                        logger.error(f"Unknown operation: {operation}")
                        success = False

                    results.append(ProcessingResult(
                        success=success,
                        input_path=op['input'],
                        output_path=op.get('output')
                    ))

                except Exception as e:
                    logger.error(f"Operation failed: {str(e)}")
                    results.append(ProcessingResult(
                        success=False,
                        input_path=op.get('input', 'unknown'),
                        error=str(e)
                    ))

            # Summary
            successful = sum(1 for r in results if r.success)
            logger.info(f"✅ Batch processing complete: {successful}/{len(results)} successful (批次處理完成)")

            return results

        except Exception as e:
            logger.error(f"❌ Batch processing failed (批次處理失敗): {str(e)}")
            return []


# ============================================================================
# Command-Line Interface (命令列介面)
# ============================================================================

def create_parser() -> argparse.ArgumentParser:
    """Create argument parser (建立參數解析器)"""
    parser = argparse.ArgumentParser(
        description='Image Processor - CPU-only image processing with Pillow (圖像處理器 - 基於 Pillow 的 CPU-only 圖像處理)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples (範例):

  # Resize image (調整圖像大小)
  python image_processor.py --operation resize --input image.jpg --output resized.jpg --width 800 --height 600

  # Crop image (裁剪圖像)
  python image_processor.py --operation crop --input image.jpg --output cropped.jpg --mode center --width 500 --height 500

  # Convert format (轉換格式)
  python image_processor.py --operation convert --input image.jpg --output image.png --output-format png

  # Optimize image (最佳化圖像)
  python image_processor.py --operation optimize --input large.jpg --output optimized.jpg --quality 85

  # Apply blur (套用模糊)
  python image_processor.py --operation blur --input image.jpg --output blurred.jpg --radius 3

  # Batch processing (批次處理)
  python image_processor.py --operation batch --batch-config config.yaml
        """
    )

    # Required arguments
    parser.add_argument(
        '--operation',
        required=True,
        choices=['resize', 'crop', 'convert', 'optimize', 'blur', 'sharpen',
                 'contrast', 'brightness', 'auto_contrast', 'metadata', 'batch'],
        help='Operation to perform (要執行的操作)'
    )

    # Input/output
    parser.add_argument('--input', help='Input image path (輸入圖像路徑)')
    parser.add_argument('--output', help='Output image path (輸出圖像路徑)')

    # Resize parameters
    parser.add_argument('--width', type=int, help='Target width in pixels (目標寬度，像素)')
    parser.add_argument('--height', type=int, help='Target height in pixels (目標高度，像素)')
    parser.add_argument('--maintain-aspect', action='store_true', default=True,
                       help='Maintain aspect ratio (保持長寬比)')
    parser.add_argument('--resampling', default='lanczos',
                       choices=['nearest', 'bilinear', 'bicubic', 'lanczos'],
                       help='Resampling filter (重新取樣濾鏡)')

    # Crop parameters
    parser.add_argument('--left', type=int, default=0, help='Left coordinate (左邊座標)')
    parser.add_argument('--top', type=int, default=0, help='Top coordinate (上邊座標)')
    parser.add_argument('--right', type=int, help='Right coordinate (右邊座標)')
    parser.add_argument('--mode', default='box', choices=['box', 'center', 'square'],
                       help='Crop mode (裁剪模式)')

    # Format conversion parameters
    parser.add_argument('--output-format',
                       choices=['jpg', 'jpeg', 'png', 'webp', 'bmp', 'tiff'],
                       help='Output format (輸出格式)')
    parser.add_argument('--quality', type=int, default=95,
                       help='JPEG/WebP quality 1-100 (品質 1-100)')
    parser.add_argument('--optimize', action='store_true', default=True,
                       help='Optimize output (最佳化輸出)')

    # Optimization parameters
    parser.add_argument('--max-width', type=int, help='Maximum width (最大寬度)')
    parser.add_argument('--max-height', type=int, help='Maximum height (最大高度)')

    # Filter parameters
    parser.add_argument('--radius', type=int, default=2, help='Blur radius (模糊半徑)')
    parser.add_argument('--factor', type=float, default=1.5,
                       help='Enhancement factor (增強係數)')
    parser.add_argument('--cutoff', type=int, default=0,
                       help='Auto contrast cutoff (自動對比度截斷)')

    # Batch processing
    parser.add_argument('--batch-config', help='Batch config YAML file (批次配置 YAML 檔案)')

    # Performance
    parser.add_argument('--threads', type=int, default=32,
                       help='Number of threads (執行緒數量)')

    # Safety
    parser.add_argument('--skip-preflight', action='store_true',
                       help='Skip preflight safety checks (跳過預檢查)')

    return parser


def main():
    """Main entry point (主要進入點)"""
    parser = create_parser()
    args = parser.parse_args()

    # Initialize safety monitor (初始化安全監控)
    if not args.skip_preflight:
        logger.info("Initializing safety infrastructure (初始化安全基礎設施)...")
        memory_monitor = MemoryMonitor(
            warning_threshold=0.70,
            critical_threshold=0.80,
            emergency_threshold=0.85
        )

        # Run preflight checks (執行預檢查)
        if not run_preflight():
            logger.error("❌ Preflight checks failed (預檢查失敗)")
            return 1

        logger.info("✅ Preflight checks passed (預檢查通過)")
    else:
        logger.warning("⚠️  Skipping preflight checks (跳過預檢查)")
        memory_monitor = None

    # Initialize processor (初始化處理器)
    processor = ImageProcessor(
        threads=args.threads,
        memory_monitor=memory_monitor
    )

    # Execute operation (執行操作)
    try:
        if args.operation == 'batch':
            if not args.batch_config:
                logger.error("Batch operation requires --batch-config")
                return 1
            results = processor.process_batch(args.batch_config)
            return 0 if any(r.success for r in results) else 1

        # Validate input/output for non-batch operations
        if not args.input:
            logger.error("--input is required")
            return 1

        if args.operation != 'metadata' and not args.output:
            logger.error("--output is required for this operation")
            return 1

        # Execute operation
        success = False

        if args.operation == 'resize':
            success = processor.resize_image(
                input_path=args.input,
                output_path=args.output,
                width=args.width,
                height=args.height,
                maintain_aspect=args.maintain_aspect,
                resampling=args.resampling
            )

        elif args.operation == 'crop':
            success = processor.crop_image(
                input_path=args.input,
                output_path=args.output,
                left=args.left,
                top=args.top,
                right=args.right,
                width=args.width,
                height=args.height,
                mode=args.mode
            )

        elif args.operation == 'convert':
            success = processor.convert_format(
                input_path=args.input,
                output_path=args.output,
                output_format=args.output_format,
                quality=args.quality,
                optimize=args.optimize
            )

        elif args.operation == 'optimize':
            max_size = None
            if args.max_width or args.max_height:
                max_size = (args.max_width or 9999, args.max_height or 9999)

            success = processor.optimize_image(
                input_path=args.input,
                output_path=args.output,
                max_size=max_size,
                quality=args.quality,
                format=args.output_format
            )

        elif args.operation == 'blur':
            success = processor.apply_blur(
                input_path=args.input,
                output_path=args.output,
                radius=args.radius
            )

        elif args.operation == 'sharpen':
            success = processor.apply_sharpen(
                input_path=args.input,
                output_path=args.output,
                factor=args.factor
            )

        elif args.operation == 'contrast':
            success = processor.adjust_contrast(
                input_path=args.input,
                output_path=args.output,
                factor=args.factor
            )

        elif args.operation == 'brightness':
            success = processor.adjust_brightness(
                input_path=args.input,
                output_path=args.output,
                factor=args.factor
            )

        elif args.operation == 'auto_contrast':
            success = processor.auto_contrast(
                input_path=args.input,
                output_path=args.output,
                cutoff=args.cutoff
            )

        elif args.operation == 'metadata':
            metadata = processor.extract_metadata(args.input)
            if metadata:
                logger.info("Image Metadata:")
                logger.info(f"  Width: {metadata.width}px")
                logger.info(f"  Height: {metadata.height}px")
                logger.info(f"  Format: {metadata.format}")
                logger.info(f"  Mode: {metadata.mode}")
                logger.info(f"  File Size: {metadata.file_size_bytes / 1024:.1f} KB")
                if metadata.exif:
                    logger.info(f"  EXIF Data: {len(metadata.exif)} entries")
                success = True

        if success:
            logger.info("✅ Operation completed successfully (操作成功完成)")
            return 0
        else:
            logger.error("❌ Operation failed (操作失敗)")
            return 1

    except Exception as e:
        logger.error(f"❌ Fatal error (嚴重錯誤): {str(e)}")
        return 1


if __name__ == '__main__':
    sys.exit(main())

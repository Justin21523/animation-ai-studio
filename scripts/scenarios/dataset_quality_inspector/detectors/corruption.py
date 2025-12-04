"""
Corruption Detector

Detects corrupted or invalid image files in dataset.
Performs file integrity and decoding validation.

Author: Animation AI Studio
Date: 2025-12-02
"""

import logging
from pathlib import Path
from typing import List, Dict, Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class CorruptionDetector:
    """
    File corruption and integrity detector

    Validates:
    - File can be opened
    - Image can be decoded
    - Format matches extension
    - Minimum file size requirements
    - Basic metadata integrity

    Features:
    - CPU-only validation
    - Fast failure detection
    - Detailed error reporting

    Example:
        detector = CorruptionDetector(
            min_file_size=1024  # 1 KB minimum
        )

        results = detector.scan_for_corruption("/path/to/dataset")
        if results["corrupted_files"]:
            print(f"Found {len(results['corrupted_files'])} corrupted files")
    """

    def __init__(
        self,
        min_file_size: int = 1024,  # 1 KB
        supported_formats: List[str] = None
    ):
        """
        Initialize Corruption Detector

        Args:
            min_file_size: Minimum file size in bytes
            supported_formats: List of supported extensions (default: common formats)
        """
        self.min_file_size = min_file_size
        self.supported_formats = supported_formats or [
            ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"
        ]

        logger.info(f"CorruptionDetector initialized: min_size={min_file_size}B")

    def scan_for_corruption(self, dataset_path: str) -> Dict[str, Any]:
        """
        Scan dataset for corrupted files

        Args:
            dataset_path: Path to dataset directory

        Returns:
            Dictionary with corruption scan results
        """
        dataset_path = Path(dataset_path)

        if not dataset_path.exists():
            logger.error(f"Dataset path does not exist: {dataset_path}")
            return {"error": "Dataset path not found"}

        # Scan for image files
        images = self._scan_images(dataset_path)

        if not images:
            logger.warning(f"No images found in {dataset_path}")
            return {
                "total_images": 0,
                "corrupted_files": [],
                "corruption_details": {}
            }

        # Check each file
        logger.info(f"Scanning {len(images)} files for corruption...")

        corrupted_files = []
        corruption_details = {}

        for img_path in images:
            issues = self._check_file(img_path)
            if issues:
                corrupted_files.append(str(img_path))
                corruption_details[str(img_path)] = issues

        logger.info(f"Found {len(corrupted_files)} corrupted files")

        return {
            "total_images": len(images),
            "corrupted_files": corrupted_files,
            "corrupted_count": len(corrupted_files),
            "corruption_details": corruption_details,
            "integrity_score": (1.0 - len(corrupted_files) / len(images)) * 100 if images else 100.0
        }

    def _scan_images(self, dataset_path: Path) -> List[Path]:
        """Scan directory for image files"""
        images = []
        for ext in self.supported_formats:
            images.extend(dataset_path.rglob(f"*{ext}"))
        return images

    def _check_file(self, file_path: Path) -> List[str]:
        """
        Check single file for corruption

        Args:
            file_path: Path to image file

        Returns:
            List of issue descriptions (empty if file is valid)
        """
        issues = []

        # Check 1: File exists
        if not file_path.exists():
            issues.append("File does not exist")
            return issues

        # Check 2: File size
        try:
            file_size = file_path.stat().st_size
            if file_size < self.min_file_size:
                issues.append(f"File too small: {file_size}B (min: {self.min_file_size}B)")
        except Exception as e:
            issues.append(f"Cannot read file stats: {e}")
            return issues

        # Check 3: File can be opened
        if not self._can_open_file(file_path):
            issues.append("Cannot open file")
            return issues

        # Check 4: Image can be decoded
        decode_issue = self._can_decode_image(file_path)
        if decode_issue:
            issues.append(decode_issue)

        # Check 5: Format validation
        format_issue = self._validate_format(file_path)
        if format_issue:
            issues.append(format_issue)

        return issues

    def _can_open_file(self, file_path: Path) -> bool:
        """
        Test if file can be opened

        Args:
            file_path: Path to file

        Returns:
            True if file can be opened
        """
        try:
            with open(file_path, 'rb') as f:
                # Try to read first byte
                f.read(1)
            return True
        except Exception as e:
            logger.debug(f"Cannot open {file_path}: {e}")
            return False

    def _can_decode_image(self, file_path: Path) -> str:
        """
        Test if image can be decoded

        Args:
            file_path: Path to image

        Returns:
            Error description or empty string if successful
        """
        try:
            img = cv2.imread(str(file_path))

            if img is None:
                return "OpenCV cannot decode image"

            # Check shape is valid
            if img.shape[0] == 0 or img.shape[1] == 0:
                return "Image has zero dimensions"

            # Check for all-zero or all-same pixels (possible corruption)
            if img.size == 0:
                return "Image has no data"

            # Basic pixel value check
            if np.all(img == img.flat[0]):
                return "Image has uniform pixel values (possible corruption)"

            return ""

        except Exception as e:
            return f"Decoding error: {str(e)}"

    def _validate_format(self, file_path: Path) -> str:
        """
        Validate image format matches extension

        Args:
            file_path: Path to image

        Returns:
            Error description or empty string if valid
        """
        try:
            # Read file header
            with open(file_path, 'rb') as f:
                header = f.read(12)

            # Check magic bytes
            extension = file_path.suffix.lower()

            if extension in [".jpg", ".jpeg"]:
                # JPEG: FF D8 FF
                if not header.startswith(b'\xff\xd8\xff'):
                    return "File header does not match JPEG format"

            elif extension == ".png":
                # PNG: 89 50 4E 47 0D 0A 1A 0A
                if not header.startswith(b'\x89PNG\r\n\x1a\n'):
                    return "File header does not match PNG format"

            elif extension == ".bmp":
                # BMP: 42 4D
                if not header.startswith(b'BM'):
                    return "File header does not match BMP format"

            elif extension in [".tiff", ".tif"]:
                # TIFF: 49 49 or 4D 4D
                if not (header.startswith(b'II') or header.startswith(b'MM')):
                    return "File header does not match TIFF format"

            elif extension == ".webp":
                # WebP: RIFF....WEBP
                if not (b'RIFF' in header and b'WEBP' in header):
                    return "File header does not match WebP format"

            return ""

        except Exception as e:
            return f"Format validation error: {str(e)}"

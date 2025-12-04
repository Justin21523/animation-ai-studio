"""
Duplicate Detector

CPU-only duplicate and near-duplicate image detection.
Uses perceptual hashing (pHash) for efficient similarity matching.

Author: Animation AI Studio
Date: 2025-12-02
"""

import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any
from collections import defaultdict

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class DuplicateDetector:
    """
    CPU-only duplicate image detector

    Uses perceptual hashing (pHash) to find:
    - Exact duplicates (identical hashes)
    - Near duplicates (similar hashes with Hamming distance)

    Features:
    - Pure Python/NumPy implementation
    - Fast O(n) hash computation
    - Efficient O(n²) comparison with early termination
    - Configurable similarity threshold

    Example:
        detector = DuplicateDetector(
            hash_size=8,
            similarity_threshold=5
        )

        results = detector.find_duplicates("/path/to/dataset")
        for group in results["duplicate_groups"]:
            print(f"Duplicates: {group}")
    """

    def __init__(
        self,
        hash_size: int = 8,
        similarity_threshold: int = 5,
        highfreq_factor: int = 4
    ):
        """
        Initialize Duplicate Detector

        Args:
            hash_size: Size of perceptual hash (8x8 = 64-bit hash)
            similarity_threshold: Max Hamming distance for near duplicates
            highfreq_factor: DCT high-frequency cutoff factor
        """
        self.hash_size = hash_size
        self.similarity_threshold = similarity_threshold
        self.highfreq_factor = highfreq_factor

        logger.info(f"DuplicateDetector initialized: hash_size={hash_size}, "
                   f"threshold={similarity_threshold}")

    def find_duplicates(self, dataset_path: str) -> Dict[str, Any]:
        """
        Find duplicate images in dataset

        Args:
            dataset_path: Path to dataset directory

        Returns:
            Dictionary with duplicate detection results
        """
        dataset_path = Path(dataset_path)

        if not dataset_path.exists():
            logger.error(f"Dataset path does not exist: {dataset_path}")
            return {"error": "Dataset path not found"}

        # Scan for images
        images = self._scan_images(dataset_path)

        if not images:
            logger.warning(f"No images found in {dataset_path}")
            return {
                "total_images": 0,
                "exact_duplicates": 0,
                "near_duplicates": 0,
                "duplicate_groups": []
            }

        # Compute perceptual hashes
        logger.info(f"Computing perceptual hashes for {len(images)} images...")
        image_hashes = {}
        for img_path in images:
            try:
                phash = self._compute_phash(img_path)
                if phash is not None:
                    image_hashes[str(img_path)] = phash
            except Exception as e:
                logger.debug(f"Failed to hash {img_path}: {e}")

        logger.info(f"Successfully hashed {len(image_hashes)} images")

        # Find duplicates
        exact_groups, near_groups = self._find_duplicate_groups(image_hashes)

        # Count total duplicates
        exact_count = sum(len(group) - 1 for group in exact_groups)
        near_count = sum(len(group) - 1 for group in near_groups)

        return {
            "total_images": len(images),
            "hashed_images": len(image_hashes),
            "exact_duplicates": exact_count,
            "near_duplicates": near_count,
            "exact_duplicate_groups": exact_groups,
            "near_duplicate_groups": near_groups
        }

    def _scan_images(self, dataset_path: Path) -> List[Path]:
        """Scan directory for image files"""
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
        images = []
        for ext in image_extensions:
            images.extend(dataset_path.rglob(f"*{ext}"))
        return images

    def _compute_phash(self, img_path: Path) -> Optional[str]:
        """
        Compute perceptual hash (pHash) for image

        Algorithm:
        1. Load and resize to 32x32 (or hash_size * highfreq_factor)
        2. Convert to grayscale
        3. Compute DCT (Discrete Cosine Transform)
        4. Extract low-frequency DCT coefficients
        5. Compute median of coefficients
        6. Generate hash: 1 if coeff > median, 0 otherwise

        Args:
            img_path: Path to image file

        Returns:
            Hexadecimal hash string or None on error
        """
        try:
            # Load image
            img = cv2.imread(str(img_path))
            if img is None:
                return None

            # Resize to standard size
            img_size = self.hash_size * self.highfreq_factor
            img = cv2.resize(img, (img_size, img_size))

            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Compute DCT
            dct = cv2.dct(np.float32(gray))

            # Extract low-frequency components (top-left corner)
            dct_low = dct[:self.hash_size, :self.hash_size]

            # Compute median
            median = np.median(dct_low)

            # Generate binary hash
            hash_bits = (dct_low > median).flatten()

            # Convert to hexadecimal string
            hash_str = self._bits_to_hex(hash_bits)

            return hash_str

        except Exception as e:
            logger.debug(f"Error computing pHash for {img_path}: {e}")
            return None

    def _bits_to_hex(self, bits: np.ndarray) -> str:
        """Convert binary array to hexadecimal string"""
        # Group bits into bytes
        hex_str = ""
        for i in range(0, len(bits), 8):
            byte_bits = bits[i:i+8]
            byte_val = 0
            for j, bit in enumerate(byte_bits):
                if bit:
                    byte_val |= (1 << (7 - j))
            hex_str += f"{byte_val:02x}"
        return hex_str

    def _hex_to_bits(self, hex_str: str) -> np.ndarray:
        """Convert hexadecimal string to binary array"""
        bits = []
        for hex_char in hex_str:
            byte_val = int(hex_char, 16)
            for i in range(4):  # 4 bits per hex char
                bits.append((byte_val >> (3 - i)) & 1)
        return np.array(bits, dtype=bool)

    def _hamming_distance(self, hash1: str, hash2: str) -> int:
        """
        Compute Hamming distance between two hashes

        Args:
            hash1: First hash (hex string)
            hash2: Second hash (hex string)

        Returns:
            Number of differing bits
        """
        if len(hash1) != len(hash2):
            return float('inf')

        bits1 = self._hex_to_bits(hash1)
        bits2 = self._hex_to_bits(hash2)

        return np.sum(bits1 != bits2)

    def _find_duplicate_groups(
        self,
        image_hashes: Dict[str, str]
    ) -> Tuple[List[List[str]], List[List[str]]]:
        """
        Find duplicate groups using hash comparison

        Args:
            image_hashes: Mapping of image path to perceptual hash

        Returns:
            (exact_duplicate_groups, near_duplicate_groups)
        """
        # Group by hash for exact duplicates
        hash_groups = defaultdict(list)
        for img_path, phash in image_hashes.items():
            hash_groups[phash].append(img_path)

        # Extract exact duplicate groups (same hash)
        exact_groups = [
            group for group in hash_groups.values()
            if len(group) > 1
        ]

        logger.info(f"Found {len(exact_groups)} exact duplicate groups")

        # Find near duplicates (similar hashes)
        near_groups = []
        processed = set()

        image_list = list(image_hashes.items())
        for i in range(len(image_list)):
            img_path1, hash1 = image_list[i]

            if img_path1 in processed:
                continue

            group = [img_path1]

            for j in range(i + 1, len(image_list)):
                img_path2, hash2 = image_list[j]

                if img_path2 in processed:
                    continue

                # Compute Hamming distance
                distance = self._hamming_distance(hash1, hash2)

                if 0 < distance <= self.similarity_threshold:
                    group.append(img_path2)

            if len(group) > 1:
                near_groups.append(group)
                processed.update(group)

        logger.info(f"Found {len(near_groups)} near duplicate groups")

        return exact_groups, near_groups

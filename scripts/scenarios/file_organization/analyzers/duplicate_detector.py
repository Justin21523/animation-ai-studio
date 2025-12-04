"""
Duplicate Detector

Identifies duplicate files using multiple hash methods:
- SHA256 content hash for exact duplicates
- Perceptual hash for near-duplicate images
- Size-based pre-filtering for optimization

Author: Animation AI Studio
Date: 2025-12-03
"""

import logging
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Set
from collections import defaultdict

try:
    from PIL import Image
    import imagehash
    IMAGEHASH_AVAILABLE = True
except ImportError:
    IMAGEHASH_AVAILABLE = False
    logging.warning("imagehash not available, perceptual hashing disabled")

from ..common import FileMetadata, DuplicateGroup, FileType

logger = logging.getLogger(__name__)


class DuplicateDetector:
    """
    Duplicate file detector using multiple hash methods

    Features:
    - SHA256 content hash for exact duplicates
    - Perceptual hash for near-duplicate images (optional)
    - Size-based pre-filtering for performance
    - Smart grouping with deduplication strategies

    Example:
        detector = DuplicateDetector(enable_perceptual=True)
        duplicates = detector.detect_duplicates(file_list)

        for group in duplicates:
            print(f"Found {len(group.files)} duplicates")
            print(f"Potential savings: {group.savings_bytes} bytes")
    """

    def __init__(
        self,
        enable_perceptual: bool = True,
        perceptual_threshold: int = 5,
        min_file_size: int = 1024,
        chunk_size: int = 65536
    ):
        """
        Initialize duplicate detector

        Args:
            enable_perceptual: Enable perceptual hashing for images
            perceptual_threshold: Hamming distance threshold for near-duplicates
            min_file_size: Minimum file size to check (bytes)
            chunk_size: Chunk size for hash computation (bytes)
        """
        self.enable_perceptual = enable_perceptual and IMAGEHASH_AVAILABLE
        self.perceptual_threshold = perceptual_threshold
        self.min_file_size = min_file_size
        self.chunk_size = chunk_size

        if enable_perceptual and not IMAGEHASH_AVAILABLE:
            logger.warning(
                "Perceptual hashing requested but imagehash not available. "
                "Install with: pip install imagehash pillow"
            )

        logger.info(
            f"DuplicateDetector initialized "
            f"(perceptual={'enabled' if self.enable_perceptual else 'disabled'}, "
            f"threshold={perceptual_threshold})"
        )

    def detect_duplicates(
        self,
        files: List[FileMetadata]
    ) -> List[DuplicateGroup]:
        """
        Detect duplicate files

        Args:
            files: List of file metadata

        Returns:
            List of duplicate groups

        Process:
        1. Filter files by minimum size
        2. Group by exact size (optimization)
        3. Compute SHA256 for files with same size
        4. Group exact duplicates
        5. (Optional) Compute perceptual hash for images
        6. Create DuplicateGroup objects
        """
        logger.info(f"Detecting duplicates in {len(files)} files")

        # Filter by minimum size
        filtered_files = [
            f for f in files
            if f.size_bytes >= self.min_file_size
        ]

        logger.info(
            f"Filtered to {len(filtered_files)} files "
            f"(>= {self.min_file_size} bytes)"
        )

        # Detect exact duplicates
        exact_duplicates = self._detect_exact_duplicates(filtered_files)

        # Detect near-duplicates in images (if enabled)
        near_duplicates = []
        if self.enable_perceptual:
            image_files = [
                f for f in filtered_files
                if f.file_type == FileType.IMAGE
            ]
            if image_files:
                near_duplicates = self._detect_near_duplicates(image_files)

        all_duplicates = exact_duplicates + near_duplicates

        logger.info(
            f"Found {len(all_duplicates)} duplicate groups "
            f"({len(exact_duplicates)} exact, {len(near_duplicates)} near)"
        )

        return all_duplicates

    def _detect_exact_duplicates(
        self,
        files: List[FileMetadata]
    ) -> List[DuplicateGroup]:
        """Detect exact duplicates using SHA256"""

        # Group by size first (optimization)
        by_size: Dict[int, List[FileMetadata]] = defaultdict(list)
        for file_meta in files:
            by_size[file_meta.size_bytes].append(file_meta)

        # Only check files with same size
        duplicate_groups = []

        for size, size_group in by_size.items():
            if len(size_group) < 2:
                continue  # No duplicates possible

            # Compute SHA256 for files with same size
            by_hash: Dict[str, List[Path]] = defaultdict(list)

            for file_meta in size_group:
                # Use cached hash if available
                if file_meta.hash_sha256:
                    file_hash = file_meta.hash_sha256
                else:
                    try:
                        file_hash = self.compute_sha256(file_meta.path)
                    except Exception as e:
                        logger.warning(
                            f"Failed to hash {file_meta.path}: {e}"
                        )
                        continue

                by_hash[file_hash].append(file_meta.path)

            # Create duplicate groups
            for file_hash, paths in by_hash.items():
                if len(paths) >= 2:
                    # Calculate savings (keep one, remove others)
                    total_size = size * len(paths)
                    savings = size * (len(paths) - 1)

                    duplicate_groups.append(DuplicateGroup(
                        files=paths,
                        hash=file_hash,
                        total_size_bytes=total_size,
                        savings_bytes=savings,
                        strategy="keep_oldest"
                    ))

        return duplicate_groups

    def _detect_near_duplicates(
        self,
        files: List[FileMetadata]
    ) -> List[DuplicateGroup]:
        """Detect near-duplicate images using perceptual hashing"""

        if not IMAGEHASH_AVAILABLE:
            return []

        logger.info(f"Computing perceptual hashes for {len(files)} images")

        # Compute perceptual hashes
        hash_map: Dict[str, List[tuple[str, Path]]] = defaultdict(list)

        for file_meta in files:
            # Use cached hash if available
            if file_meta.hash_perceptual:
                phash_str = file_meta.hash_perceptual
            else:
                try:
                    phash_str = self.compute_perceptual_hash(file_meta.path)
                    if not phash_str:
                        continue
                except Exception as e:
                    logger.debug(
                        f"Failed to compute perceptual hash for {file_meta.path}: {e}"
                    )
                    continue

            hash_map[phash_str].append((phash_str, file_meta.path))

        # Find near-duplicates by comparing hashes
        duplicate_groups = []
        processed_hashes: Set[str] = set()

        for phash_str, entries in hash_map.items():
            if phash_str in processed_hashes:
                continue

            # Convert hash string to ImageHash object
            try:
                phash = imagehash.hex_to_hash(phash_str)
            except Exception as e:
                logger.debug(f"Invalid hash string {phash_str}: {e}")
                continue

            # Find all similar hashes
            similar_group: List[Path] = []

            for other_hash_str, other_entries in hash_map.items():
                if other_hash_str in processed_hashes:
                    continue

                try:
                    other_phash = imagehash.hex_to_hash(other_hash_str)
                    distance = phash - other_phash  # Hamming distance

                    if distance <= self.perceptual_threshold:
                        similar_group.extend([path for _, path in other_entries])
                        processed_hashes.add(other_hash_str)

                except Exception as e:
                    logger.debug(f"Hash comparison error: {e}")
                    continue

            # Create duplicate group if we found near-duplicates
            if len(similar_group) >= 2:
                # Estimate savings (conservative: assume 50% similarity)
                total_size = sum(
                    p.stat().st_size for p in similar_group if p.exists()
                )
                savings = total_size // 2

                duplicate_groups.append(DuplicateGroup(
                    files=similar_group,
                    hash=f"perceptual:{phash_str}",
                    total_size_bytes=total_size,
                    savings_bytes=savings,
                    strategy="keep_largest"  # Keep highest quality
                ))

        return duplicate_groups

    def compute_sha256(self, path: Path) -> str:
        """
        Compute SHA256 hash of a file

        Args:
            path: Path to file

        Returns:
            Hex string of SHA256 hash

        Raises:
            IOError: If file cannot be read
        """
        sha256_hash = hashlib.sha256()

        try:
            with open(path, 'rb') as f:
                # Read in chunks for memory efficiency
                while True:
                    chunk = f.read(self.chunk_size)
                    if not chunk:
                        break
                    sha256_hash.update(chunk)

            return sha256_hash.hexdigest()

        except IOError as e:
            logger.error(f"Failed to read file {path}: {e}")
            raise

    def compute_perceptual_hash(
        self,
        path: Path,
        hash_size: int = 8
    ) -> Optional[str]:
        """
        Compute perceptual hash of an image

        Args:
            path: Path to image file
            hash_size: Hash size (default: 8x8 = 64-bit hash)

        Returns:
            Hex string of perceptual hash, or None if failed

        Note:
            Uses average hash (aHash) algorithm which is:
            - Fast
            - Good for detecting scaled/cropped versions
            - Robust to small changes
        """
        if not IMAGEHASH_AVAILABLE:
            return None

        try:
            # Open image
            with Image.open(path) as img:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                # Compute average hash
                phash = imagehash.average_hash(img, hash_size=hash_size)

                return str(phash)

        except Exception as e:
            logger.debug(f"Failed to compute perceptual hash for {path}: {e}")
            return None

    def group_by_hash(
        self,
        files: List[FileMetadata],
        use_perceptual: bool = False
    ) -> Dict[str, List[Path]]:
        """
        Group files by hash

        Args:
            files: List of file metadata
            use_perceptual: Use perceptual hash instead of SHA256

        Returns:
            Dictionary mapping hash to list of file paths
        """
        groups: Dict[str, List[Path]] = defaultdict(list)

        for file_meta in files:
            if use_perceptual and file_meta.hash_perceptual:
                groups[file_meta.hash_perceptual].append(file_meta.path)
            elif file_meta.hash_sha256:
                groups[file_meta.hash_sha256].append(file_meta.path)
            else:
                # Compute hash on demand
                try:
                    if use_perceptual and file_meta.file_type == FileType.IMAGE:
                        file_hash = self.compute_perceptual_hash(file_meta.path)
                    else:
                        file_hash = self.compute_sha256(file_meta.path)

                    if file_hash:
                        groups[file_hash].append(file_meta.path)

                except Exception as e:
                    logger.warning(
                        f"Failed to hash {file_meta.path}: {e}"
                    )
                    continue

        # Filter out unique files
        return {h: paths for h, paths in groups.items() if len(paths) >= 2}

    def estimate_savings(
        self,
        duplicate_groups: List[DuplicateGroup]
    ) -> int:
        """
        Estimate total space savings from removing duplicates

        Args:
            duplicate_groups: List of duplicate groups

        Returns:
            Total bytes that could be saved
        """
        return sum(group.savings_bytes for group in duplicate_groups)

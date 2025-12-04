"""
File Classifier

Classifies files into categories using multiple detection methods:
- Magic bytes detection (file headers)
- MIME type detection
- Extension-based classification
- Special handling for AI models, datasets, configs

Author: Animation AI Studio
Date: 2025-12-03
"""

import logging
import mimetypes
from pathlib import Path
from typing import Optional, Dict, Set

try:
    import magic
    MAGIC_AVAILABLE = True
except ImportError:
    MAGIC_AVAILABLE = False
    logging.warning("python-magic not available, falling back to extension-based classification")

from ..common import FileType

logger = logging.getLogger(__name__)


class FileClassifier:
    """
    File type classifier using multiple detection methods

    Features:
    - Magic bytes detection for accurate type identification
    - MIME type detection for cross-validation
    - Extension-based fallback
    - Special handling for AI/ML files

    Example:
        classifier = FileClassifier()
        file_type = classifier.classify_file(Path("/path/to/image.jpg"))
        # Returns: FileType.IMAGE
    """

    # Extension mappings for each file type
    EXTENSION_MAP = {
        FileType.IMAGE: {
            '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif',
            '.webp', '.svg', '.ico', '.heic', '.heif', '.raw', '.cr2',
            '.nef', '.arw', '.dng', '.exr', '.hdr'
        },
        FileType.VIDEO: {
            '.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm',
            '.m4v', '.mpg', '.mpeg', '.3gp', '.ts', '.vob', '.ogv',
            '.mts', '.m2ts', '.f4v', '.rm', '.rmvb', '.divx'
        },
        FileType.AUDIO: {
            '.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a', '.wma',
            '.opus', '.ape', '.alac', '.aiff', '.au', '.mid', '.midi',
            '.ra', '.ac3', '.dts'
        },
        FileType.DOCUMENT: {
            '.pdf', '.doc', '.docx', '.txt', '.rtf', '.odt', '.tex',
            '.md', '.markdown', '.rst', '.org', '.epub', '.mobi',
            '.azw', '.azw3', '.djvu', '.xps'
        },
        FileType.CODE: {
            '.py', '.js', '.java', '.cpp', '.c', '.h', '.hpp', '.cs',
            '.go', '.rs', '.rb', '.php', '.swift', '.kt', '.scala',
            '.r', '.m', '.sh', '.bash', '.zsh', '.fish', '.ps1',
            '.html', '.htm', '.css', '.scss', '.sass', '.less',
            '.xml', '.json', '.yaml', '.yml', '.toml', '.ini', '.cfg',
            '.sql', '.pl', '.lua', '.vim', '.el', '.clj', '.hs',
            '.ml', '.fs', '.erl', '.ex', '.exs', '.dart', '.ts', '.tsx',
            '.jsx', '.vue', '.svelte'
        },
        FileType.ARCHIVE: {
            '.zip', '.tar', '.gz', '.bz2', '.xz', '.7z', '.rar', '.iso',
            '.dmg', '.pkg', '.deb', '.rpm', '.apk', '.cab', '.msi',
            '.tgz', '.tbz2', '.txz', '.lz', '.lzma', '.z', '.lzo',
            '.zst', '.br'
        },
        FileType.MODEL: {
            '.safetensors', '.pt', '.pth', '.ckpt', '.bin', '.pb',
            '.onnx', '.h5', '.keras', '.tflite', '.mlmodel', '.pkl',
            '.pickle', '.joblib', '.pmml', '.mar'
        },
        FileType.CONFIG: {
            '.conf', '.config', '.env', '.properties', '.prefs',
            '.settings', '.editorconfig', '.gitignore', '.dockerignore',
            '.npmrc', '.pylintrc', '.flake8', '.prettierrc', '.eslintrc'
        },
        FileType.LOG: {
            '.log', '.out', '.err', '.trace', '.debug'
        }
    }

    # AI model file patterns
    AI_MODEL_PATTERNS = {
        'safetensors', 'checkpoint', 'lora', 'embedding',
        'model', 'weights', 'state_dict', 'trained'
    }

    # Dataset directory indicators
    DATASET_INDICATORS = {
        'dataset', 'training_data', 'test_data', 'validation',
        'train', 'test', 'val', 'images', 'labels', 'annotations'
    }

    def __init__(self):
        """Initialize the file classifier"""
        # Initialize MIME types database
        mimetypes.init()

        # Create reverse mapping: extension -> FileType
        self._ext_to_type: Dict[str, FileType] = {}
        for file_type, extensions in self.EXTENSION_MAP.items():
            for ext in extensions:
                self._ext_to_type[ext.lower()] = file_type

        # Initialize magic detector if available
        self._magic_detector = None
        if MAGIC_AVAILABLE:
            try:
                self._magic_detector = magic.Magic(mime=True)
                logger.info("Magic bytes detection enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize magic detector: {e}")
                self._magic_detector = None

        logger.info("FileClassifier initialized")

    def classify_file(self, path: Path) -> FileType:
        """
        Classify a file into a FileType category

        Args:
            path: Path to the file

        Returns:
            FileType enum value

        Example:
            >>> classifier = FileClassifier()
            >>> classifier.classify_file(Path("photo.jpg"))
            FileType.IMAGE
        """
        if not path.exists():
            logger.warning(f"File does not exist: {path}")
            return FileType.OTHER

        if path.is_dir():
            # Check if it's a dataset directory
            if self.is_dataset(path):
                return FileType.DATASET
            return FileType.OTHER

        # Special checks first
        if self.is_ai_model(path):
            return FileType.MODEL

        # Try magic bytes detection
        if self._magic_detector:
            try:
                file_type = self._classify_by_magic(path)
                if file_type != FileType.OTHER:
                    return file_type
            except Exception as e:
                logger.debug(f"Magic bytes detection failed for {path}: {e}")

        # Try MIME type detection
        try:
            file_type = self._classify_by_mime(path)
            if file_type != FileType.OTHER:
                return file_type
        except Exception as e:
            logger.debug(f"MIME type detection failed for {path}: {e}")

        # Fallback to extension-based classification
        return self._classify_by_extension(path)

    def _classify_by_magic(self, path: Path) -> FileType:
        """Classify using magic bytes"""
        if not self._magic_detector:
            return FileType.OTHER

        try:
            mime_type = self._magic_detector.from_file(str(path))
            return self._mime_to_filetype(mime_type)
        except Exception as e:
            logger.debug(f"Magic detection error for {path}: {e}")
            return FileType.OTHER

    def _classify_by_mime(self, path: Path) -> FileType:
        """Classify using MIME type"""
        mime_type, _ = mimetypes.guess_type(str(path))
        if not mime_type:
            return FileType.OTHER

        return self._mime_to_filetype(mime_type)

    def _mime_to_filetype(self, mime_type: str) -> FileType:
        """Convert MIME type to FileType"""
        if not mime_type:
            return FileType.OTHER

        mime_lower = mime_type.lower()

        # Image types
        if mime_lower.startswith('image/'):
            return FileType.IMAGE

        # Video types
        if mime_lower.startswith('video/'):
            return FileType.VIDEO

        # Audio types
        if mime_lower.startswith('audio/'):
            return FileType.AUDIO

        # Document types
        if any(doc_type in mime_lower for doc_type in [
            'pdf', 'document', 'text/plain', 'rtf', 'msword',
            'officedocument', 'opendocument'
        ]):
            return FileType.DOCUMENT

        # Archive types
        if any(arc_type in mime_lower for arc_type in [
            'zip', 'tar', 'gzip', 'bzip', 'compress', 'archive',
            'x-7z', 'x-rar', 'x-iso'
        ]):
            return FileType.ARCHIVE

        # Code/text types
        if any(code_type in mime_lower for code_type in [
            'text/', 'application/json', 'application/xml',
            'application/javascript'
        ]):
            # Check extension to distinguish code from documents
            return FileType.CODE

        return FileType.OTHER

    def _classify_by_extension(self, path: Path) -> FileType:
        """Classify using file extension"""
        ext = path.suffix.lower()

        if not ext:
            # No extension - check if it's a common script
            if self._is_script_file(path):
                return FileType.CODE
            return FileType.OTHER

        return self._ext_to_type.get(ext, FileType.OTHER)

    def _is_script_file(self, path: Path) -> bool:
        """Check if file is a script (no extension but has shebang)"""
        try:
            with open(path, 'rb') as f:
                first_line = f.read(256)
                return first_line.startswith(b'#!')
        except Exception:
            return False

    def is_ai_model(self, path: Path) -> bool:
        """
        Check if file is an AI model

        Args:
            path: Path to check

        Returns:
            True if file appears to be an AI model
        """
        if not path.is_file():
            return False

        # Check extension first (most reliable)
        ext_lower = path.suffix.lower()
        if ext_lower in {'.safetensors', '.pt', '.pth', '.ckpt', '.onnx', '.h5', '.keras', '.pb', '.bin'}:
            return True

        # Only check filename patterns if extension is also model-related
        # This prevents false positives like "not_a_model.jpg"
        if ext_lower in {'.bin', '.pkl', '.pickle', '.joblib'}:
            name_lower = path.name.lower()
            if any(pattern in name_lower for pattern in self.AI_MODEL_PATTERNS):
                return True

        return False

    def is_dataset(self, path: Path) -> bool:
        """
        Check if directory is a dataset

        Args:
            path: Path to check

        Returns:
            True if directory appears to be a dataset
        """
        if not path.is_dir():
            return False

        # Check directory name
        name_lower = path.name.lower()
        if any(indicator in name_lower for indicator in self.DATASET_INDICATORS):
            return True

        # Check for common dataset structures
        has_images = any(path.glob('*.jpg')) or any(path.glob('*.png'))
        has_labels = any(path.glob('*.txt')) or any(path.glob('*.json'))

        if has_images and has_labels:
            return True

        return False

    def detect_mime_type(self, path: Path) -> Optional[str]:
        """
        Detect MIME type of a file

        Args:
            path: Path to file

        Returns:
            MIME type string or None
        """
        # Try magic bytes first
        if self._magic_detector:
            try:
                return self._magic_detector.from_file(str(path))
            except Exception as e:
                logger.debug(f"Magic MIME detection failed: {e}")

        # Fallback to mimetypes module
        mime_type, _ = mimetypes.guess_type(str(path))
        return mime_type

    def get_supported_extensions(self, file_type: FileType) -> Set[str]:
        """
        Get all supported extensions for a file type

        Args:
            file_type: FileType to query

        Returns:
            Set of extensions (with leading dot)
        """
        return self.EXTENSION_MAP.get(file_type, set())

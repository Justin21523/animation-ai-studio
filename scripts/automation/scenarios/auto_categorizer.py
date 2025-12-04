"""
Auto Categorizer

Analyzes images/videos using Claude API to automatically categorize, tag, and extract
semantic information. All processing is CPU-only via remote LLM APIs.

Features:
  - Image analysis using Claude 3.5 Sonnet with vision capabilities
  - Batch processing with rate limiting
  - Category suggestions based on content
  - Tag extraction (objects, scenes, moods, styles)
  - Content description generation
  - JSON output with structured metadata

Usage:
  python scripts/automation/scenarios/auto_categorizer.py \
    --input-dir /path/to/images/ \
    --output /path/to/categorization_report.json \
    --categories "character,scene,object,style" \
    --batch-size 10

Author: Animation AI Studio Team
Last Modified: 2025-12-02
"""

import sys
import os
import argparse
import json
import logging
import base64
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
import time

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import safety infrastructure
from scripts.core.safety import (
    enforce_cpu_only,
    verify_no_gpu_usage,
    MemoryMonitor,
    RuntimeMonitor,
    run_preflight,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class ImageCategory:
    """Categorization result for a single image."""
    image_path: str
    primary_category: str
    sub_categories: List[str]
    tags: List[str]
    description: str
    confidence: float
    metadata: Dict[str, Any]
    timestamp: str


# ============================================================================
# Claude API Client
# ============================================================================

class ClaudeVisionAnalyzer:
    """
    Claude API client for image analysis using vision capabilities.

    Uses Claude 3.5 Sonnet with vision for comprehensive image understanding.
    """

    def __init__(self, api_key: Optional[str] = None, max_retries: int = 3):
        """
        Initialize Claude API client.

        Args:
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            max_retries: Maximum retry attempts for failed requests
        """
        self.api_key = api_key or os.environ.get('ANTHROPIC_API_KEY')
        if not self.api_key:
            raise ValueError(
                "Anthropic API key required. Set ANTHROPIC_API_KEY environment variable "
                "or pass api_key parameter."
            )

        self.max_retries = max_retries
        self.request_count = 0
        self.last_request_time = 0

        # Rate limiting (50 requests per minute for Anthropic)
        self.min_request_interval = 1.2  # seconds

        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=self.api_key)
        except ImportError:
            raise ImportError(
                "anthropic package not installed. Install with: pip install anthropic"
            )

    def _rate_limit(self):
        """Apply rate limiting between requests."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()

    def _encode_image(self, image_path: Path) -> str:
        """
        Encode image to base64 for Claude API.

        Args:
            image_path: Path to image file

        Returns:
            Base64-encoded image string
        """
        try:
            with open(image_path, 'rb') as f:
                image_data = f.read()
            return base64.standard_b64encode(image_data).decode('utf-8')
        except Exception as e:
            logger.error(f"Failed to encode image {image_path}: {e}")
            raise

    def _get_media_type(self, image_path: Path) -> str:
        """
        Determine media type from file extension.

        Args:
            image_path: Path to image file

        Returns:
            Media type string (e.g., "image/jpeg")
        """
        ext = image_path.suffix.lower()
        media_types = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.webp': 'image/webp',
        }
        return media_types.get(ext, 'image/jpeg')

    def analyze_image(
        self,
        image_path: Path,
        categories: List[str],
        prompt_template: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze image using Claude vision API.

        Args:
            image_path: Path to image file
            categories: List of category options
            prompt_template: Optional custom prompt template

        Returns:
            Dict with analysis results (category, tags, description)
        """
        logger.info(f"Analyzing image: {image_path.name}")

        # Apply rate limiting
        self._rate_limit()

        # Encode image
        image_base64 = self._encode_image(image_path)
        media_type = self._get_media_type(image_path)

        # Build prompt
        if prompt_template is None:
            prompt = self._build_default_prompt(categories)
        else:
            prompt = prompt_template

        # Make API request with retries
        for attempt in range(self.max_retries):
            try:
                response = self.client.messages.create(
                    model="claude-3-5-haiku-20241022",
                    max_tokens=1024,
                    messages=[{
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": image_base64,
                                },
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ],
                    }],
                )

                self.request_count += 1

                # Parse response
                result = self._parse_response(response.content[0].text)
                return result

            except Exception as e:
                logger.warning(f"API request failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error(f"Failed to analyze image after {self.max_retries} attempts")
                    raise

    def _build_default_prompt(self, categories: List[str]) -> str:
        """
        Build default analysis prompt.

        Args:
            categories: List of category options

        Returns:
            Prompt string
        """
        categories_str = ", ".join(categories)

        prompt = f"""Analyze this image and provide a structured categorization.

Available categories: {categories_str}

Please respond in JSON format with the following structure:
{{
  "primary_category": "the most relevant category from the list above",
  "sub_categories": ["additional relevant categories"],
  "tags": ["descriptive tags about objects, scenes, mood, style, etc."],
  "description": "a brief 1-2 sentence description of the image content",
  "confidence": 0.95
}}

Focus on:
- Accurate categorization based on the primary subject
- Descriptive tags that capture visual elements, mood, and style
- Clear, concise description
- Confidence score (0.0-1.0) based on how well the image fits the categories

Respond with ONLY the JSON object, no additional text."""

        return prompt

    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse Claude API response into structured format.

        Args:
            response_text: Raw response text from Claude

        Returns:
            Parsed dict with analysis results
        """
        try:
            # Try to extract JSON from response
            # Claude might wrap JSON in markdown code blocks
            if '```json' in response_text:
                start = response_text.find('```json') + 7
                end = response_text.find('```', start)
                json_str = response_text[start:end].strip()
            elif '```' in response_text:
                start = response_text.find('```') + 3
                end = response_text.find('```', start)
                json_str = response_text[start:end].strip()
            else:
                json_str = response_text.strip()

            result = json.loads(json_str)

            # Validate required fields
            required_fields = ['primary_category', 'tags', 'description']
            for field in required_fields:
                if field not in result:
                    raise ValueError(f"Missing required field: {field}")

            # Set defaults for optional fields
            result.setdefault('sub_categories', [])
            result.setdefault('confidence', 0.8)

            return result

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.debug(f"Response text: {response_text}")
            # Return fallback result
            return {
                'primary_category': 'unknown',
                'sub_categories': [],
                'tags': ['parse_error'],
                'description': 'Failed to parse API response',
                'confidence': 0.0
            }
        except Exception as e:
            logger.error(f"Error parsing response: {e}")
            return {
                'primary_category': 'unknown',
                'sub_categories': [],
                'tags': ['error'],
                'description': str(e),
                'confidence': 0.0
            }


# ============================================================================
# Batch Processing
# ============================================================================

def categorize_images(
    input_dir: Path,
    output_path: Path,
    categories: List[str],
    batch_size: int = 10,
    api_key: Optional[str] = None,
    file_extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.webp'),
    memory_monitor: Optional[MemoryMonitor] = None,
) -> Dict[str, Any]:
    """
    Batch categorize images in directory.

    Args:
        input_dir: Directory containing images
        output_path: Path to save categorization report
        categories: List of category options
        batch_size: Number of images to process before saving checkpoint
        api_key: Anthropic API key
        file_extensions: Tuple of valid image extensions
        memory_monitor: Optional memory monitor for safety checks

    Returns:
        Categorization report dict
    """
    logger.info("=" * 80)
    logger.info("AUTO CATEGORIZER")
    logger.info("=" * 80)
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output: {output_path}")
    logger.info(f"Categories: {', '.join(categories)}")

    # Initialize analyzer
    analyzer = ClaudeVisionAnalyzer(api_key=api_key)

    # Find all images
    image_files = []
    for ext in file_extensions:
        image_files.extend(input_dir.glob(f"**/*{ext}"))

    logger.info(f"Found {len(image_files)} images")

    if len(image_files) == 0:
        logger.warning("No images found in input directory")
        return {
            'timestamp': datetime.now().isoformat(),
            'input_dir': str(input_dir),
            'categories': categories,
            'total_images': 0,
            'processed_images': 0,
            'results': []
        }

    # Process images
    results = []
    errors = []

    for i, image_path in enumerate(image_files):
        try:
            # Memory safety check
            if memory_monitor:
                is_safe, level, info = memory_monitor.check_safety()
                if not is_safe:
                    logger.warning(f"Memory level {level} - saving checkpoint and exiting")
                    break

            # Analyze image
            logger.info(f"[{i+1}/{len(image_files)}] Processing: {image_path.name}")

            analysis = analyzer.analyze_image(
                image_path,
                categories=categories
            )

            # Create result
            result = ImageCategory(
                image_path=str(image_path),
                primary_category=analysis['primary_category'],
                sub_categories=analysis.get('sub_categories', []),
                tags=analysis['tags'],
                description=analysis['description'],
                confidence=analysis.get('confidence', 0.8),
                metadata={
                    'file_size_bytes': image_path.stat().st_size,
                    'file_name': image_path.name,
                },
                timestamp=datetime.now().isoformat()
            )

            results.append(asdict(result))

            # Save checkpoint every batch_size images
            if (i + 1) % batch_size == 0:
                logger.info(f"Checkpoint: Processed {i+1} images, saving...")
                _save_checkpoint(output_path, results, categories, input_dir)

        except Exception as e:
            logger.error(f"Failed to process {image_path}: {e}")
            errors.append({
                'image_path': str(image_path),
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })

    # Build final report
    report = {
        'timestamp': datetime.now().isoformat(),
        'input_dir': str(input_dir),
        'categories': categories,
        'total_images': len(image_files),
        'processed_images': len(results),
        'failed_images': len(errors),
        'api_requests': analyzer.request_count,
        'category_distribution': _compute_category_distribution(results),
        'results': results,
        'errors': errors,
    }

    # Save final report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)

    logger.info(f"\n✓ Categorization complete: {output_path}")
    logger.info(f"  Total images: {len(image_files)}")
    logger.info(f"  Successfully processed: {len(results)}")
    logger.info(f"  Failed: {len(errors)}")
    logger.info(f"  API requests: {analyzer.request_count}")

    return report


def _save_checkpoint(
    output_path: Path,
    results: List[Dict[str, Any]],
    categories: List[str],
    input_dir: Path
):
    """Save intermediate checkpoint."""
    checkpoint_path = output_path.parent / f"{output_path.stem}_checkpoint.json"
    checkpoint = {
        'timestamp': datetime.now().isoformat(),
        'input_dir': str(input_dir),
        'categories': categories,
        'processed_images': len(results),
        'results': results,
    }
    with open(checkpoint_path, 'w') as f:
        json.dump(checkpoint, f, indent=2)


def _compute_category_distribution(results: List[Dict[str, Any]]) -> Dict[str, int]:
    """Compute distribution of primary categories."""
    distribution = {}
    for result in results:
        category = result['primary_category']
        distribution[category] = distribution.get(category, 0) + 1
    return distribution


# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Auto-categorize images using Claude API with vision'
    )

    # Input/output
    parser.add_argument('--input-dir', type=Path, required=True,
                       help='Directory containing images to categorize')
    parser.add_argument('--output', type=Path, required=True,
                       help='Output path for categorization report (JSON)')

    # Categories
    parser.add_argument('--categories', type=str, required=True,
                       help='Comma-separated list of category options')

    # Processing
    parser.add_argument('--batch-size', type=int, default=10,
                       help='Checkpoint save interval (default: 10)')
    parser.add_argument('--file-extensions', type=str, default='.jpg,.jpeg,.png,.webp',
                       help='Comma-separated file extensions to process')

    # API
    parser.add_argument('--api-key', type=str,
                       help='Anthropic API key (defaults to ANTHROPIC_API_KEY env var)')

    # Safety
    parser.add_argument('--skip-preflight', action='store_true',
                       help='Skip preflight safety checks (not recommended)')

    args = parser.parse_args()

    # Parse categories
    categories = [c.strip() for c in args.categories.split(',')]

    # Parse file extensions
    file_extensions = tuple(
        ext.strip() if ext.startswith('.') else f'.{ext.strip()}'
        for ext in args.file_extensions.split(',')
    )

    # Enforce CPU-only
    enforce_cpu_only()

    # Run preflight checks
    if not args.skip_preflight:
        logger.info("Running preflight checks...")
        try:
            run_preflight(strict=True)
        except Exception as e:
            logger.warning(f"Preflight checks failed: {e}")
            logger.warning("Continuing anyway (use --skip-preflight to suppress this)")

    # Create memory monitor
    memory_monitor = MemoryMonitor()

    # Start runtime monitoring
    with RuntimeMonitor(check_interval=30.0) as monitor:
        # Run categorization
        categorize_images(
            input_dir=args.input_dir,
            output_path=args.output,
            categories=categories,
            batch_size=args.batch_size,
            api_key=args.api_key,
            file_extensions=file_extensions,
            memory_monitor=memory_monitor,
        )


if __name__ == '__main__':
    main()

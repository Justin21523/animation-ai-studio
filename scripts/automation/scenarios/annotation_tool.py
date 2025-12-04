#!/usr/bin/env python3
"""
Annotation Tool Integration
Supports format conversion, validation, and analysis for object detection annotations.

Supported formats:
- COCO (Common Objects in Context)
- YOLO (You Only Look Once)
- Pascal VOC (Visual Object Classes)
- LabelMe

Features:
- Format conversion between different annotation formats
- Annotation quality validation
- Statistics and analysis
- Batch processing support

Author: AI Team
Date: 2025-12-02
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
import xml.etree.ElementTree as ET
from collections import defaultdict
import shutil

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from PIL import Image
    import numpy as np
except ImportError as e:
    print(f"Error: Required package not found: {e}")
    print("Please install: pip install Pillow numpy")
    sys.exit(1)


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class BoundingBox:
    """Bounding box annotation"""
    x: float
    y: float
    width: float
    height: float

    def to_xyxy(self) -> Tuple[float, float, float, float]:
        """Convert to (x1, y1, x2, y2) format"""
        return (self.x, self.y, self.x + self.width, self.y + self.height)

    def to_xywh(self) -> Tuple[float, float, float, float]:
        """Convert to (x, y, width, height) format"""
        return (self.x, self.y, self.width, self.height)

    def to_yolo(self, img_width: int, img_height: int) -> Tuple[float, float, float, float]:
        """Convert to YOLO format (normalized x_center, y_center, width, height)"""
        x_center = (self.x + self.width / 2) / img_width
        y_center = (self.y + self.height / 2) / img_height
        norm_width = self.width / img_width
        norm_height = self.height / img_height
        return (x_center, y_center, norm_width, norm_height)

    @classmethod
    def from_yolo(cls, x_center: float, y_center: float, width: float, height: float,
                  img_width: int, img_height: int) -> 'BoundingBox':
        """Create from YOLO format"""
        abs_width = width * img_width
        abs_height = height * img_height
        x = (x_center * img_width) - (abs_width / 2)
        y = (y_center * img_height) - (abs_height / 2)
        return cls(x=x, y=y, width=abs_width, height=abs_height)

    def area(self) -> float:
        """Calculate box area"""
        return self.width * self.height

    def is_valid(self, img_width: int, img_height: int) -> bool:
        """Check if box is valid"""
        if self.width <= 0 or self.height <= 0:
            return False
        if self.x < 0 or self.y < 0:
            return False
        if self.x + self.width > img_width or self.y + self.height > img_height:
            return False
        return True


@dataclass
class Annotation:
    """Single object annotation"""
    image_id: str
    category_id: int
    category_name: str
    bbox: BoundingBox
    area: float = 0.0
    iscrowd: int = 0
    segmentation: Optional[List] = None

    def __post_init__(self):
        if self.area == 0.0:
            self.area = self.bbox.area()


@dataclass
class ImageInfo:
    """Image metadata"""
    id: str
    file_name: str
    width: int
    height: int
    annotations: List[Annotation] = field(default_factory=list)


@dataclass
class AnnotationDataset:
    """Complete annotation dataset"""
    images: List[ImageInfo] = field(default_factory=list)
    categories: Dict[int, str] = field(default_factory=dict)
    format: str = "coco"

    def get_statistics(self) -> Dict:
        """Get dataset statistics"""
        stats = {
            'total_images': len(self.images),
            'total_annotations': sum(len(img.annotations) for img in self.images),
            'categories': self.categories,
            'category_counts': defaultdict(int),
            'images_per_category': defaultdict(int),
            'avg_annotations_per_image': 0.0,
            'images_without_annotations': 0,
        }

        for img in self.images:
            if not img.annotations:
                stats['images_without_annotations'] += 1
            for ann in img.annotations:
                stats['category_counts'][ann.category_name] += 1
                stats['images_per_category'][ann.category_name] += 1

        if stats['total_images'] > 0:
            stats['avg_annotations_per_image'] = stats['total_annotations'] / stats['total_images']

        # Convert defaultdict to regular dict for JSON serialization
        stats['category_counts'] = dict(stats['category_counts'])
        stats['images_per_category'] = dict(stats['images_per_category'])

        return stats


@dataclass
class ValidationReport:
    """Annotation validation report"""
    total_images: int = 0
    total_annotations: int = 0
    valid_annotations: int = 0
    invalid_annotations: int = 0
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    issues: List[Dict] = field(default_factory=list)

    def add_error(self, image_id: str, message: str):
        """Add error"""
        self.errors.append(f"Image {image_id}: {message}")
        self.issues.append({
            'type': 'error',
            'image_id': image_id,
            'message': message
        })

    def add_warning(self, image_id: str, message: str):
        """Add warning"""
        self.warnings.append(f"Image {image_id}: {message}")
        self.issues.append({
            'type': 'warning',
            'image_id': image_id,
            'message': message
        })


# ============================================================================
# Format Converters
# ============================================================================

class COCOConverter:
    """COCO format converter"""

    @staticmethod
    def load(annotation_file: str) -> AnnotationDataset:
        """Load COCO format annotations"""
        with open(annotation_file, 'r') as f:
            coco_data = json.load(f)

        dataset = AnnotationDataset(format='coco')

        # Build category mapping
        for cat in coco_data.get('categories', []):
            dataset.categories[cat['id']] = cat['name']

        # Build image info dict
        image_dict = {}
        for img in coco_data.get('images', []):
            image_info = ImageInfo(
                id=str(img['id']),
                file_name=img['file_name'],
                width=img['width'],
                height=img['height']
            )
            image_dict[img['id']] = image_info
            dataset.images.append(image_info)

        # Add annotations
        for ann in coco_data.get('annotations', []):
            image_id = ann['image_id']
            if image_id not in image_dict:
                continue

            bbox = BoundingBox(
                x=ann['bbox'][0],
                y=ann['bbox'][1],
                width=ann['bbox'][2],
                height=ann['bbox'][3]
            )

            annotation = Annotation(
                image_id=str(image_id),
                category_id=ann['category_id'],
                category_name=dataset.categories.get(ann['category_id'], 'unknown'),
                bbox=bbox,
                area=ann.get('area', bbox.area()),
                iscrowd=ann.get('iscrowd', 0),
                segmentation=ann.get('segmentation')
            )

            image_dict[image_id].annotations.append(annotation)

        return dataset

    @staticmethod
    def save(dataset: AnnotationDataset, output_file: str):
        """Save to COCO format"""
        coco_data = {
            'images': [],
            'annotations': [],
            'categories': []
        }

        # Add categories
        for cat_id, cat_name in dataset.categories.items():
            coco_data['categories'].append({
                'id': cat_id,
                'name': cat_name,
                'supercategory': 'object'
            })

        # Add images and annotations
        # Create numeric ID mapping for string IDs
        id_mapping = {}
        numeric_id = 0

        ann_id = 1
        for img in dataset.images:
            # Handle string or int IDs
            try:
                img_numeric_id = int(img.id)
            except (ValueError, TypeError):
                # Generate numeric ID for string IDs
                if img.id not in id_mapping:
                    id_mapping[img.id] = numeric_id
                    numeric_id += 1
                img_numeric_id = id_mapping[img.id]

            coco_data['images'].append({
                'id': img_numeric_id,
                'file_name': img.file_name,
                'width': img.width,
                'height': img.height
            })

            for ann in img.annotations:
                # Handle string or int image IDs
                try:
                    ann_image_id = int(ann.image_id)
                except (ValueError, TypeError):
                    ann_image_id = id_mapping.get(ann.image_id, 0)

                bbox = ann.bbox.to_xywh()
                coco_data['annotations'].append({
                    'id': ann_id,
                    'image_id': ann_image_id,
                    'category_id': ann.category_id,
                    'bbox': list(bbox),
                    'area': ann.area,
                    'iscrowd': ann.iscrowd,
                    'segmentation': ann.segmentation or []
                })
                ann_id += 1

        with open(output_file, 'w') as f:
            json.dump(coco_data, f, indent=2)


class YOLOConverter:
    """YOLO format converter"""

    @staticmethod
    def load(images_dir: str, labels_dir: str, classes_file: str) -> AnnotationDataset:
        """Load YOLO format annotations"""
        dataset = AnnotationDataset(format='yolo')

        # Load class names
        with open(classes_file, 'r') as f:
            class_names = [line.strip() for line in f.readlines()]

        for idx, name in enumerate(class_names):
            dataset.categories[idx] = name

        # Process each image
        images_path = Path(images_dir)
        labels_path = Path(labels_dir)

        for img_file in images_path.glob('*'):
            if img_file.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
                continue

            # Get image dimensions
            try:
                with Image.open(img_file) as img:
                    img_width, img_height = img.size
            except Exception as e:
                logging.warning(f"Cannot read image {img_file}: {e}")
                continue

            image_info = ImageInfo(
                id=img_file.stem,
                file_name=img_file.name,
                width=img_width,
                height=img_height
            )

            # Load corresponding label file
            label_file = labels_path / f"{img_file.stem}.txt"
            if label_file.exists():
                with open(label_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) != 5:
                            continue

                        class_id = int(parts[0])
                        x_center, y_center, width, height = map(float, parts[1:])

                        bbox = BoundingBox.from_yolo(
                            x_center, y_center, width, height,
                            img_width, img_height
                        )

                        annotation = Annotation(
                            image_id=img_file.stem,
                            category_id=class_id,
                            category_name=dataset.categories.get(class_id, 'unknown'),
                            bbox=bbox
                        )

                        image_info.annotations.append(annotation)

            dataset.images.append(image_info)

        return dataset

    @staticmethod
    def save(dataset: AnnotationDataset, output_dir: str, classes_file: str):
        """Save to YOLO format"""
        output_path = Path(output_dir)
        images_path = output_path / 'images'
        labels_path = output_path / 'labels'

        images_path.mkdir(parents=True, exist_ok=True)
        labels_path.mkdir(parents=True, exist_ok=True)

        # Save class names
        with open(classes_file, 'w') as f:
            for cat_id in sorted(dataset.categories.keys()):
                f.write(f"{dataset.categories[cat_id]}\n")

        # Save labels
        for img in dataset.images:
            label_file = labels_path / f"{img.id}.txt"

            with open(label_file, 'w') as f:
                for ann in img.annotations:
                    yolo_bbox = ann.bbox.to_yolo(img.width, img.height)
                    f.write(f"{ann.category_id} {yolo_bbox[0]:.6f} {yolo_bbox[1]:.6f} "
                           f"{yolo_bbox[2]:.6f} {yolo_bbox[3]:.6f}\n")


class PascalVOCConverter:
    """Pascal VOC format converter"""

    @staticmethod
    def load(annotations_dir: str, images_dir: str) -> AnnotationDataset:
        """Load Pascal VOC format annotations"""
        dataset = AnnotationDataset(format='voc')

        annotations_path = Path(annotations_dir)
        images_path = Path(images_dir)

        category_id_map = {}
        next_cat_id = 0

        for xml_file in annotations_path.glob('*.xml'):
            tree = ET.parse(xml_file)
            root = tree.getroot()

            # Get image info
            filename = root.find('filename').text
            size = root.find('size')
            img_width = int(size.find('width').text)
            img_height = int(size.find('height').text)

            image_info = ImageInfo(
                id=xml_file.stem,
                file_name=filename,
                width=img_width,
                height=img_height
            )

            # Parse objects
            for obj in root.findall('object'):
                cat_name = obj.find('name').text

                # Build category mapping
                if cat_name not in category_id_map:
                    category_id_map[cat_name] = next_cat_id
                    dataset.categories[next_cat_id] = cat_name
                    next_cat_id += 1

                cat_id = category_id_map[cat_name]

                # Parse bounding box
                bndbox = obj.find('bndbox')
                xmin = float(bndbox.find('xmin').text)
                ymin = float(bndbox.find('ymin').text)
                xmax = float(bndbox.find('xmax').text)
                ymax = float(bndbox.find('ymax').text)

                bbox = BoundingBox(
                    x=xmin,
                    y=ymin,
                    width=xmax - xmin,
                    height=ymax - ymin
                )

                annotation = Annotation(
                    image_id=xml_file.stem,
                    category_id=cat_id,
                    category_name=cat_name,
                    bbox=bbox
                )

                image_info.annotations.append(annotation)

            dataset.images.append(image_info)

        return dataset

    @staticmethod
    def save(dataset: AnnotationDataset, output_dir: str):
        """Save to Pascal VOC format"""
        output_path = Path(output_dir)
        annotations_path = output_path / 'Annotations'
        annotations_path.mkdir(parents=True, exist_ok=True)

        for img in dataset.images:
            # Create XML tree
            root = ET.Element('annotation')

            # Add filename
            ET.SubElement(root, 'folder').text = 'images'
            ET.SubElement(root, 'filename').text = img.file_name

            # Add source
            source = ET.SubElement(root, 'source')
            ET.SubElement(source, 'database').text = 'Unknown'

            # Add size
            size = ET.SubElement(root, 'size')
            ET.SubElement(size, 'width').text = str(img.width)
            ET.SubElement(size, 'height').text = str(img.height)
            ET.SubElement(size, 'depth').text = '3'

            ET.SubElement(root, 'segmented').text = '0'

            # Add objects
            for ann in img.annotations:
                obj = ET.SubElement(root, 'object')
                ET.SubElement(obj, 'name').text = ann.category_name
                ET.SubElement(obj, 'pose').text = 'Unspecified'
                ET.SubElement(obj, 'truncated').text = '0'
                ET.SubElement(obj, 'difficult').text = '0'

                bndbox = ET.SubElement(obj, 'bndbox')
                x1, y1, x2, y2 = ann.bbox.to_xyxy()
                ET.SubElement(bndbox, 'xmin').text = str(int(x1))
                ET.SubElement(bndbox, 'ymin').text = str(int(y1))
                ET.SubElement(bndbox, 'xmax').text = str(int(x2))
                ET.SubElement(bndbox, 'ymax').text = str(int(y2))

            # Write XML file
            tree = ET.ElementTree(root)
            xml_file = annotations_path / f"{img.id}.xml"
            tree.write(xml_file, encoding='utf-8', xml_declaration=True)


# ============================================================================
# Annotation Tool
# ============================================================================

class AnnotationTool:
    """Main annotation tool class"""

    def __init__(self, skip_preflight: bool = False):
        """Initialize annotation tool"""
        self.skip_preflight = skip_preflight
        self.logger = self._setup_logger()

        if not skip_preflight:
            self._run_preflight_checks()

    def _setup_logger(self) -> logging.Logger:
        """Setup logger"""
        logger = logging.getLogger('AnnotationTool')
        logger.setLevel(logging.INFO)

        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        return logger

    def _run_preflight_checks(self):
        """Run preflight checks"""
        self.logger.info("Running preflight checks...")

        # Check for required packages
        required_packages = ['PIL', 'numpy']
        missing_packages = []

        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)

        if missing_packages:
            raise RuntimeError(f"Missing required packages: {', '.join(missing_packages)}")

        self.logger.info("✓ Preflight checks passed")

    def convert(self, input_path: str, output_path: str,
                input_format: str, output_format: str, **kwargs) -> Dict:
        """
        Convert annotations between formats

        Args:
            input_path: Input file or directory path
            output_path: Output file or directory path
            input_format: Input format (coco, yolo, voc)
            output_format: Output format (coco, yolo, voc)
            **kwargs: Additional format-specific arguments

        Returns:
            Conversion result dictionary
        """
        self.logger.info(f"Converting {input_format} → {output_format}")
        self.logger.info(f"Input: {input_path}")
        self.logger.info(f"Output: {output_path}")

        # Load from input format
        if input_format == 'coco':
            dataset = COCOConverter.load(input_path)
        elif input_format == 'yolo':
            images_dir = kwargs.get('images_dir', input_path)
            labels_dir = kwargs.get('labels_dir', input_path)
            classes_file = kwargs.get('classes_file')
            if not classes_file:
                raise ValueError("YOLO format requires classes_file parameter")
            dataset = YOLOConverter.load(images_dir, labels_dir, classes_file)
        elif input_format == 'voc':
            annotations_dir = input_path
            images_dir = kwargs.get('images_dir', input_path)
            dataset = PascalVOCConverter.load(annotations_dir, images_dir)
        else:
            raise ValueError(f"Unsupported input format: {input_format}")

        # Save to output format
        if output_format == 'coco':
            COCOConverter.save(dataset, output_path)
        elif output_format == 'yolo':
            classes_file = kwargs.get('output_classes_file',
                                     str(Path(output_path).parent / 'classes.txt'))
            YOLOConverter.save(dataset, output_path, classes_file)
        elif output_format == 'voc':
            PascalVOCConverter.save(dataset, output_path)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")

        stats = dataset.get_statistics()

        self.logger.info("✓ Conversion complete")
        self.logger.info(f"  Total images: {stats['total_images']}")
        self.logger.info(f"  Total annotations: {stats['total_annotations']}")
        self.logger.info(f"  Categories: {len(stats['categories'])}")

        return {
            'success': True,
            'input_format': input_format,
            'output_format': output_format,
            'statistics': stats
        }

    def validate(self, annotation_path: str, format: str,
                images_dir: Optional[str] = None, **kwargs) -> ValidationReport:
        """
        Validate annotations

        Args:
            annotation_path: Annotation file or directory path
            format: Annotation format (coco, yolo, voc)
            images_dir: Optional images directory for validation
            **kwargs: Additional format-specific arguments

        Returns:
            ValidationReport
        """
        self.logger.info(f"Validating {format} format annotations...")
        self.logger.info(f"Path: {annotation_path}")

        # Load dataset
        if format == 'coco':
            dataset = COCOConverter.load(annotation_path)
        elif format == 'yolo':
            labels_dir = annotation_path
            classes_file = kwargs.get('classes_file')
            if not classes_file:
                raise ValueError("YOLO format requires classes_file parameter")
            if not images_dir:
                raise ValueError("YOLO format requires images_dir parameter")
            dataset = YOLOConverter.load(images_dir, labels_dir, classes_file)
        elif format == 'voc':
            if not images_dir:
                images_dir = annotation_path
            dataset = PascalVOCConverter.load(annotation_path, images_dir)
        else:
            raise ValueError(f"Unsupported format: {format}")

        # Validate
        report = ValidationReport()
        report.total_images = len(dataset.images)

        for img in dataset.images:
            report.total_annotations += len(img.annotations)

            for ann in img.annotations:
                # Check bbox validity
                if not ann.bbox.is_valid(img.width, img.height):
                    report.invalid_annotations += 1
                    report.add_error(img.id,
                                    f"Invalid bbox: {ann.bbox.to_xywh()}")
                else:
                    report.valid_annotations += 1

                # Check category exists
                if ann.category_id not in dataset.categories:
                    report.add_warning(img.id,
                                      f"Unknown category ID: {ann.category_id}")

                # Check bbox size
                if ann.bbox.area() < 100:
                    report.add_warning(img.id,
                                      f"Very small bbox (area: {ann.bbox.area():.1f})")

            # Check if image has annotations
            if not img.annotations:
                report.add_warning(img.id, "No annotations")

        self.logger.info("✓ Validation complete")
        self.logger.info(f"  Valid: {report.valid_annotations}/{report.total_annotations}")
        self.logger.info(f"  Invalid: {report.invalid_annotations}")
        self.logger.info(f"  Warnings: {len(report.warnings)}")
        self.logger.info(f"  Errors: {len(report.errors)}")

        return report

    def analyze(self, annotation_path: str, format: str,
                output_file: Optional[str] = None, **kwargs) -> Dict:
        """
        Analyze annotation statistics

        Args:
            annotation_path: Annotation file or directory path
            format: Annotation format (coco, yolo, voc)
            output_file: Optional output file for statistics
            **kwargs: Additional format-specific arguments

        Returns:
            Statistics dictionary
        """
        self.logger.info(f"Analyzing {format} format annotations...")

        # Load dataset
        if format == 'coco':
            dataset = COCOConverter.load(annotation_path)
        elif format == 'yolo':
            images_dir = kwargs.get('images_dir')
            labels_dir = annotation_path
            classes_file = kwargs.get('classes_file')
            if not classes_file or not images_dir:
                raise ValueError("YOLO format requires images_dir and classes_file parameters")
            dataset = YOLOConverter.load(images_dir, labels_dir, classes_file)
        elif format == 'voc':
            images_dir = kwargs.get('images_dir', annotation_path)
            dataset = PascalVOCConverter.load(annotation_path, images_dir)
        else:
            raise ValueError(f"Unsupported format: {format}")

        # Get statistics
        stats = dataset.get_statistics()

        self.logger.info("✓ Analysis complete")
        self.logger.info(f"  Total images: {stats['total_images']}")
        self.logger.info(f"  Total annotations: {stats['total_annotations']}")
        self.logger.info(f"  Categories: {len(stats['categories'])}")
        self.logger.info(f"  Avg annotations/image: {stats['avg_annotations_per_image']:.2f}")

        # Save to file if specified
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(stats, f, indent=2)
            self.logger.info(f"✓ Statistics saved to {output_file}")

        return stats


# ============================================================================
# CLI Interface
# ============================================================================

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description='Annotation Tool Integration - Format conversion, validation, and analysis'
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    # Convert command
    convert_parser = subparsers.add_parser('convert', help='Convert annotation format')
    convert_parser.add_argument('--input', required=True, help='Input file or directory')
    convert_parser.add_argument('--output', required=True, help='Output file or directory')
    convert_parser.add_argument('--input-format', required=True,
                               choices=['coco', 'yolo', 'voc'], help='Input format')
    convert_parser.add_argument('--output-format', required=True,
                               choices=['coco', 'yolo', 'voc'], help='Output format')
    convert_parser.add_argument('--images-dir', help='Images directory (for YOLO/VOC)')
    convert_parser.add_argument('--labels-dir', help='Labels directory (for YOLO)')
    convert_parser.add_argument('--classes-file', help='Classes file (for YOLO)')
    convert_parser.add_argument('--output-classes-file', help='Output classes file (for YOLO)')

    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate annotations')
    validate_parser.add_argument('--input', required=True, help='Input file or directory')
    validate_parser.add_argument('--format', required=True,
                                choices=['coco', 'yolo', 'voc'], help='Annotation format')
    validate_parser.add_argument('--images-dir', help='Images directory')
    validate_parser.add_argument('--classes-file', help='Classes file (for YOLO)')
    validate_parser.add_argument('--report-file', help='Output validation report file')

    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze annotation statistics')
    analyze_parser.add_argument('--input', required=True, help='Input file or directory')
    analyze_parser.add_argument('--format', required=True,
                               choices=['coco', 'yolo', 'voc'], help='Annotation format')
    analyze_parser.add_argument('--images-dir', help='Images directory')
    analyze_parser.add_argument('--classes-file', help='Classes file (for YOLO)')
    analyze_parser.add_argument('--output', help='Output statistics file')

    # Global options
    parser.add_argument('--skip-preflight', action='store_true',
                       help='Skip preflight checks (for testing)')
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create tool
    tool = AnnotationTool(skip_preflight=args.skip_preflight)

    try:
        if args.command == 'convert':
            result = tool.convert(
                input_path=args.input,
                output_path=args.output,
                input_format=args.input_format,
                output_format=args.output_format,
                images_dir=args.images_dir,
                labels_dir=args.labels_dir,
                classes_file=args.classes_file,
                output_classes_file=args.output_classes_file
            )
            print("\n✅ Conversion successful!")
            print(f"   Output: {args.output}")
            print(f"   Images: {result['statistics']['total_images']}")
            print(f"   Annotations: {result['statistics']['total_annotations']}")

        elif args.command == 'validate':
            report = tool.validate(
                annotation_path=args.input,
                format=args.format,
                images_dir=args.images_dir,
                classes_file=args.classes_file
            )

            if args.report_file:
                with open(args.report_file, 'w') as f:
                    json.dump(asdict(report), f, indent=2)
                print(f"\n✅ Report saved to {args.report_file}")

            print("\n✅ Validation complete!")
            print(f"   Valid: {report.valid_annotations}/{report.total_annotations}")
            print(f"   Errors: {len(report.errors)}")
            print(f"   Warnings: {len(report.warnings)}")

        elif args.command == 'analyze':
            stats = tool.analyze(
                annotation_path=args.input,
                format=args.format,
                output_file=args.output,
                images_dir=args.images_dir,
                classes_file=args.classes_file
            )

            print("\n✅ Analysis complete!")
            print(f"   Images: {stats['total_images']}")
            print(f"   Annotations: {stats['total_annotations']}")
            print(f"   Categories: {len(stats['categories'])}")
            if args.output:
                print(f"   Statistics saved to: {args.output}")

        return 0

    except Exception as e:
        logging.error(f"Error: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())

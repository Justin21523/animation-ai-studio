"""
Character Segmentation Module for Video Editing

Integrates with 3d-animation-lora-pipeline's SAM2 implementation.
Provides video-specific character segmentation and tracking capabilities.

Key Features:
- Reuses proven SAM2 implementation from LoRA pipeline
- Video-specific character tracking
- Temporal consistency validation
- Integration with Agent Framework

Shared Resources:
- SAM2 models: /mnt/c/AI_LLM_projects/ai_warehouse/models/segmentation/
- Core utilities from LoRA pipeline

Author: Animation AI Studio
Date: 2025-11-17
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import time
import json
import cv2
import numpy as np

# Add both project roots to path
project_root = Path(__file__).parent.parent.parent.parent
lora_pipeline_root = Path("/mnt/c/AI_LLM_projects/3d-animation-lora-pipeline")
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(lora_pipeline_root))

# Import from LoRA pipeline
try:
    from scripts.generic.segmentation.instance_segmentation import SAM2InstanceSegmenter
    LORA_PIPELINE_AVAILABLE = True
except ImportError:
    LORA_PIPELINE_AVAILABLE = False
    logging.warning("LoRA pipeline not available, using standalone mode")


logger = logging.getLogger(__name__)


@dataclass
class CharacterSegment:
    """Single character segment in a frame"""
    frame_index: int
    timestamp: float  # seconds
    character_id: int
    mask: np.ndarray  # Binary mask (H x W)
    bounding_box: Tuple[int, int, int, int]  # (x, y, w, h)
    confidence: float
    area: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self, include_mask: bool = False) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = {
            "frame_index": self.frame_index,
            "timestamp": self.timestamp,
            "character_id": self.character_id,
            "bounding_box": list(self.bounding_box),
            "confidence": self.confidence,
            "area": self.area,
            "metadata": self.metadata
        }

        if include_mask:
            result["mask_shape"] = self.mask.shape
            # Optionally compress mask for storage

        return result


@dataclass
class CharacterTrack:
    """Character track across video"""
    character_id: int
    character_name: Optional[str] = None
    start_frame: int = 0
    end_frame: int = 0
    segments: List[CharacterSegment] = field(default_factory=list)
    is_consistent: bool = True
    avg_confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "character_id": self.character_id,
            "character_name": self.character_name,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "total_segments": len(self.segments),
            "is_consistent": self.is_consistent,
            "avg_confidence": self.avg_confidence,
            "segments": [s.to_dict() for s in self.segments],
            "metadata": self.metadata
        }


@dataclass
class VideoSegmentationResult:
    """Complete video segmentation result"""
    video_path: str
    total_frames: int
    fps: float
    duration: float
    character_tracks: List[CharacterTrack]
    segmentation_time: float
    model_name: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "video_path": self.video_path,
            "total_frames": self.total_frames,
            "fps": self.fps,
            "duration": self.duration,
            "character_tracks": [ct.to_dict() for ct in self.character_tracks],
            "segmentation_time": self.segmentation_time,
            "model_name": self.model_name,
            "metadata": self.metadata
        }

    def save_json(self, output_path: str):
        """Save to JSON"""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info(f"Saved segmentation result to: {output_path}")


class CharacterSegmenter:
    """
    Character Segmenter for Video Editing

    Integrates with LoRA pipeline's SAM2 implementation.
    Provides video-specific segmentation and tracking.

    Usage:
        segmenter = CharacterSegmenter(model_size="base")
        result = segmenter.segment_video(
            video_path="video.mp4",
            sample_interval=1
        )
    """

    def __init__(
        self,
        model_size: str = "base",  # tiny, small, base, large
        device: str = "cuda",
        min_mask_size: int = 64 * 64,
        max_characters_per_frame: int = 15
    ):
        """
        Initialize character segmenter

        Args:
            model_size: SAM2 model size (base = 6GB VRAM, large = 16GB)
            device: cuda or cpu
            min_mask_size: Minimum character area
            max_characters_per_frame: Maximum characters to track per frame
        """
        self.model_size = model_size
        self.device = device
        self.min_mask_size = min_mask_size
        self.max_characters_per_frame = max_characters_per_frame

        # Map model size to LoRA pipeline naming
        model_mapping = {
            "tiny": "sam2_hiera_tiny",
            "small": "sam2_hiera_small",
            "base": "sam2_hiera_base",
            "large": "sam2_hiera_large"
        }
        self.model_type = model_mapping.get(model_size, "sam2_hiera_base")

        logger.info(f"CharacterSegmenter initialized (model={self.model_type}, device={device})")

        # Initialize SAM2 segmenter
        self.segmenter = None
        self._load_segmenter()

    def _load_segmenter(self):
        """Load SAM2 segmenter from LoRA pipeline"""
        if not LORA_PIPELINE_AVAILABLE:
            raise RuntimeError(
                "LoRA pipeline not available. Please ensure:\n"
                "1. /mnt/c/AI_LLM_projects/3d-animation-lora-pipeline exists\n"
                "2. SAM2 is installed: pip install git+https://github.com/facebookresearch/sam2.git"
            )

        try:
            logger.info("Loading SAM2 from LoRA pipeline...")
            self.segmenter = SAM2InstanceSegmenter(
                model_type=self.model_type,
                device=self.device,
                min_mask_size=self.min_mask_size,
                max_instances=self.max_characters_per_frame
            )
            logger.info("SAM2 segmenter loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load SAM2 segmenter: {e}")
            raise

    def segment_video(
        self,
        video_path: str,
        sample_interval: int = 1,
        output_masks_dir: Optional[str] = None,
        track_characters: bool = True
    ) -> VideoSegmentationResult:
        """
        Segment all characters in video

        Args:
            video_path: Path to video file
            sample_interval: Process every Nth frame
            output_masks_dir: Directory to save masks (optional)
            track_characters: Whether to track characters across frames

        Returns:
            VideoSegmentationResult
        """
        start_time = time.time()

        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        logger.info(f"Segmenting video: {video_path}")

        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps

        logger.info(f"Video: {fps:.2f} FPS, {total_frames} frames, {duration:.2f}s")

        # Create output directory if requested
        if output_masks_dir:
            Path(output_masks_dir).mkdir(parents=True, exist_ok=True)

        # Process frames
        all_segments = []
        frame_idx = 0
        processed_frames = 0

        from PIL import Image

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_interval == 0:
                timestamp = frame_idx / fps

                # Convert to PIL Image (RGB)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)

                # Segment frame
                instances = self.segmenter.segment_instances(pil_image, return_scores=True)

                # Convert to CharacterSegment format
                for instance in instances:
                    segment = CharacterSegment(
                        frame_index=frame_idx,
                        timestamp=timestamp,
                        character_id=instance['instance_id'],
                        mask=instance['mask'],
                        bounding_box=tuple(instance['bbox']),
                        confidence=instance.get('stability_score', 1.0),
                        area=instance['area']
                    )
                    all_segments.append(segment)

                    # Save mask if requested
                    if output_masks_dir:
                        mask_filename = f"frame_{frame_idx:06d}_char_{instance['instance_id']}.png"
                        mask_path = Path(output_masks_dir) / mask_filename
                        mask_uint8 = (instance['mask'] * 255).astype(np.uint8)
                        cv2.imwrite(str(mask_path), mask_uint8)

                processed_frames += 1

                if processed_frames % 10 == 0:
                    logger.info(f"Processed {processed_frames} frames ({frame_idx}/{total_frames})")

            frame_idx += 1

        cap.release()

        logger.info(f"Segmented {len(all_segments)} character instances across {processed_frames} frames")

        # Build character tracks
        if track_characters:
            character_tracks = self._build_tracks(all_segments)
        else:
            # No tracking, each segment is its own track
            character_tracks = [
                CharacterTrack(
                    character_id=i,
                    start_frame=seg.frame_index,
                    end_frame=seg.frame_index,
                    segments=[seg]
                )
                for i, seg in enumerate(all_segments)
            ]

        segmentation_time = time.time() - start_time

        result = VideoSegmentationResult(
            video_path=video_path,
            total_frames=total_frames,
            fps=fps,
            duration=duration,
            character_tracks=character_tracks,
            segmentation_time=segmentation_time,
            model_name=f"sam2_{self.model_size}",
            metadata={
                "sample_interval": sample_interval,
                "processed_frames": processed_frames,
                "total_segments": len(all_segments)
            }
        )

        logger.info(f"Segmentation completed in {segmentation_time:.2f}s")
        logger.info(f"Tracked {len(character_tracks)} characters")

        return result

    def _build_tracks(self, segments: List[CharacterSegment]) -> List[CharacterTrack]:
        """Build character tracks from segments using simple IoU matching"""
        if not segments:
            return []

        # Group segments by frame
        frames_segments = {}
        for seg in segments:
            if seg.frame_index not in frames_segments:
                frames_segments[seg.frame_index] = []
            frames_segments[seg.frame_index].append(seg)

        # Sort frame indices
        sorted_frames = sorted(frames_segments.keys())

        # Initialize tracks with first frame
        tracks = []
        next_track_id = 0

        if sorted_frames:
            first_frame_segs = frames_segments[sorted_frames[0]]
            for seg in first_frame_segs:
                track = CharacterTrack(
                    character_id=next_track_id,
                    start_frame=seg.frame_index,
                    end_frame=seg.frame_index,
                    segments=[seg]
                )
                tracks.append(track)
                next_track_id += 1

        # Match subsequent frames
        for frame_idx in sorted_frames[1:]:
            current_segs = frames_segments[frame_idx]

            # Match each segment to existing tracks
            matched_tracks = set()

            for seg in current_segs:
                # Find best matching track (highest IoU)
                best_track = None
                best_iou = 0.3  # Minimum IoU threshold

                for track in tracks:
                    if track.character_id in matched_tracks:
                        continue

                    # Get last segment in track
                    last_seg = track.segments[-1]

                    # Calculate IoU
                    iou = self._calculate_bbox_iou(seg.bounding_box, last_seg.bounding_box)

                    if iou > best_iou:
                        best_iou = iou
                        best_track = track

                if best_track:
                    # Add to existing track
                    best_track.segments.append(seg)
                    best_track.end_frame = seg.frame_index
                    matched_tracks.add(best_track.character_id)
                else:
                    # Create new track
                    new_track = CharacterTrack(
                        character_id=next_track_id,
                        start_frame=seg.frame_index,
                        end_frame=seg.frame_index,
                        segments=[seg]
                    )
                    tracks.append(new_track)
                    next_track_id += 1

        # Calculate track statistics
        for track in tracks:
            if track.segments:
                track.avg_confidence = np.mean([s.confidence for s in track.segments])

                # Check consistency (area variance)
                areas = [s.area for s in track.segments]
                area_std = np.std(areas)
                area_mean = np.mean(areas)
                track.is_consistent = (area_std / area_mean) < 0.5 if area_mean > 0 else True

        logger.info(f"Built {len(tracks)} character tracks")

        return tracks

    def _calculate_bbox_iou(
        self,
        bbox1: Tuple[int, int, int, int],
        bbox2: Tuple[int, int, int, int]
    ) -> float:
        """Calculate IoU between two bounding boxes (x, y, w, h format)"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        # Convert to (x1, y1, x2, y2)
        box1 = [x1, y1, x1 + w1, y1 + h1]
        box2 = [x2, y2, x2 + w2, y2 + h2]

        # Calculate intersection
        x_left = max(box1[0], box2[0])
        y_top = max(box1[1], box2[1])
        x_right = min(box1[2], box2[2])
        y_bottom = min(box1[3], box2[3])

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # Calculate union
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - intersection_area

        iou = intersection_area / union_area if union_area > 0 else 0.0

        return iou


def main():
    """Example usage"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    video_path = "path/to/your/video.mp4"

    if not Path(video_path).exists():
        logger.warning(f"Example video not found: {video_path}")
        logger.info("Please provide a valid video path")
        return

    # Create segmenter
    segmenter = CharacterSegmenter(
        model_size="base",  # 6GB VRAM
        device="cuda"
    )

    # Segment video
    result = segmenter.segment_video(
        video_path=video_path,
        sample_interval=1,
        output_masks_dir="outputs/segmentation/masks",
        track_characters=True
    )

    # Print results
    print("\n" + "=" * 60)
    print("CHARACTER SEGMENTATION RESULTS")
    print("=" * 60)
    print(f"Video: {result.video_path}")
    print(f"Duration: {result.duration:.2f}s")
    print(f"Segmentation Time: {result.segmentation_time:.2f}s")
    print(f"Characters Tracked: {len(result.character_tracks)}")

    for track in result.character_tracks:
        print(f"\nCharacter {track.character_id}:")
        print(f"  Frames: {track.start_frame} - {track.end_frame}")
        print(f"  Segments: {len(track.segments)}")
        print(f"  Avg Confidence: {track.avg_confidence:.3f}")
        print(f"  Consistent: {track.is_consistent}")

    # Save to JSON
    output_json = "outputs/segmentation/character_tracks.json"
    result.save_json(output_json)
    print(f"\nResults saved to: {output_json}")


if __name__ == "__main__":
    main()

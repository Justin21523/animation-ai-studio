"""
Video Editing Engine using MoviePy

High-level video editing operations for AI-driven video editing:
- Cut and trim clips
- Composite multiple layers
- Apply transitions and effects
- Speed manipulation
- Text and graphics overlay

Built on MoviePy for professional video editing capabilities.

Author: Animation AI Studio
Date: 2025-11-17
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
import time
import json
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# MoviePy imports
try:
    from moviepy.editor import (
        VideoFileClip, AudioFileClip, ImageClip, TextClip, CompositeVideoClip,
        concatenate_videoclips, vfx, afx
    )
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False
    logging.warning("MoviePy not installed. Install with: pip install moviepy")


logger = logging.getLogger(__name__)


@dataclass
class EditOperation:
    """Single edit operation"""
    operation_type: str  # "cut", "composite", "speed", "effect", "transition"
    start_time: float
    end_time: float
    parameters: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "operation_type": self.operation_type,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "parameters": self.parameters,
            "metadata": self.metadata
        }


@dataclass
class EditSequence:
    """Sequence of edit operations"""
    sequence_id: str
    operations: List[EditOperation] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "sequence_id": self.sequence_id,
            "operations": [op.to_dict() for op in self.operations],
            "metadata": self.metadata
        }


@dataclass
class EditResult:
    """Result of video editing operation"""
    output_path: str
    success: bool
    edit_time: float
    input_duration: float
    output_duration: float
    operations_applied: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "output_path": self.output_path,
            "success": self.success,
            "edit_time": self.edit_time,
            "input_duration": self.input_duration,
            "output_duration": self.output_duration,
            "operations_applied": self.operations_applied,
            "metadata": self.metadata
        }


class VideoEditor:
    """
    Video Editor using MoviePy

    Provides high-level video editing operations optimized for AI-driven editing.

    Features:
    - Cut and trim clips
    - Speed manipulation (slow motion, fast forward)
    - Composite multiple layers (character + background)
    - Apply transitions (crossfade, slide)
    - Add effects (blur, brightness, contrast)
    - Text and graphics overlay

    Usage:
        editor = VideoEditor()
        result = editor.cut_clip(
            video_path="video.mp4",
            start_time=10.0,
            end_time=20.0,
            output_path="cut.mp4"
        )
    """

    def __init__(
        self,
        temp_dir: Optional[str] = None,
        enable_preview: bool = False
    ):
        """
        Initialize video editor

        Args:
            temp_dir: Directory for temporary files
            enable_preview: Enable preview generation
        """
        if not MOVIEPY_AVAILABLE:
            raise ImportError("MoviePy not installed. Install with: pip install moviepy")

        self.temp_dir = temp_dir or "outputs/editing/temp"
        self.enable_preview = enable_preview

        Path(self.temp_dir).mkdir(parents=True, exist_ok=True)

        logger.info(f"VideoEditor initialized (temp_dir={self.temp_dir})")

    def cut_clip(
        self,
        video_path: str,
        start_time: float,
        end_time: float,
        output_path: str,
        codec: str = "libx264",
        audio_codec: str = "aac"
    ) -> EditResult:
        """
        Cut a clip from video

        Args:
            video_path: Input video path
            start_time: Start time in seconds
            end_time: End time in seconds
            output_path: Output video path
            codec: Video codec
            audio_codec: Audio codec

        Returns:
            EditResult
        """
        start = time.time()

        logger.info(f"Cutting clip: {start_time:.2f}s - {end_time:.2f}s")

        try:
            # Load video
            clip = VideoFileClip(video_path)
            input_duration = clip.duration

            # Cut
            cut_clip = clip.subclip(start_time, end_time)

            # Write output
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            cut_clip.write_videofile(
                output_path,
                codec=codec,
                audio_codec=audio_codec,
                logger=None  # Suppress MoviePy logging
            )

            output_duration = cut_clip.duration

            # Cleanup
            clip.close()
            cut_clip.close()

            edit_time = time.time() - start

            result = EditResult(
                output_path=output_path,
                success=True,
                edit_time=edit_time,
                input_duration=input_duration,
                output_duration=output_duration,
                operations_applied=1,
                metadata={
                    "operation": "cut",
                    "start_time": start_time,
                    "end_time": end_time
                }
            )

            logger.info(f"Clip cut successfully: {output_duration:.2f}s")

            return result

        except Exception as e:
            logger.error(f"Failed to cut clip: {e}")
            return EditResult(
                output_path=output_path,
                success=False,
                edit_time=time.time() - start,
                input_duration=0.0,
                output_duration=0.0,
                operations_applied=0,
                metadata={"error": str(e)}
            )

    def change_speed(
        self,
        video_path: str,
        speed_factor: float,
        output_path: str,
        codec: str = "libx264"
    ) -> EditResult:
        """
        Change video speed

        Args:
            video_path: Input video path
            speed_factor: Speed multiplier (0.5 = slow motion, 2.0 = fast forward)
            output_path: Output video path
            codec: Video codec

        Returns:
            EditResult
        """
        start = time.time()

        logger.info(f"Changing speed: {speed_factor}x")

        try:
            clip = VideoFileClip(video_path)
            input_duration = clip.duration

            # Change speed
            if speed_factor > 1.0:
                # Speed up
                speed_clip = clip.fx(vfx.speedx, speed_factor)
            else:
                # Slow down
                speed_clip = clip.fx(vfx.speedx, speed_factor)

            # Write output
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            speed_clip.write_videofile(
                output_path,
                codec=codec,
                audio_codec="aac",
                logger=None
            )

            output_duration = speed_clip.duration

            clip.close()
            speed_clip.close()

            edit_time = time.time() - start

            result = EditResult(
                output_path=output_path,
                success=True,
                edit_time=edit_time,
                input_duration=input_duration,
                output_duration=output_duration,
                operations_applied=1,
                metadata={
                    "operation": "speed",
                    "speed_factor": speed_factor
                }
            )

            logger.info(f"Speed changed successfully: {output_duration:.2f}s")

            return result

        except Exception as e:
            logger.error(f"Failed to change speed: {e}")
            return EditResult(
                output_path=output_path,
                success=False,
                edit_time=time.time() - start,
                input_duration=0.0,
                output_duration=0.0,
                operations_applied=0,
                metadata={"error": str(e)}
            )

    def composite_layers(
        self,
        background_path: str,
        foreground_path: str,
        output_path: str,
        position: Tuple[int, int] = (0, 0),
        opacity: float = 1.0,
        codec: str = "libx264"
    ) -> EditResult:
        """
        Composite foreground onto background

        Args:
            background_path: Background video path
            foreground_path: Foreground video path (with alpha channel)
            output_path: Output video path
            position: (x, y) position of foreground
            opacity: Foreground opacity (0.0 to 1.0)
            codec: Video codec

        Returns:
            EditResult
        """
        start = time.time()

        logger.info(f"Compositing layers: {foreground_path} onto {background_path}")

        try:
            background = VideoFileClip(background_path)
            foreground = VideoFileClip(foreground_path)

            input_duration = background.duration

            # Set position and opacity
            foreground = foreground.set_position(position)
            if opacity < 1.0:
                foreground = foreground.set_opacity(opacity)

            # Composite
            composite = CompositeVideoClip([background, foreground])

            # Write output
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            composite.write_videofile(
                output_path,
                codec=codec,
                audio_codec="aac",
                logger=None
            )

            output_duration = composite.duration

            background.close()
            foreground.close()
            composite.close()

            edit_time = time.time() - start

            result = EditResult(
                output_path=output_path,
                success=True,
                edit_time=edit_time,
                input_duration=input_duration,
                output_duration=output_duration,
                operations_applied=1,
                metadata={
                    "operation": "composite",
                    "position": position,
                    "opacity": opacity
                }
            )

            logger.info(f"Layers composited successfully")

            return result

        except Exception as e:
            logger.error(f"Failed to composite layers: {e}")
            return EditResult(
                output_path=output_path,
                success=False,
                edit_time=time.time() - start,
                input_duration=0.0,
                output_duration=0.0,
                operations_applied=0,
                metadata={"error": str(e)}
            )

    def concatenate_clips(
        self,
        clip_paths: List[str],
        output_path: str,
        transition: Optional[str] = None,
        transition_duration: float = 0.5,
        codec: str = "libx264"
    ) -> EditResult:
        """
        Concatenate multiple clips

        Args:
            clip_paths: List of video paths to concatenate
            output_path: Output video path
            transition: Transition type ("crossfade", None)
            transition_duration: Transition duration in seconds
            codec: Video codec

        Returns:
            EditResult
        """
        start = time.time()

        logger.info(f"Concatenating {len(clip_paths)} clips")

        try:
            # Load clips
            clips = [VideoFileClip(path) for path in clip_paths]

            total_input_duration = sum(clip.duration for clip in clips)

            # Apply transitions
            if transition == "crossfade":
                # Apply crossfade between clips
                for i in range(len(clips) - 1):
                    clips[i+1] = clips[i+1].crossfadein(transition_duration)

            # Concatenate
            final_clip = concatenate_videoclips(clips, method="compose")

            # Write output
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            final_clip.write_videofile(
                output_path,
                codec=codec,
                audio_codec="aac",
                logger=None
            )

            output_duration = final_clip.duration

            # Cleanup
            for clip in clips:
                clip.close()
            final_clip.close()

            edit_time = time.time() - start

            result = EditResult(
                output_path=output_path,
                success=True,
                edit_time=edit_time,
                input_duration=total_input_duration,
                output_duration=output_duration,
                operations_applied=len(clip_paths),
                metadata={
                    "operation": "concatenate",
                    "num_clips": len(clip_paths),
                    "transition": transition,
                    "transition_duration": transition_duration
                }
            )

            logger.info(f"Clips concatenated successfully: {output_duration:.2f}s")

            return result

        except Exception as e:
            logger.error(f"Failed to concatenate clips: {e}")
            return EditResult(
                output_path=output_path,
                success=False,
                edit_time=time.time() - start,
                input_duration=0.0,
                output_duration=0.0,
                operations_applied=0,
                metadata={"error": str(e)}
            )

    def add_text_overlay(
        self,
        video_path: str,
        text: str,
        output_path: str,
        position: Tuple[int, int] = (100, 100),
        font_size: int = 50,
        color: str = "white",
        duration: Optional[float] = None,
        codec: str = "libx264"
    ) -> EditResult:
        """
        Add text overlay to video

        Args:
            video_path: Input video path
            text: Text to overlay
            output_path: Output video path
            position: (x, y) position of text
            font_size: Font size
            color: Text color
            duration: Text duration (None = entire video)
            codec: Video codec

        Returns:
            EditResult
        """
        start = time.time()

        logger.info(f"Adding text overlay: {text}")

        try:
            video = VideoFileClip(video_path)
            input_duration = video.duration

            # Create text clip
            txt_duration = duration if duration else video.duration
            txt_clip = TextClip(
                text,
                fontsize=font_size,
                color=color,
                font="Arial"
            ).set_position(position).set_duration(txt_duration)

            # Composite
            final = CompositeVideoClip([video, txt_clip])

            # Write output
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            final.write_videofile(
                output_path,
                codec=codec,
                audio_codec="aac",
                logger=None
            )

            output_duration = final.duration

            video.close()
            txt_clip.close()
            final.close()

            edit_time = time.time() - start

            result = EditResult(
                output_path=output_path,
                success=True,
                edit_time=edit_time,
                input_duration=input_duration,
                output_duration=output_duration,
                operations_applied=1,
                metadata={
                    "operation": "text_overlay",
                    "text": text,
                    "position": position,
                    "font_size": font_size
                }
            )

            logger.info(f"Text overlay added successfully")

            return result

        except Exception as e:
            logger.error(f"Failed to add text overlay: {e}")
            return EditResult(
                output_path=output_path,
                success=False,
                edit_time=time.time() - start,
                input_duration=0.0,
                output_duration=0.0,
                operations_applied=0,
                metadata={"error": str(e)}
            )

    def apply_effect(
        self,
        video_path: str,
        effect_type: str,
        output_path: str,
        parameters: Optional[Dict[str, Any]] = None,
        codec: str = "libx264"
    ) -> EditResult:
        """
        Apply video effect

        Args:
            video_path: Input video path
            effect_type: Effect type ("mirror_x", "mirror_y", "blackwhite", "invert_colors")
            output_path: Output video path
            parameters: Effect parameters
            codec: Video codec

        Returns:
            EditResult
        """
        start = time.time()

        logger.info(f"Applying effect: {effect_type}")

        try:
            clip = VideoFileClip(video_path)
            input_duration = clip.duration

            # Apply effect
            if effect_type == "mirror_x":
                effect_clip = clip.fx(vfx.mirror_x)
            elif effect_type == "mirror_y":
                effect_clip = clip.fx(vfx.mirror_y)
            elif effect_type == "blackwhite":
                effect_clip = clip.fx(vfx.blackwhite)
            elif effect_type == "invert_colors":
                effect_clip = clip.fx(vfx.invert_colors)
            else:
                raise ValueError(f"Unknown effect type: {effect_type}")

            # Write output
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            effect_clip.write_videofile(
                output_path,
                codec=codec,
                audio_codec="aac",
                logger=None
            )

            output_duration = effect_clip.duration

            clip.close()
            effect_clip.close()

            edit_time = time.time() - start

            result = EditResult(
                output_path=output_path,
                success=True,
                edit_time=edit_time,
                input_duration=input_duration,
                output_duration=output_duration,
                operations_applied=1,
                metadata={
                    "operation": "effect",
                    "effect_type": effect_type,
                    "parameters": parameters or {}
                }
            )

            logger.info(f"Effect applied successfully")

            return result

        except Exception as e:
            logger.error(f"Failed to apply effect: {e}")
            return EditResult(
                output_path=output_path,
                success=False,
                edit_time=time.time() - start,
                input_duration=0.0,
                output_duration=0.0,
                operations_applied=0,
                metadata={"error": str(e)}
            )


def main():
    """Example usage"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Example: Cut a clip
    editor = VideoEditor()

    video_path = "path/to/your/video.mp4"

    if not Path(video_path).exists():
        logger.warning(f"Example video not found: {video_path}")
        logger.info("Please provide a valid video path")
        return

    # Cut clip
    result = editor.cut_clip(
        video_path=video_path,
        start_time=10.0,
        end_time=20.0,
        output_path="outputs/editing/cut_clip.mp4"
    )

    print("\n" + "=" * 60)
    print("VIDEO EDITING RESULT")
    print("=" * 60)
    print(f"Success: {result.success}")
    print(f"Output: {result.output_path}")
    print(f"Edit Time: {result.edit_time:.2f}s")
    print(f"Input Duration: {result.input_duration:.2f}s")
    print(f"Output Duration: {result.output_duration:.2f}s")
    print(f"Operations: {result.operations_applied}")


if __name__ == "__main__":
    main()

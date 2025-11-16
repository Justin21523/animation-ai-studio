"""
Utility functions for LLM client
"""

import base64
from pathlib import Path
from typing import Union, List
import cv2
import numpy as np


def encode_image_to_base64(image_path: Union[str, Path]) -> str:
    """
    Encode image file to base64 string

    Args:
        image_path: Path to image file

    Returns:
        Base64 encoded string
    """
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def encode_numpy_to_base64(image: np.ndarray) -> str:
    """
    Encode numpy array (image) to base64

    Args:
        image: Numpy array (BGR format from OpenCV)

    Returns:
        Base64 encoded JPEG string
    """
    # Convert to JPEG bytes
    success, buffer = cv2.imencode('.jpg', image)
    if not success:
        raise ValueError("Failed to encode image")

    # Encode to base64
    return base64.b64encode(buffer).decode('utf-8')


def extract_video_frames(
    video_path: Union[str, Path],
    num_frames: int = 10,
    method: str = 'uniform'
) -> List[np.ndarray]:
    """
    Extract frames from video

    Args:
        video_path: Path to video file
        num_frames: Number of frames to extract
        method: 'uniform' or 'keyframe'

    Returns:
        List of frames as numpy arrays
    """
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames = []

    if method == 'uniform':
        # Extract uniformly spaced frames
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)

    elif method == 'keyframe':
        # Extract key frames (simplified: every Nth frame)
        step = max(1, total_frames // num_frames)
        for i in range(0, total_frames, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
            if len(frames) >= num_frames:
                break

    cap.release()
    return frames


def frames_to_base64(frames: List[np.ndarray]) -> List[str]:
    """
    Convert list of frames to base64 strings

    Args:
        frames: List of numpy arrays

    Returns:
        List of base64 encoded strings
    """
    return [encode_numpy_to_base64(frame) for frame in frames]


def create_prompt_with_images(
    text: str,
    image_paths: List[Union[str, Path]]
) -> List[Dict]:
    """
    Create multimodal prompt content

    Args:
        text: Text prompt
        image_paths: List of image paths

    Returns:
        Formatted content for LLM API
    """
    content = [{"type": "text", "text": text}]

    for img_path in image_paths:
        img_base64 = encode_image_to_base64(img_path)
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}
        })

    return content


def parse_json_from_markdown(text: str) -> dict:
    """
    Parse JSON from markdown code block

    Args:
        text: Text possibly containing JSON in markdown

    Returns:
        Parsed JSON dict
    """
    import json

    # Try to extract JSON from markdown code block
    if "```json" in text:
        json_str = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        json_str = text.split("```")[1].split("```")[0].strip()
    else:
        json_str = text

    return json.loads(json_str)


def extract_code_from_markdown(text: str, language: str = "python") -> str:
    """
    Extract code from markdown code block

    Args:
        text: Text containing code in markdown
        language: Programming language

    Returns:
        Extracted code
    """
    marker = f"```{language}"
    if marker in text:
        code = text.split(marker)[1].split("```")[0].strip()
        return code
    elif "```" in text:
        # Try generic code block
        code = text.split("```")[1].split("```")[0].strip()
        return code
    else:
        # Return as-is
        return text

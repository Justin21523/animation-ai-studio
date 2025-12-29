#!/usr/bin/env python3
"""
Video Semantic Builder (P3)

Builds a structured, RAG-friendly index for video content:
- Video → Scenes → Shots → (Characters) → Dialogue

Inputs are existing analysis artifacts produced by Module 7/8 and voice pipelines:
- Scene detection JSON (Module 7)
- Camera tracking JSON (Module 7)
- Character segmentation JSON (Module 8, optional)
- Transcript/diarization JSON (optional)

Outputs:
- `video_semantic.json`: compact structured summary
- `documents/*.json`: JSON lists designed for scripts/rag ingestion

This is intentionally "best-effort": missing inputs are tolerated so the pipeline can
incrementally improve as more modalities become available.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


logger = logging.getLogger(__name__)

SCHEMA_VERSION = "1.0"


def _normalize_tag(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value = str(value).strip()
    return value.lower() if value else None


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: Optional[int] = 0) -> Optional[int]:
    try:
        return int(value)
    except Exception:
        return default


def _read_json_if_exists(path: Optional[Path]) -> Any:
    if not path:
        return None
    path = Path(path)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@dataclass(frozen=True)
class TimeRange:
    start_time: float
    end_time: float

    def overlaps(self, other: "TimeRange") -> bool:
        return self.end_time > other.start_time and self.start_time < other.end_time


def _find_scene_id(scenes: List[Dict[str, Any]], t: float) -> Optional[int]:
    for scene in scenes:
        s = _safe_float(scene.get("start_time"))
        e = _safe_float(scene.get("end_time"))
        if s <= t < e:
            return _safe_int(scene.get("scene_id"), None)
    return None


def _find_shot_id(shots: List[Dict[str, Any]], t: float) -> Optional[int]:
    for shot in shots:
        s = _safe_float(shot.get("start_time"))
        e = _safe_float(shot.get("end_time"))
        if s <= t < e:
            return _safe_int(shot.get("shot_id"), None)
    return None


def _normalize_transcript_segments(data: Any) -> List[Dict[str, Any]]:
    """
    Normalize various transcript formats to a flat list of segments:
    {start, end, text, speaker, character?, emotion?}
    """
    if data is None:
        return []

    # Format: list of segments
    if isinstance(data, list):
        segments: List[Dict[str, Any]] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            segments.append(
                {
                    "start": _safe_float(item.get("start", item.get("start_time"))),
                    "end": _safe_float(item.get("end", item.get("end_time"))),
                    "text": str(item.get("text", "")).strip(),
                    "speaker": str(item.get("speaker", "")).strip() or None,
                    "character": (str(item.get("character")).strip().lower() if item.get("character") else None),
                    "emotion": (str(item.get("emotion")).strip().lower() if item.get("emotion") else None),
                }
            )
        return [s for s in segments if s["end"] > s["start"] and s["text"]]

    # Format: {speaker: [segments...], ...}
    if isinstance(data, dict):
        # Common output: {"SPEAKER_00": [{start,end,text,...}, ...], ...}
        segments: List[Dict[str, Any]] = []
        for speaker, items in data.items():
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict):
                    continue
                segments.append(
                    {
                        "start": _safe_float(item.get("start", item.get("start_time"))),
                        "end": _safe_float(item.get("end", item.get("end_time"))),
                        "text": str(item.get("text", "")).strip(),
                        "speaker": str(item.get("speaker", speaker)).strip() or str(speaker),
                        "character": (str(item.get("character")).strip().lower() if item.get("character") else None),
                        "emotion": (str(item.get("emotion")).strip().lower() if item.get("emotion") else None),
                    }
                )
        return [s for s in segments if s["end"] > s["start"] and s["text"]]

    return []


def _convert_camera_shots(camera_tracking: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """
    Convert camera tracking shots to include start/end times (seconds).
    """
    if not camera_tracking:
        return [], None

    fps = _safe_float(camera_tracking.get("fps"), 0.0)
    camera_style = _normalize_tag(camera_tracking.get("camera_style"))

    shots = []
    for shot in camera_tracking.get("shots", []) or []:
        if not isinstance(shot, dict):
            continue
        start_frame = _safe_int(shot.get("start_frame"), 0)
        end_frame = _safe_int(shot.get("end_frame"), 0)
        start_time = (start_frame / fps) if fps > 0 else 0.0
        end_time = (end_frame / fps) if fps > 0 else 0.0
        duration = _safe_float(shot.get("duration"), max(0.0, end_time - start_time))

        shots.append(
            {
                "shot_id": _safe_int(shot.get("shot_id"), 0),
                "start_time": start_time,
                "end_time": end_time,
                "duration": duration,
                "dominant_movement": _normalize_tag(shot.get("dominant_movement")) or "unknown",
                "avg_movement_intensity": _safe_float(shot.get("avg_movement_intensity"), 0.0),
                "is_handheld": bool(shot.get("is_handheld", False)),
                "is_mostly_static": bool(shot.get("is_mostly_static", False)),
                "smoothness_score": _safe_float(shot.get("smoothness_score"), 0.0),
            }
        )

    # Sort by time for stable IDs.
    shots.sort(key=lambda s: (s["start_time"], s["shot_id"]))
    return shots, camera_style


def build_video_semantics(
    *,
    video_path: Path,
    film: Optional[str],
    scene_detection_path: Optional[Path],
    camera_tracking_path: Optional[Path],
    transcript_path: Optional[Path],
    segmentation_path: Optional[Path],
    output_dir: Path,
    emit_documents: bool = True,
    extract_reference_frames: bool = False,
    max_reference_frames_per_character: int = 10,
) -> Dict[str, Any]:
    """
    Build structured semantic index + RAG documents.
    """
    video_id = video_path.stem
    film = _normalize_tag(film)

    scene_detection = _read_json_if_exists(scene_detection_path) or {}
    scenes = scene_detection.get("scenes", []) if isinstance(scene_detection, dict) else []
    scenes = [s for s in scenes if isinstance(s, dict)]

    camera_tracking = _read_json_if_exists(camera_tracking_path) or {}
    shots, camera_style = _convert_camera_shots(camera_tracking if isinstance(camera_tracking, dict) else {})

    transcript_raw = _read_json_if_exists(transcript_path)
    transcript_segments = _normalize_transcript_segments(transcript_raw)

    segmentation = _read_json_if_exists(segmentation_path) or {}
    tracks = []
    if isinstance(segmentation, dict):
        for track in segmentation.get("character_tracks", []) or []:
            if not isinstance(track, dict):
                continue
            segments = track.get("segments", []) or []
            seg_infos: List[Tuple[float, int, float, Optional[List[Any]]]] = []
            for seg in segments[:300]:
                if not isinstance(seg, dict):
                    continue
                ts = _safe_float(seg.get("timestamp"), 0.0)
                if ts <= 0:
                    continue
                frame_index = int(_safe_int(seg.get("frame_index"), 0) or 0)
                conf = _safe_float(seg.get("confidence"), 0.0)
                bbox = seg.get("bounding_box")
                bbox_list = bbox if isinstance(bbox, list) else None
                seg_infos.append((ts, frame_index, conf, bbox_list))

            seg_infos.sort(key=lambda x: (x[0], x[1]))

            max_refs = max(1, int(max_reference_frames_per_character))
            if len(seg_infos) <= max_refs:
                ref_infos = seg_infos
            else:
                ref_infos = []
                for j in range(max_refs):
                    idx = round(j * (len(seg_infos) - 1) / max(1, max_refs - 1))
                    ref_infos.append(seg_infos[int(idx)])

            reference_segments = [
                {
                    "timestamp": ts,
                    "frame_index": fi,
                    "confidence": conf,
                    "bounding_box": bbox,
                }
                for ts, fi, conf, bbox in ref_infos
            ]
            reference_timestamps = [s["timestamp"] for s in reference_segments]

            tracks.append(
                {
                    "character_id": _safe_int(track.get("character_id"), 0),
                    "character_name": _normalize_tag(track.get("character_name")),
                    "start_time": float(seg_infos[0][0]) if seg_infos else 0.0,
                    "end_time": float(seg_infos[-1][0]) if seg_infos else 0.0,
                    "total_segments": _safe_int(track.get("total_segments", len(segments)), len(segments)),
                    "avg_confidence": _safe_float(track.get("avg_confidence"), 0.0),
                    "reference_segments": reference_segments,
                    "reference_timestamps": reference_timestamps,
                }
            )

    # Assign scene_id / shot_id to shots & transcript segments
    for shot in shots:
        shot["scene_id"] = _find_scene_id(scenes, _safe_float(shot.get("start_time")))

    for seg in transcript_segments:
        seg_mid = (seg["start"] + seg["end"]) / 2.0
        seg["scene_id"] = _find_scene_id(scenes, seg_mid)
        seg["shot_id"] = _find_shot_id(shots, seg_mid)

    if extract_reference_frames and tracks:
        try:
            _extract_reference_frames(
                video_path=video_path,
                tracks=tracks,
                output_dir=output_dir,
                max_frames_per_character=int(max_reference_frames_per_character),
            )
        except Exception as e:
            logger.warning(f"Reference frame extraction skipped/failed: {e}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    semantic_summary = {
        "schema_version": SCHEMA_VERSION,
        "video_id": video_id,
        "film": film,
        "video_path": str(video_path),
        "sources": {
            "scene_detection_json": str(scene_detection_path) if scene_detection_path else None,
            "camera_tracking_json": str(camera_tracking_path) if camera_tracking_path else None,
            "transcript_json": str(transcript_path) if transcript_path else None,
            "segmentation_json": str(segmentation_path) if segmentation_path else None,
        },
        "stats": {
            "scenes": len(scenes),
            "shots": len(shots),
            "dialogue_segments": len(transcript_segments),
            "character_tracks": len(tracks),
            "camera_style": camera_style,
            "built_at": int(time.time()),
        },
        "scenes": scenes,
        "shots": shots,
        "dialogue": transcript_segments,
        "character_tracks": tracks,
    }

    with open(output_dir / "video_semantic.json", "w", encoding="utf-8") as f:
        json.dump(semantic_summary, f, ensure_ascii=False, indent=2)

    if not emit_documents:
        return {"output_dir": str(output_dir), "video_semantic": str(output_dir / "video_semantic.json")}

    docs_dir = output_dir / "documents"
    docs_dir.mkdir(parents=True, exist_ok=True)

    scene_docs: List[Dict[str, Any]] = []
    for scene in scenes:
        scene_id = _safe_int(scene.get("scene_id"), 0)
        start_time = _safe_float(scene.get("start_time"), 0.0)
        end_time = _safe_float(scene.get("end_time"), 0.0)
        keyframe_path = scene.get("keyframe_path")

        content = (
            f"[VideoScene] film={film or 'unknown'} video={video_id} scene={scene_id} "
            f"t={start_time:.2f}-{end_time:.2f}s keyframe={keyframe_path}"
        )
        scene_docs.append(
            {
                "doc_kind": "video_scene",
                "film": film,
                "video_id": video_id,
                "scene_id": scene_id,
                "start_time": start_time,
                "end_time": end_time,
                "duration": _safe_float(scene.get("duration"), max(0.0, end_time - start_time)),
                "keyframe_path": keyframe_path,
                "camera_style": camera_style,
                "content": content,
            }
        )

    shot_docs: List[Dict[str, Any]] = []
    for shot in shots:
        shot_id = _safe_int(shot.get("shot_id"), 0)
        scene_id = shot.get("scene_id")
        start_time = _safe_float(shot.get("start_time"), 0.0)
        end_time = _safe_float(shot.get("end_time"), 0.0)
        dominant_movement = str(shot.get("dominant_movement", "unknown"))
        handheld = bool(shot.get("is_handheld", False))
        smoothness = _safe_float(shot.get("smoothness_score"), 0.0)

        content = (
            f"[VideoShot] film={film or 'unknown'} video={video_id} scene={scene_id} shot={shot_id} "
            f"t={start_time:.2f}-{end_time:.2f}s movement={dominant_movement} handheld={handheld} smooth={smoothness:.2f}"
        )
        shot_docs.append(
            {
                "doc_kind": "video_shot",
                "film": film,
                "video_id": video_id,
                "scene_id": scene_id,
                "shot_id": shot_id,
                "start_time": start_time,
                "end_time": end_time,
                "duration": _safe_float(shot.get("duration"), max(0.0, end_time - start_time)),
                "camera_style": camera_style,
                "dominant_movement": dominant_movement,
                "is_handheld": handheld,
                "smoothness_score": smoothness,
                "content": content,
            }
        )

    dialogue_docs: List[Dict[str, Any]] = []
    for i, seg in enumerate(transcript_segments):
        start_t = _safe_float(seg.get("start"), 0.0)
        end_t = _safe_float(seg.get("end"), 0.0)
        character = seg.get("character") or None
        speaker = seg.get("speaker") or None
        text = str(seg.get("text", "")).strip()
        emotion = seg.get("emotion") or None
        scene_id = seg.get("scene_id")
        shot_id = seg.get("shot_id")

        actor = character or speaker or "unknown"
        content = (
            f"[Dialogue] film={film or 'unknown'} video={video_id} scene={scene_id} shot={shot_id} "
            f"t={start_t:.2f}-{end_t:.2f}s speaker={actor} emotion={emotion or 'unknown'} text={text}"
        )
        dialogue_docs.append(
            {
                "doc_kind": "dialogue",
                "film": film,
                "video_id": video_id,
                "scene_id": scene_id,
                "shot_id": shot_id,
                "line_index": i,
                "character": character,
                "speaker": speaker,
                "emotion": emotion,
                "start_time": start_t,
                "end_time": end_t,
                "text": text,
                "content": content,
            }
        )

    character_docs: List[Dict[str, Any]] = []
    for tr in tracks:
        char_name = tr.get("character_name")
        char_id = tr.get("character_id")
        ref_paths = tr.get("reference_frame_paths") or []
        content = (
            f"[CharacterTrack] film={film or 'unknown'} video={video_id} character={char_name or char_id} "
            f"t={tr.get('start_time', 0.0):.2f}-{tr.get('end_time', 0.0):.2f}s refs={len(ref_paths)}"
        )
        character_docs.append(
            {
                "doc_kind": "character_track",
                "film": film,
                "video_id": video_id,
                "character_id": char_id,
                "character": char_name,
                "start_time": tr.get("start_time", 0.0),
                "end_time": tr.get("end_time", 0.0),
                "avg_confidence": tr.get("avg_confidence", 0.0),
                "total_segments": tr.get("total_segments", 0),
                "reference_timestamps": tr.get("reference_timestamps") or [],
                "reference_frame_paths": ref_paths,
                "content": content,
            }
        )

    (docs_dir / "scenes.json").write_text(json.dumps(scene_docs, ensure_ascii=False, indent=2), encoding="utf-8")
    (docs_dir / "shots.json").write_text(json.dumps(shot_docs, ensure_ascii=False, indent=2), encoding="utf-8")
    (docs_dir / "dialogue.json").write_text(json.dumps(dialogue_docs, ensure_ascii=False, indent=2), encoding="utf-8")
    (docs_dir / "characters.json").write_text(json.dumps(character_docs, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "output_dir": str(output_dir),
        "video_semantic": str(output_dir / "video_semantic.json"),
        "documents_dir": str(docs_dir),
        "documents": {
            "scenes": str(docs_dir / "scenes.json"),
            "shots": str(docs_dir / "shots.json"),
            "dialogue": str(docs_dir / "dialogue.json"),
            "characters": str(docs_dir / "characters.json"),
        },
    }


def _default_output_dir(video_path: Path, film: Optional[str]) -> Path:
    if film:
        return Path("data/films") / _normalize_tag(film) / "rag" / "video_index" / video_path.stem
    return Path("outputs/rag/video_index") / video_path.stem


def _extract_reference_frames(
    *,
    video_path: Path,
    tracks: List[Dict[str, Any]],
    output_dir: Path,
    max_frames_per_character: int,
) -> None:
    try:
        import cv2  # type: ignore
    except Exception as e:
        raise RuntimeError("OpenCV (cv2) not available") from e

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    max_frames = max(1, int(max_frames_per_character))
    frames_root = Path(output_dir) / "reference_frames"
    frames_root.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    try:
        for tr in tracks:
            ref_segments = tr.get("reference_segments") or []
            if not isinstance(ref_segments, list) or not ref_segments:
                tr["reference_frame_paths"] = []
                continue

            character_slug = tr.get("character_name") or f"id_{tr.get('character_id')}"
            character_slug = str(character_slug).replace("/", "_").replace("\\", "_")
            char_dir = frames_root / character_slug
            char_dir.mkdir(parents=True, exist_ok=True)

            saved: List[str] = []
            for seg in ref_segments[:max_frames]:
                if not isinstance(seg, dict):
                    continue
                ts = _safe_float(seg.get("timestamp"), 0.0)
                frame_idx = _safe_int(seg.get("frame_index"), None)

                if frame_idx is not None and frame_idx > 0:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
                else:
                    cap.set(cv2.CAP_PROP_POS_MSEC, float(ts) * 1000.0)

                ok, frame = cap.read()
                if not ok:
                    continue

                filename = f"{character_slug}_{ts:.2f}s.jpg"
                out_path = char_dir / filename
                if cv2.imwrite(str(out_path), frame):
                    saved.append(str(out_path))

            tr["reference_frame_paths"] = saved
    finally:
        cap.release()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build RAG-friendly semantic index for a video")
    parser.add_argument("--video", required=True, help="Path to source video (used for IDs/metadata)")
    parser.add_argument("--film", help="Film name (e.g., luca) for metadata/path conventions")
    parser.add_argument("--analysis-dir", help="Directory containing analysis JSONs (outputs/analysis/<video>)")
    parser.add_argument("--scene-json", help="Scene detection JSON path (overrides --analysis-dir)")
    parser.add_argument("--camera-json", help="Camera tracking JSON path (overrides --analysis-dir)")
    parser.add_argument("--transcript-json", help="Transcript/diarization JSON path (optional)")
    parser.add_argument("--segmentation-json", help="Character segmentation JSON path (optional)")
    parser.add_argument("--output-dir", help="Output directory (default: data/films/<film>/rag/video_index/<video_id>)")
    parser.add_argument("--no-documents", action="store_true", help="Only write video_semantic.json (skip documents)")
    parser.add_argument(
        "--extract-reference-frames",
        action="store_true",
        help="Extract per-character reference frames from the source video (requires --segmentation-json)",
    )
    parser.add_argument(
        "--max-reference-frames-per-character",
        type=int,
        default=10,
        help="Max reference frames per character (default: 10)",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    video_path = Path(args.video)
    film = _normalize_tag(args.film)

    analysis_dir = Path(args.analysis_dir) if args.analysis_dir else Path("outputs/analysis") / video_path.stem
    scene_json = Path(args.scene_json) if args.scene_json else (analysis_dir / "scene_detection.json")
    camera_json = Path(args.camera_json) if args.camera_json else (analysis_dir / "camera_tracking.json")

    output_dir = Path(args.output_dir) if args.output_dir else _default_output_dir(video_path, film)

    result = build_video_semantics(
        video_path=video_path,
        film=film,
        scene_detection_path=scene_json if scene_json.exists() else None,
        camera_tracking_path=camera_json if camera_json.exists() else None,
        transcript_path=Path(args.transcript_json) if args.transcript_json else None,
        segmentation_path=Path(args.segmentation_json) if args.segmentation_json else None,
        output_dir=output_dir,
        emit_documents=not args.no_documents,
        extract_reference_frames=bool(args.extract_reference_frames),
        max_reference_frames_per_character=int(args.max_reference_frames_per_character),
    )

    print(json.dumps({"success": True, **result}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

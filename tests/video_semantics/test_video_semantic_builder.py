import json
import sys
from pathlib import Path


def test_build_video_semantics_emits_documents(tmp_path: Path) -> None:
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))

    from scripts.processing.video_semantics.video_semantic_builder import build_video_semantics

    video_path = tmp_path / "demo_video.mp4"
    video_path.write_bytes(b"")  # Builder doesn't decode unless reference extraction is enabled.

    scene_json = tmp_path / "scene_detection.json"
    scene_json.write_text(
        json.dumps(
            {
                "video_path": str(video_path),
                "scenes": [
                    {"scene_id": 1, "start_time": 0.0, "end_time": 10.0, "duration": 10.0, "keyframe_path": "kf1.jpg"},
                    {"scene_id": 2, "start_time": 10.0, "end_time": 20.0, "duration": 10.0, "keyframe_path": "kf2.jpg"},
                ],
            }
        ),
        encoding="utf-8",
    )

    camera_json = tmp_path / "camera_tracking.json"
    camera_json.write_text(
        json.dumps(
            {
                "video_path": str(video_path),
                "fps": 10.0,
                "camera_style": "Smooth",
                "shots": [
                    {"shot_id": 1, "start_frame": 0, "end_frame": 100, "duration": 10.0, "dominant_movement": "Pan"},
                    {"shot_id": 2, "start_frame": 100, "end_frame": 200, "duration": 10.0, "dominant_movement": "Static"},
                ],
            }
        ),
        encoding="utf-8",
    )

    transcript_json = tmp_path / "transcript.json"
    transcript_json.write_text(
        json.dumps(
            [
                {"start": 1.0, "end": 2.5, "text": "Hello there!", "speaker": "SPEAKER_00"},
                {"start": 12.0, "end": 13.0, "text": "Over here!", "speaker": "SPEAKER_01", "character": "Luca"},
            ]
        ),
        encoding="utf-8",
    )

    segmentation_json = tmp_path / "segmentation.json"
    segmentation_json.write_text(
        json.dumps(
            {
                "character_tracks": [
                    {
                        "character_id": 1,
                        "character_name": "Luca",
                        "total_segments": 3,
                        "avg_confidence": 0.9,
                        "segments": [
                            {"frame_index": 10, "timestamp": 1.0, "confidence": 0.95, "bounding_box": [0, 0, 10, 10]},
                            {"frame_index": 50, "timestamp": 5.0, "confidence": 0.9, "bounding_box": [1, 1, 11, 11]},
                            {"frame_index": 150, "timestamp": 15.0, "confidence": 0.85, "bounding_box": [2, 2, 12, 12]},
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    output_dir = tmp_path / "out"
    result = build_video_semantics(
        video_path=video_path,
        film="Luca",
        scene_detection_path=scene_json,
        camera_tracking_path=camera_json,
        transcript_path=transcript_json,
        segmentation_path=segmentation_json,
        output_dir=output_dir,
        emit_documents=True,
        extract_reference_frames=False,
    )

    assert Path(result["video_semantic"]).exists()
    docs_dir = Path(result["documents_dir"])
    assert (docs_dir / "scenes.json").exists()
    assert (docs_dir / "shots.json").exists()
    assert (docs_dir / "dialogue.json").exists()
    assert (docs_dir / "characters.json").exists()

    dialogue_docs = json.loads((docs_dir / "dialogue.json").read_text(encoding="utf-8"))
    assert len(dialogue_docs) == 2
    assert dialogue_docs[0]["doc_kind"] == "dialogue"
    assert dialogue_docs[0]["film"] == "luca"
    assert dialogue_docs[0]["scene_id"] == 1
    assert dialogue_docs[0]["shot_id"] == 1

    assert dialogue_docs[1]["scene_id"] == 2
    assert dialogue_docs[1]["shot_id"] == 2
    assert dialogue_docs[1]["character"] == "luca"

    shot_docs = json.loads((docs_dir / "shots.json").read_text(encoding="utf-8"))
    assert shot_docs[0]["doc_kind"] == "video_shot"
    assert shot_docs[0]["film"] == "luca"
    assert shot_docs[0]["camera_style"] == "smooth"

    character_docs = json.loads((docs_dir / "characters.json").read_text(encoding="utf-8"))
    assert character_docs[0]["doc_kind"] == "character_track"
    assert character_docs[0]["character"] == "luca"
    assert isinstance(character_docs[0]["reference_timestamps"], list)

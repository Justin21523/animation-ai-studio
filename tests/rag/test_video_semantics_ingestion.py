import sys
from pathlib import Path


def test_item_to_document_is_deterministic(tmp_path: Path) -> None:
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))

    from scripts.rag.ingest_video_semantics import _item_to_document

    item = {
        "doc_kind": "dialogue",
        "film": "luca",
        "video_id": "clip_001",
        "line_index": 7,
        "start_time": 1.0,
        "end_time": 2.0,
        "character": "luca",
        "content": "[Dialogue] film=luca video=clip_001 scene=1 shot=1 t=1.00-2.00s speaker=luca emotion=unknown text=Hello!",
        "text": "Hello!",
    }

    source = tmp_path / "dialogue.json"
    source.write_text("[]", encoding="utf-8")

    doc = _item_to_document(
        item=item,
        source_path=source,
        film="luca",
        video_id="clip_001",
        fallback_index=0,
    )

    assert doc.doc_id == "luca:clip_001:dialogue:line_7"
    assert doc.metadata["doc_kind"] == "dialogue"
    assert doc.metadata["film"] == "luca"
    assert doc.metadata["video_id"] == "clip_001"
    assert doc.content.startswith("[Dialogue]")

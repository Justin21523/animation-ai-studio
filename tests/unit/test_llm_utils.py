import sys
from pathlib import Path


# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def test_extract_text_from_chat_response_variants() -> None:
    from scripts.core.llm_client.utils import extract_text_from_chat_response

    assert extract_text_from_chat_response({"content": "hello"}) == "hello"

    assert (
        extract_text_from_chat_response(
            {"choices": [{"message": {"content": "hi"}}]}
        )
        == "hi"
    )

    assert extract_text_from_chat_response({"choices": [{"text": "ok"}]}) == "ok"

    assert (
        extract_text_from_chat_response(
            {"choices": [{"delta": {"content": "stream"}}]}
        )
        == "stream"
    )


import pytest
from transformers import AutoProcessor


PROCESSOR_CASES = [
    (
        "Qwen/Qwen2.5-Omni-3B",
        [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ],
        None,
    ),
    (
        "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ],
        None,
    ),
    (
        "google/gemma-4-E2B-it",
        [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ],
        None,
    ),
]


@pytest.mark.parametrize(("model_id", "messages", "fallback_text"), PROCESSOR_CASES)
def test_real_processor_builds_input_ids_without_model_weights(
    model_id: str,
    messages: list[dict[str, str]] | None,
    fallback_text: str | None,
) -> None:
    processor = AutoProcessor.from_pretrained(
        model_id,
        local_files_only=True,
        trust_remote_code=True,
    )

    if messages is not None:
        rendered = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    else:
        rendered = fallback_text

    assert isinstance(rendered, str)
    assert rendered

    batch = processor(text=[rendered], return_tensors="pt")

    assert "input_ids" in batch
    assert batch["input_ids"].ndim == 2
    assert batch["input_ids"].shape[0] == 1
    assert batch["input_ids"].shape[1] > 0

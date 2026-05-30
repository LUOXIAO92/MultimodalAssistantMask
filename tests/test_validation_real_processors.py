import pytest
from transformers import AutoProcessor

from mm_assistant_mask import (
    AssistantFrameSpec,
    assert_end_in_labels,
    assert_no_header_in_labels,
    build_assistant_frame_masks,
    build_labels_from_frame_mask,
    decode_supervised_tokens,
    frame_token_ids,
)


def _qwen_labels():
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2.5-Omni-3B",
        local_files_only=True,
        trust_remote_code=True,
    )
    spec = AssistantFrameSpec(
        assistant_header="<|im_start|>assistant\n",
        assistant_end="<|im_end|>",
    )
    rendered = processor.apply_chat_template(
        [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ],
        tokenize=False,
        add_generation_prompt=False,
    )
    batch = processor(text=[rendered], return_tensors="pt")
    masks = build_assistant_frame_masks(batch["input_ids"], processor.tokenizer, spec)
    labels = build_labels_from_frame_mask(batch["input_ids"], masks, batch.get("attention_mask"))
    return processor, spec, batch["input_ids"], labels


def test_decode_supervised_tokens_and_assertions_use_real_processor() -> None:
    processor, spec, input_ids, labels = _qwen_labels()

    decoded = decode_supervised_tokens(processor.tokenizer, input_ids[0], labels[0])

    assert "hi" in decoded
    assert_no_header_in_labels(decoded, spec)
    assert_end_in_labels(decoded, spec)


def test_label_assertions_fail_with_clear_errors() -> None:
    spec = AssistantFrameSpec(
        assistant_header="<|im_start|>assistant\n",
        assistant_end="<|im_end|>",
    )

    with pytest.raises(AssertionError, match="assistant header"):
        assert_no_header_in_labels(f"{spec.assistant_header}hi{spec.assistant_end}", spec)

    with pytest.raises(AssertionError, match="assistant end"):
        assert_end_in_labels("hi", spec)


def test_frame_token_ids_use_real_tokenizer() -> None:
    processor, spec, _, _ = _qwen_labels()

    ids = frame_token_ids(processor.tokenizer, spec)

    assert ids.header_ids
    assert ids.end_ids
    assert ids.stop_ids == ids.end_ids


def test_decode_supervised_tokens_rejects_shape_mismatch() -> None:
    processor, _, input_ids, labels = _qwen_labels()

    with pytest.raises(ValueError, match="input_ids shape"):
        decode_supervised_tokens(processor.tokenizer, input_ids, labels[0])

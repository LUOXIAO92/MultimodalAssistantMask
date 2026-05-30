import numpy as np
import pytest
from PIL import Image
from transformers import AutoProcessor

from mm_assistant_mask import AssistantFrameSpec, build_assistant_frame_mask_for_one


@pytest.fixture(scope="module")
def qwen25_omni_processor():
    return AutoProcessor.from_pretrained(
        "Qwen/Qwen2.5-Omni-3B",
        local_files_only=True,
        trust_remote_code=True,
    )


def _qwen_spec() -> AssistantFrameSpec:
    return AssistantFrameSpec(
        assistant_header="<|im_start|>assistant\n",
        assistant_end="<|im_end|>",
    )


def test_image_text_mixed_user_content_uses_final_processor_input_ids(
    qwen25_omni_processor,
) -> None:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "describe"},
            ],
        },
        {"role": "assistant", "content": "ok"},
    ]
    rendered = qwen25_omni_processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    image = Image.new("RGB", (32, 32), (255, 0, 0))
    batch = qwen25_omni_processor(text=[rendered], images=[image], return_tensors="pt")

    mask = build_assistant_frame_mask_for_one(
        batch["input_ids"][0],
        qwen25_omni_processor.tokenizer,
        _qwen_spec(),
    )
    supervised = qwen25_omni_processor.tokenizer.decode(
        batch["input_ids"][0][mask],
        skip_special_tokens=False,
    )

    assert "pixel_values" in batch
    assert "ok" in supervised
    assert "<|im_start|>assistant\n" not in supervised
    assert "<|im_end|>" in supervised


def test_audio_text_mixed_user_content_uses_final_processor_input_ids(
    qwen25_omni_processor,
) -> None:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "audio"},
                {"type": "text", "text": "transcribe"},
            ],
        },
        {"role": "assistant", "content": "ok"},
    ]
    rendered = qwen25_omni_processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    audio = np.zeros(16000, dtype=np.float32)
    batch = qwen25_omni_processor(text=[rendered], audio=[audio], return_tensors="pt")

    mask = build_assistant_frame_mask_for_one(
        batch["input_ids"][0],
        qwen25_omni_processor.tokenizer,
        _qwen_spec(),
    )
    supervised = qwen25_omni_processor.tokenizer.decode(
        batch["input_ids"][0][mask],
        skip_special_tokens=False,
    )

    assert "input_features" in batch
    assert "ok" in supervised
    assert "<|im_start|>assistant\n" not in supervised
    assert "<|im_end|>" in supervised


def test_processor_assistant_masks_are_diagnostic_not_training_path(
    qwen25_omni_processor,
) -> None:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "describe"},
            ],
        },
        {"role": "assistant", "content": "ok"},
    ]
    diagnostic = qwen25_omni_processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_assistant_tokens_mask=True,
        add_generation_prompt=False,
    )

    rendered = qwen25_omni_processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    image = Image.new("RGB", (32, 32), (255, 0, 0))
    batch = qwen25_omni_processor(text=[rendered], images=[image], return_tensors="pt")
    mask = build_assistant_frame_mask_for_one(
        batch["input_ids"][0],
        qwen25_omni_processor.tokenizer,
        _qwen_spec(),
    )

    assert "assistant_masks" in diagnostic
    assert sum(diagnostic["assistant_masks"][0]) == 0
    assert len(diagnostic["input_ids"][0]) != batch["input_ids"].shape[1]
    assert int(mask.sum()) > 0

import pytest
import torch
from transformers import AutoProcessor

from mm_assistant_mask import (
    AssistantFrameSpec,
    AssistantMaskSpec,
    build_assistant_frame_mask_for_one,
    build_assistant_frame_masks,
    build_assistant_labels,
    build_assistant_mask,
    build_labels_from_frame_mask,
)


CASES = [
    (
        "Qwen/Qwen2.5-Omni-3B",
        AssistantFrameSpec(
            assistant_header="<|im_start|>assistant\n",
            assistant_end="<|im_end|>",
        ),
        [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ],
        None,
    ),
    (
        "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        AssistantFrameSpec(
            assistant_header="<|im_start|>assistant\n",
            assistant_end="<|im_end|>",
        ),
        [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ],
        None,
    ),
    (
        "google/gemma-4-E2B-it",
        AssistantFrameSpec(
            assistant_header="<|turn>model\n",
            assistant_end="<turn|>",
        ),
        [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ],
        None,
    ),
]


@pytest.mark.parametrize(("model_id", "spec", "messages", "fallback_text"), CASES)
def test_frame_mask_uses_real_processor_input_ids(
    model_id: str,
    spec: AssistantFrameSpec,
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

    batch = processor(text=[rendered], return_tensors="pt")
    input_ids = batch["input_ids"]
    mask = build_assistant_frame_mask_for_one(
        input_ids=input_ids[0].tolist(),
        tokenizer=processor.tokenizer,
        spec=spec,
    )

    supervised = processor.tokenizer.decode(
        input_ids[0][mask],
        skip_special_tokens=False,
    )

    assert spec.assistant_header not in supervised
    assert "hi" in supervised
    assert spec.assistant_end in supervised
    assert not supervised.endswith("\n")


@pytest.mark.parametrize(("model_id", "spec", "messages", "fallback_text"), CASES)
def test_labels_keep_only_frame_masked_tokens(
    model_id: str,
    spec: AssistantFrameSpec,
    messages: list[dict[str, str]] | None,
    fallback_text: str | None,
) -> None:
    processor = AutoProcessor.from_pretrained(
        model_id,
        local_files_only=True,
        trust_remote_code=True,
    )

    rendered = (
        processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        if messages is not None
        else fallback_text
    )
    batch = processor(text=[rendered], return_tensors="pt")
    masks = build_assistant_frame_masks(
        input_ids=batch["input_ids"],
        tokenizer=processor.tokenizer,
        spec=spec,
    )
    labels = build_labels_from_frame_mask(
        input_ids=batch["input_ids"],
        frame_mask=masks,
        attention_mask=batch.get("attention_mask"),
    )

    assert labels.shape == batch["input_ids"].shape
    assert (labels[masks] == batch["input_ids"][masks]).all()
    assert (labels[~masks] == -100).all()


def test_public_assistant_labels_api_matches_lower_level_path() -> None:
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2.5-Omni-3B",
        local_files_only=True,
        trust_remote_code=True,
    )
    spec = AssistantMaskSpec(
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

    assistant_mask = build_assistant_mask(batch, processor.tokenizer, spec)
    public_labels = build_assistant_labels(batch, processor.tokenizer, spec)
    lower_level_labels = build_labels_from_frame_mask(
        input_ids=batch["input_ids"],
        frame_mask=assistant_mask,
        attention_mask=batch.get("attention_mask"),
    )

    assert assistant_mask.dtype == torch.bool
    assert public_labels.equal(lower_level_labels)


def test_public_assistant_labels_ignore_padding_internally() -> None:
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2.5-Omni-3B",
        local_files_only=True,
        trust_remote_code=True,
    )
    spec = AssistantMaskSpec(
        assistant_header="<|im_start|>assistant\n",
        assistant_end="<|im_end|>",
    )
    rendered = [
        processor.apply_chat_template(
            [
                {"role": "user", "content": "short"},
                {"role": "assistant", "content": "ok"},
            ],
            tokenize=False,
            add_generation_prompt=False,
        ),
        processor.apply_chat_template(
            [
                {"role": "user", "content": "longer prompt " * 8},
                {"role": "assistant", "content": "longer answer"},
            ],
            tokenize=False,
            add_generation_prompt=False,
        ),
    ]
    batch = processor(text=rendered, return_tensors="pt", padding=True)

    labels = build_assistant_labels(batch, processor.tokenizer, spec)
    padding = batch["attention_mask"] == 0

    assert padding.any()
    assert (labels[padding] == -100).all()


def test_frame_mask_can_exclude_assistant_prefix_from_loss() -> None:
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        local_files_only=True,
        trust_remote_code=True,
    )
    rendered = (
        "<|im_start|>user\nhello<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\nhi<|im_end|>\n"
    )
    batch = processor(text=[rendered], return_tensors="pt")
    spec = AssistantFrameSpec(
        assistant_header="<|im_start|>assistant\n",
        assistant_end="<|im_end|>",
        excluded_assistant_prefixes=("<think>\n\n</think>\n\n",),
    )

    mask = build_assistant_frame_mask_for_one(
        input_ids=batch["input_ids"][0],
        tokenizer=processor.tokenizer,
        spec=spec,
    )
    supervised = processor.tokenizer.decode(
        batch["input_ids"][0][mask],
        skip_special_tokens=False,
    )

    assert supervised == "hi<|im_end|>"


def test_frame_mask_include_flags_use_real_processor_input_ids() -> None:
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2.5-Omni-3B",
        local_files_only=True,
        trust_remote_code=True,
    )
    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]
    rendered = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    input_ids = processor(text=[rendered], return_tensors="pt")["input_ids"][0]

    include_header = AssistantFrameSpec(
        assistant_header="<|im_start|>assistant\n",
        assistant_end="<|im_end|>",
        include_header_in_loss=True,
    )
    no_end = AssistantFrameSpec(
        assistant_header="<|im_start|>assistant\n",
        assistant_end="<|im_end|>",
        include_end_in_loss=False,
    )
    include_separator = AssistantFrameSpec(
        assistant_header="<|im_start|>assistant\n",
        assistant_end="<|im_end|>",
        include_post_end_separator_in_loss=True,
    )

    header_mask = build_assistant_frame_mask_for_one(input_ids, processor.tokenizer, include_header)
    no_end_mask = build_assistant_frame_mask_for_one(input_ids, processor.tokenizer, no_end)
    separator_mask = build_assistant_frame_mask_for_one(
        input_ids,
        processor.tokenizer,
        include_separator,
    )

    header_supervised = processor.tokenizer.decode(input_ids[header_mask], skip_special_tokens=False)
    no_end_supervised = processor.tokenizer.decode(input_ids[no_end_mask], skip_special_tokens=False)
    separator_supervised = processor.tokenizer.decode(
        input_ids[separator_mask],
        skip_special_tokens=False,
    )

    assert include_header.assistant_header in header_supervised
    assert no_end.assistant_end not in no_end_supervised
    assert separator_supervised.endswith("\n")


def test_frame_mask_errors_use_real_processor_input_ids() -> None:
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2.5-Omni-3B",
        local_files_only=True,
        trust_remote_code=True,
    )
    spec = AssistantFrameSpec(
        assistant_header="<|im_start|>assistant\n",
        assistant_end="<|im_end|>",
    )

    no_frame_ids = processor(text=["hello"], return_tensors="pt")["input_ids"][0]
    with pytest.raises(ValueError, match="no assistant frame found"):
        build_assistant_frame_mask_for_one(no_frame_ids, processor.tokenizer, spec)

    header_ids = processor.tokenizer(spec.assistant_header, add_special_tokens=False).input_ids
    with pytest.raises(ValueError, match="no following assistant end"):
        build_assistant_frame_mask_for_one(header_ids, processor.tokenizer, spec)


def test_multiturn_repeated_payload_uses_frame_matching() -> None:
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2.5-Omni-3B",
        local_files_only=True,
        trust_remote_code=True,
    )
    spec = AssistantFrameSpec(
        assistant_header="<|im_start|>assistant\n",
        assistant_end="<|im_end|>",
    )
    messages = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "same"},
        {"role": "user", "content": "second"},
        {"role": "assistant", "content": "same"},
    ]
    rendered = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    input_ids = processor(text=[rendered], return_tensors="pt")["input_ids"][0]
    mask = build_assistant_frame_mask_for_one(input_ids, processor.tokenizer, spec)
    supervised = processor.tokenizer.decode(input_ids[mask], skip_special_tokens=False)

    assert supervised.count("same") == 2
    assert supervised.count(spec.assistant_end) == 2
    assert spec.assistant_header not in supervised


@pytest.mark.parametrize(
    ("model_id", "spec"),
    [
        (
            "Qwen/Qwen3-Omni-30B-A3B-Instruct",
            AssistantFrameSpec(
                assistant_header="<|im_start|>assistant\n",
                assistant_end="<|im_end|>",
            ),
        ),
        (
            "google/gemma-4-E2B-it",
            AssistantFrameSpec(
                assistant_header="<|turn>model\n",
                assistant_end="<turn|>",
            ),
        ),
    ],
)
def test_multiturn_text_only_for_remaining_real_processors(
    model_id: str,
    spec: AssistantFrameSpec,
) -> None:
    processor = AutoProcessor.from_pretrained(
        model_id,
        local_files_only=True,
        trust_remote_code=True,
    )
    rendered = processor.apply_chat_template(
        [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "alpha"},
            {"role": "user", "content": "second"},
            {"role": "assistant", "content": "beta"},
        ],
        tokenize=False,
        add_generation_prompt=False,
    )
    input_ids = processor(text=[rendered], return_tensors="pt")["input_ids"][0]
    mask = build_assistant_frame_mask_for_one(input_ids, processor.tokenizer, spec)
    supervised = processor.tokenizer.decode(input_ids[mask], skip_special_tokens=False)

    assert "alpha" in supervised
    assert "beta" in supervised
    assert supervised.count(spec.assistant_end) == 2
    assert spec.assistant_header not in supervised


def test_labels_reject_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="frame_mask shape"):
        build_labels_from_frame_mask(
            input_ids=torch.tensor([[1, 2, 3]]),
            frame_mask=torch.tensor([[True, False]]),
        )

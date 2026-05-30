import json
from pathlib import Path

import pytest
from transformers import AutoProcessor


ROOT = Path(__file__).resolve().parents[1]


def _read_text(path: str) -> str:
    return (ROOT / path).read_text()


def _read_json_template(path: str) -> str:
    return json.loads(_read_text(path))["chat_template"]


TEMPLATE_CASES = [
    pytest.param(
        "Qwen/Qwen2.5-Omni-3B",
        _read_json_template("templates/qwen/qwen2_5-omni-3b.json"),
        _read_text("templates/qwen/qwen2_5-omni-3b-train.jinja"),
        "<|im_start|>assistant\n",
        "<|im_end|>",
        "hi<|im_end|>\n",
        id="qwen2_5_omni_3b",
    ),
    pytest.param(
        "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        _read_json_template("templates/qwen/qwen3-omni-30b-a3b-instruct.json"),
        _read_text("templates/qwen/qwen3-omni-30b-a3b-instruct-train.jinja"),
        "<|im_start|>assistant\n",
        "<|im_end|>",
        "hi<|im_end|>\n",
        id="qwen3_omni_30b_a3b_instruct",
    ),
    pytest.param(
        "google/gemma-4-E2B-it",
        _read_text("templates/gemma/gemma-4-e2b-it.jinja"),
        _read_text("templates/gemma/gemma-4-e2b-it-train.jinja"),
        "<|turn>model\n",
        "<turn|>",
        "hi<turn|>\n",
        id="gemma_4_e2b_it",
    ),
]

TOOL_TEMPLATE_CASES = [
    pytest.param(
        "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        _read_json_template("templates/qwen/qwen3-omni-30b-a3b-instruct.json"),
        _read_text("templates/qwen/qwen3-omni-30b-a3b-instruct-train.jinja"),
        "<|im_start|>assistant\n",
        "<tool_call>",
        id="qwen3_omni_tool_call",
    ),
    pytest.param(
        "google/gemma-4-E2B-it",
        _read_text("templates/gemma/gemma-4-e2b-it.jinja"),
        _read_text("templates/gemma/gemma-4-e2b-it-train.jinja"),
        "<|turn>model\n",
        "<|tool_call>",
        id="gemma_4_e2b_tool_call",
    ),
]


@pytest.mark.parametrize(
    ("model_id", "source_template", "train_template", "assistant_header", "assistant_end", "expected_mask_text"),
    TEMPLATE_CASES,
)
def test_train_templates_match_text_only_rendering(
    model_id: str,
    source_template: str,
    train_template: str,
    assistant_header: str,
    assistant_end: str,
    expected_mask_text: str,
) -> None:
    processor = AutoProcessor.from_pretrained(
        model_id,
        local_files_only=True,
        trust_remote_code=True,
    )
    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]

    source_rendered = processor.tokenizer.apply_chat_template(
        messages,
        chat_template=source_template,
        tokenize=False,
        add_generation_prompt=False,
    )
    train_rendered = processor.tokenizer.apply_chat_template(
        messages,
        chat_template=train_template,
        tokenize=False,
        add_generation_prompt=False,
    )

    assert train_rendered == source_rendered
    assert assistant_header in train_rendered
    assert assistant_end in train_rendered


@pytest.mark.parametrize(
    ("model_id", "source_template", "train_template", "assistant_header", "assistant_end", "expected_mask_text"),
    TEMPLATE_CASES,
)
def test_train_templates_return_text_only_assistant_masks(
    model_id: str,
    source_template: str,
    train_template: str,
    assistant_header: str,
    assistant_end: str,
    expected_mask_text: str,
) -> None:
    processor = AutoProcessor.from_pretrained(
        model_id,
        local_files_only=True,
        trust_remote_code=True,
    )
    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]

    encoded = processor.tokenizer.apply_chat_template(
        messages,
        chat_template=train_template,
        tokenize=True,
        return_dict=True,
        return_assistant_tokens_mask=True,
        add_generation_prompt=False,
    )
    supervised = processor.tokenizer.decode(
        [
            token_id
            for token_id, mask_value in zip(
                encoded["input_ids"],
                encoded["assistant_masks"],
            )
            if mask_value
        ],
        skip_special_tokens=False,
    )

    assert supervised == expected_mask_text
    assert assistant_header not in supervised
    assert assistant_end in supervised


@pytest.mark.parametrize(
    ("model_id", "source_template", "train_template", "assistant_header", "assistant_end", "expected_mask_text"),
    TEMPLATE_CASES,
)
def test_train_templates_generation_prompt_has_no_payload_mask(
    model_id: str,
    source_template: str,
    train_template: str,
    assistant_header: str,
    assistant_end: str,
    expected_mask_text: str,
) -> None:
    processor = AutoProcessor.from_pretrained(
        model_id,
        local_files_only=True,
        trust_remote_code=True,
    )
    messages = [{"role": "user", "content": "hello"}]

    encoded = processor.tokenizer.apply_chat_template(
        messages,
        chat_template=train_template,
        tokenize=True,
        return_dict=True,
        return_assistant_tokens_mask=True,
        add_generation_prompt=True,
    )
    rendered = processor.tokenizer.decode(encoded["input_ids"], skip_special_tokens=False)

    assert assistant_header in rendered
    assert sum(encoded["assistant_masks"]) == 0


@pytest.mark.parametrize(
    ("model_id", "source_template", "train_template", "assistant_header", "tool_call_tag"),
    TOOL_TEMPLATE_CASES,
)
def test_train_templates_preserve_tool_call_rendering_and_masks(
    model_id: str,
    source_template: str,
    train_template: str,
    assistant_header: str,
    tool_call_tag: str,
) -> None:
    processor = AutoProcessor.from_pretrained(
        model_id,
        local_files_only=True,
        trust_remote_code=True,
    )
    messages = [
        {"role": "user", "content": "weather?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "function": {
                        "name": "get_weather",
                        "arguments": {"city": "Tokyo"},
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
    ]

    source_rendered = processor.tokenizer.apply_chat_template(
        messages,
        chat_template=source_template,
        tokenize=False,
        add_generation_prompt=False,
    )
    train_rendered = processor.tokenizer.apply_chat_template(
        messages,
        chat_template=train_template,
        tokenize=False,
        add_generation_prompt=False,
    )
    encoded = processor.tokenizer.apply_chat_template(
        messages,
        chat_template=train_template,
        tokenize=True,
        return_dict=True,
        return_assistant_tokens_mask=True,
        add_generation_prompt=False,
    )
    supervised = processor.tokenizer.decode(
        [
            token_id
            for token_id, mask_value in zip(
                encoded["input_ids"],
                encoded["assistant_masks"],
            )
            if mask_value
        ],
        skip_special_tokens=False,
    )

    assert train_rendered == source_rendered
    assert tool_call_tag in train_rendered
    assert tool_call_tag in supervised
    assert assistant_header not in supervised

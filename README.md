# MultimodalAssistantMask

Assistant-only label masking for multimodal and text-only SFT/RL pipelines. The core path is:

```text
final processor input_ids
-> token-id assistant mask matching
-> assistant-only labels
```

`{% generation %}` in a chat template is useful for declaring assistant spans at the template level, especially for text-only template validation. It does not guarantee correct assistant token masks after a multimodal processor expands image, audio, or video placeholders. For multimodal training, use the processor's default chat template and build labels from the final `input_ids`; do not use processor-returned `assistant_masks` as the main label path.

The token-id assistant mask path is not limited to images, audio, or video. It can also be used for text-only training whenever you want masks from the final tokenized `input_ids`. Unlike template-level Jinja `assistant_masks`, this path does not include the separator newline after the assistant end/eos token unless `include_post_end_separator_in_loss=True` is set in `AssistantMaskSpec`.

## Inspect Rendered Frames

To choose `assistant_header` and `assistant_end` for a model, render one tiny chat with obvious marker strings and inspect the raw template output. For Qwen3 instruct templates with empty thinking blocks, check whether extra thinking cue text appears between the assistant header and payload.

```python
from transformers import AutoProcessor

model_id = "Qwen/Qwen2.5-Omni-3B"  # replace with your target model

processor = AutoProcessor.from_pretrained(
    model_id,
    local_files_only=True,
    trust_remote_code=True,
)

user_marker = "USER_PROMPT_MARKER"
assistant_marker = "ASSISTANT_PAYLOAD_MARKER"
messages = [
    {"role": "user", "content": user_marker},
    {"role": "assistant", "content": assistant_marker},
]

rendered = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=False,
)
rendered_without_assistant = processor.apply_chat_template(
    messages[:1],
    tokenize=False,
    add_generation_prompt=False,
)

assistant_start = rendered.index(assistant_marker)
assistant_end_start = assistant_start + len(assistant_marker)
before = rendered[len(rendered_without_assistant):assistant_start]
after = rendered[assistant_end_start:]

print("rendered =", repr(rendered))
print("rendered_without_assistant =", repr(rendered_without_assistant))
print("before =", repr(before))
print("after =", repr(after))
```

## Minimal Usage

```python
from mm_assistant_mask import (
    AssistantMaskSpec,
    build_assistant_labels,
)

spec = AssistantMaskSpec(
    assistant_header="<|im_start|>assistant\n",
    assistant_end="<|im_end|>",
    generation_stop="<|im_end|>",
)

rendered = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=False,
)

batch = processor(
    text=[rendered],
    images=images,
    audio=audios,
    videos=videos,
    return_tensors="pt",
    padding=True,
    truncation=True,
)

batch["labels"] = build_assistant_labels(
    batch,
    tokenizer=processor.tokenizer,
    spec=spec,
)
```

Assistant headers are not supervised by default. Assistant payload and the assistant end token are supervised by default. The separator newline after the assistant end token is not supervised by default. Padding tokens are ignored internally when the processor returns an `attention_mask`.

If truncation leaves an assistant header without a following assistant end, the mask builder raises an error. Training code should skip that sample or adjust tokenization limits; it should not silently train on a partial assistant response.

See `examples/generic_processor_collator.py` for a generic collator skeleton.

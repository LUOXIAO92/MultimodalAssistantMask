# MultimodalAssistantMask

Assistant-only label masking for multimodal SFT/RL pipelines.

The core path is:

```text
final processor input_ids
-> token-id assistant frame matching
-> assistant-only labels
```

`{% generation %}` in a chat template is useful for declaring assistant spans at
the template level. It does not guarantee correct assistant token masks after a
multimodal processor expands image, audio, or video placeholders.

For multimodal training, do not use processor-returned `assistant_masks` as the
main label path. Build labels from the final `input_ids` instead.

## Minimal Usage

```python
from mm_assistant_mask import (
    AssistantFrameSpec,
    build_assistant_frame_masks,
    build_labels_from_frame_mask,
)

spec = AssistantFrameSpec(
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

frame_masks = build_assistant_frame_masks(
    input_ids=batch["input_ids"],
    tokenizer=processor.tokenizer,
    spec=spec,
)

batch["labels"] = build_labels_from_frame_mask(
    input_ids=batch["input_ids"],
    frame_mask=frame_masks,
    attention_mask=batch.get("attention_mask"),
)
```

Assistant headers are not supervised by default. Assistant payload and the
assistant end token are supervised by default. The separator newline after the
assistant end token is not supervised by default.

If truncation leaves an assistant header without a following assistant end, the
mask builder raises an error. Training code should skip that sample or adjust
tokenization limits; it should not silently train on a partial frame.

See `examples/generic_processor_collator.py` for a generic collator skeleton.

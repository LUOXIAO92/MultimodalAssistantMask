from collections.abc import Callable
from typing import Any

from mm_assistant_mask import (
    AssistantMaskSpec,
    build_assistant_labels,
)


def build_multimodal_assistant_collator(
    processor,
    spec: AssistantMaskSpec,
    *,
    max_length: int | None = None,
) -> Callable[[list[dict[str, Any]]], dict[str, Any]]:
    def collate(features: list[dict[str, Any]]) -> dict[str, Any]:
        # Multimodal training uses the processor's own chat template. Training
        # Jinja templates are for text-only tokenizer assistant_masks checks.
        rendered_texts = [
            processor.apply_chat_template(
                feature["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
            for feature in features
        ]
        media_kwargs = {}
        for key in ("images", "audio", "videos"):
            values = [feature.get(key) for feature in features]
            if any(value is not None for value in values):
                media_kwargs[key] = values

        batch = processor(
            text=rendered_texts,
            **media_kwargs,
            return_tensors="pt",
            padding=True,
            truncation=max_length is not None,
            max_length=max_length,
        )

        batch["labels"] = build_assistant_labels(
            batch,
            tokenizer=processor.tokenizer,
            spec=spec,
        )
        return batch

    return collate

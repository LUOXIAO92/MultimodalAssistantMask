from collections.abc import Callable
from typing import Any

from mm_assistant_mask import (
    AssistantFrameSpec,
    build_assistant_frame_masks,
    build_labels_from_frame_mask,
)


def build_multimodal_assistant_collator(
    processor,
    frame_spec: AssistantFrameSpec,
    *,
    max_length: int | None = None,
    chat_template: str | None = None,
) -> Callable[[list[dict[str, Any]]], dict[str, Any]]:
    def collate(features: list[dict[str, Any]]) -> dict[str, Any]:
        rendered_texts = [
            processor.apply_chat_template(
                feature["messages"],
                tokenize=False,
                add_generation_prompt=False,
                chat_template=chat_template,
            )
            for feature in features
        ]

        batch = processor(
            text=rendered_texts,
            images=[feature.get("images") for feature in features],
            audio=[feature.get("audio") for feature in features],
            videos=[feature.get("videos") for feature in features],
            return_tensors="pt",
            padding=True,
            truncation=max_length is not None,
            max_length=max_length,
        )

        frame_masks = build_assistant_frame_masks(
            input_ids=batch["input_ids"],
            tokenizer=processor.tokenizer,
            spec=frame_spec,
        )
        batch["labels"] = build_labels_from_frame_mask(
            input_ids=batch["input_ids"],
            frame_mask=frame_masks,
            attention_mask=batch.get("attention_mask"),
        )
        return batch

    return collate

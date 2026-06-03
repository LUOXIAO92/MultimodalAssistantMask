from collections.abc import Mapping
from typing import Any

import torch

from .frame_spec import AssistantFrameSpec
from .token_frame_matcher import build_assistant_frame_masks


def build_assistant_mask(
    batch: Mapping[str, Any],
    tokenizer,
    spec: AssistantFrameSpec,
) -> torch.BoolTensor:
    """Return the assistant tokens that should be supervised."""
    return build_assistant_frame_masks(
        input_ids=batch["input_ids"],
        tokenizer=tokenizer,
        spec=spec,
    )


def build_assistant_labels(
    batch: Mapping[str, Any],
    tokenizer,
    spec: AssistantFrameSpec,
) -> torch.Tensor:
    """Build labels with non-assistant and padding tokens set to -100."""
    assistant_mask = build_assistant_mask(batch, tokenizer=tokenizer, spec=spec)
    return build_labels_from_frame_mask(
        input_ids=batch["input_ids"],
        frame_mask=assistant_mask,
        attention_mask=batch.get("attention_mask"),
    )


def build_labels_from_frame_mask(
    input_ids: torch.Tensor,
    frame_mask: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if frame_mask.shape != input_ids.shape:
        raise ValueError(
            f"frame_mask shape {tuple(frame_mask.shape)} != input_ids shape {tuple(input_ids.shape)}"
        )
    if attention_mask is not None and attention_mask.shape != input_ids.shape:
        raise ValueError(
            f"attention_mask shape {tuple(attention_mask.shape)} != input_ids shape {tuple(input_ids.shape)}"
        )

    supervised = frame_mask.bool()
    if attention_mask is not None:
        supervised = supervised & attention_mask.bool()

    labels = input_ids.clone()
    labels[~supervised] = -100
    return labels

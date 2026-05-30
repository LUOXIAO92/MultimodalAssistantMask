"""Utilities for multimodal assistant-only label masking."""

from .frame_spec import AssistantFrameSpec
from .labels import build_labels_from_frame_mask
from .stopping import StopOnTokenSequence, stop_on_assistant_end
from .token_frame_matcher import (
    build_assistant_frame_mask_for_one,
    build_assistant_frame_masks,
    find_subsequence,
)
from .validation import (
    FrameTokenIds,
    assert_end_in_labels,
    assert_no_header_in_labels,
    decode_supervised_tokens,
    frame_token_ids,
)

__all__ = [
    "AssistantFrameSpec",
    "build_assistant_frame_mask_for_one",
    "build_assistant_frame_masks",
    "build_labels_from_frame_mask",
    "StopOnTokenSequence",
    "stop_on_assistant_end",
    "decode_supervised_tokens",
    "assert_no_header_in_labels",
    "assert_end_in_labels",
    "find_subsequence",
    "frame_token_ids",
    "FrameTokenIds",
]

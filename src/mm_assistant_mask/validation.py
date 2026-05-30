from dataclasses import dataclass

import torch

from .frame_spec import AssistantFrameSpec
from .token_frame_matcher import _to_ids


@dataclass(frozen=True)
class FrameTokenIds:
    header_ids: list[int]
    end_ids: list[int]
    stop_ids: list[int]


def decode_supervised_tokens(
    tokenizer,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
) -> str:
    if input_ids.shape != labels.shape:
        raise ValueError(
            f"input_ids shape {tuple(input_ids.shape)} != labels shape {tuple(labels.shape)}"
        )

    supervised_ids = input_ids[labels != -100]
    return tokenizer.decode(supervised_ids.tolist(), skip_special_tokens=False)


def assert_no_header_in_labels(decoded: str, spec: AssistantFrameSpec) -> None:
    if spec.assistant_header in decoded:
        raise AssertionError("assistant header found in supervised labels")


def assert_end_in_labels(decoded: str, spec: AssistantFrameSpec) -> None:
    if spec.assistant_end not in decoded:
        raise AssertionError("assistant end not found in supervised labels")


def frame_token_ids(tokenizer, spec: AssistantFrameSpec) -> FrameTokenIds:
    return FrameTokenIds(
        header_ids=_to_ids(tokenizer, spec.assistant_header),
        end_ids=_to_ids(tokenizer, spec.assistant_end),
        stop_ids=_to_ids(tokenizer, spec.stop_text),
    )

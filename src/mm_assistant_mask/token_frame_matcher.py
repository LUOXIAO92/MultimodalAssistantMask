from collections.abc import Sequence

import torch

from .frame_spec import AssistantFrameSpec


def _to_ids(tokenizer, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    if isinstance(ids, torch.Tensor):
        ids = ids.view(-1).tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(token_id) for token_id in ids]


def find_subsequence(seq: Sequence[int], pat: Sequence[int], start: int = 0) -> int:
    if not pat:
        raise ValueError("pattern must not be empty")
    limit = len(seq) - len(pat)
    for index in range(start, limit + 1):
        if list(seq[index : index + len(pat)]) == list(pat):
            return index
    return -1


def build_assistant_frame_mask_for_one(
    input_ids: Sequence[int],
    tokenizer,
    spec: AssistantFrameSpec,
) -> torch.BoolTensor:
    header_ids = _to_ids(tokenizer, spec.assistant_header)
    end_ids = _to_ids(tokenizer, spec.assistant_end)
    separator_ids = (
        _to_ids(tokenizer, spec.post_end_separator)
        if spec.include_post_end_separator_in_loss
        else []
    )

    if not header_ids:
        raise ValueError("assistant_header encodes to an empty sequence")
    if not end_ids:
        raise ValueError("assistant_end encodes to an empty sequence")

    ids = [int(token_id) for token_id in input_ids]
    mask = torch.zeros(len(ids), dtype=torch.bool)
    cursor = 0
    found = 0

    while True:
        header_start = find_subsequence(ids, header_ids, cursor)
        if header_start < 0:
            break

        content_start = header_start
        if not spec.include_header_in_loss:
            content_start = header_start + len(header_ids)

        end_start = find_subsequence(ids, end_ids, header_start + len(header_ids))
        if end_start < 0:
            raise ValueError("assistant header found but no following assistant end")

        end_end = end_start + len(end_ids)
        content_end = end_end if spec.include_end_in_loss else end_start
        if content_end > content_start:
            mask[content_start:content_end] = True

        if separator_ids:
            separator_end = end_end + len(separator_ids)
            if ids[end_end:separator_end] == separator_ids:
                mask[end_end:separator_end] = True

        found += 1
        cursor = end_end

    if found == 0:
        raise ValueError("no assistant frame found in input_ids")

    return mask


def build_assistant_frame_masks(
    input_ids: torch.Tensor,
    tokenizer,
    spec: AssistantFrameSpec,
) -> torch.BoolTensor:
    if input_ids.ndim != 2:
        raise ValueError(f"input_ids must be 2D, got shape {tuple(input_ids.shape)}")

    masks = [
        build_assistant_frame_mask_for_one(row.tolist(), tokenizer, spec)
        for row in input_ids
    ]
    return torch.stack(masks, dim=0)

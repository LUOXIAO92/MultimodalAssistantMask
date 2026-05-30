from collections.abc import Sequence

import torch
from transformers import StoppingCriteria

from .frame_spec import AssistantFrameSpec
from .token_frame_matcher import _to_ids


class StopOnTokenSequence(StoppingCriteria):
    def __init__(self, stop_ids: Sequence[int]) -> None:
        if not stop_ids:
            raise ValueError("stop_ids must not be empty")
        self.stop_ids = [int(token_id) for token_id in stop_ids]

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor | None = None,
        **kwargs,
    ) -> bool:
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must be 2D, got shape {tuple(input_ids.shape)}")
        if input_ids.shape[1] < len(self.stop_ids):
            return False

        stop = torch.tensor(self.stop_ids, device=input_ids.device, dtype=input_ids.dtype)
        return bool(torch.any(torch.all(input_ids[:, -len(self.stop_ids) :] == stop, dim=1)))


def stop_on_assistant_end(tokenizer, spec: AssistantFrameSpec) -> StopOnTokenSequence:
    return StopOnTokenSequence(_to_ids(tokenizer, spec.stop_text))

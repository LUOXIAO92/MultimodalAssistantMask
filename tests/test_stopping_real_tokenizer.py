import pytest
import torch
from transformers import AutoProcessor

from mm_assistant_mask import AssistantFrameSpec, StopOnTokenSequence, stop_on_assistant_end


def test_stop_on_assistant_end_uses_real_tokenizer_without_newline() -> None:
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2.5-Omni-3B",
        local_files_only=True,
        trust_remote_code=True,
    )
    spec = AssistantFrameSpec(
        assistant_header="<|im_start|>assistant\n",
        assistant_end="<|im_end|>",
        generation_stop="<|im_end|>",
    )
    stopping = stop_on_assistant_end(processor.tokenizer, spec)
    stop_ids = processor.tokenizer(spec.assistant_end, add_special_tokens=False).input_ids
    newline_ids = processor.tokenizer("\n", add_special_tokens=False).input_ids

    assert stopping(torch.tensor([[1, 2, *stop_ids]], dtype=torch.long))
    assert not stopping(torch.tensor([[1, 2, *stop_ids, *newline_ids]], dtype=torch.long))


def test_stop_on_token_sequence_checks_any_batch_row() -> None:
    stopping = StopOnTokenSequence([7, 8])

    assert stopping(torch.tensor([[1, 2], [3, 7]], dtype=torch.long)) is False
    assert stopping(torch.tensor([[1, 2], [7, 8]], dtype=torch.long)) is True


def test_stop_on_token_sequence_rejects_empty_sequence() -> None:
    with pytest.raises(ValueError, match="stop_ids"):
        StopOnTokenSequence([])

import torch


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

from dataclasses import dataclass


@dataclass(frozen=True)
class AssistantFrameSpec:
    assistant_header: str
    assistant_end: str
    generation_stop: str | None = None
    include_header_in_loss: bool = False
    include_end_in_loss: bool = True
    include_post_end_separator_in_loss: bool = False
    post_end_separator: str = "\n"
    excluded_assistant_prefixes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.assistant_header:
            raise ValueError("assistant_header must not be empty")
        if not self.assistant_end:
            raise ValueError("assistant_end must not be empty")
        if self.include_post_end_separator_in_loss and not self.post_end_separator:
            raise ValueError("post_end_separator must not be empty when included in loss")
        if any(not prefix for prefix in self.excluded_assistant_prefixes):
            raise ValueError("excluded_assistant_prefixes must not contain empty strings")

    @property
    def stop_text(self) -> str:
        return self.generation_stop or self.assistant_end

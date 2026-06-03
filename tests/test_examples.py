import py_compile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_FILES = [
    ROOT / "examples/minimal_qwen2_5_omni_assistant_mask.py",
    ROOT / "examples/minimal_qwen3_omni_assistant_mask.py",
    ROOT / "examples/minimal_gemma4_e2b_it_assistant_mask.py",
]
GENERIC_COLLATOR_EXAMPLE = ROOT / "examples/generic_processor_collator.py"


def test_minimal_examples_compile() -> None:
    for path in [*EXAMPLE_FILES, GENERIC_COLLATOR_EXAMPLE]:
        py_compile.compile(str(path), doraise=True)


def test_minimal_examples_do_not_hide_multimodal_usage() -> None:
    for path in EXAMPLE_FILES:
        source = path.read_text()
        assert "RUN_REAL_MULTIMODAL_EXAMPLE" not in source
        assert "Multimodal section skipped" not in source
        assert "urlopen" not in source
        assert "_download_" not in source
        assert "Multimodal decoded supervised text:" in source
        assert "Text-template decoded supervised text:" in source


def test_generic_collator_uses_processor_default_template() -> None:
    source = GENERIC_COLLATOR_EXAMPLE.read_text()
    assert "chat_template:" not in source
    assert "chat_template=chat_template" not in source
    assert "processor.apply_chat_template" in source
    assert "media_kwargs" in source
    assert "build_assistant_labels" in source
    assert "build_labels_from_frame_mask" not in source
    assert "frame_masks" not in source
    assert "attention_mask=batch.get" not in source


def test_minimal_examples_use_public_labels_api() -> None:
    for path in EXAMPLE_FILES:
        source = path.read_text()
        assert "AssistantMaskSpec" in source
        assert "build_assistant_labels" in source
        assert "build_assistant_frame_masks" not in source
        assert "build_labels_from_frame_mask" not in source
        assert "frame_masks" not in source
        assert "attention_mask=batch.get" not in source

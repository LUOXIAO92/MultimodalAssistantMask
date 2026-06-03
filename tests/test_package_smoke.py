def test_package_imports() -> None:
    import mm_assistant_mask

    assert "AssistantFrameSpec" in mm_assistant_mask.__all__
    assert "AssistantMaskSpec" in mm_assistant_mask.__all__
    assert "build_assistant_labels" in mm_assistant_mask.__all__
    assert "build_assistant_mask" in mm_assistant_mask.__all__

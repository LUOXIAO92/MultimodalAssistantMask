def test_package_imports() -> None:
    import mm_assistant_mask

    assert "AssistantFrameSpec" in mm_assistant_mask.__all__

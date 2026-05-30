# Multimodal Processor Regression Tests Progress

## 0. 基准文件

本子任务以以下文件为基准：

1. `docs/multimodal_assistant_mask_design.md`
2. `plans/multimodal_assistant_mask_implementation_plan.md`
3. `progress/multimodal_assistant_mask_progress.md`

## 1. 任务表

| 任务 | 状态 | 验收 |
| --- | --- | --- |
| 创建多模态 processor regression 子任务计划 | 已完成 | `plans/multimodal_processor_regression_tests_plan.md` 存在 |
| 创建多模态 processor regression 子任务进度 | 已完成 | 本文件存在 |
| 探测 image mixed 输入格式 | 已完成 | Qwen2.5-Omni 真实 processor 成功 |
| 增加 image mixed 测试 | 已完成 | final input_ids + frame matching |
| 探测 audio mixed 输入格式 | 已完成 | Qwen2.5-Omni 真实 processor 成功 |
| 增加 audio mixed 测试 | 已完成 | final input_ids + frame matching |
| 增加 processor assistant mask diagnostic/regression | 已完成 | 不作为主路径 |
| 运行标准 pytest | 已完成 | `conda run -n makesense python -m pytest` |

## 2. 当前记录

1. `Qwen/Qwen2.5-Omni-3B` image mixed 路径可用：`apply_chat_template` 渲染 `<|vision_bos|><|IMAGE|><|vision_eos|>`，processor 产出 `input_ids`、`pixel_values`、`image_grid_thw`。
2. `Qwen/Qwen2.5-Omni-3B` audio mixed 路径可用：`apply_chat_template` 渲染 `<|audio_bos|><|AUDIO|><|audio_eos|>`，processor 产出 `input_ids`、`input_features`、`feature_attention_mask`。
3. 已新增 `tests/test_multimodal_processor_real_inputs.py`。
4. 已新增 processor assistant mask diagnostic regression，确认 processor `assistant_masks` 不作为训练主路径。
5. 标准测试命令 `conda run -n makesense python -m pytest` 通过：26 passed，4 warnings。

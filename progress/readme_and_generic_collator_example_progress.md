# README And Generic Collator Example Progress

## 0. 基准文件

本子任务以以下文件为基准：

1. `docs/multimodal_assistant_mask_design.md`
2. `plans/multimodal_assistant_mask_implementation_plan.md`
3. `progress/multimodal_assistant_mask_progress.md`

## 1. 任务表

| 任务 | 状态 | 验收 |
| --- | --- | --- |
| 创建 README/示例子任务计划 | 已完成 | `plans/readme_and_generic_collator_example_plan.md` 存在 |
| 创建 README/示例子任务进度 | 已完成 | 本文件存在 |
| README 说明主路径 | 已完成 | final processor input_ids + frame matching |
| README 说明 `{% generation %}` 边界 | 已完成 | 不把 processor assistant_masks 当主路径 |
| README 说明 frame matching 通用性 | 已完成 | 不限多模态，纯文本也可使用 |
| 新增 generic collator 示例 | 已完成 | 不写死具体模型 |
| generic collator 使用 processor 默认模板 | 已完成 | 不接收/透传 train Jinja |
| 文档说明 truncation 策略 | 已完成 | 不能静默训练 |
| 运行验证 | 已完成 | pytest + compile 示例 |

## 2. 当前记录

1. 已更新 `README.md`。
2. 已新增 `examples/generic_processor_collator.py`。
3. `conda run -n makesense python -m py_compile examples/generic_processor_collator.py` 通过。
4. README 已说明 token-id frame matching 不限于图片/音频/视频，纯文本 final `input_ids` 也可使用；与 Jinja `assistant_masks` 的差异是默认不包含 assistant end/eos 后 separator newline。
5. generic collator 已确认使用 processor 默认 `apply_chat_template`，不接收/透传 train Jinja；训练 Jinja 只用于纯文本模板线。
6. `conda run -n makesense python -m pytest tests/test_examples.py` 通过：3 passed。
7. 标准测试命令 `conda run -n makesense python -m pytest` 通过：41 passed，9 warnings。

# Validation Debug Helpers Progress

## 0. 基准文件

本子任务以以下文件为基准：

1. `docs/multimodal_assistant_mask_design.md`
2. `plans/multimodal_assistant_mask_implementation_plan.md`
3. `progress/multimodal_assistant_mask_progress.md`

## 1. 任务表

| 任务 | 状态 | 验收 |
| --- | --- | --- |
| 创建 validation 子任务计划 | 已完成 | `plans/validation_debug_helpers_plan.md` 存在 |
| 创建 validation 子任务进度 | 已完成 | 本文件存在 |
| 实现 `decode_supervised_tokens` | 已完成 | 可解码 labels 非 `-100` 部分 |
| 实现 `assert_no_header_in_labels` | 已完成 | header 出现时报错 |
| 实现 `assert_end_in_labels` | 已完成 | end 缺失时报错 |
| 实现 frame token ids debug helper | 已完成 | 暴露 header/end token ids |
| 增加真实 processor 测试 | 已完成 | 不使用 fake tokenizer |
| 运行标准 pytest | 已完成 | `conda run -n makesense python -m pytest` |

## 2. 当前记录

1. 已新增 `src/mm_assistant_mask/validation.py`。
2. 已新增真实 processor 测试 `tests/test_validation_real_processors.py`。
3. 标准测试命令 `conda run -n makesense python -m pytest` 通过：18 passed，3 warnings。

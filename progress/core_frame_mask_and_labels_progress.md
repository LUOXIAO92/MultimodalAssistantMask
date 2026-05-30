# Core Frame Mask And Labels Progress

## 0. 基准文件

本子任务以以下文件为基准：

1. `docs/multimodal_assistant_mask_design.md`
2. `plans/multimodal_assistant_mask_implementation_plan.md`
3. `progress/multimodal_assistant_mask_progress.md`

## 1. 任务表

| 任务 | 状态 | 验收 |
| --- | --- | --- |
| 创建核心 mask 子任务计划 | 已完成 | `plans/core_frame_mask_and_labels_plan.md` 存在 |
| 创建核心 mask 子任务进度 | 已完成 | 本文件存在 |
| 实现 `AssistantFrameSpec` | 已完成 | 默认策略符合设计文档 |
| 实现 token frame matcher | 已完成 | 真实 processor input_ids 上可构造 mask |
| 实现 assistant 起始前缀排除 | 已完成 | `excluded_assistant_prefixes` 可排除结构前缀 |
| 实现 labels 构造 | 已完成 | mask 外为 `-100` |
| 增加真实 processor 测试 | 已完成 | 不使用 fake tokenizer |
| 运行标准 pytest | 已完成 | `conda run -n makesense python -m pytest` |

## 2. 当前记录

1. 已新增 `src/mm_assistant_mask/frame_spec.py`。
2. 已新增 `src/mm_assistant_mask/token_frame_matcher.py`。
3. 已新增 `src/mm_assistant_mask/labels.py`。
4. 已通过真实 processor 覆盖：
   - `Qwen/Qwen2.5-Omni-3B`
   - `Qwen/Qwen3-Omni-30B-A3B-Instruct`
   - `google/gemma-4-E2B-it`
5. `google/gemma-4-E2B` 已纠正为 `google/gemma-4-E2B-it`，使用 instruction-tuned processor 的真实 chat template 路径。
6. 已补齐 `Qwen/Qwen3-Omni-30B-A3B-Instruct` 与 `google/gemma-4-E2B-it` 的多轮 text-only 真实 processor 测试。
7. 已新增 `AssistantFrameSpec.excluded_assistant_prefixes`，用于排除 assistant header 后自动插入但不应进入 loss 的结构前缀。
8. 标准测试命令 `conda run -n makesense python -m pytest` 通过：41 passed，9 warnings。

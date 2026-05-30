# Training Chat Templates Progress

## 0. 基准文件

本子任务以以下文件为基准：

1. `docs/multimodal_assistant_mask_design.md`
2. `plans/multimodal_assistant_mask_implementation_plan.md`
3. `progress/multimodal_assistant_mask_progress.md`

## 1. 任务表

| 任务 | 状态 | 验收 |
| --- | --- | --- |
| 创建 training template 子任务计划 | 已完成 | `plans/training_chat_templates_plan.md` 存在 |
| 创建 training template 子任务进度 | 已完成 | 本文件存在 |
| 新增 Gemma train Jinja 模板 | 已完成 | `templates/gemma/gemma-4-e2b-it-train.jinja` |
| 新增 Qwen2.5-Omni train Jinja 模板 | 已完成 | `templates/qwen/qwen2_5-omni-3b-train.jinja` |
| 新增 Qwen3-Omni train Jinja 模板 | 已完成 | `templates/qwen/qwen3-omni-30b-a3b-instruct-train.jinja` |
| 增加 text-only render parity 测试 | 已完成 | 与源模板渲染一致 |
| 增加 tokenizer-only assistant mask 测试 | 已完成 | header 不进 mask，payload/end 进 mask |
| 增加 add_generation_prompt 测试 | 已完成 | prompt cue 无 payload mask |
| 增加 tool call 模板测试 | 已完成 | tool call 渲染与源模板一致，并进入 assistant mask |
| 运行标准 pytest | 已完成 | `conda run -n makesense python -m pytest` |

## 2. 当前记录

1. 训练模板文件命名使用 `*-train.jinja`。
2. Qwen 系列训练模板输出为 `.jinja`，不使用 JSON 包装。
3. 本子任务已验收 text-only tokenizer assistant mask 和 Qwen3/Gemma tool call 路径；多模态训练 labels 仍使用 final processor `input_ids` + frame matching。
4. 已新增 `tests/test_training_templates.py`，覆盖 Gemma 4 E2B it、Qwen2.5-Omni-3B、Qwen3-Omni-30B-A3B-Instruct 的 text-only render parity、assistant mask、`add_generation_prompt`，以及 Qwen3/Gemma tool call render parity 与 assistant mask。
5. Gemma 与 Qwen3 train 模板保留源模板的 tool call、tool response、thinking/internal tag 等分支，只在 assistant 生成区域增加 `{% generation %}` / `{% endgeneration %}`。
6. Qwen3-Omni-Instruct train 模板插入的空 `<think>` 闭合前缀已排除在 tokenizer-only assistant mask 外。
7. 标准测试命令 `conda run -n makesense python -m pytest` 通过：41 passed，9 warnings。

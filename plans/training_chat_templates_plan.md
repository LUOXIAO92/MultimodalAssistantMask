# Training Chat Templates Plan

## 0. 基准文件

本子任务以以下文件为基准：

1. `docs/multimodal_assistant_mask_design.md`
2. `plans/multimodal_assistant_mask_implementation_plan.md`
3. `progress/multimodal_assistant_mask_progress.md`

服务的主线验收标准：

1. 模板线只负责 tokenizer-only assistant span declaration。
2. 多模态训练主路径仍然是 final processor `input_ids` + token-id frame matching。
3. assistant header 不进 generation block。
4. assistant payload 和 assistant end 进 generation block。

## 1. 目标

基于当前加入的 Gemma 4 E2B it、Qwen2.5-Omni、Qwen3-Omni 模板素材，新增 `*-train.jinja` 训练模板，并验证 text-only tokenizer assistant mask。

## 2. 范围

本阶段只做以下事情：

1. 新增 `templates/gemma/gemma-4-e2b-it-train.jinja`。
2. 新增 `templates/qwen/qwen2_5-omni-3b-train.jinja`。
3. 新增 `templates/qwen/qwen3-omni-30b-a3b-instruct-train.jinja`。
4. Qwen 训练模板使用 `.jinja` 文件，不使用 JSON 包装。
5. 增加 text-only render parity、assistant mask、`add_generation_prompt` 测试。
6. 增加 tool call render parity 与 assistant mask 测试，防止训练模板退化成只支持普通对话。

## 3. 不做范围

1. 不创建 `src/mm_assistant_mask/templates.py`。
2. 不把 tokenizer-only `assistant_masks` 作为多模态训练主路径。
3. 不在本阶段完整验收 media 复杂分支。

## 4. 验证

标准命令：

```bash
conda run -n makesense python -m pytest
```

成功标准：

1. text-only 渲染与源模板一致。
2. assistant header 不在 tokenizer-only assistant mask 中。
3. assistant payload 和 assistant end 在 tokenizer-only assistant mask 中。
4. `add_generation_prompt=True` 只渲染 assistant header/prompt cue，不产生 assistant payload mask。
5. tool call 渲染与源模板一致，且 assistant tool call 内容进入 tokenizer-only assistant mask。

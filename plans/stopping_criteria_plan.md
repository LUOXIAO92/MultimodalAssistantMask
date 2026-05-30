# Stopping Criteria Plan

## 0. 基准文件

本子任务以以下文件为基准：

1. `docs/multimodal_assistant_mask_design.md`
2. `plans/multimodal_assistant_mask_implementation_plan.md`
3. `progress/multimodal_assistant_mask_progress.md`

服务的主线验收标准：

1. generation / rollout 必须在 `generation_stop` 或 `assistant_end` 本身停止。
2. 不等待 `assistant_end + "\n"`。
3. 使用真实 tokenizer 编码 stop sequence，不使用 fake tokenizer。

## 1. 目标

实现最小 `StopOnTokenSequence`，用于 Hugging Face generation `StoppingCriteria`。

## 2. 不做范围

1. 不实现 reward。
2. 不实现 response extraction。
3. 不加载模型权重做真实 generation。
4. 不做复杂 per-sample stopped state 管理。

## 3. 验证

标准命令：

```bash
conda run -n makesense python -m pytest
```

测试使用真实 processor/tokenizer 编码 stop token ids，但不加载模型。

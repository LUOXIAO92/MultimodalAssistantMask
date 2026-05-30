# Multimodal Processor Regression Tests Plan

## 0. 基准文件

本子任务以以下文件为基准：

1. `docs/multimodal_assistant_mask_design.md`
2. `plans/multimodal_assistant_mask_implementation_plan.md`
3. `progress/multimodal_assistant_mask_progress.md`

服务的主线验收标准：

1. 真实 processor 处理 image/audio mixed user content 后，仍在最终 `input_ids` 上构造 assistant-only labels。
2. processor 返回的 `assistant_masks` 只能作为 diagnostic，对错不影响本项目主路径。
3. 测试不使用 fake tokenizer，不加载模型权重。

## 1. 目标

补齐阶段 6 剩余多模态 processor 覆盖：

1. image + text mixed user content。
2. audio + text mixed user content。
3. processor assistant mask diagnostic 或等价 expansion regression。

## 2. 不做范围

1. 不维护 media expansion 公式。
2. 不加载模型权重。
3. 不要求所有 processor 都支持所有 media 类型。
4. 不用 fake media processor 替代真实 processor。

## 3. 验证

标准命令：

```bash
conda run -n makesense python -m pytest
```

如果某个 processor 不支持对应 media 类型，测试应明确跳过或只覆盖支持该 media 的首批 processor。

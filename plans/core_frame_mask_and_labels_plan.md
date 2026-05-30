# Core Frame Mask And Labels Plan

## 0. 基准文件

本子任务以以下文件为基准：

1. `docs/multimodal_assistant_mask_design.md`
2. `plans/multimodal_assistant_mask_implementation_plan.md`
3. `progress/multimodal_assistant_mask_progress.md`

服务的主线验收标准：

1. 在最终 processor `input_ids` 上做 token-id assistant frame matching。
2. assistant header 默认不进 loss。
3. assistant payload 和 assistant end 默认进 loss。
4. assistant end 后的 separator newline 默认不进 loss。
5. 测试使用真实 processor/tokenizer，不使用 fake tokenizer。

## 1. 目标

实现第一版核心 SFT label builder：

1. `frame_spec.py`
2. `token_frame_matcher.py`
3. `labels.py`

## 2. 不做范围

1. 不实现 `validation.py`。
2. 不实现 `stopping.py`。
3. 不实现 template 文件。
4. 不加入模型专属分支。

## 3. 验证

标准命令：

```bash
conda run -n makesense python -m pytest
```

测试必须使用首批真实 processor/tokenizer 的至少一个 Qwen case 和一个 Gemma case。

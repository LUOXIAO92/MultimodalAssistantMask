# Validation Debug Helpers Plan

## 0. 基准文件

本子任务以以下文件为基准：

1. `docs/multimodal_assistant_mask_design.md`
2. `plans/multimodal_assistant_mask_implementation_plan.md`
3. `progress/multimodal_assistant_mask_progress.md`

服务的主线验收标准：

1. 让用户能解码 labels 中真正参与 loss 的 token。
2. 能断言 assistant header 不在 supervised tokens 中。
3. 能断言 assistant end token 在 supervised tokens 中。
4. 暴露 assistant header/end 的真实 tokenizer ids，辅助排查 frame spec 与 processor/tokenizer 不一致。

## 1. 目标

实现 `validation.py` 中的最小 debug helper，不改变训练数据，只做可观测性和断言。

## 2. 不做范围

1. 不修复 user content 中原样出现 assistant frame string 的 escaping 问题。
2. 不实现 stopping。
3. 不实现模板加载。
4. 不创建泛用诊断框架。

## 3. 验证

标准命令：

```bash
conda run -n makesense python -m pytest
```

测试必须使用真实 processor/tokenizer 构造 input ids 和 labels。

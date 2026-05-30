# README And Generic Collator Example Plan

## 0. 基准文件

本子任务以以下文件为基准：

1. `docs/multimodal_assistant_mask_design.md`
2. `plans/multimodal_assistant_mask_implementation_plan.md`
3. `progress/multimodal_assistant_mask_progress.md`

服务的主线验收标准：

1. README 必须强调多模态训练主路径是 final processor `input_ids` + assistant frame matching。
2. README 必须说明 `{% generation %}` 是模板层 span declaration，不保证 processor expansion 后的 mask 正确。
3. 示例必须展示 generic processor collator，不写死具体模型。
4. 文档必须说明 truncation 不能静默训练。

## 1. 目标

补充 README 和最小 generic collator 示例，让用户能按当前 API 连接 processor batch、frame mask 和 labels。

## 2. 不做范围

1. 不维护具体模型模板。
2. 不加载模型权重。
3. 不实现数据集读取框架。
4. 不写模型专属 collator。

## 3. 验证

标准命令：

```bash
conda run -n makesense python -m pytest
```

示例文件至少应能被 Python 编译检查。

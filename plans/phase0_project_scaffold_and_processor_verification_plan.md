# Phase 0 Project Scaffold And Processor Verification Plan

## 0. 基准文件

本子任务以以下文件为基准：

1. `docs/multimodal_assistant_mask_design.md`
2. `plans/multimodal_assistant_mask_implementation_plan.md`
3. `progress/multimodal_assistant_mask_progress.md`

服务的主线验收标准：

1. 所有验证在 `makesense` conda 环境中执行。
2. 核心测试必须使用真实 processor/tokenizer，不使用 fake tokenizer。
3. 首批 processor 只加载 processor/tokenizer，不加载大模型权重。
4. 后续实现必须围绕最终 processor `input_ids` 上的 token-id assistant frame matching。

## 1. 目标

建立最小 Python 项目骨架，并确认真实 processor 验证路径可执行。

## 2. 范围

本阶段只做以下事情：

1. 新增最小 `pyproject.toml`。
2. 新增最小 `src/mm_assistant_mask/__init__.py`，确保本地包可导入。
3. 新增 pytest 配置和阶段 0 smoke test。
4. 检查 `makesense` conda 环境。
5. 检查首批 processor/tokenizer 是否能在不加载模型权重的情况下完成最小加载。

本阶段不实现：

1. `AssistantFrameSpec`。
2. token frame matcher。
3. labels 构造。
4. validation/debug 工具。
5. stopping 工具。

这些属于后续阶段，避免提前创建没有实现内容的模块。

## 3. 首批 processor

1. `Qwen/Qwen2.5-Omni-3B`
2. `Qwen/Qwen3-Omni-30B-A3B-Instruct`
3. `google/gemma-4-E2B-it`

测试原则：

1. 使用真实 `AutoProcessor` 或对应 processor 加载路径。
2. 不调用 `AutoModel`、`from_pretrained` 模型类或任何模型权重加载。
3. processor/tokenizer 文件可来自本地缓存；如果缺缓存且需要网络，记录为环境阻塞，不用 fake tokenizer 替代。

## 4. 验证命令

标准命令：

```bash
conda run -n makesense python -m pytest
```

辅助检查命令：

```bash
conda run -n makesense python -c "import transformers; print(transformers.__version__)"
```

## 5. 成功标准

1. `conda run -n makesense python -m pytest` 可以启动 pytest。
2. `import mm_assistant_mask` 成功。
3. pytest 中不存在 fake tokenizer。
4. 首批 processor 的加载测试以真实 processor/tokenizer 为准。
5. 如果 processor 文件缺失或网络受限，测试应明确报告依赖缺失，而不是静默退化为 fake tokenizer。

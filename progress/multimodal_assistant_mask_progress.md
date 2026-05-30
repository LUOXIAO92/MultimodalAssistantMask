# Multimodal Assistant Mask 进度表

本进度表对应 `plans/multimodal_assistant_mask_implementation_plan.md`。状态说明：

- `未开始`：尚未实现。
- `进行中`：已有部分工作，但未通过验收。
- `已完成`：实现和验证都已完成。
- `后续`：不属于第一阶段核心交付，等待主路径稳定后处理。

## 总览

| 阶段 | 内容 | 状态 | 验收方式 |
| --- | --- | --- | --- |
| 0 | 项目骨架确认 | 已完成 | `conda run -n makesense python -m pytest` 可运行，本地包可导入 |
| 1 | `AssistantFrameSpec` | 已完成 | 默认 loss 策略符合设计文档 |
| 2 | token-id frame matcher | 已完成 | 单轮、多轮、重复 payload 测试通过 |
| 3 | labels 构造 | 已完成 | mask、padding、shape 测试通过 |
| 4 | validation/debug 工具 | 已完成 | supervised tokens 可解码，可断言 header/end |
| 5 | stopping 工具 | 已完成 | stop sequence 行为测试通过 |
| 6 | 核心测试覆盖 | 已完成 | 真实 processor 驱动的核心测试稳定通过 |
| 7 | 示例与 README | 已完成 | README 与 generic collator 示例可读 |
| 8 | Jinja/template 修改线 | 已完成 | text-only 与 tool call render parity / tokenizer-only mask 测试通过 |
| 9 | 最小真实使用示例 | 已完成 | 一模型一脚本，多步多模态与多步纯文本示例已通过 |

## 详细任务

### 阶段 0：项目骨架确认

| 任务 | 状态 | 备注 |
| --- | --- | --- |
| 检查 repo 当前结构 | 已完成 | 已确认存在 `docs/`、`plans/`、`progress/` |
| 确认第一阶段模块职责 | 已完成 | 已在计划书中写明每个模块用途、边界和保留条件 |
| 删除无明确用途的 `templates.py` 规划 | 已完成 | 不创建 Python 模块；后续只在有加载 API 时再加入 |
| 检查包管理文件 | 已完成 | 已新增最小 `pyproject.toml` |
| 确定 src layout | 已完成 | 使用 `src/mm_assistant_mask/` |
| 配置 pytest | 已完成 | 标准命令可运行 |
| 确认 `makesense` conda 环境 | 已完成 | Python 3.13.13 |
| 确认真实 tokenizer/processor 测试依赖 | 已完成 | 核心验收不使用 fake tokenizer |
| 确认 `Qwen/Qwen2.5-Omni-3B` processor | 已完成 | 只加载 processor/tokenizer，不加载模型权重 |
| 确认 `Qwen/Qwen3-Omni-30B-A3B-Instruct` processor | 已完成 | 只加载 processor/tokenizer，不加载模型权重 |
| 确认 `google/gemma-4-E2B-it` processor | 已完成 | 只加载 processor/tokenizer，不加载模型权重 |
| 验证 `conda run -n makesense python -m pytest` | 已完成 | 4 passed，1 warning |

### 阶段 1：`AssistantFrameSpec`

| 任务 | 状态 | 备注 |
| --- | --- | --- |
| 新增 `frame_spec.py` | 已完成 | 定义 dataclass |
| 添加 header/end 字段 | 已完成 | 必需字段 |
| 添加 generation stop 字段 | 已完成 | 为空时可回退到 assistant end |
| 添加 include flags | 已完成 | 默认不含 header，包含 end，不含 end 后 newline |
| 添加 assistant 起始前缀排除 | 已完成 | 可排除 Qwen3 空 think 前缀等结构文本 |
| 添加基础 validation | 已完成 | header/end 不允许为空 |
| 真实 processor 测试默认策略 | 已完成 | 覆盖 Qwen/Gemma 风格配置表达能力，不使用 fake tokenizer |

### 阶段 2：token-id frame matcher

| 任务 | 状态 | 备注 |
| --- | --- | --- |
| 实现 tokenizer 输出兼容转换 | 已完成 | 支持 dict、BatchEncoding、tensor、list |
| 实现 `find_subsequence` | 已完成 | 空 pattern 抛错 |
| 实现单样本 frame mask | 已完成 | `build_assistant_frame_mask_for_one` |
| 支持多轮 assistant frame | 已完成 | cursor 顺序推进 |
| 处理重复 payload | 已完成 | 通过 frame 匹配避免错位 |
| header found but no end 抛错 | 已完成 | 防止截断静默训练 |
| no frame found 抛错 | 已完成 | 防止无监督样本静默进入训练 |

### 阶段 3：labels 构造

| 任务 | 状态 | 备注 |
| --- | --- | --- |
| 新增 `labels.py` | 已完成 | 构造训练 labels |
| mask 外置 `-100` | 已完成 | PyTorch tensor 实现 |
| mask 内保留 input ids | 已完成 | labels 等于原 input ids |
| 支持 attention mask | 已完成 | 排除 padding |
| shape mismatch 抛错 | 已完成 | 明确错误信息 |

### 阶段 4：validation/debug 工具

| 任务 | 状态 | 备注 |
| --- | --- | --- |
| 新增 `validation.py` | 已完成 | 调试辅助 |
| 实现 `decode_supervised_tokens` | 已完成 | 解码 labels 非 `-100` 部分 |
| 实现 header 断言 | 已完成 | header 不应出现在 supervised tokens |
| 实现 end 断言 | 已完成 | end token 应出现在 supervised tokens |
| 可选 raw frame string 检测 | 后续 | 用于 user content serialization 风险提示 |

### 阶段 5：stopping 工具

| 任务 | 状态 | 备注 |
| --- | --- | --- |
| 新增 `stopping.py` | 已完成 | 可选依赖 Transformers |
| 实现 token sequence stop | 已完成 | 默认 stop at assistant end |
| 支持 batch 基础行为 | 已完成 | 任一 batch row 末尾出现 stop sequence 即停止 |
| 测试不等待 separator newline | 已完成 | stop token 本身出现即可 |

### 阶段 6：核心测试覆盖

| 测试 | 状态 | 备注 |
| --- | --- | --- |
| `Qwen/Qwen2.5-Omni-3B` 单轮 text-only | 已完成 | 使用真实 processor/tokenizer |
| `Qwen/Qwen2.5-Omni-3B` 多轮 text-only | 已完成 | 多个 assistant frame |
| `Qwen/Qwen3-Omni-30B-A3B-Instruct` 单轮 text-only | 已完成 | 使用真实 processor/tokenizer |
| `Qwen/Qwen3-Omni-30B-A3B-Instruct` 多轮 text-only | 已完成 | 多个 assistant frame |
| `google/gemma-4-E2B-it` 单轮 text-only | 已完成 | 使用真实 processor/tokenizer |
| `google/gemma-4-E2B-it` 多轮 text-only | 已完成 | 多个 assistant frame |
| 首批 processor 不加载模型权重 | 已完成 | processor 测试不应初始化大模型 |
| 重复 assistant payload | 已完成 | 防止裸 payload 匹配问题 |
| header 不进 labels | 已完成 | 解码或索引断言 |
| end token 进 labels | 已完成 | 默认策略 |
| end 后 newline 默认不进 labels | 已完成 | 关键设计决策 |
| include header flag | 已完成 | 可选策略 |
| exclude end flag | 已完成 | 可选策略 |
| include post-end separator flag | 已完成 | 可选策略 |
| truncation 错误 | 已完成 | header found but no end |
| no frame found 错误 | 已完成 | 无 assistant frame |
| padding + attention mask | 已完成 | batch labels |
| shape mismatch | 已完成 | labels 参数校验 |
| processor assistant mask 错位复现或等价 regression | 已完成 | processor `assistant_masks` 只作 diagnostic |
| image + text mixed user content | 已完成 | Qwen2.5-Omni 真实 processor |
| audio + text mixed user content | 已完成 | Qwen2.5-Omni 真实 processor |
| 禁止 fake tokenizer 核心验收 | 已完成 | tests/src 中无 fake tokenizer，核心测试均使用真实 processor/tokenizer |

### 阶段 7：示例与 README

| 任务 | 状态 | 备注 |
| --- | --- | --- |
| README 说明主路径 | 已完成 | final input_ids + frame matching |
| README 说明 `{% generation %}` 边界 | 已完成 | 模板线不是多模态主路径 |
| 新增 generic collator 示例 | 已完成 | 不写死具体模型 |
| generic collator 模板边界 | 已完成 | 使用 processor 默认模板，不接收/透传 train Jinja |
| 文档说明 truncation 策略 | 已完成 | 抛错、跳过或调整 max_length |

### 阶段 8：Jinja/template 修改线

| 任务 | 状态 | 备注 |
| --- | --- | --- |
| Qwen 风格 training template | 已完成 | 已新增 Qwen2.5/Qwen3 Omni `*-train.jinja`，text-only 与 Qwen3 tool call 已验收 |
| Gemma 风格 training template | 已完成 | 已新增 Gemma 4 E2B it `*-train.jinja`，text-only 与 tool call 已验收 |
| render parity 测试 | 已完成 | text-only 与 tool call 均与源模板渲染一致 |
| tokenizer-only assistant mask 测试 | 已完成 | header 不进 mask，payload/end/tool call 进 mask；Qwen3 空 think 前缀不进 mask |
| add_generation_prompt 测试 | 已完成 | text-only prompt cue 无 payload mask |

### 阶段 9：真实使用示例

| 任务 | 状态 | 备注 |
| --- | --- | --- |
| 旧真实多模态示例计划 | 已废弃 | `plans/real_multimodal_usage_example_plan.md` 已标注不合格 |
| 旧真实多模态示例进度 | 已废弃 | `progress/real_multimodal_usage_example_progress.md` 已标注不合格 |
| 最小用例新计划 | 已完成 | `plans/minimal_assistant_mask_usage_examples_plan.md` |
| 最小用例新进度 | 已完成 | `progress/minimal_assistant_mask_usage_examples_progress.md` |
| Qwen2.5-Omni 一脚本用例 | 已完成 | 一个脚本内包含多步多模态与多步纯文本两个 section |
| Qwen3-Omni 一脚本用例 | 已完成 | 一个脚本内包含多步多模态与多步纯文本两个 section |
| Gemma 4 E2B it 一脚本用例 | 已完成 | 一个脚本内包含多步多模态与多步纯文本两个 section |
| 示例轻量测试 | 已完成 | 编译、检查无旧分支/下载 helper |
| 真实 URL 多模态运行 | 已完成 | 三个最小脚本已直接运行通过 |

## 当前已完成事项

1. 已阅读并理解 `docs/multimodal_assistant_mask_design.md`。
2. 已确认项目指导约束来自 `AGENTS.md`。
3. 已创建实施计划书。
4. 已创建本进度表。
5. 已补充第一阶段模块职责、边界和保留条件。
6. 已从第一阶段代码目录规划中移除无明确用途的 `templates.py`。
7. 已在计划书第 0 节加入施工基准文件与子任务新文件规则。
8. 已将首批 Gemma processor 从 base 模型 `google/gemma-4-E2B` 纠正为 instruction-tuned 模型 `google/gemma-4-E2B-it`，并通过真实 processor 测试。
9. 已在计划书第 0 节加入“完成任务后必须回表标记，再继续未完成任务”的执行规则。
10. 已完成 validation/debug 工具，并通过真实 processor 测试。
11. 已完成 stopping 工具，并通过真实 tokenizer 测试。
12. 已补齐 `Qwen/Qwen3-Omni-30B-A3B-Instruct` 与 `google/gemma-4-E2B-it` 的多轮 text-only 真实 processor 测试。
13. 已确认核心测试没有 fake tokenizer。
14. 已完成 README 和 generic processor collator 示例，并通过 pytest 与 py_compile 验证。
15. 已完成 Qwen2.5-Omni image/audio mixed user content 真实 processor 测试。
16. 已完成 processor assistant mask diagnostic regression，并确认核心测试覆盖阶段完成。
17. 已完成阶段 8 training chat template 修改线，新增 Gemma/Qwen Omni `*-train.jinja` 模板，并通过 text-only 与 tool call assistant mask 测试。
18. 旧真实使用示例因抽象层过重已废弃；已按最小用例计划新增一模型一脚本示例，每个脚本内直接展示多步多模态与多步纯文本两条路径。
19. 已新增 `AssistantFrameSpec.excluded_assistant_prefixes`，用于排除 Qwen3-Omni-Instruct 自动插入的空 `<think>` 闭合前缀等不应进入 loss 的 assistant 起始结构文本。

## 后续工作

阶段 0 到阶段 9 当前计划内工作均已完成。旧 `real_multimodal*` 示例已标注废弃，后续由用户移除。

后续可选工作只剩非阻塞增强：例如为 validation/debug 增加 raw frame string 风险提示，或在确有调用方后再设计模板发现/加载 API。

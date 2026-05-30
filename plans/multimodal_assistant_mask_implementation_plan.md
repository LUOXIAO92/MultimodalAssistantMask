# Multimodal Assistant Mask 实施计划书

## 0. 施工基准与子任务文件规则

当前所有施工都必须以以下文件为基准：

1. 设计文档：`docs/multimodal_assistant_mask_design.md`
2. 实施计划书：`plans/multimodal_assistant_mask_implementation_plan.md`
3. 进度表：`progress/multimodal_assistant_mask_progress.md`

执行规则：

1. 任何实现、测试、示例和文档更新都必须服务于设计文档定义的主线：在最终 processor `input_ids` 上用 token-id assistant frame matching 构造 assistant-only labels。
2. 如果需要增加新的子任务，不能直接把子任务细节追加到当前三份基准文件里。
3. 新子任务必须新建独立文件，并根据类别放入合适目录，例如：
   - 子任务计划：`plans/<task_name>_plan.md`
   - 子任务设计或说明：`docs/<task_name>.md`
   - 子任务进度：`progress/<task_name>_progress.md`
4. 当子任务类别发生变化时，必须新开一个文件。例如 processor 验证、template 修改、RL stopping、README 示例分别属于不同类别时，应分别维护各自的计划或进度文件。
5. 子任务文件必须引用本节列出的三份基准文件，并说明自己服务于哪一条主线验收标准。
6. 子任务不能偏离主线，不能引入与 assistant frame matching、真实 processor 验证、assistant-only labels、generation stop 或 training template 兼容性无关的工作。
7. 当前三份基准文件只维护项目级原则、阶段划分、验收标准和总体进度；子任务细节放入独立文件。
8. 每完成一个任务，必须回到对应子任务进度文件和总进度表，把已完成任务标记为 `已完成`，再继续下一个 `未开始` 或 `进行中` 任务。

## 1. 目标

本计划基于 `docs/multimodal_assistant_mask_design.md`，目标是实现一个通用的多模态 assistant-only label mask 工具。项目不绑定某个模型，而是围绕最终 `processor input_ids` 上的 token-id assistant frame matching 构造可靠的 SFT/RL 训练 mask。

核心交付物：

1. 一个可配置的 `AssistantFrameSpec`。
2. 一个在最终 `input_ids` 上工作的 assistant frame matcher。
3. 一个从 frame mask 构造 labels 的工具函数。
4. 基础 validation/debug 工具。
5. generation / rollout stopping 工具。
6. 针对 text-only、多轮、多模态 processor、padding、truncation、newline 策略等关键行为的测试。
7. 最小示例和 README 说明。

## 2. 基本假设

1. 用户会提供或选择正确的 assistant frame 规格，例如 assistant header 与 assistant end。
2. 最终训练路径会先通过 processor 生成真实 `input_ids`，再调用本项目的 mask builder。
3. 本项目不推断 image/audio/video 展开 token 数量。
4. 本项目不把 processor 返回的 `assistant_masks` 作为多模态训练主路径。
5. 第一阶段优先完成通用 Python API 与真实 processor 驱动的测试，不优先维护大量模型模板。
6. 具体模型模板可以后续加入，但必须作为配置或模板资源存在，不能污染核心算法。
7. 所有本地验证默认在名为 `makesense` 的 conda 环境中执行。
8. 本项目不接受 fake tokenizer 作为核心正确性测试依据；fake tokenizer 容易掩盖 tokenizer special token、chat template、processor expansion 与真实 `input_ids` 对齐问题。
9. 首批 processor 测试优先覆盖 `Qwen/Qwen2.5-Omni-3B`、`Qwen/Qwen3-Omni-30B-A3B-Instruct`、`google/gemma-4-E2B-it`；测试应只加载 processor/tokenizer，不加载大模型权重。

## 3. 不做范围

1. 不写死 Qwen、Gemma 或任何单一模型的特殊逻辑。
2. 不维护每个 processor 的 media expansion 公式。
3. 不用字符 offset mapping 构造多模态 label mask。
4. 不匹配裸 assistant payload。
5. 不用自然语言黑名单判断哪些 token 应该训练。
6. 不在第一阶段实现完整上游 Transformers / TRL patch。

## 4. 模块职责与目录规划

### 4.1 模块保留原则

代码模块必须先有明确用途，再进入目录规划。不能为了“看起来完整”提前创建空文件或泛用工具箱。每个模块都必须满足：

1. 有清晰输入和输出。
2. 被核心流程、测试或示例直接调用。
3. 不与其他模块职责重叠。
4. 如果当前阶段没有实际用途，就不创建；后续出现真实需求时再加入。

### 4.2 第一阶段保留的 Python 模块

| 模块 | 是否创建 | 实际用途 | 边界 |
| --- | --- | --- | --- |
| `frame_spec.py` | 是 | 定义 `AssistantFrameSpec`，描述 assistant header、assistant end、generation stop 以及 loss inclusion 策略。它是模型差异进入算法的唯一配置入口。 | 不内置某个模型的硬编码分支；Qwen/Gemma 只能作为测试用 spec。 |
| `token_frame_matcher.py` | 是 | 在真实 processor 产出的最终 `input_ids` 上，把 `assistant_header ... assistant_end` 编码成 token ids 后做 frame matching，输出 assistant loss bool mask。 | 不处理 image/audio/video expansion 公式；不使用字符 offset；不匹配裸 payload。 |
| `labels.py` | 是 | 把 frame mask 转成训练 labels：mask 外置 `-100`，mask 内保留 `input_ids`，并用 `attention_mask` 排除 padding。 | 不负责寻找 assistant span；只消费已经构造好的 bool mask。 |
| `validation.py` | 是 | 提供调试和测试断言，例如解码 supervised tokens、确认 header 不进 labels、确认 end token 进 labels、暴露 header/end token ids。 | 不改变训练数据；不修复 serialization/escaping 问题，只报告风险。 |
| `stopping.py` | 是 | 为 RL / rollout / generation 提供 stop-at-token-sequence 工具，确保生成在 `generation_stop` 或 `assistant_end` 本身停止。 | 不做 reward 计算；不等待 `assistant_end + "\n"`；如果第一版不实现 generation/rollout，就不创建该文件。 |

### 4.3 暂不创建的模块

| 模块 | 是否创建 | 原因 | 后续创建条件 |
| --- | --- | --- | --- |
| `templates.py` | 否 | 当前没有独立 Python API 需要它。模板线真正需要的是具体 `.jinja` 文件和 render parity 测试；提前创建 `templates.py` 容易变成无边界的模板工具箱。 | 只有当项目需要一个明确的模板加载/发现 API，例如 `list_templates()` 或 `load_template(name)`，并且有测试和示例直接调用时，才创建。 |

### 4.4 目录规划

建议目录结构：

```text
src/mm_assistant_mask/
  __init__.py
  frame_spec.py
  token_frame_matcher.py
  labels.py
  stopping.py
  validation.py

tests/
  test_token_frame_matcher.py
  test_multiturn.py
  test_no_newline_in_loss.py
  test_padding_and_truncation.py
  test_labels.py
  test_validation.py
  test_stopping.py

examples/
  generic_processor_collator.py

templates/
  qwen/
  gemma/
```

第一阶段只创建实际实现和测试需要的文件，避免空目录、占位模块和没有调用方的抽象。`templates/` 目录只在加入具体 `.jinja` 模板时创建；不会为了未来可能的模板线提前创建 `src/mm_assistant_mask/templates.py`。

## 5. 实施阶段

### 阶段 0：项目骨架确认

目标：

确认当前 repo 的包管理方式、测试框架、Python 版本约束，以及 `makesense` conda 环境中的真实 processor 依赖。

任务：

1. 检查是否已有 `pyproject.toml`、测试依赖和 src layout。
2. 如果没有，新增最小 `pyproject.toml`。
3. 明确包名为 `mm_assistant_mask`。
4. 配置 pytest。
5. 确认 `conda run -n makesense python -m pytest` 是项目标准验证命令。
6. 确认 `makesense` 环境中可加载首批真实 processor/tokenizer：
   - `Qwen/Qwen2.5-Omni-3B`
   - `Qwen/Qwen3-Omni-30B-A3B-Instruct`
   - `google/gemma-4-E2B-it`
7. 确认 processor 测试路径不调用模型加载，不下载或初始化大模型权重。

验收标准：

1. `conda run -n makesense python -m pytest` 能运行。
2. 包可以从本地源码导入。
3. 没有引入不必要依赖。
4. 测试不依赖 fake tokenizer 作为核心验收路径。
5. 首批 processor 至少能完成 processor/tokenizer 加载和最小 chat template/input_ids 构造测试。

### 阶段 1：核心 frame spec

目标：

定义模型无关的 assistant frame 配置。

任务：

1. 新增 `AssistantFrameSpec` dataclass。
2. 字段包含：
   - `assistant_header`
   - `assistant_end`
   - `generation_stop`
   - `include_header_in_loss`
   - `include_end_in_loss`
   - `include_post_end_separator_in_loss`
   - `post_end_separator`
3. 增加轻量 validation，确保 header/end 不为空字符串。

验收标准：

1. 可以表达 Qwen 风格 frame。
2. 可以表达 Gemma 风格 frame。
3. 默认行为符合设计文档：不 mask header，mask end，不 mask end 后 newline。

### 阶段 2：token-id frame matcher

目标：

在最终 processor `input_ids` 上顺序查找 `assistant_header ... assistant_end`，并构造 bool mask。

任务：

1. 实现 tokenizer text 到 token ids 的兼容转换函数。
2. 实现 `find_subsequence`。
3. 实现 `build_assistant_frame_mask_for_one`。
4. 支持多轮 assistant frame。
5. 找到 header 但找不到 end 时抛出明确错误。
6. 默认至少找到一个 frame，否则抛出明确错误。
7. 保持算法只依赖 token ids，不依赖 processor-specific 逻辑。

验收标准：

1. 单轮 text-only frame mask 正确。
2. 多轮 frame mask 正确。
3. 重复 payload 不错位。
4. header 不进 mask。
5. end token 进 mask。
6. end 后 newline 默认不进 mask。

### 阶段 3：labels 构造

目标：

把 bool frame mask 转成训练 labels。

任务：

1. 实现 `build_labels` 或 `build_labels_from_frame_mask`。
2. `labels = input_ids.clone()`。
3. 非 supervised 位置置为 `-100`。
4. 如果提供 `attention_mask`，只保留非 padding 的 supervised token。
5. 校验 tensor shape。

验收标准：

1. mask 外 labels 全是 `-100`。
2. mask 内 labels 等于原始 `input_ids`。
3. padding token 即使 mask 为 True，也会被 attention mask 排除。
4. shape 不匹配时抛出明确错误。

### 阶段 4：validation/debug 工具

目标：

提供调试 label mask 的轻量工具，方便用户确认 supervised token 内容。

任务：

1. 实现 `decode_supervised_tokens`。
2. 实现 `assert_no_header_in_labels`。
3. 实现 `assert_end_in_labels`。
4. 可选实现 user raw frame string 检测，用于提示 serialization/escaping 风险。

验收标准：

1. 可以解码 labels 中非 `-100` token。
2. header 出现在 supervised tokens 时能报错。
3. end token 缺失时能报错。

### 阶段 5：stopping 工具

目标：

为 RL / rollout / generation 提供 stop-at-end-token 工具。

任务：

1. 实现基于 token sequence 的 stopping criteria。
2. 默认使用 `frame_spec.generation_stop`，如果为空则使用 `assistant_end`。
3. 支持 batch 维度下的基础行为。
4. 不等待 `assistant_end + "\n"`。

验收标准：

1. 生成序列末尾出现 stop token sequence 时停止。
2. 仅出现 stop 后 separator newline 不是停止条件的必要部分。
3. stop sequence 为空时抛出明确错误。

### 阶段 6：测试覆盖

目标：

把设计文档中的关键正确性要求转成自动化测试。

优先测试：

1. `Qwen/Qwen2.5-Omni-3B` processor 下的单轮 text-only。
2. `Qwen/Qwen2.5-Omni-3B` processor 下的多轮 text-only。
3. `Qwen/Qwen3-Omni-30B-A3B-Instruct` processor 下的单轮 text-only。
4. `Qwen/Qwen3-Omni-30B-A3B-Instruct` processor 下的多轮 text-only。
5. `google/gemma-4-E2B-it` processor 下的单轮 text-only。
6. `google/gemma-4-E2B-it` processor 下的多轮 text-only。
7. 首批 processor 的最小 image + text mixed user content 测试，前提是对应 processor 支持该输入形态。
8. 首批 processor 的最小 audio + text mixed user content 测试，前提是对应 processor 支持该输入形态。
9. 首批 processor 不加载模型权重，只验证 processor/tokenizer 到最终 `input_ids` 的路径。
10. processor assistant mask 错位复现或等价的真实 processor expansion regression。
11. 重复 assistant payload。
12. header 不进 labels。
13. end token 进 labels。
14. end 后 newline 默认不进 labels。
15. include flags 行为：
   - `include_header_in_loss=True`
   - `include_end_in_loss=False`
   - `include_post_end_separator_in_loss=True`
16. header found but no end 的 truncation 错误。
17. no assistant frame found 的错误。
18. padding + attention mask。
19. labels shape mismatch。

后续测试：

1. video processor 示例，如果目标 processor 支持。
2. 更多模型 processor 覆盖。
3. Jinja training template render parity。

验收标准：

1. `conda run -n makesense python -m pytest` 稳定通过。
2. 核心正确性测试必须经过真实 tokenizer/processor，不使用 fake tokenizer 代替。
3. 首批 processor 测试不得加载大模型权重；如果需要下载 processor/tokenizer 文件，必须在文档中明确缓存和网络依赖。
4. processor 返回的 `assistant_masks` 可以作为 diagnostic 对照，但不能作为通过标准。

### 阶段 7：示例与文档

目标：

让用户能快速理解正确使用路径。

任务：

1. 更新 README，强调：
   - `{% generation %}` 是模板层 span declaration。
   - 多模态训练 labels 应在最终 `input_ids` 上用 frame matching 构造。
2. 新增 generic processor collator 示例。
3. 示例中展示：
   - render chat template
   - processor 构造 batch
   - build frame mask
   - build labels
4. 文档中明确 truncation 样本处理建议：抛错、跳过或重新 tokenize，不能静默训练。

验收标准：

1. README 有最小代码示例。
2. 示例不写死某个模型。
3. 示例能清楚显示用户如何提供 `AssistantFrameSpec`。

### 阶段 8：模板线后续工作

目标：

在核心库稳定后，再维护 training-compatible Jinja templates。

任务：

1. 选择一个 Qwen 风格模板作为首个模板案例。
2. 选择一个 Gemma 风格模板作为第二个模板案例。
3. 测试 render parity。
4. 测试 tokenizer-only assistant mask。
5. 测试 add_generation_prompt。
6. 明确模板线不是多模态 label builder 主路径。

验收标准：

1. 模板渲染与原始模板保持一致，或差异可解释。
2. assistant header 不进 tokenizer-only mask。
3. payload + end token 进 tokenizer-only mask。

## 6. API 草案

```python
from mm_assistant_mask import AssistantFrameSpec
from mm_assistant_mask import build_assistant_frame_mask_for_one
from mm_assistant_mask import build_labels

spec = AssistantFrameSpec(
    assistant_header="<|im_start|>assistant\n",
    assistant_end="<|im_end|>",
    generation_stop="<|im_end|>",
)

mask = build_assistant_frame_mask_for_one(
    input_ids=batch["input_ids"][0].tolist(),
    tokenizer=processor.tokenizer,
    spec=spec,
)

labels = build_labels(
    input_ids=batch["input_ids"],
    masks=frame_masks,
    attention_mask=batch.get("attention_mask"),
)
```

## 7. 风险与处理策略

### 风险 1：tokenizer 对 header/end 的编码与最终 processor input_ids 不一致

处理：

在 validation 中暴露 header/end ids，并提供 decode/debug 工具。文档要求用户用同一个 processor/tokenizer 配置构造 batch 与 frame spec。

### 风险 2：user 内容中原样出现 assistant header/end

处理：

视为 serialization/escaping 风险，不通过弱化 frame matching 解决。可以提供检测工具提醒用户。

### 风险 3：样本被 truncation 截断

处理：

找到 header 但找不到 end 时抛错。训练 collator 可以选择跳过样本或调整 max_length。

### 风险 4：真实 processor 测试依赖较重

处理：

本项目接受真实 processor 测试作为必要成本。核心正确性测试必须在 `makesense` conda 环境中使用真实 tokenizer/processor 执行，首批覆盖 `Qwen/Qwen2.5-Omni-3B`、`Qwen/Qwen3-Omni-30B-A3B-Instruct`、`google/gemma-4-E2B-it`。processor 测试不应加载模型权重；若需要下载 processor/tokenizer 文件，应显式标注缓存路径、网络依赖与跳过条件。不能用 fake tokenizer 替代核心验收。

### 风险 5：模板线扩大范围

处理：

先完成核心 Python mask builder。模板线作为后续独立里程碑，不阻塞主路径。

## 8. 成功标准

第一版成功标准：

1. 用户可以提供任意 `AssistantFrameSpec`。
2. 项目能在最终 `input_ids` 上构造 assistant-only mask。
3. labels 只覆盖 assistant payload 和 assistant end token。
4. 多轮对话与重复 payload 正确。
5. 截断样本不会静默训练。
6. 真实 processor 驱动的测试覆盖核心行为。
7. README 清楚说明为什么不能依赖 multimodal processor 的 `assistant_masks`。

## 9. 推荐执行顺序

1. 建项目骨架和测试入口。
2. 实现 `AssistantFrameSpec`。
3. 实现 token frame matcher。
4. 实现 labels 构造。
5. 写真实 processor 驱动的核心测试。
6. 补 validation/debug 工具。
7. 补 stopping 工具。
8. 写 README 和 generic collator 示例。
9. 再启动 Jinja template 修改线。

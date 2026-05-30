# 多模态 Assistant Mask 设计文档

## 0. 摘要

这个文档总结一个通用的多模态 SFT/RL 训练问题：在包含 image / audio / video 的多模态对话中，如何可靠地构造 **assistant-only loss mask**。

当前问题不应被限定为某个具体模型，例如 Gemma4、Qwen3-VL、Qwen3.6 或 Qwen-Omni。它是一个更普遍的多模态 processor 问题：

> Chat template 可以用 `{% generation %}` / `{% endgeneration %}` 标记 assistant 生成区域；  
> 但当 processor 把 image/audio/video 占位符展开成真实 multimodal 输入后，基于字符 span 和 offset mapping 推断 assistant mask 可能错位。

因此设计分为两条线：

1. **Jinja training chat template 修改线**  
   给模型聊天模板补 `{% generation %}` / `{% endgeneration %}`，使 tokenizer-only / text SFT 能得到正确 `assistant_masks`，并为社区贡献 training-compatible templates。

2. **Python 多模态 assistant label builder 主线**  
   用最终 processor `input_ids` 上的 **token-id frame matching** 构造 SFT/RL 训练用 assistant mask。  
   这条线是主线，不能依赖某一两个模型特例，也不能依赖 processor 返回的 `assistant_masks`。

---

## 1. 背景问题

### 1.1 Assistant-only loss 的目标

在对话 SFT 中，loss 应该只打在 assistant 真正要生成的 token 上：

```text
system/user/context/media prompt      -> labels = -100
assistant generated payload + EOS     -> labels = input_ids
```

对于多轮对话，每个 assistant turn 都应该有一个独立的 supervised span。

例如 Qwen 风格：

```text
<|im_start|>assistant\n
<think>\n\n</think>\n\n<src>...</src><tgt>...</tgt><|im_end|>
```

loss 应该覆盖：

```text
<think>\n\n</think>\n\n<src>...</src><tgt>...</tgt><|im_end|>
```

不应该覆盖：

```text
<|im_start|>assistant\n
```

例如 Gemma4 风格：

```text
<|turn>model\n
<src>...</src><tgt>...</tgt><turn|>
```

loss 应该覆盖：

```text
<src>...</src><tgt>...</tgt><turn|>
```

不应该覆盖：

```text
<|turn>model\n
```

### 1.2 多模态下的核心困难

在 tokenizer-only 路径中，image/audio/video 只是文本占位符，例如：

```text
<|vision_start|><|image_pad|><|vision_end|>
<|audio|>
```

但在 processor 路径中，真实 image/audio/video 会被 processor 处理成模型输入的一部分。最终的 `input_ids` 可能包含展开后的 image pad token、audio token、特殊 media token，或者与 `pixel_values` / audio features 对齐的特殊结构。

因此，下面这条路径不可靠：

```text
Jinja generation char span
→ processor(text + image/audio/video)
→ offset_mapping
→ assistant_masks
```

实验证明：同一个 training template 下，`tokenizer.apply_chat_template(..., return_assistant_tokens_mask=True)` 可能返回正确 mask；但 `processor.apply_chat_template(..., return_assistant_tokens_mask=True)` 在 image 输入下可能把 assistant mask 错映射到 `<|image_pad|>`。

---

## 2. 总体设计原则

### 2.1 不把问题限定在某个模型

这个项目不应该写成：

```text
Gemma4 assistant mask fix
Qwen3-VL assistant mask fix
```

而应该写成：

```text
multimodal assistant-only label masking
```

模型差异只应该体现在配置层，例如：

```python
AssistantFrameSpec(
    assistant_header="<|im_start|>assistant\n",
    assistant_end="<|im_end|>",
)
```

或者：

```python
AssistantFrameSpec(
    assistant_header="<|turn>model\n",
    assistant_end="<turn|>",
)
```

算法本身应对任意多模态模型通用。

### 2.2 Jinja template 和 Python mask builder 分离

Jinja training template 的职责：

```text
声明 assistant-generated span；
保证 tokenizer-only assistant_masks 正确；
方便社区维护 training-compatible templates。
```

Python mask builder 的职责：

```text
在最终 processor input_ids 上构造可靠 label mask；
不依赖 processor 自动返回的 assistant_masks；
不依赖字符 offset；
不需要知道 image/audio/video 展开成多少 token。
```

### 2.3 主线使用 token-id frame matching

不要用裸 payload 匹配：

```text
<src>...</src><tgt>...</tgt>
```

而要匹配完整 assistant frame：

```text
assistant_header
assistant_generated_payload
assistant_end
```

最终在 token id 层做匹配：

```text
assistant_header_ids ... assistant_end_ids
```

mask 范围：

```text
header 之后，到 assistant_end 结束为止
```

默认规则：

```text
不 mask assistant header
mask assistant payload
mask assistant EOS / turn-end token
不 mask EOS 后面的 separator newline
```

---

## 3. 设计线一：Jinja training chat template 修改

### 3.1 目的

给聊天模板加入：

```jinja
{% generation %}
{% endgeneration %}
```

用于标记 assistant 生成内容。

这些 Jinja 标签不是文本 token，不会出现在最终 prompt 中。它们是 template control tags，用于让 tokenizer/processor 在渲染时记录 generation span。

### 3.2 放置原则

核心原则：

```text
assistant role header / generation prompt cue 不放进 generation block；
assistant generated payload + turn end 放进 generation block。
```

通用形式：

```jinja
{{ assistant_header }}
{%- generation -%}
{{ assistant_generated_payload }}
{{ assistant_turn_end }}
{%- endgeneration -%}
```

不要写成：

```jinja
{%- generation -%}
{{ assistant_header }}
{{ assistant_generated_payload }}
{{ assistant_turn_end }}
{%- endgeneration -%}
```

### 3.3 Qwen 风格示例

```jinja
{{- '<|im_start|>assistant\n' }}
{%- generation %}
{{- '<think>\n' + reasoning_content + '\n</think>\n\n' + content }}
{{- '<|im_end|>\n' }}
{%- endgeneration %}
```

这里：

```text
不进 generation:
<|im_start|>assistant\n

进 generation:
<think>...</think>\n\n + content + <|im_end|>\n
```

### 3.4 Gemma 风格示例

```jinja
{{ '<|turn>model\n' }}
{%- generation -%}
{{ assistant_content }}
{{ '<turn|>\n' }}
{%- endgeneration -%}
```

这里：

```text
不进 generation:
<|turn>model\n

进 generation:
assistant_content + <turn|>\n
```

### 3.5 换行符处理

上游 training template 可以为了 render parity 把 EOS 后的换行放在 generation block 中，例如：

```jinja
{{ '<|im_end|>\n' }}
```

或者：

```jinja
{{ '<turn|>\n' }}
```

这对 tokenizer-only `assistant_masks` 是可接受的。

但本项目的 Python 主线不直接照搬这个范围。主线使用 token-id frame matching，默认只 mask 到 EOS / turn-end token 本身，不包含后续 separator newline。

因此：

```text
template 层:
    可以包含 <EOS>\n，以匹配现有 TRL training template 风格。

Python label builder 层:
    assistant_end = "<EOS>"，不包含 "\n"。
```

### 3.6 Template 修改线的测试

每个 template 至少测试：

1. **Render parity**  
   training template 在 `tokenize=False` 下渲染文本应与原模板一致，或者差异必须可解释。

2. **Header 不进 mask**  
   tokenizer 解码 `assistant_masks == 1` 的 token，不应包含 assistant header。

3. **Payload + EOS 进 mask**  
   应包含 assistant content 和 `<|im_end|>` / `<turn|>` 等 turn end token。

4. **多轮对话**  
   每个 assistant turn 都应产生独立 mask span。

5. **thinking/tool-call 场景**  
   如果模型模板支持 `<think>`、tool calls、tool responses，应保证这些属于 assistant 输出的内容被正确包进 generation block。

6. **add_generation_prompt**  
   `add_generation_prompt=True` 时应只渲染 assistant header / prompt cue，不应渲染 assistant payload。

---

## 4. 设计线二：Python 多模态 Assistant Label Builder 主线

### 4.1 主线目标

为 SFT / RL / preference training 等训练流程提供可靠 assistant mask：

```text
输入:
    processor
    rendered_text 或 messages
    final processor input_ids
    AssistantFrameSpec

输出:
    label_mask 或 labels
```

其中 `label_mask=True` 的位置表示应参与 assistant loss。

### 4.2 为什么不使用 processor assistant_masks

不直接信任：

```python
processor.apply_chat_template(
    messages,
    tokenize=True,
    return_dict=True,
    return_assistant_tokens_mask=True,
)
```

原因：

```text
processor 生成 assistant_masks 时通常依赖:
    Jinja generation char span + offset_mapping

多模态输入展开后:
    char span 不再可靠对应 final input_ids token index
```

因此 processor 返回的 `assistant_masks` 可以作为 diagnostic 信息，但不能作为多模态训练主路径。

### 4.3 核心抽象：AssistantFrameSpec

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class AssistantFrameSpec:
    assistant_header: str
    assistant_end: str
    generation_stop: str | None = None

    include_header_in_loss: bool = False
    include_end_in_loss: bool = True

    # 默认不包含 EOS 后面的 separator newline。
    # 如果某些训练目标确实需要包含换行，可显式打开。
    include_post_end_separator_in_loss: bool = False
    post_end_separator: str = "\n"
```

示例：

```python
QWEN_IM_FRAME = AssistantFrameSpec(
    assistant_header="<|im_start|>assistant\n",
    assistant_end="<|im_end|>",
    generation_stop="<|im_end|>",
)
```

```python
GEMMA4_TURN_FRAME = AssistantFrameSpec(
    assistant_header="<|turn>model\n",
    assistant_end="<turn|>",
    generation_stop="<turn|>",
)
```

注意：这些只是示例。repo 不应该内置只支持 Qwen/Gemma。实际项目应允许用户提供任意 frame spec，或从 model card / tokenizer config / rendered template 中推断。

### 4.4 Token-id frame matching 算法

#### 输入

```python
input_ids: list[int]
tokenizer
frame_spec: AssistantFrameSpec
```

#### 步骤

1. 将 assistant header 编码成 token id：

```python
header_ids = tokenizer(
    frame_spec.assistant_header,
    add_special_tokens=False,
).input_ids
```

2. 将 assistant end 编码成 token id：

```python
end_ids = tokenizer(
    frame_spec.assistant_end,
    add_special_tokens=False,
).input_ids
```

3. 在最终 processor `input_ids` 中顺序查找：

```text
header_ids ... end_ids
```

4. mask 范围：

```text
payload_start = header_start + len(header_ids)
payload_end = end_start + len(end_ids)
```

5. 标记：

```text
mask[payload_start:payload_end] = True
```

6. 从 `payload_end` 之后继续查找下一轮 assistant turn。

#### 伪代码

```python
def find_subsequence(seq: list[int], pat: list[int], start: int = 0) -> int:
    if not pat:
        raise ValueError("empty pattern")
    limit = len(seq) - len(pat)
    for i in range(start, limit + 1):
        if seq[i : i + len(pat)] == pat:
            return i
    return -1


def build_assistant_frame_mask(
    input_ids: list[int],
    tokenizer,
    spec: AssistantFrameSpec,
) -> list[bool]:
    header_ids = tokenizer(spec.assistant_header, add_special_tokens=False).input_ids
    end_ids = tokenizer(spec.assistant_end, add_special_tokens=False).input_ids

    if not header_ids:
        raise ValueError("assistant_header encodes to empty token sequence")
    if not end_ids:
        raise ValueError("assistant_end encodes to empty token sequence")

    mask = [False] * len(input_ids)
    cursor = 0

    while True:
        header_start = find_subsequence(input_ids, header_ids, cursor)
        if header_start < 0:
            break

        payload_start = header_start
        if not spec.include_header_in_loss:
            payload_start = header_start + len(header_ids)

        end_start = find_subsequence(input_ids, end_ids, header_start + len(header_ids))
        if end_start < 0:
            raise ValueError("assistant header found but no following assistant end")

        end_end = end_start + len(end_ids)

        if spec.include_end_in_loss:
            payload_end = end_end
        else:
            payload_end = end_start

        for i in range(payload_start, payload_end):
            mask[i] = True

        cursor = end_end

    return mask
```

### 4.5 Batch labels 构造

```python
def labels_from_frame_mask(input_ids: torch.Tensor, frame_masks: torch.Tensor) -> torch.Tensor:
    labels = input_ids.clone()
    labels[~frame_masks] = -100
    return labels
```

如果有 `attention_mask`：

```python
frame_masks = frame_masks & attention_mask.bool()
```

### 4.6 SFT collator 使用流程

```python
# 1. 用 processor 渲染最终 prompt 文本
rendered = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=False,
    chat_template=training_template_or_default,
    **template_kwargs,
)

# 2. 调用 processor 构造最终多模态 input_ids
batch = processor(
    text=[rendered],
    images=images,
    audio=audios,
    videos=videos,
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=max_length,
)

# 3. 在最终 input_ids 上做 assistant frame token-id matching
mask = build_assistant_frame_mask(
    input_ids=batch["input_ids"][0].tolist(),
    tokenizer=processor.tokenizer,
    spec=frame_spec,
)

# 4. 构造 labels
labels = batch["input_ids"].clone()
labels[0, ~torch.tensor(mask, dtype=torch.bool)] = -100
batch["labels"] = labels
```

### 4.7 多轮对话支持

多轮对话天然支持，因为算法会顺序查找多个 assistant frame：

```text
assistant_header ... assistant_end
assistant_header ... assistant_end
assistant_header ... assistant_end
```

每个 frame 的 payload 都会被 mask。

这不是单轮 prompt-completion 机制。

### 4.8 重复内容支持

如果两个 assistant turn 内容相同，例如：

```text
<src><|wait|></src><tgt><|wait|></tgt>
<src><|wait|></src><tgt><|wait|></tgt>
```

不会产生问题，因为算法不是匹配裸 payload，而是匹配 frame：

```text
assistant_header ... assistant_end
```

并且按 cursor 顺序推进。

### 4.9 User 内容中出现 header/end token

如果 user content 能形成完整 assistant frame，例如：

```text
<|im_start|>assistant\n...<|im_end|>
```

并且 tokenizer 把它当成真正 special token frame，那么这是 template / serialization / escaping 问题，而不是 mask builder 的“误伤”。

本项目可以提供检测：

```text
user text contains raw assistant header/end special strings
```

但不应通过弱化 assistant frame matching 来规避这个问题。

### 4.10 换行符策略

主线默认：

```text
mask 到 assistant_end token 结束；
不 mask assistant_end 后面的 newline。
```

原因：

```text
模型应该学习在 <EOS> / <turn-end> 本身停止；
newline 是 turn separator，不是停止语义。
```

推理时也应在 `generation_stop` 出现时立即停止：

```python
generation_stop = "<|im_end|>"  # or "<turn|>"
```

而不是等待：

```text
<|im_end|>\n
<turn|>\n
```

---

## 5. RL 使用场景

虽然这个设计最先用于 SFT labels，但同样适用于 RL / preference training。

### 5.1 Rollout response extraction

从生成结果中提取 assistant response：

```text
assistant_header
response
assistant_end
```

提取 `response` 时：

```text
去掉 assistant_header
保留或去掉 assistant_end 取决于 reward 设计
默认 stop 在 assistant_end
```

### 5.2 Reward 计算

对于严格格式任务，例如 streaming ASR/translation：

```text
<src>...</src><tgt>...</tgt>
```

reward 应只基于 assistant payload，不基于 role header，也不基于 separator newline。

### 5.3 RL generation stop

所有 rollout 都必须配置 stop condition：

```text
stop at assistant_end / generation_stop
```

否则模型可能在当前 assistant turn 结束后继续生成下一轮 user/model/thought/template 内容。

---

## 6. 与 Transformers / TRL 的关系

### 6.1 Template 线可以 upstream

training template 修改可以贡献到 TRL / Transformers：

```text
添加或修正模型的 training-compatible jinja template
保证 prefix-preserving
添加 {% generation %} / {% endgeneration %}
```

### 6.2 Processor mask 问题需要 upstream 支持

更彻底的 upstream 方向是：

```text
processor 暴露多模态 placeholder expansion map
或者提供 media-expansion-aware assistant mask
或者用 token-level alignment 代替 char span + offset_mapping
```

但这些需要 Transformers 社区和上游模型 provider 共同配合。外部库不应维护每个 processor 的私有 image/audio/video expansion 公式。

### 6.3 本项目主线不依赖 expansion formula

本项目不尝试推断：

```text
一张图展开成多少 image_pad token
一段音频展开成多少 audio token
一个视频采样多少帧、每帧多少 token
```

而是在最终 `input_ids` 上直接匹配 assistant frame token。

---

## 7. Repo 设计建议

### 7.1 目录结构

```text
mm-assistant-mask/
  README.md
  pyproject.toml
  src/mm_assistant_mask/
    __init__.py
    frame_spec.py
    token_frame_matcher.py
    labels.py
    stopping.py
    validation.py
    templates.py
  templates/
    qwen/
      qwen3_6_training.jinja
      qwen3_vl_training.jinja
    gemma/
      gemma4_training.jinja
  examples/
    qwen3vl_image_sft_mask.py
    gemma4_audio_sft_mask.py
    generic_processor_collator.py
  tests/
    test_token_frame_matcher.py
    test_multiturn.py
    test_no_newline_in_loss.py
    test_processor_assistant_mask_failure.py
    test_render_parity.py
    test_padding_and_truncation.py
```

### 7.2 核心模块

#### `frame_spec.py`

```python
@dataclass(frozen=True)
class AssistantFrameSpec:
    assistant_header: str
    assistant_end: str
    generation_stop: str | None = None
    include_header_in_loss: bool = False
    include_end_in_loss: bool = True
    include_post_end_separator_in_loss: bool = False
    post_end_separator: str = "\n"
```

#### `token_frame_matcher.py`

```python
def build_assistant_frame_mask(
    input_ids: list[int],
    tokenizer,
    spec: AssistantFrameSpec,
) -> list[bool]:
    ...
```

#### `labels.py`

```python
def build_labels_from_frame_mask(
    input_ids: torch.Tensor,
    frame_mask: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    ...
```

#### `stopping.py`

```python
class StopOnTokenSequence(StoppingCriteria):
    ...
```

#### `validation.py`

```python
def decode_supervised_tokens(tokenizer, input_ids, labels) -> str:
    ...

def assert_no_header_in_labels(decoded: str, spec: AssistantFrameSpec) -> None:
    ...
```

---

## 8. 测试计划

### 8.1 基础测试

1. 单轮 text-only。
2. 多轮 text-only。
3. 多轮 image。
4. 多轮 audio。
5. image + text mixed user content。
6. audio + text mixed user content。
7. video，如果目标 processor 支持。

### 8.2 关键正确性测试

#### Header 不进 labels

```text
decoded(labels != -100)
```

不得包含：

```text
<|im_start|>assistant\n
<|turn>model\n
<start_of_turn>model\n
```

#### End token 进 labels

必须包含：

```text
<|im_end|>
<turn|>
<end_of_turn>
```

#### End 后 newline 默认不进 labels

如果最终模板渲染：

```text
<|im_end|>\n
```

labels 默认只覆盖到：

```text
<|im_end|>
```

#### 多轮全部覆盖

每个 assistant turn 的 payload 都应进入 labels。

#### 重复 payload 不错位

多个 assistant turn 内容相同时，仍应正确按 frame 顺序定位。

#### Processor assistant_masks 错位复现

建立 regression test：

```text
processor.apply_chat_template(... image ..., return_assistant_tokens_mask=True)
```

可能 mask 到 media token；本项目 frame matcher 应正确 mask assistant payload。

### 8.3 Truncation 测试

如果 assistant frame 被截断：

```text
找到 header 但找不到 end
```

应抛错或跳过样本，不能静默训练。

### 8.4 Padding 测试

对 batch 内不同长度样本，mask 应只作用于非 padding token。

### 8.5 Generation stop 测试

模型生成时必须在：

```text
generation_stop
```

出现时停止，而不是等 `generation_stop + "\n"`。

---

## 9. 不做什么

本项目不做：

1. 维护每个模型 processor 的 image/audio/video expansion 公式。
2. 依赖 processor 返回的 `assistant_masks` 作为多模态训练主路径。
3. 针对某一个模型写死 mask 逻辑。
4. 用自然语言黑名单过滤，例如 `"Thinking Process"`、`"assistant"`、`"user"`。
5. 匹配裸 payload，例如只匹配 `<src>...</src><tgt>...</tgt>`。

---

## 10. 当前决策记录

### Decision 1

Jinja training template 修改值得做，但它不是多模态训练 label mask 的主路径。

### Decision 2

多模态 SFT/RL 的主线是：

```text
final processor input_ids
→ token-id assistant frame matching
→ assistant-only labels
```

### Decision 3

Assistant EOS / turn-end token 必须进 loss。

### Decision 4

EOS / turn-end 后面的 separator newline 默认不进 loss。

### Decision 5

Generation / rollout 必须在 EOS / turn-end token 本身停止。

### Decision 6

这个问题是通用多模态问题，不是 Gemma4 或 Qwen 的个别问题。

---

## 11. 最小可用实现草案

```python
from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass(frozen=True)
class AssistantFrameSpec:
    assistant_header: str
    assistant_end: str
    generation_stop: str | None = None
    include_header_in_loss: bool = False
    include_end_in_loss: bool = True


def _to_ids(tokenizer, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    if isinstance(ids, torch.Tensor):
        return [int(x) for x in ids.view(-1).tolist()]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(x) for x in ids]


def find_subsequence(seq: Sequence[int], pat: Sequence[int], start: int = 0) -> int:
    if not pat:
        raise ValueError("pattern must not be empty")
    limit = len(seq) - len(pat)
    for i in range(start, limit + 1):
        if list(seq[i : i + len(pat)]) == list(pat):
            return i
    return -1


def build_assistant_frame_mask_for_one(
    input_ids: Sequence[int],
    tokenizer,
    spec: AssistantFrameSpec,
) -> torch.BoolTensor:
    header_ids = _to_ids(tokenizer, spec.assistant_header)
    end_ids = _to_ids(tokenizer, spec.assistant_end)

    if not header_ids:
        raise ValueError("assistant_header encodes to an empty sequence")
    if not end_ids:
        raise ValueError("assistant_end encodes to an empty sequence")

    mask = torch.zeros(len(input_ids), dtype=torch.bool)
    cursor = 0
    found = 0

    while True:
        header_start = find_subsequence(input_ids, header_ids, cursor)
        if header_start < 0:
            break

        content_start = header_start if spec.include_header_in_loss else header_start + len(header_ids)

        end_start = find_subsequence(input_ids, end_ids, header_start + len(header_ids))
        if end_start < 0:
            raise ValueError("assistant header found but no following assistant end")

        end_end = end_start + len(end_ids)
        content_end = end_end if spec.include_end_in_loss else end_start

        if content_end > content_start:
            mask[content_start:content_end] = True
            found += 1

        cursor = end_end

    if found == 0:
        raise ValueError("no assistant frame found in input_ids")

    return mask


def build_labels(
    input_ids: torch.Tensor,
    masks: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if masks.shape != input_ids.shape:
        raise ValueError(f"mask shape {masks.shape} != input_ids shape {input_ids.shape}")

    supervised = masks.bool()
    if attention_mask is not None:
        supervised = supervised & attention_mask.bool()

    labels = input_ids.clone()
    labels[~supervised] = -100
    return labels
```

---

## 12. README 中应强调的一句话

> `{% generation %}` solves assistant span declaration at the template level.  
> It does not guarantee correct assistant token masks after multimodal processor expansion.  
> For multimodal SFT/RL, labels should be built on final processor `input_ids` using assistant-frame token matching.

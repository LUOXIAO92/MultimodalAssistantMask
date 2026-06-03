# MultimodalAssistantMask

MultimodalAssistantMask 用来为多模态和纯文本 SFT/RL 构造 label mask，让 loss 只落在 assistant 回复上。核心思路是：不要依赖模板层的字符区间，而是在 processor 产出的最终 `input_ids` 上构造 assistant mask。

```text
final processor input_ids
-> token-id assistant mask matching
-> assistant-only labels
```

多模态训练时，建议先用 processor 默认 chat template 渲染对话，再基于最终 `input_ids` 构造 labels。processor 返回的 `assistant_masks` 可以用于检查，但不要拿它来生成训练 labels，因为 image、audio、video 占位符在 processor 里展开以后，模板层记录的 assistant mask 可能已经对不齐了。

chat template 里的 `{% generation %}` 仍然有用：它可以在模板层标出 assistant 生成区间，适合用来验证纯文本模板的 `assistant_masks`。只是它解决的是模板层的问题，不等于解决了多模态 processor 展开后的 label 对齐问题。

token-id assistant mask 不只适用于图像、音频或视频。只要你希望以最终 token 化后的 `input_ids` 为准来构造 mask，纯文本训练也可以使用同一套方法。和 Jinja `assistant_masks` 不同，默认情况下这里不会把 assistant end/eos token 后面的分隔换行放进 loss；如果确实需要，可以在 `AssistantMaskSpec` 里设置 `include_post_end_separator_in_loss=True`。

## 检查模板实际渲染结果

为某个模型选择 `assistant_header` 和 `assistant_end` 时，最稳妥的方法是渲染一条带 marker 的极简对话，直接查看 chat template 实际吐出的字符串。Qwen3 instruct 这类模板可能会在 assistant header 和 payload 中间插入空 thinking 标签，需要留意。

```python
from transformers import AutoProcessor

model_id = "Qwen/Qwen2.5-Omni-3B"  # replace with your target model

processor = AutoProcessor.from_pretrained(
    model_id,
    local_files_only=True,
    trust_remote_code=True,
)

user_marker = "USER_PROMPT_MARKER"
assistant_marker = "ASSISTANT_PAYLOAD_MARKER"
messages = [
    {"role": "user", "content": user_marker},
    {"role": "assistant", "content": assistant_marker},
]

rendered = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=False,
)
rendered_without_assistant = processor.apply_chat_template(
    messages[:1],
    tokenize=False,
    add_generation_prompt=False,
)

assistant_start = rendered.index(assistant_marker)
assistant_end_start = assistant_start + len(assistant_marker)
before = rendered[len(rendered_without_assistant):assistant_start]
after = rendered[assistant_end_start:]

print("rendered =", repr(rendered))
print("rendered_without_assistant =", repr(rendered_without_assistant))
print("before =", repr(before))
print("after =", repr(after))
```

## 最小用法

```python
from mm_assistant_mask import (
    AssistantMaskSpec,
    build_assistant_labels,
)

spec = AssistantMaskSpec(
    assistant_header="<|im_start|>assistant\n",
    assistant_end="<|im_end|>",
    generation_stop="<|im_end|>",
)

rendered = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=False,
)

batch = processor(
    text=[rendered],
    images=images,
    audio=audios,
    videos=videos,
    return_tensors="pt",
    padding=True,
    truncation=True,
)

batch["labels"] = build_assistant_labels(
    batch,
    tokenizer=processor.tokenizer,
    spec=spec,
)
```

默认只给 assistant payload 和 assistant end token 计算 loss，不给 assistant header 计算 loss。assistant end token 后面的分隔换行默认不放进 labels。如果 processor 返回了 `attention_mask`，padding token 会在内部自动忽略。

如果 truncation 只留下 assistant header，却把对应的 assistant end 截掉了，mask builder 会直接报错。训练代码应该跳过这条样本，或者调大 tokenization 长度限制；不要在不完整的 assistant 回复上继续训练。

通用 collator 示例见 `examples/generic_processor_collator.py`。

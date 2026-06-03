# MultimodalAssistantMask

MultimodalAssistantMask は、マルチモーダル/テキストのみの SFT や RL で、assistant の応答部分だけに loss をかけるための label mask を作るライブラリです。基本的には、processor が出力した最終的な `input_ids` を見て、assistant mask を token id レベルで作ります。

```text
final processor input_ids
-> token-id assistant mask matching
-> assistant-only labels
```

chat template の `{% generation %}` を使うと、assistant が生成する範囲をテンプレート側で示せます。テキストのみの template を確認する用途では便利です。ただし、image、audio、video のプレースホルダーが processor によって展開されたあとも、その mask が正しいとは限りません。マルチモーダル学習では、processor のデフォルト chat template でレンダリングしたあと、最終的な `input_ids` を見て labels を作ります。processor が返す `assistant_masks` は確認用にとどめ、学習用 labels の作成には使わないでください。

token-id assistant mask は、画像・音声・動画がある場合だけの仕組みではありません。最終的な `input_ids` を基準に mask を作りたいなら、テキストのみの学習でも同じ方法を使えます。Jinja の `assistant_masks` と違い、このライブラリの標準設定では assistant end/eos token の直後にある区切り用の改行には loss をかけません。必要な場合だけ、`AssistantMaskSpec` で `include_post_end_separator_in_loss=True` を指定してください。

## レンダリング結果を確認する

モデルごとの `assistant_header` と `assistant_end` を決めるときは、短い会話を実際にレンダリングして、chat template が出す文字列を確認するのが確実です。分かりやすい marker を入れておくと、assistant payload の前後を切り出しやすくなります。Qwen3 instruct 系では、assistant header と payload の間に空の thinking cue が入ることがあります。

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

## 最小例

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

デフォルトでは、assistant header には loss をかけません。assistant payload と assistant end token には loss をかけます。assistant end token の直後にある区切り用の改行は、標準では labels に含めません。processor が `attention_mask` を返す場合、padding token は内部で無視されます。

truncation によって assistant header だけが残り、対応する assistant end が消えてしまった場合、mask builder はエラーを出します。そのサンプルはスキップするか、tokenization の長さ制限を調整してください。途中で切れた assistant response をそのまま学習に使うのは避けてください。

汎用 collator の例は `examples/generic_processor_collator.py` を参照してください。

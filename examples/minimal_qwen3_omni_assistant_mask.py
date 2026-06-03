import sys
sys.path.append("src")

from pathlib import Path
from transformers.audio_utils import load_audio
from transformers.image_utils import load_image
from transformers import AutoProcessor

from mm_assistant_mask import (
    AssistantMaskSpec,
    build_assistant_labels,
)

MODEL_ID = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
TEMPLATE_PATH = "templates/qwen/qwen3-omni-30b-a3b-instruct-train.jinja"
IMAGE_URL = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/demo/cars.jpg"
AUDIO_URL = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/demo/cough.wav"
USER_TEXT = "What can you see and hear? Answer in one short sentence."
ASSISTANT_RESPONSE_1 = "I see cars and hear someone coughing."
FOLLOW_UP_TEXT = "Repeat the answer as a text-only follow-up."
ASSISTANT_RESPONSE_2 = "Cars are visible, and coughing is audible."
EXPECTED_MULTIMODAL_SUPERVISED_TEXT = (
    "I see cars and hear someone coughing.<|im_end|>"
    "Cars are visible, and coughing is audible.<|im_end|>"
)
EXPECTED_TEXT_SUPERVISED_TEXT = (
    "I see cars and hear someone coughing.<|im_end|>\n"
    "Cars are visible, and coughing is audible.<|im_end|>"
)


processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    local_files_only=True,
    trust_remote_code=True,
)
template = Path(TEMPLATE_PATH).read_text()

print(f"Expected multimodal supervised text: {EXPECTED_MULTIMODAL_SUPERVISED_TEXT!r}")
print(f"Expected text-template supervised text: {EXPECTED_TEXT_SUPERVISED_TEXT!r}")

# Multimodal path: build labels from the final processor input_ids.
# This multi-turn dialog checks that every assistant turn is supervised, not
# just the first answer after the image/audio user message.
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": IMAGE_URL},
            {"type": "audio", "audio": AUDIO_URL},
            {"type": "text", "text": USER_TEXT},
        ],
    },
    {"role": "assistant", "content": ASSISTANT_RESPONSE_1},
    {"role": "user", "content": FOLLOW_UP_TEXT},
    {"role": "assistant", "content": ASSISTANT_RESPONSE_2},
]

# Multimodal training does not use the train Jinja template. Use the
# processor's own chat template, then build labels from final input_ids.
rendered = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=False,
)
batch = processor(
    text=[rendered],
    images=[load_image(IMAGE_URL)],
    audio=[load_audio(AUDIO_URL)],
    return_tensors="pt",
)

# Build assistant-only labels from the final processor input_ids.
spec = AssistantMaskSpec(
    assistant_header="<|im_start|>assistant\n",
    assistant_end="<|im_end|>",
    generation_stop="<|im_end|>",
    excluded_assistant_prefixes=("<think>\n\n</think>\n\n",),
)
labels = build_assistant_labels(
    batch,
    tokenizer=processor.tokenizer,
    spec=spec,
)

# Verify the supervised labels decode to exactly the fixed assistant target.
supervised_ids = batch["input_ids"][0][labels[0] != -100]
supervised_text = processor.tokenizer.decode(
    supervised_ids.tolist(),
    skip_special_tokens=False,
)
print("Multimodal decoded supervised text:", repr(supervised_text))
assert supervised_text == EXPECTED_MULTIMODAL_SUPERVISED_TEXT

# Text-only template path: this is where the train Jinja template is used.
messages = [
    {"role": "user", "content": USER_TEXT},
    {"role": "assistant", "content": ASSISTANT_RESPONSE_1},
    {"role": "user", "content": FOLLOW_UP_TEXT},
    {"role": "assistant", "content": ASSISTANT_RESPONSE_2},
]
encoded = processor.tokenizer.apply_chat_template(
    messages,
    chat_template=template,
    tokenize=True,
    return_dict=True,
    return_assistant_tokens_mask=True,
    add_generation_prompt=False,
)
supervised_ids = [
    token_id
    for token_id, mask in zip(encoded["input_ids"], encoded["assistant_masks"])
    if mask
]

# Verify tokenizer assistant_masks supervise the same fixed assistant target.
supervised_text = processor.tokenizer.decode(supervised_ids, skip_special_tokens=False)
print("Text-template decoded supervised text:", repr(supervised_text))
assert supervised_text.rstrip("\n") == EXPECTED_TEXT_SUPERVISED_TEXT

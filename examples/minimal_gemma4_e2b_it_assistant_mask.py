import sys
sys.path.append("src")

from pathlib import Path
from transformers.audio_utils import load_audio
from transformers.image_utils import load_image
from transformers import AutoProcessor

from mm_assistant_mask import (
    AssistantFrameSpec,
    build_assistant_frame_masks,
    build_labels_from_frame_mask,
)

MODEL_ID = "google/gemma-4-E2B-it"
TEMPLATE_PATH = "templates/gemma/gemma-4-e2b-it-train.jinja"
IMAGE_URL = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/demo/cars.jpg"
AUDIO_URL = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/demo/cough.wav"
USER_TEXT = "What can you see and hear? Answer in one short sentence."
ASSISTANT_RESPONSE_1 = "I see cars and hear someone coughing."
FOLLOW_UP_TEXT = "Repeat the answer as a text-only follow-up."
ASSISTANT_RESPONSE_2 = "Cars are visible, and coughing is audible."
EXPECTED_MULTIMODAL_SUPERVISED_TEXT = (
    "I see cars and hear someone coughing.<turn|>"
    "Cars are visible, and coughing is audible.<turn|>"
)
EXPECTED_TEXT_SUPERVISED_TEXT = (
    "I see cars and hear someone coughing.<turn|>\n"
    "Cars are visible, and coughing is audible.<turn|>"
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

# Build the training mask on final input_ids, then convert it to labels.
frame_spec = AssistantFrameSpec(
    assistant_header="<|turn>model\n",
    assistant_end="<turn|>",
    generation_stop="<turn|>",
)
frame_masks = build_assistant_frame_masks(
    input_ids=batch["input_ids"],
    tokenizer=processor.tokenizer,
    spec=frame_spec,
)
labels = build_labels_from_frame_mask(
    input_ids=batch["input_ids"],
    frame_mask=frame_masks,
    attention_mask=batch.get("attention_mask"),
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

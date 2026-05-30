# Minimal Assistant Mask Usage Examples Plan

## 0. 基准文件

本子任务以以下文件为基准：

1. `docs/multimodal_assistant_mask_design.md`
2. `plans/multimodal_assistant_mask_implementation_plan.md`
3. `progress/multimodal_assistant_mask_progress.md`

本计划取代已废弃的 `plans/real_multimodal_usage_example_plan.md`。

## 1. 目标

重新实现 examples，使它们成为 SFT/RL 用户能直接复制到 notebook cell 或训练脚本里的最小用例。

示例只回答两个问题：

1. 多模态输入：如何从 final processor `input_ids` 构造 assistant-only labels。
2. 纯文本输入：如何用 training chat template 的 tokenizer-only `assistant_masks` 验证 `{% generation %}` 是否标对。

## 2. 核心原则

1. 不新增共享配置模块。
2. 不新增 dataclass、case registry、runner、framework、generic helper 层。
3. 不用循环隐藏每个模型的关键代码路径。
4. 允许少量重复代码。重复比抽象更适合作为用户示例。
5. 每个示例必须把以下步骤直接写出来：
   - 构造 `messages`
   - 加载 processor/tokenizer
   - 读取 training template
   - 渲染 prompt
   - 构造 processor batch 或 tokenizer encoding
   - 构造 mask / labels
   - 解码 supervised text
   - 与 expected supervised text 比较
6. 多模态示例不得使用 processor 返回的 `assistant_masks` 作为训练主路径。
7. 纯文本示例可以使用 tokenizer-only `assistant_masks`，但必须明确它只用于模板线验证。

## 3. 示例文件形态

新增三个最小脚本，一个脚本对应一个模型：

1. `examples/minimal_qwen2_5_omni_assistant_mask.py`
2. `examples/minimal_qwen3_omni_assistant_mask.py`
3. `examples/minimal_gemma4_e2b_it_assistant_mask.py`

每个脚本内部包含两个顺序 section：

1. 多模态使用例子：final processor `input_ids` + frame matching + labels。
2. 纯文本使用例子：training template + tokenizer-only `assistant_masks`。

每个脚本必须在文件顶部直接写出该模型的 fixed assistant response 和 expected supervised text。两个 section 都复用这个 expected supervised text 作观测参考。

每个 section 直接写完整代码，不通过共享 case 表、runner、配置模块或循环分发。

## 4. 共同输入

三个 processor 使用同一份 user 输入。

多模态 user input：

```python
{
    "role": "user",
    "content": [
        {
            "type": "image",
            "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/demo/cars.jpg",
        },
        {
            "type": "audio",
            "audio": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/demo/cough.wav",
        },
        {
            "type": "text",
            "text": "What can you see and hear? Answer in one short sentence.",
        },
    ],
}
```

纯文本 user input：

```python
{
    "role": "user",
    "content": "What can you see and hear? Answer in one short sentence.",
}
```

固定 assistant responses：

```python
{"role": "assistant", "content": "I see cars and hear someone coughing."}
{"role": "user", "content": "Repeat the answer as a text-only follow-up."}
{"role": "assistant", "content": "Cars are visible, and coughing is audible."}
```

assistant end token 由 chat template 渲染，示例数据本身不手写 `<|im_end|>` 或 `<turn|>`。

## 5. 每个模型脚本中的 expected supervised text

示例必须在每个模型脚本顶部直接写出 expected supervised text，并在多模态 section 和纯文本 section 都打印 actual decoded supervised text。

Qwen2.5-Omni 多模态 frame matching：

```text
I see cars and hear someone coughing.<|im_end|>Cars are visible, and coughing is audible.<|im_end|>
```

Qwen2.5-Omni 纯文本 template assistant mask：

```text
I see cars and hear someone coughing.<|im_end|>
Cars are visible, and coughing is audible.<|im_end|>
```

Qwen3-Omni 多模态 frame matching：

```text
I see cars and hear someone coughing.<|im_end|>Cars are visible, and coughing is audible.<|im_end|>
```

Qwen3-Omni 纯文本 template assistant mask：

```text
I see cars and hear someone coughing.<|im_end|>
Cars are visible, and coughing is audible.<|im_end|>
```

Gemma 4 E2B it 多模态 frame matching：

```text
I see cars and hear someone coughing.<turn|>Cars are visible, and coughing is audible.<turn|>
```

Gemma 4 E2B it 纯文本 template assistant mask：

```text
I see cars and hear someone coughing.<turn|>
Cars are visible, and coughing is audible.<turn|>
```

如果 tokenizer-only text template mask 包含 assistant end 后 separator newline，验证时只允许这一种差异，例如比较 `actual.rstrip("\n") == expected`。多模态 frame mask 默认不应包含 end 后 separator newline。

## 6. 每个脚本的多模态使用例子要求

每个模型脚本的多模态 section 必须展示本项目核心路径：

```text
processor final input_ids
-> build_assistant_frame_masks
-> build_labels_from_frame_mask
-> decoded supervised text
```

每个模型脚本必须直接写出：

1. `AutoProcessor.from_pretrained(..., local_files_only=True, trust_remote_code=True)`
2. processor 默认 `apply_chat_template` 渲染多模态 messages
3. 对应 `AssistantFrameSpec`
4. 同一份 image/audio/text messages
5. processor 调用
6. `build_assistant_frame_masks`
7. `build_labels_from_frame_mask`
8. 解码 `labels != -100` 的 token
9. 打印并断言 expected supervised text

如果某个 processor 不支持同一份 image/audio/text mixed input，该 section 可以捕获异常并打印明确原因，但不能删掉该模型、删掉模态或改用 fake processor。

多模态示例默认直接使用用户给出的真实 URL。无法联网时，用户可自行把 URL 改成本地图片/音频路径；示例中不增加联网/非联网分支。

多模态 section 不使用训练 Jinja 模板。训练 Jinja 模板只用于纯文本 section 的 tokenizer-only assistant mask 验证。

## 7. 每个脚本的纯文本使用例子要求

每个模型脚本的纯文本 section 必须展示模板线：

```text
tokenizer.apply_chat_template(..., return_assistant_tokens_mask=True)
-> decoded assistant mask text
```

每个模型脚本必须直接写出：

1. `AutoProcessor.from_pretrained(..., local_files_only=True, trust_remote_code=True)`
2. 对应 template path
3. 同一份 text-only messages
4. tokenizer-only `apply_chat_template`
5. 解码 `assistant_masks == 1` 的 token
6. 打印并断言 expected supervised text

该示例必须显式说明它不用于多模态训练 label 主路径。

## 8. 旧实现处理

旧方案引入的抽象文件不再作为验收依据：

1. `examples/real_multimodal_example_config.py`
2. `examples/real_multimodal_processor_masks.py`
3. `examples/text_template_assistant_masks.py`
4. 旧方案对应的配置完整性测试

用户会在后续移除旧的 `real_multimodal*` 文件；本计划实施时不依赖它们，也不继续扩展它们。

保留或改写测试时，测试目标应变成：

1. 三个最小模型脚本可以编译。
2. 测试确认示例没有环境变量跳过分支、下载 helper 或额外抽象层。
3. 标准 pytest 不联网运行示例；真实多模态路径由用户直接运行脚本验证。

## 9. 验证

标准验证：

```bash
conda run -n makesense python -m py_compile examples/minimal_qwen2_5_omni_assistant_mask.py examples/minimal_qwen3_omni_assistant_mask.py examples/minimal_gemma4_e2b_it_assistant_mask.py
conda run -n makesense python -m pytest
```

真实多模态验证：

```bash
conda run -n makesense python examples/minimal_qwen2_5_omni_assistant_mask.py
conda run -n makesense python examples/minimal_qwen3_omni_assistant_mask.py
conda run -n makesense python examples/minimal_gemma4_e2b_it_assistant_mask.py
```

标准 pytest 不应依赖网络。真实 URL 无法访问时，用户可把示例中的 URL 改成本地路径；不得退化为 fake media。

## 10. 禁止事项

在用户允许实施前，不写或修改示例代码。

实施时也禁止：

1. 新增示例抽象层。
2. 新增共享示例配置模块。
3. 为了减少重复而隐藏模型差异。
4. 把示例写成库或框架。
5. 用 processor `assistant_masks` 构造多模态训练 labels。

# Real Multimodal Usage Example Plan

> 状态：不合格，已废弃。
>
> 废弃原因：
>
> 1. 该计划允许新增共享配置模块、case dataclass、runner/helper 层，导致实际示例更像可复用部署代码，而不是给 SFT/RL 用户直接照抄的最小用例。
> 2. 该计划把“如何直接得到 assistant mask / labels”的核心动作藏进抽象层，违背本项目的简单和外显原则。
> 3. 该计划的测试重点偏向配置表完整性，而不是示例代码本身是否以最短路径展示 expected supervised text 与 decoded supervised text 的对应关系。
> 4. 该计划没有足够强地约束示例形态：用例应像 notebook cell 一样直接构造 messages、processor batch、mask、labels 和验证输出，而不是构造示例框架。
>
> 后续实施必须改用新计划：`plans/minimal_assistant_mask_usage_examples_plan.md`。在获得用户确认前，不再基于本计划继续写示例代码。

## 0. 基准文件

本子任务以以下文件为基准：

1. `docs/multimodal_assistant_mask_design.md`
2. `plans/multimodal_assistant_mask_implementation_plan.md`
3. `progress/multimodal_assistant_mask_progress.md`

服务的主线验收标准：

1. 示例必须展示同一份真实多模态输入经过不同 processor 得到 final `input_ids` 后，再用 token-id assistant frame matching 构造 assistant-only labels。
2. 示例必须单独展示纯文本输入路径，用 training chat template 的 tokenizer-only `assistant_masks` 验证 generation block。
3. 示例不能把 processor 返回的 `assistant_masks` 作为多模态训练主路径。
4. 示例必须包含真实 image + audio + text mixed user content。
5. 示例应覆盖首批三个 processor：`Qwen/Qwen2.5-Omni-3B`、`Qwen/Qwen3-Omni-30B-A3B-Instruct`、`google/gemma-4-E2B-it`。
6. 示例不得加载大模型权重；只加载 processor/tokenizer。

## 1. 目标

新增面向用户参考的真实使用示例，分开展示两条路径：

1. **多模态输入路径**：同一份 image/audio/text user 输入接入三个真实 processor，并用本项目的 final `input_ids` frame matching 构造 assistant-only labels。这是本项目的核心功能路径。
2. **纯文本输入路径**：同一份 text-only messages 使用 `*-train.jinja` 模板，通过 tokenizer-only `assistant_masks` 验证 `{% generation %}` 标注。这是模板线的用途，不替代多模态 label builder 主路径。

下面的输入来自 Qwen3-Omni demo，但它在本子任务中作为通用 image/audio/text mixed user content 使用，不代表示例只服务于 Qwen3-Omni。

示例输入必须包含：

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

示例 assistant response 使用一个短文本，确保可以构造完整 assistant frame 并验证 labels，例如：

```python
{
    "role": "assistant",
    "content": "I see cars and hear someone coughing.",
}
```

assistant end token 应由所选 chat template 渲染，示例数据本身不手写 `<|im_end|>` 或 `<turn|>`。

固定 assistant response 使每个示例都有稳定的期望输出。掩码提取后的 supervised text 应符合：

Qwen 系列：

```text
I see cars and hear someone coughing.<|im_end|>
```

Gemma：

```text
I see cars and hear someone coughing.<turn|>
```

如果所选 chat template 在 assistant end 后渲染 separator newline，纯文本 tokenizer-only mask 可能包含该 newline；多模态核心路径的 frame mask 默认不包含 end 后 separator newline。

## 2. 范围

本阶段只做以下事情：

1. 新增真实多模态示例文件，例如 `examples/real_multimodal_processor_masks.py`。
2. 新增纯文本模板示例文件，例如 `examples/text_template_assistant_masks.py`。
3. 两个示例都加载三个 processor/tokenizer，均使用 `local_files_only=True`，不加载模型权重：
   - `Qwen/Qwen2.5-Omni-3B`
   - `Qwen/Qwen3-Omni-30B-A3B-Instruct`
   - `google/gemma-4-E2B-it`
4. 两个示例都为每个 processor 选择对应 training chat template：
   - `templates/qwen/qwen2_5-omni-3b-train.jinja`
   - `templates/qwen/qwen3-omni-30b-a3b-instruct-train.jinja`
   - `templates/gemma/gemma-4-e2b-it-train.jinja`
5. 多模态示例为每个 processor 选择对应 `AssistantFrameSpec`：
   - Qwen 系列：`<|im_start|>assistant\n` / `<|im_end|>`
   - Gemma：`<|turn>model\n` / `<turn|>`
6. 多模态示例构造同一份包含 image/audio/text 的 user message 和短 assistant response。
7. 纯文本示例构造同一份 text-only user message 和同一个短 assistant response。
8. 多模态示例下载或读取 image/audio 资源，按各 processor 支持的参数名交给 processor 构造 batch。
9. 如果某个 processor 不支持同一份 mixed media 输入，多模态示例应清楚报告该 processor 的不支持原因，而不是静默改用 fake processor 或删掉该模态。
10. 多模态示例调用：
   - `build_assistant_frame_masks`
   - `build_labels_from_frame_mask`
   - `decode_supervised_tokens`
   - `assert_no_header_in_labels`
   - `assert_end_in_labels`
11. 纯文本示例调用 tokenizer `apply_chat_template(..., return_assistant_tokens_mask=True)`，并解码 `assistant_masks == 1` 的 token。
12. 两个示例都打印每个 processor 的关键调试信息：
   - rendered prompt 的短预览
   - `input_ids` shape
   - supervised token 数量
   - decoded supervised text
   - expected supervised text
   - 是否匹配预期
13. 增加轻量测试，至少验证示例文件可编译，并验证三个 processor case 的配置表完整和期望输出配置完整。

## 3. 网络与资源策略

这个示例需要访问用户提供的真实 image/audio URL。由于测试环境可能没有网络，且不同 processor 对 image/audio 参数形态可能不同，本阶段采用分层验证：

1. 示例脚本本身允许联网下载 URL 资源。
2. 自动化 pytest 默认不访问网络，不把外部 URL 可用性作为必过条件。
3. 如需自动化运行真实下载路径，应使用显式环境变量打开，例如：

```bash
RUN_REAL_MULTIMODAL_EXAMPLE=1 conda run -n makesense python -m pytest
```

4. 如果 URL 下载失败，示例应报出清晰错误，而不是退化成 fake media processor 或 fake tokenizer。
5. 如果某个 processor 对同一份 image/audio/text mixed input 的接口不兼容，示例应保留该 case，并在运行结果中明确提示不支持或需要的输入格式。

## 4. 不做范围

1. 不加载 `AutoModel` 或任何模型权重。
2. 不执行真实 generation。
3. 不把 URL 下载逻辑抽象成通用数据集框架。
4. 不引入 fake tokenizer、fake processor 或 fake media expansion 公式。
5. 不把 tokenizer-only `assistant_masks` 当成多模态训练 labels。
6. 不修改核心库 API。

## 5. 验证

标准命令：

```bash
conda run -n makesense python -m pytest
```

最低验收：

1. 两个示例文件通过 `py_compile`。
2. 标准 pytest 通过。
3. 示例配置覆盖三个 processor。
4. 多模态示例中的 `AssistantFrameSpec` 与对应模型 frame 一致：
   Qwen 系列：
   - `assistant_header="<|im_start|>assistant\n"`
   - `assistant_end="<|im_end|>"`
   - `generation_stop="<|im_end|>"`
   Gemma：
   - `assistant_header="<|turn>model\n"`
   - `assistant_end="<turn|>"`
   - `generation_stop="<turn|>"`
5. 多模态示例清楚展示 labels 来自 final processor `input_ids` 上的 frame mask。
6. 纯文本示例清楚展示 supervised text 来自 tokenizer-only `assistant_masks`。
7. 两个示例都显式记录每个模型的 expected supervised text，并比较实际 decoded supervised text。

可选真实运行验收：

1. 在允许网络的环境中下载 image/audio URL。
2. 对支持该 mixed media 输入的 processor，processor 返回 `input_ids` 以及对应多模态特征字段。
3. 多模态 decoded supervised text 等于对应 expected supervised text，或只在文档允许的 separator newline 上有差异。
4. 纯文本 decoded supervised text 等于对应 expected supervised text，或只在文档允许的 separator newline 上有差异。
5. decoded supervised text 不包含 assistant header。
6. 对不支持该 mixed media 输入的 processor，输出明确说明不支持原因。

## 6. 进度回表

实现完成后，需要更新：

1. `progress/real_multimodal_usage_example_progress.md`
2. `progress/multimodal_assistant_mask_progress.md`

更新时只标记已经通过验收的项目，不把需要联网的可选真实运行验收误标为默认必过。

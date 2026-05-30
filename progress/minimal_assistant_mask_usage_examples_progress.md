# Minimal Assistant Mask Usage Examples Progress

## 0. 基准文件

本子任务以以下文件为基准：

1. `docs/multimodal_assistant_mask_design.md`
2. `plans/multimodal_assistant_mask_implementation_plan.md`
3. `progress/multimodal_assistant_mask_progress.md`

## 1. 任务表

| 任务 | 状态 | 验收 |
| --- | --- | --- |
| 创建最小用例计划 | 已完成 | `plans/minimal_assistant_mask_usage_examples_plan.md` 存在 |
| 新增 Qwen2.5-Omni 一脚本用例 | 已完成 | `examples/minimal_qwen2_5_omni_assistant_mask.py` |
| 新增 Qwen3-Omni 一脚本用例 | 已完成 | `examples/minimal_qwen3_omni_assistant_mask.py` |
| 新增 Gemma 4 E2B it 一脚本用例 | 已完成 | `examples/minimal_gemma4_e2b_it_assistant_mask.py` |
| 每个脚本包含多步多模态 section | 已完成 | final processor `input_ids` + frame matching + labels |
| 每个脚本包含多步纯文本 section | 已完成 | tokenizer-only `assistant_masks` 验证模板 |
| 每个脚本顶部标明 expected supervised text | 已完成 | 固定 assistant response 对应固定 expected 输出 |
| 移除联网/非联网分支 | 已完成 | 无环境变量跳过、无下载 helper、默认直接使用 URL |
| 示例轻量测试 | 已完成 | 编译、检查无旧分支/下载 helper |
| 运行标准 pytest | 已完成 | `conda run -n makesense python -m pytest` |
| 真实 URL 多模态运行 | 已完成 | 三个脚本已直接运行通过 |

## 2. 当前记录

1. 已新增三个最小脚本，一个模型一个脚本；没有新增共享配置模块、dataclass、case registry 或 runner。
2. 每个脚本直接写出 processor、messages、expected supervised text、mask/labels 构造和 decoded supervised text 验证；template path 只用于纯文本 section。
3. 已移除环境变量跳过逻辑、手写网络下载逻辑和下载 helper。
4. 三个脚本默认使用用户给出的 image/audio URL，并通过 Transformers 的 `load_image` / `load_audio` 转成 processor 需要的真实媒体输入；无法联网时可把 URL 改成本地路径。
5. `conda run -n makesense python -m py_compile examples/minimal_qwen2_5_omni_assistant_mask.py examples/minimal_qwen3_omni_assistant_mask.py examples/minimal_gemma4_e2b_it_assistant_mask.py tests/test_examples.py` 通过。
6. `conda run -n makesense python -m pytest tests/test_examples.py` 通过：3 passed。
7. `conda run -n makesense python examples/minimal_qwen2_5_omni_assistant_mask.py` 已联网运行通过。
8. `conda run -n makesense python examples/minimal_qwen3_omni_assistant_mask.py` 已联网运行通过；Qwen3-Omni-Instruct 默认 processor 模板与训练 Jinja 模板插入的空 `<think>` 闭合前缀均已排除在 assistant mask/loss 外，纯文本与多模态 supervised text 对齐。
9. `conda run -n makesense python examples/minimal_gemma4_e2b_it_assistant_mask.py` 已联网运行通过。
10. 三个脚本已改为多步对话：多模态 user -> assistant -> 纯文本 follow-up user -> assistant，用于验证多个 assistant frame 都被监督。
11. 已新增 `AssistantFrameSpec.excluded_assistant_prefixes`，用于排除 Qwen3-Omni-Instruct 这类模板自动插入但不应进入 loss 的 assistant 起始前缀。
12. 多模态 section 已改为使用 processor 默认 `apply_chat_template`，不传入训练 Jinja；训练 Jinja 只用于纯文本 section。
13. 标准测试命令 `conda run -n makesense python -m pytest` 通过：41 passed，9 warnings。

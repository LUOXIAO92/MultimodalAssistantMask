# Real Multimodal Usage Example Progress

> 状态：不合格，已废弃。
>
> 废弃原因：
>
> 1. 该进度对应的旧实现引入了共享配置、case dataclass 和 runner/helper 层，不符合最小用例原则。
> 2. 旧实现把 SFT/RL 用户最关心的 assistant mask / labels 构造步骤藏在抽象层后面，不适合作为用户参考示例。
> 3. 后续不再以本进度表的 `已完成` 状态作为验收依据。
>
> 新实施应遵循 `plans/minimal_assistant_mask_usage_examples_plan.md`。旧的 `real_multimodal*` 示例文件后续由用户移除。

## 0. 基准文件

本子任务以以下文件为基准：

1. `docs/multimodal_assistant_mask_design.md`
2. `plans/multimodal_assistant_mask_implementation_plan.md`
3. `progress/multimodal_assistant_mask_progress.md`

## 1. 任务表

| 任务 | 状态 | 验收 |
| --- | --- | --- |
| 创建真实多模态示例子任务计划 | 已完成 | `plans/real_multimodal_usage_example_plan.md` 存在 |
| 创建真实多模态示例子任务进度 | 已完成 | 本文件存在 |
| 新增共享示例配置 | 已完成 | 三个 processor case、模板、frame spec、expected supervised text |
| 新增纯文本模板示例 | 已完成 | tokenizer-only assistant mask 路径 |
| 新增真实多模态 processor 示例 | 已完成 | final processor `input_ids` + frame matching 路径 |
| 增加示例轻量测试 | 已完成 | 编译、配置覆盖、expected 输出配置 |
| 运行纯文本示例 | 已完成 | 三个 processor 的 decoded supervised text 均匹配预期 |
| 运行标准 pytest | 已完成 | `conda run -n makesense python -m pytest` |
| 真实 URL 多模态运行 | 后续 | 需要显式 `RUN_REAL_MULTIMODAL_EXAMPLE=1` 和网络 |

## 2. 当前记录

1. 已新增 `examples/real_multimodal_example_config.py`，集中维护三个 processor 的 model id、training template、`AssistantFrameSpec` 和 expected supervised text。
2. 已新增 `examples/text_template_assistant_masks.py`，展示纯文本输入使用 tokenizer-only `assistant_masks` 验证 training template。
3. 已新增 `examples/real_multimodal_processor_masks.py`，展示真实 image/audio/text 输入使用 final processor `input_ids` + token-id frame matching 构造 labels。
4. 已新增 `tests/test_examples.py`，默认不访问网络；真实 URL 多模态运行通过 `RUN_REAL_MULTIMODAL_EXAMPLE=1` 显式开启。
5. `conda run -n makesense python -m py_compile examples/real_multimodal_example_config.py examples/text_template_assistant_masks.py examples/real_multimodal_processor_masks.py` 通过。
6. `conda run -n makesense python examples/text_template_assistant_masks.py` 通过，三个 processor 的 decoded supervised text 均与 expected supervised text 匹配，允许模板层 end 后 separator newline。
7. 标准测试命令 `conda run -n makesense python -m pytest` 通过：41 passed，1 skipped，8 warnings。

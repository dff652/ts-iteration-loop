# 模型初始化指南 (Model Initialization Guide)

ChatTS-14B 的微调需要对原始的 Qwen 基座模型进行格式转换和参数初始化。

## 1. 格式转换步骤

如果您使用的是原始的 Qwen2.5-14B-Instruct，请按照以下步骤操作：

1. **下载模型**：从 HuggingFace 下载 [Qwen2.5-14B-Instruct](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct)。
2. **替换配置文件**：
   从 [ChatTS-14B 仓库](https://huggingface.co/bytedance-research/ChatTS-14B) 下载并替换以下文件到模型目录：
   - `*.py` (模型实现代码)
   - `config.json`
   - `added_tokens.json`
   - `tokenizer_config.json`
   - `special_tokens_map.json`

## 2. Xavier 初始化 (强烈建议)

为了保证训练稳定性，必须对 `ts_encoder` 部分进行 Xavier normal 初始化。

### 为什么需要初始化？
`ts_encoder` 是处理时序输入的关键组件。如果不进行初始化，模型在训练初期可能无法收敛，或者对时序输入产生异常响应。

### 如何执行？
我们提供了脚本 `scripts/init_model.py`。
1. 修改脚本中的 `SRC`（原始模型路径）和 `DST`（初始化后的保存路径）。
2. 运行脚本：
   ```bash
   python scripts/init_model.py
   ```
3. 训练时请使用 `DST` 路径下的模型。

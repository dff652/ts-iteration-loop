# ChatTS-14B 微调模型测试方案

本方案旨在验证微调后的 `ChatTS-14B` 模型是否成功掌握了自定义数据集（GDSH/HBSN等）中的异常检测任务逻辑、输出格式以及对特定模式的敏感度。

---

## 1. 测试环境准备

*   **Python 环境**: `chatts_train_env`
*   **模型路径**: 
    *   基座: `llm_models/ChatTS-14B`
    *   微调权重 (Adapter): `saves/chatts-14b/qlora/gdsh_tune/`

---

## 2. 方案 A：交互式命令行测试 (CLI Demo)

这是最直接的验证方式，用于测试模型的“逻辑理解”和“指令遵循”能力。

### 2.1 启动命令
在终端中执行以下命令：

```bash
conda activate chatts_train_env

llamafactory-cli chat \
    --model_name_or_path llm_models/ChatTS-14B \
    --adapter_name_or_path saves/chatts-14b/qlora/gdsh_tune/ \
    --template chatts \
    --finetuning_type lora \
    --quantization_bit 4 \
    --trust_remote_code True
```

### 2.2 测试要点
*   **格式验证**: 询问模型如何进行异常检测，观察其返回的是否为微调数据中的标准 JSON 格式。
    *   *输入示例*: `你是一位时间序列异常检测专家。请告诉你的工作流程和异常类型。`
*   **指令遵循**: 观察模型是否能正确理解“竖直尖峰”、“异常平台”等专业术语。
*   **局限性**: 命令行交互界面不适合直接粘贴 5000 个点的原始数值。此方案主要用于验证 **Text-to-JSON** 的逻辑对齐。

---

## 3. 方案 B：自动化推理测试 (Python Script) - 推荐

针对时序数据的特殊性（数据量大），建议使用脚本读取 CSV 并进行端到端测试。

### 3.1 脚本功能描述
我将为您提供一个 Python 脚本，其流程如下：
1.  加载基座模型并合并 LoRA 权重。
2.  读取 `data/chatts_tune/timeseries/` 下的某个测试 CSV 文件。
3.  自动构建 Prompt 并将时序数据转化为模型可接受的 Tensor。
4.  获取模型生成的 JSON 结果并保存。

### 3.2 验证标准
*   **成功标准**: 模型输出合法的 JSON 字符串，且 `detected_anomalies` 列表中的区间与业务常识基本吻合。
*   **Loss 参考**: 训练结束时 Loss 约为 0.68，模型应具备较强的模式识别能力。

---

## 4. 后续步骤

1.  **首选执行方案 A**: 快速确认模型没有“学废”（比如只会输出乱码）。
2.  **执行方案 B**: 如果需要批量评估 100 个点位的准确率，请向我索取推理脚本。

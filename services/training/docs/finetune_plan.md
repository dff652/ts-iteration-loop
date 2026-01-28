# ChatTS-14B GDSH 数据微调实施计划

## 1. 目标
使用 `data/chatts_tune` 目录下提供的自定义 GDSH 异常检测数据，对 `ChatTS-14B` 模型进行微调 (Fine-tuning)。

## 2. 当前现状
*   **模型**: 已位于 `./llm_models/ChatTS-14B`。
*   **数据源**:
    *   **标签/Prompt**: `data/chatts_tune/json/gdsh.json` (LLaMA-Factory 多模态格式，引用了图片路径)。
    *   **时序数据**: `data/chatts_tune/timeseries/gdsh/*.csv` (CSV 格式，包含 `value` 列)。
*   **存在问题**: 当前数据格式（JSON 引用图片路径 + 独立 CSV）与 ChatTS 训练所需的标准输入格式（包含 `<ts><ts/>` 占位符和嵌入式数值数组的 JSONL）不兼容。

## 3. 实施方案

### 第一阶段：数据预处理
**目标**: 将源数据转换为 ChatTS 标准 JSONL 格式。

*   **脚本路径**: `scripts/preprocess_gdsh.py`
*   **处理逻辑**:
    1.  读取 `data/chatts_tune/json/gdsh.json`。
    2.  遍历每个样本。
    3.  **Prompt 构建**: 提取 `user` 的指令。将原有的 `<image>` 标签替换为 ChatTS 标准占位符 `<ts><ts/>`。
    4.  **时序数据提取**:
        *   解析 `image` 字段中的文件名（例如匹配 `gdsh_second_*.PV.jpg` 对应的 `gdsh_second_*.PV.csv`）。
        *   **注意**: 原始 JSON 中的路径是 `/home/wyx/...`，需要映射到本地相对路径 `data/chatts_tune/timeseries/gdsh/`。
        *   读取对应的 CSV 文件，提取 `value` 列作为浮点数列表。
    5.  **输出构建**: 生成如下结构的 JSON 对象：
        ```json
        {
            "input": "User prompt with <ts><ts/>...",
            "output": "Assistant response JSON...",
            "timeseries": [[0.1, 0.2, ...]]  // 列表的列表 (Batch x Length)
        }
        ```
    6.  保存为 `data/chatts_tune/train.jsonl`。

### 第二阶段：数据集配置
**目标**: 在训练框架中注册新数据集。

*   **文件**: `/home/share/data/training_chatts/dataset_info.json`
*   **操作**: 添加 `chatts_tune` 条目：
    ```json
    "chatts_tune": {
        "file_name": "chatts_tune/train.jsonl",
        "columns": {
            "prompt": "input",
            "response": "output",
            "timeseries": "timeseries"
        }
    }
    ```

### 第三阶段：训练环境配置
**目标**: 创建可复现的训练脚本。

*   **脚本路径**: `scripts/chatts/lora/train_chatts_tune.sh`
*   **配置概览**:
    *   基座模型: `llm_models/ChatTS-14B`
    *   数据集: `chatts_tune`
    *   微调方法: LoRA (Low-Rank Adaptation)
    *   输出目录: `saves/chatts-14b/lora/gdsh_tune`
    *   其他参数: 根据显存情况调整 batch size 和 gradient accumulation。

## 4. 任务清单

- [ ] **数据准备**: 编写并运行 `scripts/preprocess_gdsh.py`。
- [ ] **数据验证**: 检查生成的 `data/chatts_tune/train.jsonl` 格式是否正确。
- [ ] **配置更新**: 修改 `/home/share/data/training_chatts/dataset_info.json`。
- [ ] **脚本编写**: 创建 `scripts/chatts/lora/train_chatts_tune.sh`。
- [ ] **执行训练**: 启动训练并监控 Loss。

## 5. 备注
*   **路径映射**: JSON 中的 `image` 字段包含不存在的绝对路径 `/home/wyx/...`，脚本必须通过文件名匹配来定位本地 CSV 文件。
*   **序列长度**: 原始 CSV 数据长度可能约为 5000 点，而 ChatTS 示例数据约为 256 点。模型 (Qwen2 基础) 支持长上下文，但过长的时序数据可能会导致显存溢出或性能下降。如果训练中遇到 OOM (Out of Memory)，可能需要对数据进行下采样 (Downsampling) 或切片。我们将首先尝试原始长度或简单的下采样。
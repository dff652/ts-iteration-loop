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

*   **脚本路径**: `scripts/chatts/lora/train_chatts_tune_qlora_safe.sh` (推荐用于 RTX 2080Ti)
*   **配置概览**:
    *   基座模型: `llm_models/ChatTS-14B`
    *   数据集: `chatts_tune`
    *   微调方法: QLoRA (4-bit 量化 + LoRA)
    *   输出目录: `saves/chatts-14b/qlora/2080Ti_tune_20251226`
    *   其他参数: batch_size=1, gradient_accumulation=16, epochs=3

**执行命令**:
```bash
# 1. 激活环境
conda activate chatts_train_env

# 2. 启动训练（前台运行）
bash scripts/chatts/lora/train_chatts_tune_qlora_safe.sh
```

### 微调脚本对比

| 脚本 | 训练框架 | 量化 | 显存需求 | 适用显卡 |
|-----|---------|------|---------|---------|
| `train_chatts_tune_qlora_safe.sh` | torchrun | 4-bit | ~10GB | RTX 2080Ti ✅ |
| `train_chatts_tune_simple.sh` | LlamaFactory CLI | 4-bit | ~10GB | RTX 2080Ti |
| `train_chatts_tune_qlora.sh` | LlamaFactory CLI | 4-bit | ~12GB | RTX 3060+ |
| `train_chatts_tune.sh` | DeepSpeed | 无 (FP16) | ~20GB+ | RTX 3090/A100 |

**关键区别**:
- **QLoRA 版本**: 使用 4-bit 量化，显存占用低，适合消费级显卡
- **无量化版本**: FP16 全精度，需要专业级显卡，但训练精度更高
- **DeepSpeed**: 支持 CPU Offload，可进一步降低显存压力

---

### 训练步数计算

**公式**:
```
总步数 = (样本数 / batch_size / gradient_accumulation_steps) × num_train_epochs
```

**参数说明**:
| 参数 | 含义 | 当前值 |
|-----|------|-------|
| 样本数 | 训练数据总条数 | 417 |
| batch_size | 每次前向传播处理的样本数 | 1 |
| gradient_accumulation_steps | 累积多少次梯度后更新权重 | 16 |
| num_train_epochs | 遍历完整数据集的轮数 | 3 |

**当前配置计算**:
```
每 epoch 步数 = 417 / 1 / 16 ≈ 26 步
总步数 = 26 × 3 = 78 步
```

**为什么用梯度累积?**
- 显存有限时，通过多次小 batch 累积梯度，模拟大 batch 训练效果
- `batch_size=1 × 16次累积 = 等效 batch_size=16`

---

### num_train_epochs vs max_steps

| 参数 | 含义 | 优先级 |
|-----|------|-------|
| `num_train_epochs` | 遍历整个数据集的次数 | 低 |
| `max_steps` | 最大训练步数 | 高 |

> 如果两者都设置，`max_steps` 优先，达到后停止训练。

---

### 使用建议

| 显卡 | 显存 | 推荐脚本 | 备注 |
|-----|------|---------|------|
| RTX 2080Ti | 11GB | `train_chatts_tune_qlora_safe.sh` | 必须用 4-bit 量化 |
| RTX 3090 | 24GB | `train_chatts_tune.sh` | 可用 FP16 全精度 |
| A100 | 40GB+ | `train_chatts_tune.sh` | 可加大 batch_size |

**训练时间预估** (RTX 2080Ti, 78步):
- QLoRA: 约 30-60 分钟

---

## 4. 任务清单

- [x] **数据准备**: 运行 `scripts/preprocess_tune_data.py` 生成训练数据
- [x] **数据验证**: 检查 `data/chatts_tune/train.jsonl` 格式 (417条记录)
- [x] **配置更新**: 修改 `/home/share/data/training_chatts/dataset_info.json`
- [x] **脚本编写**: 创建训练脚本
- [ ] **执行训练**: 启动训练并监控 Loss

## 5. 备注
*   **路径映射**: JSON 中的 `image` 字段包含不存在的绝对路径，脚本通过文件名匹配定位本地 CSV 文件。
*   **序列长度**: 原始 CSV 数据长度约 5000 点，模型支持长上下文，但可能增加显存消耗。
*   **数据集选择**: 由 `/home/share/data/training_chatts/dataset_info.json` 中的 `file_name` 字段决定使用哪个 JSONL 文件。
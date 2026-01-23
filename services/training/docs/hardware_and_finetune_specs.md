# 硬件环境与 ChatTS 微调技术方案

## 1. 硬件环境 (Hardware Environment)

*   **GPU 配置**: 2x NVIDIA GeForce RTX 2080 Ti
*   **显存规格**: 22GB / 卡 (共 44GB VRAM)
    *   *注：标准版 2080 Ti 为 11GB，当前设备为 22GB 大显存版本，极大提升了模型承载能力。*
*   **互联技术**: 双卡 (可能支持 NVLink，当前方案暂按通用 PCIe 互联设计以确保兼容性)。
*   **驱动环境**: CUDA 12.4, Driver 550.90.07。

## 2. 软件环境 (Software Stack)

*   **基础环境**: Conda `chatts`
*   **核心组件**:
    *   Python: 3.12
    *   PyTorch: 2.6.0+cu126
    *   DeepSpeed: 0.18.1
    *   Transformers: 4.57.3
    *   Flash-Attention: 2.8.3 (已启用，显著降低长序列显存占用)
    *   PEFT / TRL: 最新版本，支持 LoRA 微调。

## 3. 微调技术方案 (Technical Solution)

### 3.1 模型架构
*   **基座模型**: `ChatTS-14B` (基于 Qwen2.5-14B 深度定制的时序大模型)。
*   **模型规模**: FP16 精度下权重约占 **28GB** 显存。

### 3.2 微调策略: LoRA (Low-Rank Adaptation)
鉴于 14B 模型全量微调对显存要求极高，本项目采用 **LoRA** 技术。
*   **原理**: 冻结基座模型权重，仅训练低秩适配器（Adapter）。
*   **目标模块**: `q_proj, k_proj, v_proj` (注意力机制的关键投影层)。
*   **Rank**: 8 (参数量小，训练快，且足以捕捉特定领域的异常模式)。
*   **优势**: 极大降低显存占用（梯度和优化器状态仅占几十 MB），同时保持基座模型的泛化能力。

### 3.3 分布式训练策略: DeepSpeed Zero-3 Offload
为了在 44GB 总显存中放下 28GB 的模型并预留空间给计算（Activations），采用了 **DeepSpeed ZeRO Stage 3 (CPU Offload)**。

*   **参数切分 (ZeRO-3)**: 模型权重被切分到 2 张 GPU 上。
    *   理论单卡静态占用: ~14GB (28GB / 2)。
*   **CPU Offload**: 将优化器状态（Optimizer States）和部分参数计算卸载到 CPU 内存。
    *   **目的**: 这里的 Offload 主要是为了“安全兜底”。虽然 22GB 显存理论上可能刚好放下（14GB 权重 + 6-8GB 激活值），但长上下文（4096）容易导致动态显存溢出 (OOM)。Offload 牺牲了部分 PCIe 传输速度，换取了绝对的运行稳定性。
*   **NVLink 备注**: 如果后续验证显存充裕（峰值 < 20GB），可关闭 Offload 利用 NVLink 进行显卡间直连通信，训练速度预计可提升 2-3 倍。

### 3.4 数据处理
*   **数据源**: 4 个自定义数据集 (gdsh, hbsn, whlj, zhlh)，共 421 条样本。
*   **格式**: 标准 ChatTS 格式 (`input` 含 `<ts><ts/>`, `timeseries` 为数值列表)。
*   **序列长度**: 原始长度 5000 点，训练截断长度 (Cutoff) 设为 **4096**。

## 4. 训练超参数配置 (Hyperparameters)

| 参数项 | 设定值 | 说明 |
| :--- | :--- | :--- |
| **Batch Size (Per Device)** | 1 | 单卡批次为 1，防止显存溢出。 |
| **Gradient Accumulation** | 8 | 梯度累积步数。等效总 Batch Size = 1 * 8 * 2(GPUs) = **16**。 |
| **Learning Rate** | 1e-4 | LoRA 微调的典型学习率。 |
| **Epochs** | 3 | 总共遍历数据 3 轮。 |
| **Precision** | FP16 | 半精度训练。 |
| **Cutoff Length** | 4096 | 包含文本和时序 Token 的最大总长度。 |

## 5. 预期性能
*   **显存占用**: 预计单卡占用约 16GB - 20GB (在 Zero-3 Offload 模式下)。
*   **训练时长**: 预计 30 - 45 分钟 (视 PCIe 带宽和 CPU 性能而定)。

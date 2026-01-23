# ChatTS-14B RTX 6000 Ada 微调工作记录

**日期**: 2025年12月25日
**硬件**: 2x NVIDIA RTX 6000 Ada Generation (48GB VRAM)
**任务**: 基于自定义数据 (418条) 进行 ChatTS-14B LoRA 微调。

---

## 1. 环境构建与依赖解决

在 RTX 6000 Ada 环境下，主要解决了一系列软件兼容性问题：

| 问题 (Issue) | 原因分析 | 解决方案 |
| :--- | :--- | :--- |
| **DeepSpeed 编译失败** | 缺少 `nvcc` 编译器，无法构建算子。 | `conda install -c nvidia cuda-nvcc=12.1` 并设置 `CUDA_HOME=$CONDA_PREFIX`。 |
| **模型加载 ValueError** | `transformers>=4.52` 对 PyTorch < 2.6 的非安全 checkpoint 加载有限制。 | 降级 `transformers` 至 `4.49.0`。 |
| **显存溢出 (OOM)** | 14B 模型 + 8k 上下文 + Batch Size 4 超过 48GB 显存上限。 | 将 `per_device_train_batch_size` 设为 **1**，同时增加 `gradient_accumulation_steps` 至 **16**。 |
| **训练提前退出** | 脚本初版遗漏 `--do_train`。 | 在启动命令中显式添加 `--do_train`。 |

---

## 2. 最终训练配置

*   **基座模型**: `ChatTS-14B`
*   **微调策略**: LoRA (Rank 16)
*   **目标模块**: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`
*   **数据量**: 418 条 (清洗后有效样本)
*   **训练参数**:
    *   Effective Batch Size: 32 (1 BS * 16 steps * 2 GPUs)
    *   Learning Rate: 2e-5 (Cosine scheduler)
    *   Cutoff Length: 8192
    *   Precision: BF16
    *   Epochs: 5

---

## 3. 训练结果

*   **持续时长**: 约 22 分钟 (含加载时间)。
*   **最终 Loss**: **0.4721** (从 1.27 稳步下降)。
*   **保存位置**: `saves/chatts-14b/lora/rtx6000_tune/`
*   **主要产出**:
    *   `adapter_model.safetensors`: LoRA 适配器权重。
    *   `training_loss.png`: 损失函数下降曲线图。
    *   `checkpoint-65`: 训练完成的最终检查点。

---

## 4. 后续建议

1.  **权重合并**: 建议使用 LLaMA-Factory 的导出功能将 LoRA 权重与 Base 模型合并，以便进行推理加速。
2.  **验证**: 使用 `src/cli_demo.py` 加载适配器进行异常检测任务的实际测试。
3.  **长序列优化**: 当前 8k 长度显存占用极高 (接近 48GB 极限)，若需进一步增加长度或 batch size，可考虑开启 DeepSpeed ZeRO-3 Offload。
# ChatTS-8B 微调实施计划

## 1. 任务背景
在完成 ChatTS-14B 的微调工作后，现计划基于 ChatTS-8B 模型进行微调。ChatTS-8B 规模更小（约 8 Billion 参数），加载和训练对硬件资源的压力相对较小，但仍需注意内存控制。

## 2. 资源评估

### 2.1 硬件环境
*   **GPU**: 2x NVIDIA GeForce RTX 2080 Ti (22GB 显存/卡，总计 44GB)
*   **内存 (RAM)**: 46GB 物理内存 + 32GB Swap (总计约 78GB 虚拟内存)
*   **显存互联**: 支持 NVLink

### 2.2 软件环境
*   **基础环境**: 已创建的 Conda 环境 `chatts_train_env`。
*   **核心库版本**:
    *   `transformers`: 4.52.4 (与模型 config.json 要求一致)
    *   `peft`: 0.15.2
    *   `trl`: 0.9.6
    *   `torch`: 2.6.0+cu126
    *   `bitsandbytes`: 0.49.0 (支持 4-bit 量化)

### 2.3 模型与数据
*   **基座模型**: `ChatTS-8B` (位于 `/home/share/llm_models/bytedance-research/ChatTS-8B`)
*   **模型类型**: `qwen3ts`
*   **微调数据**: `/home/dff652/TS-anomaly-detection/ChatTS-Training/data/chatts_tune/train.jsonl` (共 421 条)

## 3. 技术方案规划

### 3.1 微调策略
鉴于之前的 14B 微调经验，虽然 8B 模型较小，但为了确保稳定性并充分利用双卡资源，我们有两种选择：
1.  **方案 A (推荐 - 稳健版)**: **单卡或双卡 QLoRA (4-bit)**。显存占用极低（单卡约 6-8GB），加载速度快，系统内存压力小。
2.  **方案 B (高性能版)**: **双卡 DeepSpeed ZeRO-3 (FP16)**。8B 模型 (约 16GB) 使用 ZeRO-3 分片后，每张卡仅需 8GB 权重。22GB 显存完全可以放下权重 + 激活值。

**最终决定**: 先尝试 **方案 B (双卡 DeepSpeed ZeRO-3)**。如果 16GB 的模型仍触发系统 RAM OOM，则退回到方案 A。

### 3.2 关键参数配置
*   **GPU 数量**: 2
*   **训练步数**: 约 80 Steps (3 Epochs)
*   **学习率**: 1e-4
*   **梯度累积**: 8 (等效 Batch Size = 1 * 8 * 2 = 16)
*   **截断长度 (Cutoff)**: 4096
*   **DeepSpeed 配置**: `ds_config/ds_config_3_offload.json` (开启 Offload 以保底)

## 4. 实施步骤清单

- [ ] **环境验证**: 确认 `chatts_train_env` 的可用性。
- [ ] **数据检查**: 再次确认 `train.jsonl` 中 `timeseries` 的长度与 `qwen3ts` 配置的兼容性（模型支持 8192，当前数据 5000，符合要求）。
- [ ] **脚本编写**: 创建 `scripts/chatts/lora/train_chatts_8b_ds3.sh`。
- [ ] **执行微调**: 启动训练并监控 Loss 与系统资源。
- [ ] **结果报告**: 记录微调进度与产出路径。

## 5. 代码修改建议
*   **LLaMA-Factory 适配**: 当前代码库已包含 `qwen3ts` 支持，无需修改核心逻辑。
*   **启动命令**: 必须使用 `llamafactory-cli` 或 `torchrun`。

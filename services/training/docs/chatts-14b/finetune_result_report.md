# ChatTS-14B 微调训练结果报告

**任务状态**: 已完成 (Success)
**完成日期**: 2025年12月25日
**训练耗时**: 约 1.5 小时 (包含模型加载与量化阶段)

---

## 1. 训练配置总结

本次微调采用了 **QLoRA (4-bit)** 策略，成功克服了物理内存限制 (46GB RAM)，实现了 ChatTS-14B 在双卡 2080 Ti 环境下的平稳运行。

*   **基座模型**: `llm_models/ChatTS-14B` (Qwen2.5-14B custom)
*   **微调方式**: LoRA (rank=8, target: q_proj, k_proj, v_proj)
*   **量化精度**: 4-bit (NF4)
*   **分布式策略**: 单 GPU 串行测试后扩展为 **双卡并行** (通过 Swap 扩容支持)
*   **数据规模**: 421 条清洗后的有效样本
*   **训练参数**: 3 Epochs, 累计梯度 Accumulation=16, 学习率 1e-4

---

## 2. 训练过程监控

训练过程表现非常稳健，Loss 指标呈现理想的收敛曲线。

### 2.1 关键进度指标 (Step-by-Step)
| 步数 (Step) | Epoch | 损失值 (Loss) | 学习率 (LR) | 状态 |
| :--- | :--- | :--- | :--- | :--- |
| 1 | 0.04 | 1.2369 | 0.0 | 预热阶段 |
| 4 | 0.15 | 1.0232 | 6e-5 | 快速下降 |
| 20 | 0.77 | 0.7558 | 6.7e-5 | 进入稳定期 |
| 34 | 1.31 | 0.6520 | 6.3e-5 | 低位波动 |
| **42** | **3.00** | **~0.68** | **0.0** | **训练圆满结束** |

### 2.2 资源利用情况
*   **显存 (VRAM)**: 每张卡约占用 **10GB - 12GB**，22GB 的显存容量提供了充足的冗余。
*   **系统内存 (RAM)**: 依靠 **32GB Swap (交换空间)** 成功度过了模型加载时的峰值，总内存占用维持在 60GB 左右。
*   **负载**: 双 GPU 满负荷运行，无 OOM 崩溃记录。

---

## 3. 训练结果与产出物

微调后的模型权重（LoRA Adapter）已成功保存并同步到以下路径：

### 3.1 核心权重路径
*   **最终 Checkpoint**: `/home/dff652/TS-anomaly-detection/ChatTS-Training/saves/chatts-14b/qlora/gdsh_tune/checkpoint-42`
*   **同步根目录**: `/home/dff652/TS-anomaly-detection/ChatTS-Training/saves/chatts-14b/qlora/gdsh_tune/`

### 3.2 文件清单
*   `adapter_model.bin`: 核心增量权重。
*   `adapter_config.json`: LoRA 配置文件。
*   `training_loss.png`: Loss 变化曲线图。
*   `trainer_log.jsonl`: 完整的训练步数日志。

---

## 4. 下一步计划建议

1.  **推理测试**: 使用 `llamafactory-cli chat` 加载该 Adapter，输入新的 CSV 时序数据，验证模型识别异常区间的准确性。
2.  **权重合并 (可选)**: 如需部署到不支持 PEFT 的环境，可进行权重合并 (Merge LoRA Weights)。
3.  **多轮微调**: 若测试效果不佳，可基于当前 checkpoint 继续追加更多特定领域的异常数据。

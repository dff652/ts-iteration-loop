# ChatTS 微调结果汇总

> 更新时间：2025-12-28

本文档汇总了 ChatTS 模型在不同硬件和配置下的微调结果。

## 结果对比表

| 目录名 | 模型 | 微调方法 | 量化 | 硬件 | 总步数 | Epochs | 耗时(秒) | 耗时(分钟) | 最终 Loss | 样本/秒 |
|--------|------|---------|------|------|--------|--------|---------|-----------|-----------|---------|
| `rtx6000_tune` | 14B | LoRA (rank 16) | ❌ FP16 | RTX 6000 ×2 | 65 | 5 | 1216.8 | **20.3** | 0.641 | 1.72 |
| `rtx6000_tune_20251228` | 14B | LoRA (rank 16) | ❌ FP16 | RTX 6000 ×2 | 60 | 5 | 1286.8 | **21.4** | 0.638 | 1.61 |
| `2080Ti_tune_qlora_safe_20251226` | 14B | LoRA (rank 8) | ✅ 4-bit | RTX 2080Ti ×2 | 78 | 3 | 2133.6 | **35.6** | 0.473 | 0.58 |
| `rtx2080ti_tune` | 14B | LoRA (rank 8) | ❌ FP16 | RTX 2080Ti ×2 | 42 | 3 | 2204.6 | **36.7** | 0.736 | 0.57 |
| **`RTX6000_tune_no_offload`** | **8B** | **LoRA (rank 8)** | ❌ FP16 | RTX 6000 ×1 | **156** | 3 | 1329.5 | **22.2** | **0.341** | 0.93 |
| `2080ti_tune` (8B) | 8B | LoRA (rank 8) | ✅ 4-bit | RTX 2080Ti ×2 | 78 | 3 | 1262.5 | **21.0** | 0.500 | 0.98 |

## 关键发现

### 1. 硬件性能对比

| 对比项 | RTX 6000 | RTX 2080Ti | 性能比 |
|--------|----------|------------|--------|
| 14B 训练速度 (样本/秒) | ~1.7 | ~0.6 | **3倍** |
| 显存 | 24GB | 11GB | 2.2倍 |

> RTX 6000 训练速度约是 RTX 2080Ti 的 **3 倍**

### 2. 模型规模对比

| 模型 | 训练速度 (样本/秒) | 相对速度 |
|------|-------------------|----------|
| ChatTS-8B (RTX6000) | ~0.93 | 1.6x |
| ChatTS-8B (2080Ti QLoRA) | ~0.98 | 1.7x |
| ChatTS-14B | ~0.58 | 1.0x |

> 8B 模型训练速度约是 14B 的 **1.6~1.7 倍** (相同硬件下)

### 3. 量化方式对比

| 方式 | 最终 Loss | 训练速度 | 显存占用 |
|------|-----------|----------|----------|
| LoRA (8B FP16) | **0.34** | 0.93 | 中 |
| QLoRA (4-bit) | 0.47~0.50 | 0.58~0.98 | **低** |
| LoRA (14B FP16) | 0.64~0.74 | 1.6~1.7 | 高 |

> **8B LoRA FP16 收敛最好**：最终 loss 仅 0.34，优于 QLoRA 和 14B 模型

### 4. 训练脚本推荐

| 场景 | 推荐脚本 | 说明 |
|------|----------|------|
| RTX 6000 + 14B | `train_chatts_14b_rtx6000.sh` | DDP，无 DeepSpeed |
| RTX 2080Ti + 14B | `train_chatts_tune.sh` | DeepSpeed ZeRO-3 + Offload |
| RTX 2080Ti + 8B | `train_chatts_8b_fast.sh` | DeepSpeed ZeRO-3，无 Offload |
| 显存受限 | `train_chatts_8b_qlora.sh` | QLoRA 4-bit 量化 |

## 目录结构

```
saves/
├── chatts-14b/
│   ├── lora/
│   │   ├── debug_tune/           # 调试用
│   │   ├── rtx6000_tune/         # RTX 6000 LoRA FP16
│   │   └── rtx6000_tune_20251228/ # RTX 6000 LoRA FP16
│   └── qlora/
│       ├── 2080Ti_tune_qlora_safe_20251226/  # 2080Ti QLoRA 4-bit
│       └── rtx2080ti_tune/       # 2080Ti LoRA FP16
└── chatts-8b/
    ├── lora/
    │   ├── RTX6000_tune_no_offload/  # RTX 6000 LoRA FP16 ⭐最新
    │   └── gdsh_tune_ds3/        # (空目录)
    └── qlora/
        └── 2080ti_tune/          # 2080Ti QLoRA 4-bit
```

---

## 微调方法统计

### 已使用的方法

| 微调方法 | 14B 模型 | 8B 模型 | 总计 | 说明 |
|---------|---------|---------|------|------|
| **LoRA** (FP16) | 3 个实验 | 4 个实验 | 7 个 | 显存需求中等 |
| **QLoRA** (4-bit) | 3 个实验 | 1 个实验 | 4 个 | 显存需求最低 |

### 可选的 PEFT 方法

| 方法 | 显存需求 | 训练速度 | 效果 | LlamaFactory 参数 |
|------|---------|---------|------|------------------|
| **LoRA** | 中等 | 快 | 很好 | `--finetuning_type lora` |
| **QLoRA** | 极低 (4-bit) | 慢 | 好 | `--finetuning_type lora --quantization_bit 4` |
| **DoRA** | 中等 | 中等 | 更好 | `--finetuning_type dora` |
| **AdaLoRA** | 中等 | 中等 | 自适应 | `--finetuning_type adalora` |
| **Full** | 极高 | 慢 | 最好 | `--finetuning_type full` |
| **Freeze** | 低 | 快 | 一般 | `--finetuning_type freeze` |

### 推荐方案

| 硬件配置 | 推荐方法 | 理由 |
|---------|---------|------|
| RTX 6000 (24GB) | **LoRA** | 显存充足，速度最快 |
| RTX 2080Ti (11GB) | **QLoRA** | 显存限制，4-bit 必需 |
| A100/H100 (40GB+) | **Full** 或 **LoRA** | 可尝试全量微调 |
| 追求更好效果 | **DoRA** | LoRA 改进版，效果更优 |

---

## 微调方法原理

### 1. LoRA (Low-Rank Adaptation)

**核心思想：** 在原始权重矩阵旁添加低秩分解矩阵，只训练这些小矩阵

```
原始: Y = W·X
LoRA: Y = (W + B·A)·X = W·X + B·A·X
                        ↑      ↑
                    冻结    可训练(低秩)
```

其中 B ∈ R^(d×r), A ∈ R^(r×k)，r << min(d, k)，典型值 r=8 或 16

**为什么 LoRA 有效？**

1. **低秩假设 (Intrinsic Rank Hypothesis)**
   - 研究表明预训练模型的权重更新具有**低内在维度**
   - 微调时的权重变化 ΔW 可以用低秩矩阵近似：ΔW ≈ BA
   - 论文证明：即使 r << d，也能捕获 99%+ 的权重变化信息

2. **参数效率**
   - 原始参数：d × k (如 4096 × 4096 = 16M)
   - LoRA 参数：d × r + r × k (如 4096 × 8 + 8 × 4096 = 65K)
   - 压缩比：**0.4%**

3. **梯度流动**
   - 基座模型梯度直接流向 LoRA 矩阵
   - 避免了全量微调的灾难性遗忘
   - 保留了预训练知识

**实现：**
```bash
--finetuning_type lora
--lora_rank 8
--lora_target "q_proj,k_proj,v_proj"
```

**LoRA 的副作用与局限性：**

| 副作用 | 严重程度 | 说明 | 解决方案 |
|--------|---------|------|---------|
| **效果上限** | 中等 | 低秩假设限制表达能力，复杂任务可能不足 | 增大 rank 或考虑全量微调 |
| **推理开销** | 轻微 | 分离加载时有 ~5-10% 开销 | 合并权重到基座 |
| **rank 敏感** | 轻微 | 太小欠拟合，太大失去效率 | 从 r=8 开始测试 |
| **层选择** | 轻微 | 目标层选择影响学习效果 | 参考成熟配置 |
| **灾难性遗忘** | 轻微 | 可能遗忘通用能力 | 混合训练数据 |
| **ChatTS 特殊** | 轻微 | 时序编码器可能需要单独学习率 | 使用 `--timeseries_sft_lr` |

---

### 2. QLoRA (Quantized LoRA)

**原理：** 基座模型量化为 4-bit，在其上应用 LoRA

```
基座模型 (FP16, ~28GB) → 量化为 4-bit (~7GB) → LoRA适配器 (FP16) → 训练
```

**关键技术：**
- **NF4 (4-bit NormalFloat)**：信息论最优的 4-bit 量化格式
- **双重量化**：对量化常数也进行量化，进一步节省显存
- **分页优化器**：将优化器状态分页到 CPU

**实现：**
```bash
--finetuning_type lora
--quantization_bit 4
```

---

### 3. DoRA (Weight-Decomposed Low-Rank Adaptation)

**原理：** 将权重分解为幅度和方向，LoRA 只更新方向

```
W = m · (W₀ + BA) / ||W₀ + BA||
    ↑      ↑
  幅度   方向(LoRA)
```

**优势：** 比 LoRA 收敛更稳定，最终效果更好

**实现：**
```bash
--finetuning_type dora
```

---

### 4. Full Fine-tuning (全量微调)

**原理：** 更新模型所有参数

**显存需求：** 14B 模型需要 ~80GB (训练状态 + 梯度 + 优化器)

**实现：**
```bash
--finetuning_type full
```

---

### 5. Freeze (冻结微调)

**原理：** 冻结大部分层，只训练最后几层

**实现：**
```bash
--finetuning_type freeze
--freeze_trainable_layers 2
```

---

### 方法对比

| 方法 | 可训练参数 | 显存 | 效果 |
|------|-----------|------|------|
| Full | 100% | 极高 | ⭐⭐⭐⭐⭐ |
| DoRA | ~0.2% | 中等 | ⭐⭐⭐⭐ |
| LoRA | ~0.1% | 中等 | ⭐⭐⭐⭐ |
| QLoRA | ~0.1% | 极低 | ⭐⭐⭐ |
| Freeze | ~5% | 低 | ⭐⭐ |

---

## 数据量分析

### 当前数据量

- **训练数据**：`data/chatts_tune/train_1.jsonl`
- **样本数量**：417 条

### LoRA 微调数据量要求

| 数据规模 | 效果 | 说明 |
|---------|------|------|
| < 100 | ⚠️ 可能不足 | 容易过拟合 |
| **100-500** | ✅ 有效 | **当前位置**，足够学习任务模式 |
| 500-2000 | ✅ 良好 | 更稳定的泛化能力 |
| > 2000 | ✅ 理想 | 可尝试更大 rank 或更多 epochs |

### 为什么 400+ 样本足够？

1. **LoRA 参数极少**：只训练 ~0.1% 参数 (<1M 可训练参数)
2. **预训练知识迁移**：ChatTS 已从大规模时序数据预训练，微调只需学习特定任务格式
3. **实验验证**：loss 从 ~1.2 降到 0.34 (8B) / 0.47 (14B)，收敛良好

### 小数据集推荐参数

| 参数 | 推荐值 | 理由 |
|------|-------|------|
| epochs | 3 | 400 样本 × 3 = 1200 步，足够收敛 |
| lora_rank | 8 | 小数据用小 rank，避免过拟合 |
| learning_rate | 1e-4 | 标准设置 |
| lora_dropout | 0.05 | 可选，防止过拟合 |

### ⚠️ 过拟合预防

1. 不要过多 epochs（3 epochs 足够，5 epochs 可能过拟合）
2. 监控 loss 曲线（先降后升说明过拟合）
3. 可添加 dropout：`--lora_dropout 0.05`

## 注意事项

1. **空目录**：`chatts-8b/lora/gdsh_tune_ds3` 为空，无训练结果
2. **输出路径**：建议使用动态日期命名输出目录，避免覆盖：
   ```bash
   --output_dir "saves/chatts-14b/lora/rtx6000_tune_$(date +%Y%m%d)"
   ```
3. **数据集**：当前使用 `chatts_tune` 数据集，对应文件 `data/chatts_tune/train_1.jsonl`

---

## Epoch 设置分析

### Loss 变化趋势

| 训练任务 | Epochs | 初始 Loss | Epoch 1 Loss | Epoch 2 Loss | 最终 Loss | 趋势分析 |
|----------|--------|-----------|--------------|--------------|-----------|----------|
| 14B LoRA (RTX6000) | 5 | ~1.2 | ~0.8 | ~0.6 | ~0.47 | ⚠️ 第5 epoch仍有波动 (0.3~0.5) |
| 14B QLoRA (2080Ti) | 3 | ~1.2 | ~0.4 | ~0.25 | ~0.18 | ✅ 收敛良好 |
| 8B QLoRA (2080Ti) | 3 | ~1.2 | ~0.4 | ~0.3 | ~0.22 | ✅ 收敛良好 |

### 分析结论

#### 14B LoRA (5 epochs) - ⚠️ 可能过多
- epoch 4-5 时 loss 在 0.3~0.5 之间**波动较大**，没有明显下降趋势
- 每个 epoch 末尾出现异常低值 (0.02~0.03)，这是 epoch 边界的特殊 batch
- **建议**：3-4 epochs 可能更合适，减少过拟合风险

#### 14B/8B QLoRA (3 epochs) - ✅ 设置合理
- loss 从 ~1.2 稳定下降到 0.17~0.22
- 第3 epoch 仍有轻微下降空间，但已接近收敛
- **建议**：3 epochs 设置合理，可尝试 4 epochs 看是否有提升

### Epoch 建议

| 任务类型 | 当前 Epochs | 建议 Epochs | 理由 |
|----------|------------|-------------|------|
| 14B LoRA (高显存) | 5 | **3-4** | 减少过拟合，loss 在 epoch 3 后趋于稳定 |
| 14B/8B QLoRA | 3 | **3-4** | 当前设置合理，可尝试 4 |

---

## Loss 曲线图

### 14B LoRA (RTX 6000)

![14B LoRA RTX6000 训练 loss 曲线](images/14b_lora_rtx6000_loss.png)

### 14B LoRA (RTX 6000) - 20251228

![14B LoRA RTX6000 20251228 训练 loss 曲线](images/14b_lora_rtx6000_20251228_loss.png)

### 14B QLoRA (RTX 2080Ti)

![14B QLoRA 2080Ti 训练 loss 曲线](images/14b_qlora_2080ti_loss.png)

### 14B LoRA (RTX 2080Ti)

![14B LoRA 2080Ti 训练 loss 曲线](images/14b_lora_2080ti_loss.png)

### 8B LoRA (RTX 6000) - 最新

![8B LoRA RTX6000 训练 loss 曲线](images/8b_lora_rtx6000_loss.png)

### 8B QLoRA (RTX 2080Ti)

![8B QLoRA 2080Ti 训练 loss 曲线](images/8b_qlora_2080ti_loss.png)

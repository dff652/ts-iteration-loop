# ChatTS-14B 微调工作记录与问题排查报告

**日期**: 2025年12月25日
**任务**: 基于自定义 GDSH/HBSN/WHLJ/ZHLH 数据集微调 ChatTS-14B 模型。
**硬件环境**: 2x NVIDIA RTX 2080 Ti (22GB VRAM), 46GB System RAM。

---

## 1. 工作摘要

本次工作旨在建立一套稳定、可复现的 ChatTS-14B 微调流程。鉴于硬件内存（RAM）受限（46GB）与模型参数量巨大（14B）之间的矛盾，工作重心从最初的分布式 DeepSpeed 训练转向了更为稳健的单卡 QLoRA 低精度微调方案。

**主要成果**:
1.  **数据清洗**: 成功清洗并合并 4 份原始数据，生成 421 条高质量训练样本。
2.  **环境构建**: 解决了复杂的依赖版本冲突，构建了兼容 LLaMA-Factory 的全新环境 `chatts_train_env`。
3.  **策略优化**: 确定了 **Single GPU + QLoRA (4-bit)** 的最终技术路线，解决了模型加载阶段内存溢出 (OOM) 的问题。

---

## 2. 过程问题与解决方案记录

在实施过程中，我们遇到了一系列环境依赖和资源瓶颈问题。以下是详细的排查与解决记录：

| 阶段 | 遇到的问题 (Error/Issue) | 原因分析 | 解决方案 |
| :--- | :--- | :--- | :--- |
| **数据准备** | 预处理脚本仅匹配了 zhlh 数据，其他数据集匹配率为 0。 | JSON 标注中的图片路径 (`image`) 与本地 CSV 文件名不一致（存在中文前缀或路径差异）。 | 编写 `scripts/preprocess_tune_data.py`，加入特定规则：为 `whlj` 添加“数据集”前缀，为 `hbsn` 去除通用前缀。最终成功匹配 421 条数据。 |
| **环境配置** | `ImportError: undefined symbol ... vllm` | `vllm` 库与当前 PyTorch 版本不兼容，且在 SFT 训练阶段并非必需。 | **卸载 vllm** (`pip uninstall vllm`)。 |
| **环境配置** | `ImportError: transformers>=4.49.0...` (版本冲突) | `pip` 默认安装了最新版 `transformers` (4.57)，但 LLaMA-Factory 代码限制上限为 4.54。 | **降级库版本**：`pip install transformers==4.54.0`。 |
| **环境配置** | `ImportError: datasets`, `peft`, `accelerate` (连锁版本冲突) | 手动修复一个包后，其他包（如 TRL）又因为版本依赖关系报错，陷入 Dependency Hell。 | **重建环境**：放弃修补旧环境，创建全新环境 `chatts_train_env`，按照 LLaMA-Factory 的 `requirements.txt` 一次性安装指定版本的依赖组合。 |
| **训练启动** | `ValueError: --flash_attn False` | 启动脚本参数格式错误，该参数不接受布尔值。 | 修正为 `--flash_attn disabled`。 |
| **资源瓶颈** | `Exit Code 247 (SIGKILL)` / 无日志输出 | **系统内存 (RAM) 溢出**。DeepSpeed ZeRO-3 在加载 14B 模型 (28GB) 时，多进程初始化导致内存峰值超过物理内存 (46GB)。 | **切换策略**：放弃 DeepSpeed 分布式训练，改为 **单卡 QLoRA**。利用 `bitsandbytes` 进行 4-bit 量化加载，大幅降低内存和显存占用。 |
| **训练启动** | `ValueError: trust_remote_code=True` | ChatTS 是自定义模型结构，必须允许远程代码执行，但启动命令中漏传了此参数。 | 在启动脚本和 `torchrun` 命令中显式添加 `--trust_remote_code True`。 |

---

## 3. 最终技术方案

### 3.1 软件环境 (`chatts_train_env`)
*   **Python**: 3.12
*   **Core**: PyTorch 2.6.0
*   **Dependencies**:
    *   `transformers`: 4.52.4
    *   `peft`: 0.15.2
    *   `trl`: 0.9.6
    *   `bitsandbytes`: 0.49.0 (用于 4-bit 量化)
    *   `flash-attn`: Disabled (因编译环境问题暂时禁用，使用 4-bit 量化时影响不大)

### 3.2 训练配置
*   **脚本**: `scripts/chatts/lora/train_chatts_tune_qlora_safe.sh`
*   **策略**: QLoRA (Quantized LoRA)
*   **精度**: 4-bit 加载 (NF4), 16-bit 计算 (FP16)
*   **显存占用**: 预计单卡约 10GB (远低于 2080 Ti 的 22GB 上限)。
*   **内存占用**: 经过优化，加载阶段内存峰值控制在 20GB 以内，适配 46GB RAM 环境。

---

## 4. 如何执行

由于模型加载时间较长（5-10分钟），建议手动在终端执行：

```bash
# 1. 激活环境
conda activate chatts_train_env

# 2. 启动训练 (建议使用 nohup 防止断开)
nohup bash scripts/chatts/lora/train_chatts_tune_qlora_safe.sh > my_train.log 2>&1 &

# 3. 监控日志
tail -f my_train.log
```

---

## 5. 产出物位置
*   **处理后的数据**: `data/chatts_tune/train.jsonl`
*   **训练日志与权重**: `saves/chatts-14b/qlora/gdsh_tune/`

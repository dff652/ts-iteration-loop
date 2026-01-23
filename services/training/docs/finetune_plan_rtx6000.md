# ChatTS-14B RTX 6000 Ada 微调方案

**日期**: 2025年12月25日
**任务**: 基于自定义数据微调 ChatTS-14B 模型。
**硬件**: 2x NVIDIA RTX 6000 Ada Generation (48GB VRAM per GPU)。相比之前的 RTX 2080 Ti，显存充裕，不再需要激进的 4-bit 量化 (QLoRA)。

## 1. 环境准备

我们将创建一个新的 Conda 环境，参考之前的踩坑记录，但根据 RTX 6000 Ada (Ampere/Ada 架构) 进行适配。

**软件栈**:
- Python: 3.11 (或 3.12，3.11兼容性通常更好)
- PyTorch: 2.4.0+ (适配 CUDA 12/13)
- Transformers: 4.46.0+ (ChatTS 需要较新版本支持，但要避免过新导致 LLaMA-Factory 兼容问题，建议 4.46~4.48)
- DeepSpeed: 最新版

**Conda 创建命令**:
```bash
conda create -n chatts_env python=3.11 -y
conda activate chatts_env
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install "transformers>=4.46.0,<4.50.0" "peft>=0.10.0" "trl>=0.8.0" "accelerate>=0.30.0"
pip install deepspeed bitsandbytes scipy scikit-learn pandas matplotlib
# 安装当前目录下的包 (ChatTS/LLaMA-Factory)
pip install -e .
pip install -r requirements.txt
```

## 2. 数据准备

- **数据路径**: `/home/douff/ts/ChatTS-Training/data/chatts_tune/train.jsonl`
- **数据格式检查**: 已确认为标准的 JSONL 格式，包含 `<ts><ts/>` 占位符和嵌入的数值 `timeseries` 列表。
- **Dataset Info**: 需要确保 `data/dataset_info.json` 中已经注册了该数据集。

## 3. 训练配置

我们将使用 **LoRA** 微调，FP16 或 BF16 (Ada 架构支持 BF16，建议使用 BF16 以获得更好的数值稳定性)。

**脚本**: `scripts/lora/train_chatts_14b_rtx6000.sh`

**关键参数**:
- `model_name_or_path`: `/home/share/llm_models/bytedance-research/ChatTS-14B`
- `dataset`: `chatts_tune` (需在 dataset_info.json 确认)
- `finetuning_type`: `lora`
- `lora_rank`: 16 (显存足够，可以适当增加 rank)
- `per_device_train_batch_size`: 4 (根据显存调整)
- `gradient_accumulation_steps`: 8 (等效 batch size = 4 * 8 * 2(gpus) = 64)
- `cutoff_len`: 4096 (支持长时序)
- `learning_rate`: 2e-5
- `bf16`: True (开启 BF16)
- `deepspeed`: `ds_config/ds_config_2.json` (ZeRO-2 足够，甚至不用 DeepSpeed 直接 DDP 也可以，但 DS 效率通常更高)

## 4. 步骤

1.  **环境安装**: 执行上述 Conda 和 Pip 命令。
2.  **配置检查**: 检查 `data/dataset_info.json` 是否包含 `chatts_tune`。
3.  **脚本生成**: 创建 `scripts/lora/train_chatts_14b_rtx6000.sh`。
4.  **执行训练**: 运行脚本。

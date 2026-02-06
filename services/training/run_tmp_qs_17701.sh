#!/bin/bash

# Qwen3-VL-8B LoRA Fine-tuning Script
# Adapted for ts-iteration-loop platform

# Environment variables
export NCCL_DEBUG=WARN
# export CUDA_VISIBLE_DEVICES=0,1 # Managed by platform/docker

# Configuration
MODEL_PATH="/home/share/models/Qwen3-VL-8B-TR"
DATASET="qwen_converted_1_20260202"
DATASET_DIR="${DATASET_DIR:-/home/share/data/training_qwen}"
OUTPUT_DIR="/home/douff/ts/ts-iteration-loop/services/training/saves/qwen/train_qwen_vl_test"

# Run training
# Use NPROC_PER_NODE to control DDP world size (default: single GPU to avoid NCCL/NVML issues).

NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
if [ "${NPROC_PER_NODE}" -le 1 ]; then
    RUN_CMD=(python src/train.py)
else
    RUN_CMD=(torchrun --nproc_per_node="${NPROC_PER_NODE}" src/train.py)
fi

"${RUN_CMD[@]}" \
    --stage sft \
    --do_train \
    --model_name_or_path "${MODEL_PATH}" \
    --dataset "${DATASET}" \
    --dataset_dir "${DATASET_DIR}" \
    --template "qwen" \
    --finetuning_type lora \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.1 \
    --lora_target "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
    --output_dir "${OUTPUT_DIR}" \
    --overwrite_output_dir \
    --per_device_train_batch_size 2 \
    --dataloader_pin_memory False \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --save_steps 50 \
    --learning_rate 2e-5 \
    --warmup_steps 0 \
    --num_train_epochs 3 \
    --plot_loss \
    \
    --save_only_model \
    --save_safetensors False \
    --trust_remote_code True \
    --cutoff_len 4096 \
    --image_max_pixels 3200000 \
    --image_min_pixels 1024 \
    --video_max_pixels 65536 \
    --video_min_pixels 256 \
    --freeze_vision_tower False \
    --freeze_multi_modal_projector False \
    --freeze_trainable_layers 2 \
    --freeze_trainable_modules all \
    --gradient_accumulation_steps 8 \
    --cutoff_len 4096 \
    --image_max_pixels 3200000 \
    --image_min_pixels 1024 \
    --bf16

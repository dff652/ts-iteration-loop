#!/bin/bash

# Qwen3-VL-8B LoRA Fine-tuning Script
# Adapted for ts-iteration-loop platform

# Environment variables
export NCCL_DEBUG=WARN
# export CUDA_VISIBLE_DEVICES=0,1 # Managed by platform/docker

# Configuration
MODEL_PATH="/home/share/models/Qwen3-VL-8B-TR"
DATASET="picture_data" # Default dataset, logic in train.sh was dynamic, here we set a default but platform overrides args
DATASET_DIR="${DATASET_DIR:-/home/share/data/training_qwen}"
OUTPUT_DIR="${OUTPUT_DIR:-saves/qwen3-vl-8b/lora/v1}"

# Run training
# Using torchrun for distributed training compatibility with platform standards
# Note: Ensure --template matches the template supported by LLaMA-Factory version in containers

torchrun --nproc_per_node=2 src/train.py \
    --stage sft \
    --do_train \
    --model_name_or_path "${MODEL_PATH}" \
    --dataset "${DATASET}" \
    --dataset_dir "${DATASET_DIR}" \
    --template "qwen" \
    --finetuning_type lora \
    --lora_rank 16 \
    --lora_alpha 16 \
    --lora_dropout 0.1 \
    --lora_target "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
    --output_dir "${OUTPUT_DIR}" \
    --overwrite_output_dir \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --save_steps 50 \
    --learning_rate 1e-4 \
    --warmup_steps 0 \
    --num_train_epochs 3.0 \
    --plot_loss \
    --bf16 \
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
    --freeze_trainable_modules all

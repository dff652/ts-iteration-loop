#!/bin/bash

# Environment variables
export NCCL_DEBUG=WARN 
export DEEPSPEED_TIMEOUT=120

# Configuration
MODEL_PATH="/home/data1/llm_models/bytedance-research/ChatTS-8B"
DATASET="chatts_tune"
OUTPUT_DIR="saves/chatts-8b/lora/gdsh_tune_ds3"

# Run training (Dual GPU DeepSpeed ZeRO-3)
torchrun --nproc_per_node=2 src/train.py \
    --deepspeed ds_config/ds_config_3_offload.json \
    --stage sft \
    --model_name_or_path "${MODEL_PATH}" \
    --dataset "${DATASET}" \
    --interleave_probs "1.0" \
    --do_train \
    --mix_strategy "interleave_over" \
    --template "chatts" \
    --finetuning_type lora \
    --lora_rank 8 \
    --lora_target "q_proj,k_proj,v_proj" \
    --flash_attn disabled \
    --output_dir "${OUTPUT_DIR}" \
    --overwrite_output_dir \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --save_steps 50 \
    --learning_rate 1e-4 \
    --timeseries_sft_lr 1e-4 \
    --warmup_ratio 0.05 \
    --num_train_epochs 3 \
    --plot_loss \
    --fp16 \
    --save_only_model \
    --save_safetensors False \
    --preprocessing_num_workers 4 \
    --trust_remote_code True \
    --cutoff_len 4096 \
    --low_cpu_mem_usage True

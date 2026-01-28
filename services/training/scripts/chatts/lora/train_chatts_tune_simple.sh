#!/bin/bash

# Configuration
MODEL_PATH="llm_models/ChatTS-14B"
DATASET="chatts_tune"
DATASET_DIR="${DATASET_DIR:-/home/share/data/training_chatts}"
OUTPUT_DIR="${OUTPUT_DIR:-saves/chatts-14b/qlora/gdsh_tune}"

# Run training (Pure Single GPU QLoRA - No DeepSpeed/Torchrun Overhead)
llamafactory-cli train \
    --stage sft \
    --model_name_or_path "${MODEL_PATH}" \
    --dataset "${DATASET}" \
    --dataset_dir "${DATASET_DIR}" \
    --interleave_probs "1.0" \
    --do_train \
    --mix_strategy "interleave_over" \
    --template "chatts" \
    --finetuning_type lora \
    --quantization_bit 4 \
    --lora_rank 8 \
    --lora_target "q_proj,k_proj,v_proj" \
    --flash_attn disabled \
    --output_dir "${OUTPUT_DIR}" \
    --overwrite_output_dir \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
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
    --preprocessing_num_workers 1 \
    --trust_remote_code True \
    --cutoff_len 4096

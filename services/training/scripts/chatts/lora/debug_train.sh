#!/bin/bash

# Environment variables
export CUDA_VISIBLE_DEVICES=0
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
DATASET_DIR="${DATASET_DIR:-/home/share/data/training_chatts}"

# Run command with python directly (Single GPU debug)
python src/train.py \
    --stage sft \
    --model_name_or_path /home/share/llm_models/bytedance-research/ChatTS-14B \
    --dataset chatts_tune \
    --dataset_dir "${DATASET_DIR}" \
    --template chatts \
    --finetuning_type lora \
    --lora_target q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
    --lora_rank 16 \
    --output_dir saves/chatts-14b/lora/debug_tune \
    --overwrite_output_dir \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-5 \
    --timeseries_sft_lr 2e-5 \
    --num_train_epochs 5.0 \
    --logging_steps 1 \
    --save_steps 10 \
    --save_total_limit 2 \
    --bf16 True \
    --plot_loss True \
    --trust_remote_code True \
    --cutoff_len 8192 \
    --preprocessing_num_workers 4

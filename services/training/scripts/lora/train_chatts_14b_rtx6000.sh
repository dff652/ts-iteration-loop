#!/bin/bash

# Environment variables
export CUDA_VISIBLE_DEVICES=0,1
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH

# Run command with torchrun (No DeepSpeed, DDP)
torchrun --nproc_per_node=2 --master_port=19901 src/train.py \
    --stage sft \
    --do_train \
    --model_name_or_path /home/share/llm_models/bytedance-research/ChatTS-14B \
    --dataset chatts_tune \
    --template chatts \
    --finetuning_type lora \
    --lora_target q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
    --lora_rank 16 \
    --output_dir "saves/chatts-14b/lora/rtx6000_tune_$(date +%Y%m%d)" \
    --overwrite_output_dir \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 2e-5 \
    --timeseries_sft_lr 2e-5 \
    --num_train_epochs 5.0 \
    --logging_steps 1 \
    --save_steps 20 \
    --save_total_limit 2 \
    --bf16 True \
    --plot_loss True \
    --trust_remote_code True \
    --cutoff_len 8192 \
    --preprocessing_num_workers 16

# 訓練配置說明 (Training Configuration)

本文檔解釋 ChatTS 訓練腳本中的核心參數。

## 1. 核心訓練參數

在 `scripts/chatts/full/dev.sh` 或您的自定義腳本中，以下參數至關重要：

- `--template chatts`: 必須使用 `chatts` 模板，以正確處理時序占位符和特殊 Token。
- `--timeseries_sft_lr 1e-5`: 設定時序編碼器（`ts_encoder`）的獨立學習率。通常建議與主學習率保持一致或略大。
- `--finetuning_type full`: 建議進行全量微調以獲得最佳效果。如果顯存受限，可使用 `lora`。
- `--cutoff_len 4000`: 設定上下文最大長度。由於 256 點的時序數據會映射為多個 Token，建議設定較大的長度（> 2048）。

## 2. 顯存優化 (DeepSpeed)

14B 模型在全量微調時需要較大的顯存。

- **ZeRO-3**：建議使用 `ds_config/ds_config_3.json`。
- **Offload**：如果顯存仍然不足（如單卡 A100/A800 80G 以下），可以使用帶有 `offload` 的配置：
  - `--deepspeed ds_config/ds_config_3_offload.json`

## 3. 啟動命令示例

```bash
NCCL_DEBUG=WARN DEEPSPEED_TIMEOUT=120 deepspeed --num_gpus 8 src/train.py \
    --deepspeed ds_config/ds_config_3.json \
    --stage sft \
    --model_name_or_path /path/to/inited_model \
    --dataset chatts_tune_data \
    --do_train \
    --template chatts \
    --finetuning_type full \
    --output_dir /path/to/output \
    --learning_rate 1e-5 \
    --timeseries_sft_lr 1e-5 \
    --num_train_epochs 3.0 \
    --fp16 True
```


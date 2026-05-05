#!/bin/bash
# Usage:
#   bash run_train.sh                          # synthetic smoke test (GPU 1)
#   bash run_train.sh /path/to/embeddings.npy  # real data (GPU 1)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ENV_FILE="/root/workspace/.env"
if [[ -f "$ENV_FILE" ]]; then
    while IFS= read -r line; do
        [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue
        if [[ "$line" == *"="* ]]; then
            key="${line%%=*}"
            val="${line#*=}"
            key="$(echo "$key" | xargs)"
            val="$(echo "$val" | xargs)"
            [[ -n "$key" ]] && export "$key=$val"
        fi
    done < "$ENV_FILE"
fi

export CUDA_VISIBLE_DEVICES=1

LOG_DIR="logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/train_${TIMESTAMP}.log"
TRAIN_PATH="/workspace/hyeongchan/datasets/ffhq256/dino_cls_features/train_features.bin"
VAL_PATH="/workspace/hyeongchan/datasets/ffhq256/dino_cls_features/eval_features.bin"

echo "Data: FFHQ256 DINO CLS  GPU: 1"
CMD="python train_run.py \
    --data_path $TRAIN_PATH \
    --val_data_path $VAL_PATH \
    --input_dim 768 \
    --latent_dim 3072 \
    --num_layers 4 \
    --epochs 200 \
    --batch_size 2048 \
    --accum_steps 1 \
    --lr 3e-4 \
    --warmup_epochs 5 \
    --beta_kl 1e-2 \
    --beta_kl_warmup_epochs 50 \
    --lambda_sigreg 0.05 \
    --sigreg_type epps_pulley \
    --sigreg_target z \
    --ep_num_points 33 \
    --num_projections 3072 \
    --save_every 10 \
    --wandb_project vae-sigreg-sae \
    --wandb_run_name ffhq256_dino_betakl1e2_proj3072_${TIMESTAMP} \
    --wandb_tags gpu1 ffhq256 dino beta_kl_1e-2 proj3072"

echo "Log  : $LOG_FILE"
nohup bash -c "CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 $CMD" > "$LOG_FILE" 2>&1 &
PID=$!
echo "PID  : $PID"
echo "Watch: tail -f $LOG_FILE"
echo "Kill : kill $PID"

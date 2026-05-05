#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TRAIN_PATH="${TRAIN_PATH:-/workspace/hyeongchan/datasets/ffhq256/dino_cls_features/train_features.bin}"
VAL_PATH="${VAL_PATH:-/workspace/hyeongchan/datasets/ffhq256/dino_cls_features/eval_features.bin}"
PYTHON_BIN="${PYTHON_BIN:-/root/workspace/miniconda3/envs/vae-sigreg-sae/bin/python}"
EPOCHS="${EPOCHS:-50}"
BETA_KL="${BETA_KL:-1e-3}"
SIG="${SIG:-0.5}"

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

if [[ ! -f "$TRAIN_PATH" ]]; then
  echo "Missing TRAIN_PATH: $TRAIN_PATH" >&2
  exit 1
fi

if [[ ! -f "$VAL_PATH" ]]; then
  echo "Missing VAL_PATH: $VAL_PATH" >&2
  exit 1
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing PYTHON_BIN: $PYTHON_BIN" >&2
  exit 1
fi

mkdir -p logs checkpoints
TS="$(date +%Y%m%d_%H%M%S)"

declare -a RUNS=(
  "s3_latent_1536:0:1536"
  "s3_latent_3072:1:3072"
  "s3_latent_6144:2:6144"
)

echo "Launching Stage 3 latent sweep at $TS"
echo "Train: $TRAIN_PATH"
echo "Val  : $VAL_PATH"
echo "beta_kl       : $BETA_KL"
echo "lambda_sigreg : $SIG"
echo "epochs        : $EPOCHS"

for spec in "${RUNS[@]}"; do
  IFS=":" read -r RUN GPU LATENT <<< "$spec"
  CKPT_DIR="checkpoints/${RUN}_${TS}"
  LOG_FILE="logs/${RUN}_${TS}.log"
  mkdir -p "$CKPT_DIR"

  CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 nohup "$PYTHON_BIN" train_run.py \
    --data_path "$TRAIN_PATH" \
    --val_data_path "$VAL_PATH" \
    --input_dim 768 \
    --latent_dim "$LATENT" \
    --num_layers 4 \
    --epochs "$EPOCHS" \
    --batch_size 2048 \
    --accum_steps 1 \
    --lr 3e-4 \
    --warmup_epochs 5 \
    --beta_kl "$BETA_KL" \
    --beta_kl_warmup_epochs 50 \
    --lambda_sigreg "$SIG" \
    --sigreg_type epps_pulley \
    --sigreg_target z \
    --ep_num_points 33 \
    --ep_slice_chunk_size 256 \
    --num_projections "$LATENT" \
    --ckpt_dir "$CKPT_DIR" \
    --save_every 10 \
    --wandb_project vae-sigreg-sae \
    --wandb_run_name "$RUN" \
    --wandb_tags stage3 latent_sweep "$RUN" "gpu_${GPU}" "latent_${LATENT}" "beta_${BETA_KL}" "sig_${SIG}" epps_pulley \
    > "$LOG_FILE" 2>&1 &

  PID="$!"
  echo "$RUN gpu=$GPU latent=$LATENT pid=$PID log=$LOG_FILE ckpt=$CKPT_DIR"
done

echo "Use: tail -f logs/<run>_${TS}.log"
echo "Use: nvidia-smi"

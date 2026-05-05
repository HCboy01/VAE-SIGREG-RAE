#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TRAIN_PATH="${TRAIN_PATH:-/workspace/hyeongchan/datasets/ffhq256/dino_cls_features/train_features.bin}"
VAL_PATH="${VAL_PATH:-/workspace/hyeongchan/datasets/ffhq256/dino_cls_features/eval_features.bin}"
EPOCHS="${EPOCHS:-200}"

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
  echo "Set TRAIN_PATH=/path/to/train_features.bin and rerun." >&2
  exit 1
fi

if [[ ! -f "$VAL_PATH" ]]; then
  echo "Missing VAL_PATH: $VAL_PATH" >&2
  echo "Set VAL_PATH=/path/to/eval_features.bin and rerun." >&2
  exit 1
fi

mkdir -p logs checkpoints
TS="$(date +%Y%m%d_%H%M%S)"

declare -a RUNS=(
  "s1_kl_3e-5:0:3072:3e-5:0.05:3072:epps_pulley"
  "s1_kl_1e-4:1:3072:1e-4:0.05:3072:epps_pulley"
  "s1_kl_3e-4:2:3072:3e-4:0.05:3072:epps_pulley"
  "s1_kl_1e-3:3:3072:1e-3:0.05:3072:epps_pulley"
  "s1_kl_3e-3:4:3072:3e-3:0.05:3072:epps_pulley"
  "s1_kl_1e-2:5:3072:1e-2:0.05:3072:epps_pulley"
  "s1_kl_3e-2:6:3072:3e-2:0.05:3072:epps_pulley"
  "s1_kl_no_sig_probe:7:3072:1e-2:0.0:3072:none"
)

echo "Launching Stage 1 sweep at $TS"
echo "Train: $TRAIN_PATH"
echo "Val  : $VAL_PATH"

for spec in "${RUNS[@]}"; do
  IFS=":" read -r RUN GPU LATENT BETA SIG PROJ SIGREG_TYPE <<< "$spec"
  CKPT_DIR="checkpoints/${RUN}_${TS}"
  LOG_FILE="logs/${RUN}_${TS}.log"
  mkdir -p "$CKPT_DIR"

  CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 nohup python train_run.py \
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
    --beta_kl "$BETA" \
    --beta_kl_warmup_epochs 50 \
    --lambda_sigreg "$SIG" \
    --sigreg_type "$SIGREG_TYPE" \
    --sigreg_target z \
    --ep_num_points 33 \
    --ep_slice_chunk_size 256 \
    --num_projections "$PROJ" \
    --ckpt_dir "$CKPT_DIR" \
    --save_every 10 \
    --wandb_project vae-sigreg-sae \
    --wandb_run_name "$RUN" \
    --wandb_tags stage1 ablation "$RUN" "gpu_${GPU}" "latent_${LATENT}" "beta_${BETA}" "sig_${SIG}" "$SIGREG_TYPE" \
    > "$LOG_FILE" 2>&1 &

  PID="$!"
  echo "$RUN gpu=$GPU pid=$PID log=$LOG_FILE ckpt=$CKPT_DIR"
done

echo "Use: tail -f logs/<run>_${TS}.log"
echo "Use: nvidia-smi"

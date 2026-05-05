#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TRAIN_PATH="${TRAIN_PATH:-/workspace/hyeongchan/datasets/ffhq256/dino_cls_features/train_features.bin}"
VAL_PATH="${VAL_PATH:-/workspace/hyeongchan/datasets/ffhq256/dino_cls_features/eval_features.bin}"
PYTHON_BIN="${PYTHON_BIN:-/root/workspace/miniconda3/envs/vae-sigreg-sae/bin/python}"
LATENT="${LATENT:-3072}"
EPOCHS="${EPOCHS:-50}"
BETA_KL="${BETA_KL:-1e-3}"
BETA_TAG="${BETA_TAG:-$(echo "$BETA_KL" | sed 's/[^A-Za-z0-9]/_/g')}"
SIG_GRID="${SIG_GRID:-0 0.01 0.03 0.05 0.1 0.2 0.5 1.0}"
LAUNCH_SLEEP="${LAUNCH_SLEEP:-5}"
export WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-300}"

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

[[ -f "$TRAIN_PATH" ]] || { echo "Missing TRAIN_PATH: $TRAIN_PATH" >&2; exit 1; }
[[ -f "$VAL_PATH" ]] || { echo "Missing VAL_PATH: $VAL_PATH" >&2; exit 1; }
[[ -x "$PYTHON_BIN" ]] || { echo "Missing PYTHON_BIN: $PYTHON_BIN" >&2; exit 1; }

read -r -a SIGS <<< "$SIG_GRID"
if [[ "${#SIGS[@]}" -lt 1 || "${#SIGS[@]}" -gt 8 ]]; then
  echo "SIG_GRID must contain 1 to 8 values for up to 8 GPUs." >&2
  echo "Current SIG_GRID: $SIG_GRID" >&2
  exit 1
fi

mkdir -p logs checkpoints
TS="$(date +%Y%m%d_%H%M%S)"

echo "Launching per-KL SIGReg sweep at $TS"
echo "Train        : $TRAIN_PATH"
echo "Val          : $VAL_PATH"
echo "latent_dim   : $LATENT"
echo "beta_kl      : $BETA_KL"
echo "epochs       : $EPOCHS"
echo "sig_grid     : $SIG_GRID"
echo "wandb timeout: $WANDB_INIT_TIMEOUT sec"
echo "selection    : choose the largest SIG before val/active_units drops sharply"

for IDX in "${!SIGS[@]}"; do
  GPU="$IDX"
  SIG="${SIGS[$IDX]}"
  SIG_TAG="$(echo "$SIG" | sed 's/[^A-Za-z0-9]/_/g')"
  if [[ "$SIG" == "0" || "$SIG" == "0.0" ]]; then
    SIGREG_TYPE="none"
    RUN="s4_b${BETA_TAG}_sig_0"
  else
    SIGREG_TYPE="epps_pulley"
    RUN="s4_b${BETA_TAG}_sig_${SIG_TAG}"
  fi

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
    --sigreg_type "$SIGREG_TYPE" \
    --sigreg_target z \
    --ep_num_points 33 \
    --ep_slice_chunk_size 256 \
    --num_projections "$LATENT" \
    --ckpt_dir "$CKPT_DIR" \
    --save_every 10 \
    --wandb_project vae-sigreg-sae \
    --wandb_run_name "$RUN" \
    --wandb_tags stage4 per_beta_sigreg active_drop "$RUN" "gpu_${GPU}" "latent_${LATENT}" "beta_${BETA_KL}" "sig_${SIG}" "$SIGREG_TYPE" \
    > "$LOG_FILE" 2>&1 &

  PID="$!"
  echo "$RUN gpu=$GPU sig=$SIG pid=$PID log=$LOG_FILE ckpt=$CKPT_DIR"
  sleep "$LAUNCH_SLEEP"
done

echo "Use: tail -f logs/s4_b${BETA_TAG}_sig_<sig>_${TS}.log"
echo "Use: nvidia-smi"

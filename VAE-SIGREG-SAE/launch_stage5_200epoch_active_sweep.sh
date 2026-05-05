#!/usr/bin/env bash
set -euo pipefail

# Stage 5: for one beta_kl, measure which SIGReg lambda survives 200 epochs
# without active-unit collapse. Runs up to 8 lambdas on GPUs 0..7.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TRAIN_PATH="${TRAIN_PATH:-/workspace/hyeongchan/datasets/ffhq256/dino_cls_features/train_features.bin}"
VAL_PATH="${VAL_PATH:-/workspace/hyeongchan/datasets/ffhq256/dino_cls_features/eval_features.bin}"
PYTHON_BIN="${PYTHON_BIN:-/root/workspace/miniconda3/envs/vae-sigreg-sae/bin/python}"
LATENT="${LATENT:-3072}"
EPOCHS="${EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-2048}"
BETA_KL="${BETA_KL:-1e-4}"
BETA_TAG="${BETA_TAG:-$(echo "$BETA_KL" | sed 's/[^A-Za-z0-9]/_/g')}"
SIG_GRID="${SIG_GRID:-0 1 3 10 30 100 300 1000}"
MIN_ACTIVE_UNITS="${MIN_ACTIVE_UNITS:-2500}"
DEAD_THRESHOLDS="${DEAD_THRESHOLDS:-0.1 0.5}"
LAUNCH_SLEEP="${LAUNCH_SLEEP:-10}"
export WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-600}"

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
  echo "SIG_GRID must contain 1 to 8 values for GPUs 0..7." >&2
  echo "Current SIG_GRID: $SIG_GRID" >&2
  exit 1
fi

mkdir -p logs checkpoints
TS="$(date +%Y%m%d_%H%M%S)"

echo "Launching Stage 5 200-epoch active survival sweep at $TS"
echo "beta_kl         : $BETA_KL"
echo "latent_dim      : $LATENT"
echo "epochs          : $EPOCHS"
echo "batch_size      : $BATCH_SIZE"
echo "sig_grid        : $SIG_GRID"
echo "grid policy     : shared x-values across beta_kl for comparable S-curves"
echo "min_active_units: $MIN_ACTIVE_UNITS"
echo "dead target     : mu"
echo "dead thresholds : $DEAD_THRESHOLDS"
echo "checkpoint thr  : 0.1"
echo "checkpoint rule : best val/total_loss among epochs with ever_active_units@0.1 >= min_active_units"

for IDX in "${!SIGS[@]}"; do
  GPU="$IDX"
  SIG="${SIGS[$IDX]}"
  SIG_TAG="$(echo "$SIG" | sed 's/[^A-Za-z0-9]/_/g')"
  if [[ "$SIG" == "0" || "$SIG" == "0.0" ]]; then
    SIGREG_TYPE="none"
    RUN="s5_b${BETA_TAG}_sig_0_200ep"
  else
    SIGREG_TYPE="epps_pulley"
    RUN="s5_b${BETA_TAG}_sig_${SIG_TAG}_200ep"
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
    --batch_size "$BATCH_SIZE" \
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
    --best_metric active_safe_total \
    --min_active_units "$MIN_ACTIVE_UNITS" \
    --dead_activation_target mu \
    --dead_activation_threshold 0.1 \
    --dead_activation_thresholds $DEAD_THRESHOLDS \
    --wandb_project vae-sigreg-sae \
    --wandb_run_name "$RUN" \
    --wandb_tags stage5 active_survival 200epoch "$RUN" "gpu_${GPU}" "latent_${LATENT}" "beta_${BETA_KL}" "sig_${SIG}" "$SIGREG_TYPE" \
    > "$LOG_FILE" 2>&1 &

  PID="$!"
  echo "$RUN gpu=$GPU sig=$SIG pid=$PID log=$LOG_FILE ckpt=$CKPT_DIR"
  sleep "$LAUNCH_SLEEP"
done

echo "Use: tail -f logs/s5_b${BETA_TAG}_sig_<sig>_200ep_${TS}.log"
echo "Use: nvidia-smi"

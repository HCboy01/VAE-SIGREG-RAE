#!/usr/bin/env bash
# Train β=1e-4/3e-4/1e-3 with λ=0 (no SIGReg) on GPUs 0/1/2.
# When done, copies best.pt to best_ckpt/ for the heatmap figure.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/root/workspace/miniconda3/envs/vae-sigreg-sae/bin/python}"
TRAIN_PATH="${TRAIN_PATH:-/workspace/hyeongchan/datasets/ffhq256/dino_cls_features/train_features.bin}"
VAL_PATH="${VAL_PATH:-/workspace/hyeongchan/datasets/ffhq256/dino_cls_features/eval_features.bin}"

[[ -f "$TRAIN_PATH" ]] || { echo "Missing TRAIN_PATH: $TRAIN_PATH" >&2; exit 1; }
[[ -f "$VAL_PATH" ]]   || { echo "Missing VAL_PATH: $VAL_PATH" >&2; exit 1; }
[[ -x "$PYTHON_BIN" ]] || { echo "Missing PYTHON_BIN: $PYTHON_BIN" >&2; exit 1; }

mkdir -p logs checkpoints best_ckpt
TS="$(date +%Y%m%d_%H%M%S)"

declare -A GPU_FOR_BETA=([1e-4]=0 [3e-4]=1 [1e-3]=2)

PIDS=()
CKPT_DIRS=()
BETAS=(1e-4 3e-4 1e-3)

for BETA in "${BETAS[@]}"; do
    GPU="${GPU_FOR_BETA[$BETA]}"
    BETA_TAG="$(echo "$BETA" | sed 's/[^A-Za-z0-9]/_/g')"
    RUN="s5_b${BETA_TAG}_sig_0_200ep"
    CKPT_DIR="checkpoints/${RUN}_${TS}"
    LOG_FILE="logs/${RUN}_${TS}.log"
    mkdir -p "$CKPT_DIR"
    CKPT_DIRS+=("$CKPT_DIR")

    echo "Launching beta=$BETA lambda=0  gpu=$GPU  log=$LOG_FILE"
    CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 nohup "$PYTHON_BIN" train_run.py \
        --data_path    "$TRAIN_PATH" \
        --val_data_path "$VAL_PATH" \
        --input_dim    768 \
        --latent_dim   3072 \
        --num_layers   4 \
        --epochs       200 \
        --batch_size   2048 \
        --accum_steps  1 \
        --lr           3e-4 \
        --warmup_epochs 5 \
        --beta_kl      "$BETA" \
        --beta_kl_warmup_epochs 50 \
        --lambda_sigreg 0 \
        --sigreg_type  none \
        --sigreg_target z \
        --num_projections 3072 \
        --ckpt_dir     "$CKPT_DIR" \
        --save_every   10 \
        --best_metric  total_loss \
        --dead_activation_target mu \
        --dead_activation_threshold 0.1 \
        --dead_activation_thresholds 0.1 0.5 \
        --wandb_project  vae-sigreg-sae \
        --wandb_run_name "$RUN" \
        --wandb_tags stage5 null_ablation 200epoch "beta_${BETA}" sig_0 none \
        > "$LOG_FILE" 2>&1 &
    PIDS+=("$!")
done

echo ""
echo "Waiting for ${#PIDS[@]} jobs (PIDs: ${PIDS[*]})..."
FAILED=0
for i in "${!PIDS[@]}"; do
    PID="${PIDS[$i]}"
    BETA="${BETAS[$i]}"
    if wait "$PID"; then
        echo "beta=$BETA done"
    else
        echo "beta=$BETA FAILED (pid=$PID)" >&2
        FAILED=1
    fi
done

if [[ "$FAILED" -eq 1 ]]; then
    echo "One or more training jobs failed. Aborting checkpoint copy." >&2
    exit 1
fi

echo ""
echo "Copying best checkpoints to best_ckpt/..."
for i in "${!BETAS[@]}"; do
    BETA="${BETAS[$i]}"
    SRC="${CKPT_DIRS[$i]}/best.pt"
    DST="best_ckpt/beta_${BETA}_lambda_0_best.pt"
    if [[ -f "$SRC" ]]; then
        cp "$SRC" "$DST"
        echo "  $DST"
    else
        echo "  WARNING: $SRC not found, skipping" >&2
    fi
done

echo ""
echo "Done. Run the figure script:"
echo "  $PYTHON_BIN scripts/plot_stage5_paper_figures.py"

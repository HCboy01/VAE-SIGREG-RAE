#!/usr/bin/env bash
# Train NCT-CRC and WikiArt with β=3e-4, λ=3 on GPUs 0 and 1.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/root/workspace/miniconda3/envs/vae-sigreg-sae/bin/python}"

NCT_TRAIN="/workspace/NCT-CRC-HE-100K/dino_cls_features/train_features.bin"
NCT_VAL="/workspace/NCT-CRC-HE-100K/dino_cls_features/eval_features.bin"
WIKI_TRAIN="/workspace/wikiart_processed/train_features.bin"
WIKI_VAL="/workspace/wikiart_processed/eval_features.bin"

[[ -x "$PYTHON_BIN" ]] || { echo "Missing PYTHON_BIN: $PYTHON_BIN" >&2; exit 1; }
for f in "$NCT_TRAIN" "$NCT_VAL" "$WIKI_TRAIN" "$WIKI_VAL"; do
    [[ -f "$f" ]] || { echo "Missing: $f" >&2; exit 1; }
done

mkdir -p logs checkpoints
TS="$(date +%Y%m%d_%H%M%S)"

ENV_FILE="/root/workspace/.env"
if [[ -f "$ENV_FILE" ]]; then
    while IFS= read -r line; do
        [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue
        if [[ "$line" == *"="* ]]; then
            key="${line%%=*}"; val="${line#*=}"
            key="$(echo "$key" | xargs)"; val="$(echo "$val" | xargs)"
            [[ -n "$key" ]] && export "$key=$val"
        fi
    done < "$ENV_FILE"
fi

common_args=(
    --input_dim 768
    --latent_dim 3072
    --num_layers 4
    --epochs 200
    --batch_size 2048
    --accum_steps 1
    --lr 3e-4
    --warmup_epochs 5
    --beta_kl 3e-4
    --beta_kl_warmup_epochs 50
    --lambda_sigreg 3
    --sigreg_type epps_pulley
    --sigreg_target z
    --ep_num_points 33
    --ep_slice_chunk_size 256
    --num_projections 3072
    --save_every 10
    --best_metric active_safe_total
    --min_active_units 2000
    --dead_activation_target mu
    --dead_activation_threshold 0.1
    --dead_activation_thresholds 0.1 0.5
    --wandb_project vae-sigreg-sae
)

# ── NCT-CRC ──────────────────────────────────────────────────────────────────
NCT_RUN="nct_b3e-4_sig3_200ep"
NCT_CKPT="checkpoints/${NCT_RUN}_${TS}"
NCT_LOG="logs/${NCT_RUN}_${TS}.log"
mkdir -p "$NCT_CKPT"

echo "Launching NCT  gpu=0  log=$NCT_LOG"
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 nohup "$PYTHON_BIN" train_run.py \
    --data_path     "$NCT_TRAIN" \
    --val_data_path "$NCT_VAL" \
    --ckpt_dir      "$NCT_CKPT" \
    --wandb_run_name "$NCT_RUN" \
    --wandb_tags nct b3e-4 sig3 200epoch \
    "${common_args[@]}" \
    > "$NCT_LOG" 2>&1 &
NCT_PID=$!

# ── WikiArt ──────────────────────────────────────────────────────────────────
WIKI_RUN="wikiart_b3e-4_sig3_200ep"
WIKI_CKPT="checkpoints/${WIKI_RUN}_${TS}"
WIKI_LOG="logs/${WIKI_RUN}_${TS}.log"
mkdir -p "$WIKI_CKPT"

echo "Launching WikiArt  gpu=1  log=$WIKI_LOG"
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 nohup "$PYTHON_BIN" train_run.py \
    --data_path     "$WIKI_TRAIN" \
    --val_data_path "$WIKI_VAL" \
    --ckpt_dir      "$WIKI_CKPT" \
    --wandb_run_name "$WIKI_RUN" \
    --wandb_tags wikiart b3e-4 sig3 200epoch \
    "${common_args[@]}" \
    > "$WIKI_LOG" 2>&1 &
WIKI_PID=$!

echo ""
echo "Both jobs launched. Monitor:"
echo "  tail -f $NCT_LOG"
echo "  tail -f $WIKI_LOG"
echo ""

FAILED=0
wait "$NCT_PID"  && echo "NCT done"  || { echo "NCT FAILED"  >&2; FAILED=1; }
wait "$WIKI_PID" && echo "WikiArt done" || { echo "WikiArt FAILED" >&2; FAILED=1; }

[[ "$FAILED" -eq 1 ]] && exit 1

echo ""
echo "Checkpoints:"
echo "  NCT    → $NCT_CKPT/best.pt"
echo "  WikiArt → $WIKI_CKPT/best.pt"

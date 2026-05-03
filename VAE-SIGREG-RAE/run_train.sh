#!/usr/bin/env bash
set -euo pipefail

cd /scratch/x3411a10/VAE-SIGREG/VAE-SIGREG-RAE

export EXPERIMENT_NAME="${EXPERIMENT_NAME:-vae_sigreg_mu_cond}"

python src/train_vae_cond.py \
  --config configs/stage2/training/ImageNet256/DiTDH-XL_DINOv2-B_VAESIGREG.yaml \
  --data-path "${DATA_PATH:-/scratch/x3411a10/datasets/ffhq256/imagefolder/train}" \
  --results-dir "${RESULTS_DIR:-ckpts}" \
  --image-size "${IMAGE_SIZE:-256}" \
  --precision "${PRECISION:-bf16}" \
  --workers "${WORKERS:-8}" \
  "$@"

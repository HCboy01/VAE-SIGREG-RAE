#!/bin/bash
# Weaker reconstruction + compact latent regularizer:
#   0.05 * [mean(mu^2) + mean(logsigma^2) + mean(|mu * sigma|)]
# with the original KL term disabled.

set -eo pipefail

cd /scratch/x3411a10/IGAE/stage1_IGAE
module purge
module load conda/pytorch_2.5.0
source /home01/x3411a10/.bashrc

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATA_PATH="/scratch/x3411a10/datasets/ffhq256/dino_features_train.pt"
RUN_TAG="s1_grid_b0_lreg_musq0p05_logsigl2a0p05_musigl1l0p05"
CKPT_DIR="/scratch/x3411a10/IGAE/stage1_IGAE/checkpoints/${RUN_TAG}"
LOG="logs/${RUN_TAG}.out"

echo "[$(date '+%F %T')] START ${RUN_TAG}" | tee -a "$LOG"

PYTHONUNBUFFERED=1 python train_run.py \
  --data_path     "$DATA_PATH" \
  --val_fraction  0.05 \
  --input_dim     768 \
  --latent_dim    6144 \
  --num_layers    4 \
  --epochs        200 \
  --batch_size    2048 \
  --accum_steps   1 \
  --lr            3e-4 \
  --weight_decay  1e-2 \
  --beta_kl       0 \
  --beta_kl_warmup_epochs 1 \
  --lambda_sigreg 0 \
  --lambda_mu_l1 0 \
  --lambda_logsigma_l1 0 \
  --lambda_mu_sq 0.05 \
  --lambda_logsigma_l2 0.05 \
  --lambda_mu_sigma_l1 0.05 \
  --num_projections 512 \
  --dead_activation_thresholds 0.1 0.5 1.0 \
  --ckpt_dir      "$CKPT_DIR" \
  --save_every    20 \
  --wandb_project "igae-stage1-lreg" \
  --wandb_run_name "$RUN_TAG" \
  --wandb_tags    "lreg" "mu_sq" "logsigma_l2" "mu_sigma_l1" "scale0p05" "beta0" "no_sigreg" \
  2>&1 | tee -a "$LOG"

echo "[$(date '+%F %T')] DONE ${RUN_TAG}" | tee -a "$LOG"

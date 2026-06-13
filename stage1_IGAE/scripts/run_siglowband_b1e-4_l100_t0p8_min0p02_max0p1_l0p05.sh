#!/bin/bash
# Soft sigma low-fraction band regularizer.
#
# For each feature, softly estimates frac_B[sigma < 0.8] and penalizes it
# outside [0.02, 0.10]. This encourages every feature to be selectively
# confident on a small image subset without becoming globally low-sigma.
#
# Run:
#   s1_grid_b1e-4_l100_siglowband_t0p8_min0p02_max0p1_l0p05

set -eo pipefail

cd /scratch/x3411a10/IGAE/stage1_IGAE
module purge
module load conda/pytorch_2.5.0
source /home01/x3411a10/.bashrc

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATA_PATH="/scratch/x3411a10/datasets/ffhq256/dino_features_train.pt"
RUN_TAG="s1_grid_b1e-4_l100_siglowband_t0p8_min0p02_max0p1_l0p05"
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
  --beta_kl       1e-4 \
  --beta_kl_warmup_epochs 50 \
  --lambda_sigreg 100 \
  --lambda_sigma_lowfrac 0.05 \
  --sigma_lowfrac_threshold 0.8 \
  --sigma_lowfrac_min 0.02 \
  --sigma_lowfrac_target 0.10 \
  --sigma_lowfrac_temperature 0.05 \
  --num_projections 512 \
  --dead_activation_thresholds 0.1 0.5 1.0 \
  --ckpt_dir      "$CKPT_DIR" \
  --save_every    20 \
  --wandb_project "igae-stage1-siglowband" \
  --wandb_run_name "$RUN_TAG" \
  --wandb_tags    "siglowband" "tau0p8" "min0p02" "max0p1" "lambda0p05" "beta1e-4" "lambda100" \
  2>&1 | tee -a "$LOG"

echo "[$(date '+%F %T')] DONE ${RUN_TAG}" | tee -a "$LOG"

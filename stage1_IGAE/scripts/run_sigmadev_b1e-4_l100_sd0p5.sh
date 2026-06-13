#!/bin/bash
# Compare stronger batch feature-axis sigma deviation penalties at beta=1e-4, lambda_sigreg=100.
#
# Runs:
#   s1_grid_b1e-4_l100_sigmadev_l1_sd0p5_below
#   s1_grid_b1e-4_l100_sigmadev_l2_sd0p5_below

set -eo pipefail

cd /scratch/x3411a10/IGAE/stage1_IGAE
module purge
module load conda/pytorch_2.5.0
source /home01/x3411a10/.bashrc

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATA_PATH="/scratch/x3411a10/datasets/ffhq256/dino_features_train.pt"
BETA="1e-4"
LAMBDA_SIGREG="100"
LAMBDA_SIGMA_DEV="0.5"

for METRIC in l1 l2; do
  RUN_TAG="s1_grid_b1e-4_l100_sigmadev_${METRIC}_sd0p5_below"
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
    --beta_kl       "$BETA" \
    --beta_kl_warmup_epochs 50 \
    --lambda_sigreg "$LAMBDA_SIGREG" \
    --lambda_sigma_dev "$LAMBDA_SIGMA_DEV" \
    --sigma_dev_metric "$METRIC" \
    --sigma_dev_only_below \
    --num_projections 512 \
    --dead_activation_thresholds 0.1 0.5 1.0 \
    --ckpt_dir      "$CKPT_DIR" \
    --save_every    20 \
    --wandb_project "igae-stage1-sigmadev" \
    --wandb_run_name "$RUN_TAG" \
    --wandb_tags    "sigmadev" "$METRIC" "only_below" "sd0p5" "beta${BETA}" "lambda${LAMBDA_SIGREG}" \
    2>&1 | tee -a "$LOG"

  echo "[$(date '+%F %T')] DONE ${RUN_TAG}" | tee -a "$LOG"
done

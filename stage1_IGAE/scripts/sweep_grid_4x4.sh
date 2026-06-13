#!/bin/bash
# 4×4 grid sweep: β ∈ {0, 1e-3, 1e-2, 1e-1} × λ ∈ {0, 1, 10, 100}
#
# IDX  β      λ      skip?
#  0   0      0
#  1   0      1
#  2   0      10
#  3   0      100
#  4   1e-3   0
#  5   1e-3   1
#  6   1e-3   10
#  7   1e-3   100
#  8   1e-2   0
#  9   1e-2   1
# 10   1e-2   10
# 11   1e-2   100
# 12   1e-1   0
# 13   1e-1   1
# 14   1e-1   10
# 15   1e-1   100
#
# Usage:
#   for IDX in {0..15}; do SLURM_ARRAY_TASK_ID=$IDX bash scripts/sweep_grid_4x4.sh; done

set -e
module purge
module load conda/pytorch_2.5.0
source /home01/x3411a10/.bashrc

BETAS=(  0    0    0    0    1e-3  1e-3  1e-3  1e-3  1e-2  1e-2  1e-2  1e-2  1e-1  1e-1  1e-1  1e-1 )
LAMBDAS=( 0    1   10  100    0     1    10   100     0     1    10   100     0     1    10   100  )

IDX=${SLURM_ARRAY_TASK_ID:-0}
BETA=${BETAS[$IDX]}
LAMBDA=${LAMBDAS[$IDX]}

RUN_TAG="s1_grid_b${BETA}_l${LAMBDA}"
echo "[grid] idx=$IDX  beta=$BETA  lambda=$LAMBDA  tag=$RUN_TAG"

DATA_PATH="/scratch/x3411a10/datasets/ffhq256/dino_features_train.pt"
CKPT_DIR="/scratch/x3411a10/IGAE/stage1_IGAE/checkpoints/${RUN_TAG}"

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
  --lambda_sigreg "$LAMBDA" \
  --num_projections 512 \
  --dead_activation_thresholds 0.1 0.5 1.0 \
  --ckpt_dir      "$CKPT_DIR" \
  --save_every    20 \
  --wandb_project "igae-stage1-grid" \
  --wandb_run_name "$RUN_TAG" \
  --wandb_tags    "grid4x4" "beta${BETA}" "lambda${LAMBDA}"

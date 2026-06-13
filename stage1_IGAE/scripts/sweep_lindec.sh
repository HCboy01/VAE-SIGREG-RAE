#!/bin/bash
# Linear-decoder grid sweep for the default Stage-1 objective:
# sample KL + D-space Epps-Pulley SIGReg.
#
# Usage:
#   sbatch --array=0-18 scripts/sweep_lindec.sh
#   SLURM_ARRAY_TASK_ID=0 bash scripts/sweep_lindec.sh

#SBATCH -J s1_lindec
#SBATCH -p cas_v100_4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --comment=pytorch
#SBATCH -o logs/lindec_%a_%j.out
#SBATCH -e logs/lindec_%a_%j.err

set -e
module purge
module load conda/pytorch_2.5.0
source /home01/x3411a10/.bashrc

cd /scratch/x3411a10/IGAE/stage1_IGAE
mkdir -p logs

IDX=${SLURM_ARRAY_TASK_ID:-0}

NAMES=(
  s1_grid_b0_l0_lindec
  s1_grid_b0_l1_lindec
  s1_grid_b0_l10_lindec
  s1_grid_b0_l100_lindec
  s1_grid_b0_l1000_lindec
  s1_grid_b0_l10000_lindec
  s1_grid_b1e-1_l0_lindec
  s1_grid_b1e-1_l1_lindec
  s1_grid_b1e-1_l10_lindec
  s1_grid_b1e-1_l100_lindec
  s1_grid_b1e-2_l0_lindec
  s1_grid_b1e-2_l1_lindec
  s1_grid_b1e-2_l10_lindec
  s1_grid_b1e-2_l100_lindec
  s1_grid_b1e-3_l0_lindec
  s1_grid_b1e-3_l1_lindec
  s1_grid_b1e-3_l100_lindec
  s1_grid_b1e-4_l1_lindec
  s1_grid_b1e-4_l100_lindec
)
BETAS=(
  0      0      0      0      0      0
  1e-1   1e-1   1e-1   1e-1
  1e-2   1e-2   1e-2   1e-2
  1e-3   1e-3   1e-3
  1e-4   1e-4
)
LAMBDAS=(
  0      1      10     100    1000   10000
  0      1      10     100
  0      1      10     100
  0      1      100
  1      100
)
WARMUPS=(
  50     50     50     50     50     50
  50     50     50     50
  50     50     50     50
  50     50     50
  50     50
)

RUN_TAG=${NAMES[$IDX]}
BETA=${BETAS[$IDX]}
LAMBDA=${LAMBDAS[$IDX]}
WARMUP=${WARMUPS[$IDX]}

echo "[lindec] idx=$IDX  tag=$RUN_TAG  beta=$BETA  lambda=$LAMBDA  warmup=$WARMUP"

DATA_PATH="/scratch/x3411a10/datasets/ffhq256/dino_features_train.pt"
CKPT_DIR="/scratch/x3411a10/IGAE/stage1_IGAE/checkpoints/${RUN_TAG}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PYTHONUNBUFFERED=1 python train_run.py \
  --data_path     "$DATA_PATH" \
  --val_fraction  0.05 \
  --input_dim     768 \
  --latent_dim    6144 \
  --num_layers    4 \
  --linear_decoder \
  --epochs        200 \
  --batch_size    2048 \
  --accum_steps   1 \
  --lr            3e-4 \
  --weight_decay  1e-2 \
  --beta_kl       "$BETA" \
  --beta_kl_warmup_epochs "$WARMUP" \
  --lambda_sigreg "$LAMBDA" \
  --num_projections 512 \
  --dead_activation_thresholds 0.1 0.5 1.0 \
  --ckpt_dir      "$CKPT_DIR" \
  --save_every    20 \
  --wandb_project "igae-stage1-lindec" \
  --wandb_run_name "$RUN_TAG" \
  --wandb_tags    "lindec" "grid" "beta${BETA}" "lambda${LAMBDA}"

#!/bin/bash
# Linear-decoder sweep: same 32 configs as MLP sweep, decoder=nn.Linear(latent->input)
# Array index 0-31 maps to results_template_lindec.csv rows (excluding header)
#
# Usage:
#   sbatch --array=0-31 scripts/sweep_lindec.sh
#   # or single run:
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

# 32 configs: run_name  beta_kl  lambda_sigreg  warmup
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
  s1_lowb_b1e-4_l0_lindec
  s1_lowb_b1e-4_l10_lindec
  s1_lowb_b1e-4_l30_lindec
  s1_lowb_b2e-4_l0_lindec
  s1_lowb_b2e-4_l10_lindec
  s1_lowb_b2e-4_l30_lindec
  s1_tierA_b1e-3_l10_w50_lindec
  s1_tierA_b3e-4_l10_w50_lindec
  s1_tierA_b5e-4_l10_w50_lindec
  s1_tierA_b7e-4_l10_w50_lindec
  s1_tierB_b5e-4_l0_w50_lindec
  s1_tierB_b5e-4_l30_w50_lindec
  s1_tierC_b5e-4_l10_w100_lindec
)
BETAS=(
  0      0      0      0      0      0
  1e-1   1e-1   1e-1   1e-1
  1e-2   1e-2   1e-2   1e-2
  1e-3   1e-3   1e-3
  1e-4   1e-4
  1e-4   1e-4   1e-4
  2e-4   2e-4   2e-4
  1e-3   3e-4   5e-4   7e-4
  5e-4   5e-4
  5e-4
)
LAMBDAS=(
  0      1      10     100    1000   10000
  0      1      10     100
  0      1      10     100
  0      1      100
  1      100
  0      10     30
  0      10     30
  10     10     10     10
  0      30
  10
)
WARMUPS=(
  50     50     50     50     50     50
  50     50     50     50
  50     50     50     50
  50     50     50
  50     50
  50     50     50
  50     50     50
  50     50     50     50
  50     50
  100
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
  --batch_size    512 \
  --accum_steps   4 \
  --lr            3e-4 \
  --weight_decay  1e-2 \
  --beta_kl       "$BETA" \
  --beta_kl_warmup_epochs "$WARMUP" \
  --lambda_sigreg "$LAMBDA" \
  --sigreg_type   epps_pulley \
  --num_projections 512 \
  --dead_activation_thresholds 0.1 0.5 1.0 \
  --ckpt_dir      "$CKPT_DIR" \
  --save_every    20 \
  --wandb_project "igae-stage1-lindec" \
  --wandb_run_name "$RUN_TAG" \
  --wandb_tags    "lindec" "beta${BETA}" "lambda${LAMBDA}"

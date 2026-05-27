#!/bin/bash
# Stage-1 VAE-SIGREG sweep: 7 runs (Tier A × 4 + Tier B × 2 + Tier C × 1)
#
# Usage:
#   sbatch --array=0-6 sweep_stage1.sh
#   -- OR --
#   for i in {0..6}; do SLURM_ARRAY_TASK_ID=$i bash sweep_stage1.sh & done
#
# Tier A (β sweep, λ=10 fixed): β ∈ {3e-4, 5e-4, 7e-4, 1e-3}
# Tier B (λ ablation, β=5e-4):  λ ∈ {0, 30}
# Tier C (annealing, β=5e-4, λ=10): kl_warmup=100
#
#SBATCH -J s1_sweep
#SBATCH -p cas_v100_4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --comment=pytorch
#SBATCH -o logs/s1_sweep_%a_%j.out
#SBATCH -e logs/s1_sweep_%a_%j.err

set -e
module purge
module load conda/pytorch_2.5.0
source /home01/x3411a10/.bashrc

cd /scratch/x3411a10/IGAE/stage1_IGAE
mkdir -p logs

# ── sweep table ─────────────────────────────────────────────────────────────
#  idx  tier  beta_kl   lambda  kl_warmup  tag
BETAS=(  3e-4   5e-4   7e-4   1e-3   5e-4   5e-4   5e-4  )
LAMBDAS=( 10     10     10     10      0     30     10    )
WARMUPS=( 50     50     50     50     50     50    100    )
TIERS=(   A      A      A      A      B      B      C    )

IDX=${SLURM_ARRAY_TASK_ID:-0}
BETA=${BETAS[$IDX]}
LAMBDA=${LAMBDAS[$IDX]}
WARMUP=${WARMUPS[$IDX]}
TIER=${TIERS[$IDX]}

RUN_TAG="s1_tier${TIER}_b${BETA}_l${LAMBDA}_w${WARMUP}"
echo "[sweep] idx=$IDX  beta=$BETA  lambda=$LAMBDA  warmup=$WARMUP  tag=$RUN_TAG"

DATA_PATH="/scratch/x3411a10/datasets/ffhq256/dino_features_train.pt"
CKPT_DIR="/scratch/x3411a10/IGAE/stage1_IGAE/checkpoints/${RUN_TAG}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PYTHONUNBUFFERED=1 python train_run.py \
  --data_path     "$DATA_PATH" \
  --val_fraction  0.05 \
  --input_dim     768 \
  --latent_dim    6144 \
  --num_layers    4 \
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
  --wandb_project "igae-stage1-sweep" \
  --wandb_run_name "$RUN_TAG" \
  --wandb_tags    "sweep" "tier${TIER}" "beta${BETA}" "lambda${LAMBDA}"

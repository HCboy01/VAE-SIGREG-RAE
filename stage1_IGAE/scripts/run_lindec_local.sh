#!/bin/bash
# Run the linear-decoder default grid sequentially on a local GPU.
# Usage:
#   bash scripts/run_lindec_local.sh
#   bash scripts/run_lindec_local.sh 0 5
#   bash scripts/run_lindec_local.sh 5

set -e
cd /scratch/x3411a10/IGAE/stage1_IGAE
mkdir -p logs

module purge
module load conda/pytorch_2.5.0
source /home01/x3411a10/.bashrc

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

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
BETAS=( 0 0 0 0 0 0 1e-1 1e-1 1e-1 1e-1 1e-2 1e-2 1e-2 1e-2 1e-3 1e-3 1e-3 1e-4 1e-4 )
LAMBDAS=( 0 1 10 100 1000 10000 0 1 10 100 0 1 10 100 0 1 100 1 100 )
WARMUPS=( 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 )

START=${1:-0}
END=${2:-18}
if [[ $# -eq 1 && "$1" =~ ^[0-9]+$ ]]; then
    END=$1
fi

DATA_PATH="/scratch/x3411a10/datasets/ffhq256/dino_features_train.pt"

for IDX in $(seq $START $END); do
    TAG=${NAMES[$IDX]}
    BETA=${BETAS[$IDX]}
    LAMBDA=${LAMBDAS[$IDX]}
    WARMUP=${WARMUPS[$IDX]}
    CKPT_DIR="/scratch/x3411a10/IGAE/stage1_IGAE/checkpoints/${TAG}"
    LOG="logs/lindec_local_${IDX}_${TAG}.out"

    echo "========================================================"
    echo "[$(date '+%H:%M:%S')] START idx=$IDX  $TAG"
    echo "  beta=$BETA  lambda=$LAMBDA  warmup=$WARMUP"
    echo "========================================================"

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
      --save_every    200 \
      --wandb_project "igae-stage1-lindec" \
      --wandb_run_name "$TAG" \
      --wandb_tags    "lindec" "grid" "beta${BETA}" "lambda${LAMBDA}" \
      2>&1 | tee "$LOG"

    echo "[$(date '+%H:%M:%S')] DONE  idx=$IDX  $TAG"
    echo ""
done

echo "All runs complete."

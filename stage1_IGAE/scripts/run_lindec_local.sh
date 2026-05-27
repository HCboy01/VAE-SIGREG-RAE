#!/bin/bash
# Run all 32 linear-decoder configs sequentially on a local GPU.
# Usage:
#   bash scripts/run_lindec_local.sh            # run all
#   bash scripts/run_lindec_local.sh 0 5        # run idx 0..5 only
#   bash scripts/run_lindec_local.sh 26         # run single idx 26

set -e
cd /scratch/x3411a10/IGAE/stage1_IGAE
mkdir -p logs

module purge
module load conda/pytorch_2.5.0
source /home01/x3411a10/.bashrc

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

NAMES=(
  s1_grid_b0_l0_lindec        # 0
  s1_grid_b0_l1_lindec        # 1
  s1_grid_b0_l10_lindec       # 2
  s1_grid_b0_l100_lindec      # 3
  s1_grid_b0_l1000_lindec     # 4
  s1_grid_b0_l10000_lindec    # 5
  s1_grid_b1e-1_l0_lindec     # 6
  s1_grid_b1e-1_l1_lindec     # 7
  s1_grid_b1e-1_l10_lindec    # 8
  s1_grid_b1e-1_l100_lindec   # 9
  s1_grid_b1e-2_l0_lindec     # 10
  s1_grid_b1e-2_l1_lindec     # 11
  s1_grid_b1e-2_l10_lindec    # 12
  s1_grid_b1e-2_l100_lindec   # 13
  s1_grid_b1e-3_l0_lindec     # 14
  s1_grid_b1e-3_l1_lindec     # 15
  s1_grid_b1e-3_l100_lindec   # 16
  s1_grid_b1e-4_l1_lindec     # 17
  s1_grid_b1e-4_l100_lindec   # 18
  s1_lowb_b1e-4_l0_lindec     # 19
  s1_lowb_b1e-4_l10_lindec    # 20
  s1_lowb_b1e-4_l30_lindec    # 21
  s1_lowb_b2e-4_l0_lindec     # 22
  s1_lowb_b2e-4_l10_lindec    # 23
  s1_lowb_b2e-4_l30_lindec    # 24
  s1_tierA_b1e-3_l10_w50_lindec  # 25
  s1_tierA_b3e-4_l10_w50_lindec  # 26
  s1_tierA_b5e-4_l10_w50_lindec  # 27
  s1_tierA_b7e-4_l10_w50_lindec  # 28
  s1_tierB_b5e-4_l0_w50_lindec   # 29
  s1_tierB_b5e-4_l30_w50_lindec  # 30
  s1_tierC_b5e-4_l10_w100_lindec # 31
)
BETAS=(  0      0      0      0      0      0
         1e-1   1e-1   1e-1   1e-1
         1e-2   1e-2   1e-2   1e-2
         1e-3   1e-3   1e-3
         1e-4   1e-4
         1e-4   1e-4   1e-4
         2e-4   2e-4   2e-4
         1e-3   3e-4   5e-4   7e-4
         5e-4   5e-4
         5e-4 )
LAMBDAS=( 0      1      10     100    1000   10000
          0      1      10     100
          0      1      10     100
          0      1      100
          1      100
          0      10     30
          0      10     30
          10     10     10     10
          0      30
          10 )
WARMUPS=( 50     50     50     50     50     50
          50     50     50     50
          50     50     50     50
          50     50     50
          50     50
          50     50     50
          50     50     50
          50     50     50     50
          50     50
          100 )

# index range from args (default: all)
START=${1:-0}
END=${2:-31}
# single-index shortcut: run_lindec_local.sh 5  →  runs only idx 5
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
      --save_every    200 \
      --wandb_project "igae-stage1-lindec" \
      --wandb_run_name "$TAG" \
      --wandb_tags    "lindec" "beta${BETA}" "lambda${LAMBDA}" \
      2>&1 | tee "$LOG"

    echo "[$(date '+%H:%M:%S')] DONE  idx=$IDX  $TAG"
    echo ""
done

echo "All runs complete."

#!/bin/bash
# Feature-axis SIGReg sweep: --sigreg_target mu_feat
# mu.T [D, B] → D=6144 features as samples, pushed toward N(0,1)
#
# Configs: best β candidates from lindec sweep × λ ∈ {1, 10, 100}
#
# Usage:
#   nohup bash scripts/run_featsigreg_local.sh > logs/featsigreg_master.out 2>&1 &
#   bash scripts/run_featsigreg_local.sh 0      # single run
#   bash scripts/run_featsigreg_local.sh 3 8    # range

set -e
cd /scratch/x3411a10/IGAE/stage1_IGAE
mkdir -p logs

module purge
module load conda/pytorch_2.5.0
source /home01/x3411a10/.bashrc

export PYTORCH_ALLOC_CONF=expandable_segments:True

#  idx  beta    lambda
NAMES=(
  s1_featsr_b2e-4_l1      # 0
  s1_featsr_b2e-4_l10     # 1
  s1_featsr_b2e-4_l100    # 2
  s1_featsr_b3e-4_l1      # 3
  s1_featsr_b3e-4_l10     # 4
  s1_featsr_b3e-4_l100    # 5
  s1_featsr_b5e-4_l1      # 6
  s1_featsr_b5e-4_l10     # 7
  s1_featsr_b5e-4_l100    # 8
  s1_featsr_b1e-3_l1      # 9
  s1_featsr_b1e-3_l10     # 10
  s1_featsr_b1e-3_l100    # 11
)
BETAS=( 2e-4  2e-4  2e-4  3e-4  3e-4  3e-4  5e-4  5e-4  5e-4  1e-3  1e-3  1e-3 )
LAMBDAS=(  1    10   100     1    10   100     1    10   100     1    10   100 )

START=${1:-0}
END=${2:-11}
if [[ $# -eq 1 && "$1" =~ ^[0-9]+$ ]]; then END=$1; fi

DATA_PATH="/scratch/x3411a10/datasets/ffhq256/dino_features_train.pt"

for IDX in $(seq $START $END); do
    TAG=${NAMES[$IDX]}
    BETA=${BETAS[$IDX]}
    LAMBDA=${LAMBDAS[$IDX]}
    CKPT_DIR="/scratch/x3411a10/IGAE/stage1_IGAE/checkpoints/${TAG}"

    echo "========================================================"
    echo "[$(date '+%H:%M:%S')] START idx=$IDX  $TAG"
    echo "  beta=$BETA  lambda=$LAMBDA  sigreg_target=mu_feat"
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
      --beta_kl_warmup_epochs 50 \
      --lambda_sigreg "$LAMBDA" \
      --sigreg_type   epps_pulley \
      --sigreg_target mu_feat \
      --num_projections 512 \
      --dead_activation_thresholds 0.1 0.5 1.0 \
      --ckpt_dir      "$CKPT_DIR" \
      --save_every    200 \
      --wandb_project "igae-stage1-featsigreg" \
      --wandb_run_name "$TAG" \
      --wandb_tags    "featsigreg" "beta${BETA}" "lambda${LAMBDA}" \
      2>&1 | tee "logs/featsigreg_${IDX}_${TAG}.out"

    echo "[$(date '+%H:%M:%S')] DONE  idx=$IDX  $TAG"
    echo ""
done

echo "All runs complete."

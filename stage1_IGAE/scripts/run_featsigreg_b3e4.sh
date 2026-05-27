#!/bin/bash
# Feature-axis SIGReg: beta=3e-4 x lambda={0,1,10,30,50,100}
# lambda=0 is included as a no-SIGReg baseline (new run needed)
#
# Usage:
#   nohup bash scripts/run_featsigreg_b3e4.sh > logs/featsigreg_b3e4_master.out 2>&1 &
#   bash scripts/run_featsigreg_b3e4.sh 0   # single idx

set -e
cd /scratch/x3411a10/IGAE/stage1_IGAE
mkdir -p logs

module purge
module load conda/pytorch_2.5.0
source /home01/x3411a10/.bashrc

export PYTORCH_ALLOC_CONF=expandable_segments:True

NAMES=(
  s1_featsr_b3e-4_l0     # 0  (baseline: no SIGReg)
  s1_featsr_b3e-4_l1     # 1
  s1_featsr_b3e-4_l10    # 2
  s1_featsr_b3e-4_l30    # 3
  s1_featsr_b3e-4_l50    # 4
  s1_featsr_b3e-4_l100   # 5
)
LAMBDAS=( 0  1  10  30  50  100 )

DATA_PATH="/scratch/x3411a10/datasets/ffhq256/dino_features_train.pt"

# 2(λ=10)부터 먼저, 이후 나머지 순서로
for IDX in 2 3 4 5 0 1; do
    TAG=${NAMES[$IDX]}
    LAMBDA=${LAMBDAS[$IDX]}
    CKPT_DIR="/scratch/x3411a10/IGAE/stage1_IGAE/checkpoints/${TAG}"

    echo "========================================================"
    echo "[$(date '+%H:%M:%S')] START idx=$IDX  $TAG"
    echo "  beta=3e-4  lambda=$LAMBDA  sigreg_target=mu_feat"
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
      --beta_kl       3e-4 \
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
      --wandb_tags    "featsigreg" "beta3e-4" "lambda${LAMBDA}" \
      2>&1 | tee "logs/featsigreg_b3e4_${IDX}_${TAG}.out"

    echo "[$(date '+%H:%M:%S')] DONE  idx=$IDX  $TAG"
    echo ""
done

echo "All runs complete."

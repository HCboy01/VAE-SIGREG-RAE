#!/bin/bash
# feature_sq KL sweep
# kl_type=feature_sq × β ∈ {1e-3, 1e-4} × λ ∈ {10, 100}  → 4 runs, linear_decoder
#
# Usage:
#   nohup bash scripts/sweep_featklsq.sh > logs/featklsq_master.out 2>&1 &
#   bash scripts/sweep_featklsq.sh 0        # 단일 run
#   bash scripts/sweep_featklsq.sh 0 1      # 범위
#
# idx  beta   lambda
#  0   1e-3    10
#  1   1e-3   100
#  2   1e-4    10
#  3   1e-4   100

set -e
module purge
module load conda/pytorch_2.9.1_cuda12
source /home01/x3411a10/.bashrc

cd /scratch/x3411a10/IGAE/stage1_IGAE
mkdir -p logs

BETAS=(   1e-3  1e-3  1e-4  1e-4  )
LAMBDAS=(   10   100    10   100   )

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATA_PATH="/scratch/x3411a10/datasets/ffhq256/dino_features_train.pt"

if [ $# -eq 1 ]; then
    START=$1; END=$1
elif [ $# -eq 2 ]; then
    START=$1; END=$2
else
    START=0; END=$(( ${#BETAS[@]} - 1 ))
fi

for IDX in $(seq $START $END); do
    BETA=${BETAS[$IDX]}
    LAMBDA=${LAMBDAS[$IDX]}
    RUN_TAG="s1_featkl_sq_b${BETA}_l${LAMBDA}_w50_lindec"
    CKPT_DIR="/scratch/x3411a10/IGAE/stage1_IGAE/checkpoints/${RUN_TAG}"

    echo "========================================================"
    echo "[$(date '+%H:%M:%S')] START idx=$IDX  $RUN_TAG"
    echo "  kl_type=feature_sq  beta=$BETA  lambda=$LAMBDA"
    echo "========================================================"

    PYTHONUNBUFFERED=1 python -u train_run.py \
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
      --beta_kl_warmup_epochs 50 \
      --kl_type       feature_sq \
      --lambda_sigreg "$LAMBDA" \
      --sigreg_type   epps_pulley \
      --num_projections 512 \
      --dead_activation_thresholds 0.1 0.5 1.0 \
      --ckpt_dir      "$CKPT_DIR" \
      --save_every    20 \
      --wandb_project "igae-stage1-sweep" \
      --wandb_run_name "$RUN_TAG" \
      --wandb_tags    "featkl_sq" "beta${BETA}" "lambda${LAMBDA}" \
      2>&1 | tee "logs/featklsq_${IDX}_${RUN_TAG}.out"

    echo "[$(date '+%H:%M:%S')] DONE  idx=$IDX  $RUN_TAG"
    echo ""
done

echo "All runs complete."

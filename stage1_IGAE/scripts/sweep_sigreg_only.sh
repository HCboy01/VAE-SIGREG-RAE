#!/bin/bash
# SIGReg-only sweep — KL 완전 제거, SIGReg만으로 q(z) ~ N(0,I) 유도
# beta_kl=0  ×  lambda_sigreg ∈ {100, 300, 1000, 3000}  → 4 runs, linear_decoder
#
# 핵심 가설:
#   KL 없이 SIGReg만으로도 각 dim의 marginal이 N(0,1)에 수렴하는가?
#   sigma -> 0 (deterministic AE)이 되며, SIGReg는 aggregate q(mu)를 N(0,I)로 밀게 됨.
#   어떤 dim을 실제 사용하는지는 KL 대신 decoder column norm으로 측정.
#
# 배치 크기 극대화 (8192): SIGReg는 배치 내 ECF 추정 → 배치가 클수록 정확한 gradient.
# 에폭 400: batch 크면 epoch당 step 수 감소(~7 steps) → 보상.
#
# Usage:
#   nohup bash scripts/sweep_sigreg_only.sh > logs/sigreg_only_master.out 2>&1 &
#   bash scripts/sweep_sigreg_only.sh 0        # 단일 run
#   bash scripts/sweep_sigreg_only.sh 0 1      # 범위
#
# idx  lambda
#  0    100
#  1    300
#  2   1000
#  3   3000

set -e
module purge
module load conda/pytorch_2.9.1_cuda12
source /home01/x3411a10/.bashrc

cd /scratch/x3411a10/IGAE/stage1_IGAE
mkdir -p logs

LAMBDAS=( 100  300  1000  3000 )

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_ALLOC_CONF=expandable_segments:True

DATA_PATH="/scratch/x3411a10/datasets/ffhq256/dino_features_train.pt"

if [ $# -eq 1 ]; then
    START=$1; END=$1
elif [ $# -eq 2 ]; then
    START=$1; END=$2
else
    START=0; END=$(( ${#LAMBDAS[@]} - 1 ))
fi

for IDX in $(seq $START $END); do
    LAMBDA=${LAMBDAS[$IDX]}
    RUN_TAG="s1_sigreg_only_b0_l${LAMBDA}_lindec"
    CKPT_DIR="/scratch/x3411a10/IGAE/stage1_IGAE/checkpoints/${RUN_TAG}"

    echo "========================================================"
    echo "[$(date '+%H:%M:%S')] START idx=$IDX  $RUN_TAG"
    echo "  beta_kl=0  lambda=${LAMBDA}  batch=8192  epochs=400"
    echo "========================================================"

    PYTHONUNBUFFERED=1 python -u train_run.py \
      --data_path     "$DATA_PATH" \
      --val_fraction  0.05 \
      --input_dim     768 \
      --latent_dim    6144 \
      --num_layers    4 \
      --linear_decoder \
      --epochs        400 \
      --batch_size    8192 \
      --accum_steps   1 \
      --lr            3e-4 \
      --weight_decay  1e-2 \
      --beta_kl       0 \
      --beta_kl_warmup_epochs 0 \
      --kl_type       sample \
      --lambda_sigreg "$LAMBDA" \
      --sigreg_type   epps_pulley \
      --num_projections 512 \
      --dead_activation_thresholds 0.1 0.5 1.0 \
      --ckpt_dir      "$CKPT_DIR" \
      --save_every    40 \
      --wandb_project "igae-stage1-sweep" \
      --wandb_run_name "$RUN_TAG" \
      --wandb_tags    "sigreg_only" "beta0" "lambda${LAMBDA}" "batch8192" \
      2>&1 | tee "logs/sigreg_only_${IDX}_${RUN_TAG}.out"

    echo "[$(date '+%H:%M:%S')] DONE  idx=$IDX  $RUN_TAG"
    echo ""
done

echo "All runs complete."

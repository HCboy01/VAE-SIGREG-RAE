#!/bin/bash
# Very-low-β × high-λ sweep
# β ∈ {1e-5, 1e-6}  ×  λ ∈ {30, 100, 300}  → 6 runs, all linear_decoder
#
# Usage:
#   # 전체 순차 실행:
#   nohup bash scripts/sweep_vlowb_lambda.sh > logs/vlowb_lambda_master.out 2>&1 &
#
#   # 단일 run (idx 0~5):
#   bash scripts/sweep_vlowb_lambda.sh 0
#
#   # 범위 지정:
#   bash scripts/sweep_vlowb_lambda.sh 0 2
#
#   # SLURM 병렬:
#   sbatch --array=0-5 scripts/sweep_vlowb_lambda.sh
#
# idx  beta   lambda
#  0   1e-5    30
#  1   1e-5   100
#  2   1e-5   300
#  3   1e-6    30
#  4   1e-6   100
#  5   1e-6   300
#
#SBATCH -J vlowb_lambda
#SBATCH -p cas_v100_4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --comment=pytorch
#SBATCH -o logs/vlowb_lambda_%a_%j.out
#SBATCH -e logs/vlowb_lambda_%a_%j.err

set -e
module purge
module load conda/pytorch_2.5.0
source /home01/x3411a10/.bashrc

cd /scratch/x3411a10/IGAE/stage1_IGAE
mkdir -p logs

BETAS=(   1e-5  1e-5  1e-5  1e-6  1e-6  1e-6  )
LAMBDAS=(   30   100   300    30   100   300   )

export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATA_PATH="/scratch/x3411a10/datasets/ffhq256/dino_features_train.pt"

# index range
if [ -n "${SLURM_ARRAY_TASK_ID}" ]; then
    START=${SLURM_ARRAY_TASK_ID}
    END=${SLURM_ARRAY_TASK_ID}
elif [ $# -eq 1 ]; then
    START=$1; END=$1
elif [ $# -eq 2 ]; then
    START=$1; END=$2
else
    START=0; END=$(( ${#BETAS[@]} - 1 ))
fi

for IDX in $(seq $START $END); do
    BETA=${BETAS[$IDX]}
    LAMBDA=${LAMBDAS[$IDX]}
    RUN_TAG="s1_vlowb_b${BETA}_l${LAMBDA}_w50_lindec"
    CKPT_DIR="/scratch/x3411a10/IGAE/stage1_IGAE/checkpoints/${RUN_TAG}"

    echo "========================================================"
    echo "[$(date '+%H:%M:%S')] START idx=$IDX  $RUN_TAG"
    echo "  beta=$BETA  lambda=$LAMBDA"
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
      --num_projections 512 \
      --dead_activation_thresholds 0.1 0.5 1.0 \
      --ckpt_dir      "$CKPT_DIR" \
      --save_every    20 \
      --wandb_project "igae-stage1-sweep" \
      --wandb_run_name "$RUN_TAG" \
      --wandb_tags    "vlowb" "highlambda" "beta${BETA}" "lambda${LAMBDA}" \
      2>&1 | tee "logs/vlowb_lambda_${IDX}_${RUN_TAG}.out"

    echo "[$(date '+%H:%M:%S')] DONE  idx=$IDX  $RUN_TAG"
    echo ""
done

echo "All runs complete."

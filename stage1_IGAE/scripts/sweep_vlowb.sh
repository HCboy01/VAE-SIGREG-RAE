#!/bin/bash
# Very-low-β sweep: β ∈ {1e-5, 1e-6}, λ=10, linear_decoder
#
# Usage:
#   sbatch --array=0-1 scripts/sweep_vlowb.sh
#   -- OR (local sequential) --
#   nohup bash scripts/sweep_vlowb.sh > logs/vlowb_master.out 2>&1 &
#
#SBATCH -J vlowb_sweep
#SBATCH -p cas_v100_4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --comment=pytorch
#SBATCH -o logs/vlowb_%a_%j.out
#SBATCH -e logs/vlowb_%a_%j.err

set -e
module purge
module load conda/pytorch_2.5.0
source /home01/x3411a10/.bashrc

cd /scratch/x3411a10/IGAE/stage1_IGAE
mkdir -p logs

BETAS=(  1e-5   1e-6  )
LAMBDAS=( 10     10   )
WARMUPS=( 50     50   )

# local sequential: iterate all; sbatch: pick by array index
if [ -n "${SLURM_ARRAY_TASK_ID}" ]; then
    START=${SLURM_ARRAY_TASK_ID}
    END=${SLURM_ARRAY_TASK_ID}
else
    START=0
    END=$(( ${#BETAS[@]} - 1 ))
fi

DATA_PATH="/scratch/x3411a10/datasets/ffhq256/dino_features_train.pt"
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for IDX in $(seq $START $END); do
    BETA=${BETAS[$IDX]}
    LAMBDA=${LAMBDAS[$IDX]}
    WARMUP=${WARMUPS[$IDX]}
    RUN_TAG="s1_vlowb_b${BETA}_l${LAMBDA}_w${WARMUP}_lindec"
    CKPT_DIR="/scratch/x3411a10/IGAE/stage1_IGAE/checkpoints/${RUN_TAG}"

    echo "========================================================"
    echo "[$(date '+%H:%M:%S')] START idx=$IDX  $RUN_TAG"
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
      --save_every    20 \
      --wandb_project "igae-stage1-sweep" \
      --wandb_run_name "$RUN_TAG" \
      --wandb_tags    "vlowb" "beta${BETA}" "lambda${LAMBDA}" \
      2>&1 | tee "logs/vlowb_${IDX}_${RUN_TAG}.out"

    echo "[$(date '+%H:%M:%S')] DONE  idx=$IDX  $RUN_TAG"
    echo ""
done

echo "All runs complete."

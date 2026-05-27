#!/bin/bash
# Feature-axis SIGReg: beta=1e-3 x lambda={0,1,10,30,50,100}
# GPU 1

set -e
cd /scratch/x3411a10/IGAE/stage1_IGAE
mkdir -p logs

module purge
module load conda/pytorch_2.5.0
source /home01/x3411a10/.bashrc

export CUDA_VISIBLE_DEVICES=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

NAMES=(
  s1_featsr_b1e-3_l0     # 0
  s1_featsr_b1e-3_l1     # 1
  s1_featsr_b1e-3_l10    # 2
  s1_featsr_b1e-3_l30    # 3
  s1_featsr_b1e-3_l50    # 4
  s1_featsr_b1e-3_l100   # 5
)
LAMBDAS=( 0  1  10  30  50  100 )

# λ=10부터 먼저
for IDX in 2 3 4 5 0 1; do
    TAG=${NAMES[$IDX]}
    LAMBDA=${LAMBDAS[$IDX]}
    CKPT_DIR="/scratch/x3411a10/IGAE/stage1_IGAE/checkpoints/${TAG}"

    echo "========================================================"
    echo "[$(date '+%H:%M:%S')] START idx=$IDX  $TAG"
    echo "  beta=1e-3  lambda=$LAMBDA  sigreg_target=mu_feat  GPU=1"
    echo "========================================================"

    PYTHONUNBUFFERED=1 python train_run.py \
      --data_path     "/scratch/x3411a10/datasets/ffhq256/dino_features_train.pt" \
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
      --beta_kl       1e-3 \
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
      --wandb_tags    "featsigreg" "beta1e-3" "lambda${LAMBDA}" \
      2>&1 | tee "logs/featsigreg_b1e3_${IDX}_${TAG}.out"

    echo "[$(date '+%H:%M:%S')] DONE  idx=$IDX  $TAG"
    echo ""
done

echo "All runs complete."

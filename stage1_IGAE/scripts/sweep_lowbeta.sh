#!/bin/bash
# Low-β SIGReg study: β={1e-4, 2e-4} × λ={0, 10, 30}
# 목적: KL이 약한 구간에서 SIGReg가 collapse를 막는지 확인
#
# IDX  β      λ
#  0   1e-4   0
#  1   1e-4   10
#  2   1e-4   30
#  3   2e-4   0
#  4   2e-4   10
#  5   2e-4   30

set -e

BETAS=(  1e-4   1e-4   1e-4   2e-4   2e-4   2e-4 )
LAMBDAS=( 0     10     30      0     10     30   )

IDX=${SLURM_ARRAY_TASK_ID:-0}
BETA=${BETAS[$IDX]}
LAMBDA=${LAMBDAS[$IDX]}

RUN_TAG="s1_lowb_b${BETA}_l${LAMBDA}"
echo "[sweep] idx=$IDX  beta=$BETA  lambda=$LAMBDA  tag=$RUN_TAG"

DATA_PATH="/scratch/x3411a10/datasets/ffhq256/dino_features_train.pt"
CKPT_DIR="/scratch/x3411a10/IGAE/stage1_IGAE/checkpoints/${RUN_TAG}"

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
  --beta_kl_warmup_epochs 50 \
  --lambda_sigreg "$LAMBDA" \
  --sigreg_type   epps_pulley \
  --num_projections 512 \
  --dead_activation_thresholds 0.1 0.5 1.0 \
  --ckpt_dir      "$CKPT_DIR" \
  --save_every    20 \
  --wandb_project "igae-stage1-sweep" \
  --wandb_run_name "$RUN_TAG" \
  --wandb_tags    "lowbeta" "beta${BETA}" "lambda${LAMBDA}"

#!/bin/bash
# Launch the precision-signal sweep on gdebug01.
# GPU0: kappa=2, then weaker lambda=0.02/kappa=2
# GPU1: kappa=4

set -eo pipefail

cd /scratch/x3411a10/IGAE/stage1_IGAE

(
  CUDA_VISIBLE_DEVICES=0 bash scripts/run_precision_signal_b1e-4_l0p05_k2.sh
  CUDA_VISIBLE_DEVICES=0 bash scripts/run_precision_signal_b1e-4_l0p02_k2.sh
) > logs/precsig_sweep_gpu0_driver.out 2>&1 &
pid0=$!

(
  CUDA_VISIBLE_DEVICES=1 bash scripts/run_precision_signal_b1e-4_l0p05_k4.sh
) > logs/precsig_sweep_gpu1_driver.out 2>&1 &
pid1=$!

echo "GPU0 sweep pid: ${pid0}"
echo "GPU1 sweep pid: ${pid1}"
echo "Logs:"
echo "  /scratch/x3411a10/IGAE/stage1_IGAE/logs/precsig_sweep_gpu0_driver.out"
echo "  /scratch/x3411a10/IGAE/stage1_IGAE/logs/precsig_sweep_gpu1_driver.out"

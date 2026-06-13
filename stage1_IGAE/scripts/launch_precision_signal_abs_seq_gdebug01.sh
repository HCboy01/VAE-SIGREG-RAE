#!/bin/bash
# Sequential abs-mu precision-signal sweep for one GPU.

set -eo pipefail

cd /scratch/x3411a10/IGAE/stage1_IGAE

bash scripts/run_precision_signal_abs_b1e-4_l0p05_k0p25.sh
bash scripts/run_precision_signal_abs_b1e-4_l0p05_k0p5.sh
bash scripts/run_precision_signal_abs_b1e-4_l0p05_k1.sh

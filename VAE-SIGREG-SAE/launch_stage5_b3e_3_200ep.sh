#!/usr/bin/env bash
set -euo pipefail

# Optional only. Main 200-epoch S-curve comparison excludes beta=3e-3.
export BETA_KL=3e-3
export EPOCHS=200
export MIN_ACTIVE_UNITS="${MIN_ACTIVE_UNITS:-500}"
export SIG_GRID="${SIG_GRID:-0 1 3 10 30 100 300 1000}"
exec bash "$(dirname "${BASH_SOURCE[0]}")/launch_stage5_200epoch_active_sweep.sh"

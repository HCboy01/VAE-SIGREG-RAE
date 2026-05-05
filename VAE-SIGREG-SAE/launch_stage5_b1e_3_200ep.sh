#!/usr/bin/env bash
set -euo pipefail

# Shared-grid 200 epoch sweep for comparable active-vs-lambda S-curves.
export BETA_KL=1e-3
export EPOCHS=200
export MIN_ACTIVE_UNITS="${MIN_ACTIVE_UNITS:-2500}"
export SIG_GRID="${SIG_GRID:-0 1 3 10 30 100 300 1000}"
exec bash "$(dirname "${BASH_SOURCE[0]}")/launch_stage5_200epoch_active_sweep.sh"

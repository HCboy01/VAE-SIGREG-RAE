#!/usr/bin/env bash
set -euo pipefail

# Next experiment: Stage 5 200-epoch active-survival sweep.
# Start with beta=1e-4, because the previous 200-epoch final run collapsed.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/launch_stage5_b1e_4_200ep.sh"

#!/usr/bin/env bash
set -euo pipefail

REPO_PATH=${REPO_PATH:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}
PYTHON_BIN=${PYTHON_BIN:-$HOME/.venv/bin/python}
WAN_PATH=${WAN_PATH:-$HOME/.venv/wan}
CONFIG=${1:-$REPO_PATH/tests/e2e_tests/embodied/task7_real_runner_smoke.yaml}

export EMBODIED_PATH=${EMBODIED_PATH:-$REPO_PATH/examples/embodiment}
export PYTHONPATH="$REPO_PATH:$REPO_PATH/../rlix-core/src:$WAN_PATH${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}
export RAY_memory_usage_threshold=${RAY_memory_usage_threshold:-0.99}
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/rlinf-matplotlib}
export HF_HOME=${HF_HOME:-/tmp/rlinf-huggingface}

exec "$PYTHON_BIN" \
  "$REPO_PATH/tests/e2e_tests/embodied/task7_real_runner_smoke.py" \
  --config "$CONFIG"

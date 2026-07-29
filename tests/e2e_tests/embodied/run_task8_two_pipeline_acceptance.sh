#!/usr/bin/env bash
set -euo pipefail

environment=""
mode=""
scenario=""
config=""
output_dir=""
run_id=""
vla_digest=""
environment_digest=""
preflight_only=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --environment) environment="$2"; shift 2 ;;
    --mode) mode="$2"; shift 2 ;;
    --scenario) scenario="$2"; shift 2 ;;
    --config) config="$2"; shift 2 ;;
    --output-dir) output_dir="$2"; shift 2 ;;
    --run-id) run_id="$2"; shift 2 ;;
    --vla-checkpoint-digest) vla_digest="$2"; shift 2 ;;
    --environment-checkpoint-digest) environment_digest="$2"; shift 2 ;;
    --preflight-only) preflight_only=1; shift ;;
    *) echo "unknown Task 8 argument: $1" >&2; exit 2 ;;
  esac
done

[[ -n "$environment" ]] || { echo "missing required Task 8 argument: --environment" >&2; exit 2; }
[[ -n "$mode" ]] || { echo "missing required Task 8 argument: --mode" >&2; exit 2; }
[[ -n "$scenario" ]] || { echo "missing required Task 8 argument: --scenario" >&2; exit 2; }
[[ -n "$config" ]] || { echo "missing required Task 8 argument: --config" >&2; exit 2; }
[[ -n "$output_dir" ]] || { echo "missing required Task 8 argument: --output-dir" >&2; exit 2; }
[[ -n "$run_id" ]] || { echo "missing required Task 8 argument: --run-id" >&2; exit 2; }
[[ -n "$vla_digest" ]] || { echo "missing required Task 8 argument: --vla-checkpoint-digest" >&2; exit 2; }
[[ -n "$environment_digest" ]] || { echo "missing required Task 8 argument: --environment-checkpoint-digest" >&2; exit 2; }

if [[ $preflight_only -ne 1 ]]; then
  echo "The real two-driver acceptance path is not implemented yet; use --preflight-only to validate and freeze the matrix without claiming acceptance." >&2
  exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${RLINF_PYTHON:-/root/.venv/bin/python}"
exec "$python_bin" "$script_dir/task8_two_pipeline_acceptance.py" \
  --scope acceptance-preflight \
  --environment "$environment" \
  --mode "$mode" \
  --scenario "$scenario" \
  --config "$config" \
  --output-dir "$output_dir" \
  --run-id "$run_id" \
  --vla-checkpoint-digest "$vla_digest" \
  --environment-checkpoint-digest "$environment_digest"

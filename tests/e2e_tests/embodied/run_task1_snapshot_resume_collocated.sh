#!/usr/bin/env bash
set -euo pipefail

REPO_PATH=${REPO_PATH:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}
CONFIG=${1:-$REPO_PATH/tests/e2e_tests/embodied/task1_wan_snapshot_resume_collocated.yaml}

exec "$REPO_PATH/tests/e2e_tests/embodied/run_task1_snapshot_resume.sh" "$CONFIG"

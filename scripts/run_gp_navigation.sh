#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

exec "${PROJECT_ROOT}/scripts/run_autonomy.sh" \
  --planner gp_navigation \
  --assistance ground_only \
  "$@"

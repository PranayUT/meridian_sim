#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${PROJECT_ROOT}/scripts/env.sh"
exec python "${PROJECT_ROOT}/tools/place_obstacles.py" "$@"

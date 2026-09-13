#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Keep this launcher usable from a fresh terminal, just like run_sim.sh.
# The controller needs both NumPy and Gazebo's Python transport bindings.
if ! python -c "import numpy; import gz.transport13" >/dev/null 2>&1; then
  if command -v conda >/dev/null 2>&1; then
    exec conda run --no-capture-output --name rugged-ugv \
      "${PROJECT_ROOT}/scripts/run_autonomy.sh" "$@"
  fi
  echo "The rugged-ugv environment is unavailable. Run ${PROJECT_ROOT}/scripts/setup.sh first." >&2
  exit 127
fi

source "${PROJECT_ROOT}/scripts/env.sh"

SHOW_MAPS=true
AUTONOMY_ARGS=()
for argument in "$@"; do
  if [[ "${argument}" == "--no-map-viewer" ]]; then
    SHOW_MAPS=false
  else
    AUTONOMY_ARGS+=("${argument}")
  fi
done

VIEWER_PID=""
if [[ "${SHOW_MAPS}" == true ]]; then
  python -m autonomy.meridian_drive.map_viewer &
  VIEWER_PID=$!
fi

cleanup() {
  if [[ -n "${VIEWER_PID}" ]]; then
    kill "${VIEWER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM
python -m autonomy.meridian_drive.gazebo_node "${AUTONOMY_ARGS[@]}"

#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${PROJECT_ROOT}/scripts/env.sh"
# Keep test commands away from an interactive simulation on the same host.
export GZ_PARTITION="rugged_ugv_smoke_${$}"
LOG_FILE="$(mktemp /tmp/rugged-ugv-gazebo.XXXXXX.log)"
SIM_PID=""
CONTROL_PID=""

cleanup() {
  [[ -z "${CONTROL_PID}" ]] || kill "${CONTROL_PID}" 2>/dev/null || true
  [[ -z "${SIM_PID}" ]] || kill "${SIM_PID}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

gz sim --force-version 8 -r -s -v 2 "${PROJECT_ROOT}/worlds/headless_smoke.sdf" >"${LOG_FILE}" 2>&1 &
SIM_PID=$!

# No rendering plugin is loaded in this physics-and-control smoke world.
sleep 5
if ! kill -0 "${SIM_PID}" 2>/dev/null; then
  sed -n '1,240p' "${LOG_FILE}"
  exit 1
fi

python -m autonomy.meridian_drive.gazebo_node \
  --samples 64 \
  --horizon 40 \
  --world-pose-topic /world/headless_smoke/dynamic_pose/info \
  > /tmp/rugged-ugv-controller.log 2>&1 &
CONTROL_PID=$!
python "${PROJECT_ROOT}/tools/check_runtime.py" \
  --no-lidar \
  --world-pose-topic /world/headless_smoke/dynamic_pose/info
echo "Gazebo log: ${LOG_FILE}"

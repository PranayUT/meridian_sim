#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! python -c "import numpy; import gz.transport13" >/dev/null 2>&1; then
  if command -v conda >/dev/null 2>&1; then
    exec conda run --no-capture-output --name rugged-ugv \
      "${PROJECT_ROOT}/scripts/run_experiment.sh" "$@"
  fi
  echo "The rugged-ugv environment is unavailable. Run ${PROJECT_ROOT}/scripts/setup.sh first." >&2
  exit 127
fi

source "${PROJECT_ROOT}/scripts/env.sh"

if [[ ! -f "${PROJECT_ROOT}/models/hill_terrain/meshes/terrain.tif" ]]; then
  python "${PROJECT_ROOT}/tools/generate_terrain.py"
fi

# --rtf, --gui, --run-id, and --veg-root belong here; the rest go to the harness.
RTF="3"
GUI=false
RUN_ID=""
VEG_ROOT=""
HARNESS_ARGS=()
ARGS=("$@")
index=0
while ((index < ${#ARGS[@]})); do
  argument="${ARGS[index]}"
  case "${argument}" in
    --rtf)
      RTF="${ARGS[index + 1]:-}"
      ((index += 2))
      continue
      ;;
    --rtf=*)
      RTF="${argument#--rtf=}"
      ((index += 1))
      continue
      ;;
    --run-id)
      RUN_ID="${ARGS[index + 1]:-}"
      ((index += 2))
      continue
      ;;
    --run-id=*)
      RUN_ID="${argument#--run-id=}"
      ((index += 1))
      continue
      ;;
    --veg-root)
      VEG_ROOT="${ARGS[index + 1]:-}"
      ((index += 2))
      continue
      ;;
    --veg-root=*)
      VEG_ROOT="${argument#--veg-root=}"
      ((index += 1))
      continue
      ;;
    --gui)
      GUI=true
      ((index += 1))
      continue
      ;;
  esac
  HARNESS_ARGS+=("${argument}")
  ((index += 1))
done
# The variant has to go ahead of models/ so model://painted_vegetation resolves
# to this run's obstacles, and the harness has to grade drags against the same
# mesh, so it is both an environment entry and a harness argument.
if [[ -n "${VEG_ROOT}" ]]; then
  if [[ ! -d "${VEG_ROOT}/painted_vegetation" ]]; then
    echo "no painted_vegetation under ${VEG_ROOT}; build it with tools/make_vegetation.py" >&2
    exit 2
  fi
  VEG_ROOT="$(cd "${VEG_ROOT}" && pwd)"
  export GZ_SIM_RESOURCE_PATH="${VEG_ROOT}:${GZ_SIM_RESOURCE_PATH}"
  HARNESS_ARGS+=(--veg-root "${VEG_ROOT}")
fi
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_$$}"
if [[ ! "${RUN_ID}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "--run-id may only contain letters, digits, dot, dash, and underscore" >&2
  exit 2
fi
# One Gazebo bus per campaign, so several can share a machine without their
# topics, services, or set_pose calls reaching each other.
export GZ_PARTITION="${GZ_PARTITION:-rugged_ugv_${RUN_ID}}"
if [[ ! "${RTF}" =~ ^[0-9]+([.][0-9]+)?$ ]] || [[ "${RTF}" == 0 ]]; then
  echo "--rtf needs a positive number, for example --rtf 3" >&2
  exit 2
fi

SIM_LOG="${PROJECT_ROOT}/runtime/experiments/${RUN_ID}/gazebo.log"
mkdir -p "$(dirname "${SIM_LOG}")"
SIM_PID=""
cleanup() {
  [[ -z "${SIM_PID}" ]] || kill "${SIM_PID}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

SIM_FLAGS=(-r -z "$(awk -v r="${RTF}" 'BEGIN { printf "%d", r / 0.001 }')")
if [[ "${GUI}" == false ]]; then
  SIM_FLAGS+=(-s)
else
  # Same chase-camera layout run_sim.sh uses. Without it the default GUI camera
  # is fixed and the rover leaves frame within seconds of starting a route.
  SIM_FLAGS+=(--gui-config "${PROJECT_ROOT}/config/hill_country.gui.config")
fi
echo "Run ${RUN_ID} on partition ${GZ_PARTITION}"
echo "Starting Gazebo at ${RTF}x (log: ${SIM_LOG})"
gz sim --force-version 8 "${SIM_FLAGS[@]}" \
  "${PROJECT_ROOT}/worlds/hill_country.sdf" >"${SIM_LOG}" 2>&1 &
SIM_PID=$!

for _ in $(seq 1 60); do
  if gz topic -l 2>/dev/null | grep -q "/world/hill_country/dynamic_pose/info"; then
    break
  fi
  if ! kill -0 "${SIM_PID}" 2>/dev/null; then
    echo "Gazebo exited during startup. Last lines of ${SIM_LOG}:" >&2
    tail -n 20 "${SIM_LOG}" >&2
    exit 1
  fi
  sleep 1
done

python "${PROJECT_ROOT}/tools/run_experiment.py" --run-id "${RUN_ID}" \
  ${HARNESS_ARGS[@]+"${HARNESS_ARGS[@]}"}

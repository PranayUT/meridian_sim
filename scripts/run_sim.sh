#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Keep the launcher usable from a fresh terminal. setup.sh installs Gazebo in
# this environment, but requiring every caller to activate it first makes a
# missing `gz` look like a simulator failure.
if ! command -v gz >/dev/null 2>&1; then
  if command -v conda >/dev/null 2>&1; then
    exec conda run --no-capture-output --name rugged-ugv \
      "${PROJECT_ROOT}/scripts/run_sim.sh" "$@"
  fi
  echo "Gazebo was not found. Run ${PROJECT_ROOT}/scripts/setup.sh first." >&2
  exit 127
fi

source "${PROJECT_ROOT}/scripts/env.sh"

if [[ ! -f "${PROJECT_ROOT}/models/hill_terrain/meshes/terrain.tif" ]]; then
  python "${PROJECT_ROOT}/tools/generate_terrain.py"
fi

SIM_ARGS=()
SIM_SEED="${GZ_SIM_SEED:-4207}"
HAS_SEED=false
HAS_GUI_CONFIG=false
HEADLESS=false
RTF=""
# The world ships real_time_factor 1.0. Rather than edit the SDF per run, --rtf
# becomes gz sim's -z (iterations per wall second), which overrides that
# throttle: iterations = rtf / max_step_size.
STEP_SIZE=0.001
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
    --seed)
      HAS_SEED=true
      SIM_SEED="${ARGS[index + 1]:-${SIM_SEED}}"
      ;;
    --seed=*)
      HAS_SEED=true
      SIM_SEED="${argument#--seed=}"
      ;;
    --gui-config|--gui-config=*)
      HAS_GUI_CONFIG=true
      ;;
    -s)
      HEADLESS=true
      ;;
  esac
  SIM_ARGS+=("${argument}")
  ((index += 1))
done
if [[ -n "${RTF}" ]]; then
  if [[ ! "${RTF}" =~ ^[0-9]+([.][0-9]+)?$ ]] || [[ "${RTF}" == 0 ]]; then
    echo "--rtf needs a positive number, for example --rtf 5" >&2
    exit 2
  fi
  SIM_ARGS+=(-z "$(awk -v r="${RTF}" -v s="${STEP_SIZE}" 'BEGIN { printf "%d", r / s }')")
fi
if [[ "${HAS_SEED}" == false ]]; then
  SIM_ARGS+=(--seed "${SIM_SEED}")
fi
if [[ "${HAS_GUI_CONFIG}" == false && "${HEADLESS}" == false ]]; then
  SIM_ARGS+=(--gui-config "${PROJECT_ROOT}/config/hill_country.gui.config")
fi

echo "Gazebo seed: ${SIM_SEED}"
if [[ -n "${RTF}" ]]; then
  echo "Target real-time factor: ${RTF}x (autonomy paces itself on sim time)"
fi
if [[ "${HAS_GUI_CONFIG}" == false && "${HEADLESS}" == false ]]; then
  echo "GUI camera: chase view configured for hill_rover"
fi

exec gz sim --force-version 8 -r "${PROJECT_ROOT}/worlds/hill_country.sdf" "${SIM_ARGS[@]}"

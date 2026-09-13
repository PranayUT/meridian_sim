#!/usr/bin/env bash
# Drive a multi-round route campaign: every round bakes one vegetation variant
# and drives all the routes against it, then the next round moves to a new seed.
# The round seed feeds both the obstacles and the planner, so a round is one
# reproducible world and one reproducible sampler, and a route's variation
# across rounds is variation in both.
#
# Rounds are not run one after another. Every (round, route, direction) is an
# independent trial with its own simulator, so --jobs of them run at once and
# a fast route never waits on a slow one in the same round.
set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! python -c "import numpy; import gz.transport13" >/dev/null 2>&1; then
  if command -v conda >/dev/null 2>&1; then
    exec conda run --no-capture-output --name rugged-ugv \
      "${PROJECT_ROOT}/scripts/run_campaign.sh" "$@"
  fi
  echo "The rugged-ugv environment is unavailable. Run ${PROJECT_ROOT}/scripts/setup.sh first." >&2
  exit 127
fi

ROUNDS=3
JOBS=2
RTF=3
SEED=7
CAMPAIGN=""
ROUTES=(Route-11 Route-12 Route-13)
DIRECTIONS=(forward reverse)

usage() {
  cat <<'USAGE'
Usage: scripts/run_campaign.sh [options]

  --rounds N       vegetation/planner seeds to sweep (default 3)
  --jobs N         simulators to run at once (default 2)
  --rtf X          real-time factor per simulator (default 3)
  --seed S         first round's seed; round r uses S + r (default 7)
  --routes "A B"   routes from paths/from_truck (default Route-11 Route-12 Route-13)
  --directions "forward reverse"
  --campaign NAME  run-id prefix and summary filter (default a timestamp)
USAGE
}

while (($# > 0)); do
  case "$1" in
    --rounds) ROUNDS="$2"; shift 2 ;;
    --jobs) JOBS="$2"; shift 2 ;;
    --rtf) RTF="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --campaign) CAMPAIGN="$2"; shift 2 ;;
    --routes) read -ra ROUTES <<<"$2"; shift 2 ;;
    --directions) read -ra DIRECTIONS <<<"$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

for pair in "rounds:${ROUNDS}" "jobs:${JOBS}"; do
  if [[ ! "${pair#*:}" =~ ^[1-9][0-9]*$ ]]; then
    echo "--${pair%%:*} needs a positive integer" >&2
    exit 2
  fi
done
if [[ ! "${SEED}" =~ ^[0-9]+$ ]]; then
  echo "--seed needs a non-negative integer" >&2
  exit 2
fi
CAMPAIGN="${CAMPAIGN:-$(date +%Y%m%d_%H%M%S)}"
if [[ ! "${CAMPAIGN}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "--campaign may only contain letters, digits, dot, dash, and underscore" >&2
  exit 2
fi

VEG_DIR="${PROJECT_ROOT}/runtime/vegetation"
FAILURES="${PROJECT_ROOT}/runtime/experiments/${CAMPAIGN}_failures.txt"
mkdir -p "${VEG_DIR}" "$(dirname "${FAILURES}")"
: >"${FAILURES}"

TRIALS=$((ROUNDS * ${#ROUTES[@]} * ${#DIRECTIONS[@]}))
echo "campaign ${CAMPAIGN}: ${ROUNDS} round(s) x ${#ROUTES[@]} route(s) x ${#DIRECTIONS[@]} direction(s)"
echo "${TRIALS} trials, ${JOBS} simulator(s) at once, ${RTF}x each, seeds ${SEED}..$((SEED + ROUNDS - 1))"

# Bake every round's obstacles before dispatching anything. Routes in a round
# share one variant, so building up front keeps two of that round's trials from
# writing the same meshes at the same time.
for ((round = 0; round < ROUNDS; round++)); do
  seed=$((SEED + round))
  if ! python "${PROJECT_ROOT}/tools/make_vegetation.py" \
      --seed "${seed}" --out "${VEG_DIR}/seed-${seed}"; then
    echo "could not bake vegetation for seed ${seed}" >&2
    exit 1
  fi
done
echo "vegetation on disk: $(du -sh "${VEG_DIR}" | cut -f1) under ${VEG_DIR}"

run_trial() {
  local run_id="$1" route="$2" direction="$3" seed="$4" veg_root="$5"
  local run_dir="${PROJECT_ROOT}/runtime/experiments/${run_id}"
  mkdir -p "${run_dir}"
  echo "$(date +%H:%M:%S) start ${run_id}"
  if "${PROJECT_ROOT}/scripts/run_experiment.sh" \
      --run-id "${run_id}" --rtf "${RTF}" --veg-root "${veg_root}" \
      --routes "${route}" --directions "${direction}" \
      --cycles 1 --seed "${seed}" --veg-seed "${seed}" \
      >"${run_dir}/campaign.log" 2>&1; then
    echo "$(date +%H:%M:%S) done  ${run_id}"
  else
    echo "$(date +%H:%M:%S) FAILED ${run_id} (see ${run_dir}/campaign.log)"
    echo "${run_id}" >>"${FAILURES}"
  fi
}

for ((round = 0; round < ROUNDS; round++)); do
  seed=$((SEED + round))
  for route in "${ROUTES[@]}"; do
    for direction in "${DIRECTIONS[@]}"; do
      while (($(jobs -pr | wc -l) >= JOBS)); do
        wait -n
      done
      run_trial "${CAMPAIGN}_s${seed}_${route}_${direction}" \
        "${route}" "${direction}" "${seed}" "${VEG_DIR}/seed-${seed}" &
    done
  done
done
wait

failed=$(wc -l <"${FAILURES}")
if ((failed > 0)); then
  echo
  echo "${failed} trial(s) did not finish:"
  cat "${FAILURES}"
else
  rm -f "${FAILURES}"
fi

echo
python "${PROJECT_ROOT}/tools/summarize_experiments.py" \
  "${PROJECT_ROOT}/runtime/experiments/${CAMPAIGN}_"*/campaign.csv

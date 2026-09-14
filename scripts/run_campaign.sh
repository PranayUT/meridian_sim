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
# Queue trials serially by default so a large campaign is safe on a modest
# machine. Callers with measured headroom can still opt into more workers.
JOBS=1
RTF=3
SEED=7
CAMPAIGN=""
ROUTES=(Route-11 Route-12 Route-13)
DIRECTIONS=(forward reverse)
ASSISTANCE="ground_only"
UAV_THRESHOLD="0.20"
MAPPING_MATURITY="1.0"
# Zero leaves trials unpinned, which is the right default for a campaign that
# has a machine to itself. Set it to give each concurrent trial its own cores.
CPUS_PER_TRIAL=0
CPU_BASE=0
LOCKSTEP_ARGS=()

usage() {
  cat <<'USAGE'
Usage: scripts/run_campaign.sh [options]

  --rounds N       vegetation/planner seeds to sweep (default 3)
  --jobs N         simulators to run at once (default 1; trials are queued)
  --rtf X          real-time factor per simulator (default 3)
  --seed S         first round's seed; round r uses S + r (default 7)
  --routes "A B"   routes from paths/from_truck (default Route-11 Route-12 Route-13)
  --directions "forward reverse"
  --assistance MODE  ground_only, greedy_uav, counterfactual_uav,
                     explore_then_drive, or always_on_uav
  --uav-threshold X request when uncertain rollout exposure reaches X (default .20)
  --mapping-maturity X  seconds one swept cell must remain uncertain (default 1.0)
  --campaign NAME  run-id prefix and summary filter (default a timestamp)
  --cpus-per-trial N  pin each concurrent trial to its own N cores (default 0, unpinned)
  --cpu-base N     first core to hand out, so two campaigns can share a machine
  --lockstep       step the world from the planner, so a trial runs at the same
                   rate in simulator time however many trials share the machine
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
    --assistance) ASSISTANCE="$2"; shift 2 ;;
    --uav-threshold) UAV_THRESHOLD="$2"; shift 2 ;;
    --mapping-maturity) MAPPING_MATURITY="$2"; shift 2 ;;
    --cpus-per-trial) CPUS_PER_TRIAL="$2"; shift 2 ;;
    --cpu-base) CPU_BASE="$2"; shift 2 ;;
    --lockstep) LOCKSTEP_ARGS=(--lockstep); shift ;;
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
for pair in "cpus-per-trial:${CPUS_PER_TRIAL}" "cpu-base:${CPU_BASE}"; do
  if [[ ! "${pair#*:}" =~ ^[0-9]+$ ]]; then
    echo "--${pair%%:*} needs a non-negative integer" >&2
    exit 2
  fi
done
if ((CPUS_PER_TRIAL > 0)); then
  online_cpus="$(nproc)"
  if ((CPU_BASE + JOBS * CPUS_PER_TRIAL > online_cpus)); then
    echo "pinning needs $((CPU_BASE + JOBS * CPUS_PER_TRIAL)) cores but only ${online_cpus} are online" >&2
    exit 2
  fi
  if ! command -v taskset >/dev/null 2>&1; then
    echo "--cpus-per-trial needs taskset (util-linux)" >&2
    exit 2
  fi
fi
case "${ASSISTANCE}" in
  ground_only|greedy_uav|counterfactual_uav|explore_then_drive|always_on_uav) ;;
  *)
    echo "--assistance must be ground_only, greedy_uav, counterfactual_uav," \
         "explore_then_drive, or always_on_uav" >&2
    exit 2
    ;;
esac
CAMPAIGN="${CAMPAIGN:-$(date +%Y%m%d_%H%M%S)}"
if [[ ! "${CAMPAIGN}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "--campaign may only contain letters, digits, dot, dash, and underscore" >&2
  exit 2
fi

# Job control puts every trial in its own process group. A signal can then be
# aimed at a whole trial - harness, planner, and simulator together - instead of
# reaching only the script that launched it.
set -m

VEG_DIR="${PROJECT_ROOT}/runtime/vegetation"
# One directory per invocation: every trial of this campaign is a subdirectory
# of it, so a run is a single thing to inspect, archive, or delete.
CAMPAIGN_DIR="${PROJECT_ROOT}/runtime/experiments/${CAMPAIGN}"
FAILURES="${CAMPAIGN_DIR}/failures.txt"
mkdir -p "${VEG_DIR}" "${CAMPAIGN_DIR}"
: >"${FAILURES}"

TRIALS=$((ROUNDS * ${#ROUTES[@]} * ${#DIRECTIONS[@]}))
echo "campaign ${CAMPAIGN}: ${ROUNDS} round(s) x ${#ROUTES[@]} route(s) x ${#DIRECTIONS[@]} direction(s)"
echo "${TRIALS} trials, ${JOBS} simulator(s) at once, ${RTF}x each, seeds ${SEED}..$((SEED + ROUNDS - 1))"
echo "results under ${CAMPAIGN_DIR}"

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

# Ctrl+C must not leave simulators behind: an orphaned gz sim holds a core and
# keeps its partition claimed long after the campaign is gone. Signal every
# trial group, give each run_experiment.sh time to stop its own simulator, then
# insist, and sweep any simulator whose harness died before it could clean up.
sweep_sims() {
  local pid_file pid
  for pid_file in "${CAMPAIGN_DIR}"/*/sim.pid; do
    [[ -e "${pid_file}" ]] || continue
    pid="$(cat "${pid_file}" 2>/dev/null || true)"
    [[ "${pid}" =~ ^[0-9]+$ ]] || continue
    kill -0 "${pid}" 2>/dev/null || continue
    # Confirm it is still a simulator: pids get reused, and this sends SIGKILL.
    [[ "$(ps -o args= -p "${pid}" 2>/dev/null)" == *"gz sim"* ]] || continue
    kill -KILL "${pid}" 2>/dev/null || true
    rm -f "${pid_file}"
  done
}

shutdown() {
  trap - INT TERM
  local pgid pgids alive
  echo
  echo "interrupted: stopping trials and their simulators"
  pgids="$(jobs -pr)"
  # Take the job list first, then leave monitor mode: bash would otherwise
  # announce each terminated job by echoing the raw run_trial invocation.
  set +m
  for pgid in ${pgids}; do
    kill -TERM -"${pgid}" 2>/dev/null || kill -TERM "${pgid}" 2>/dev/null || true
  done
  for _ in $(seq 1 60); do
    alive=0
    for pgid in ${pgids}; do
      kill -0 -"${pgid}" 2>/dev/null && alive=1
    done
    ((alive)) || break
    sleep 0.5
  done
  for pgid in ${pgids}; do
    kill -KILL -"${pgid}" 2>/dev/null || kill -KILL "${pgid}" 2>/dev/null || true
  done
  sweep_sims
  # pgrep prints 0 and exits non-zero when nothing matches, so a `|| echo 0`
  # fallback would report the count twice.
  local still
  still="$(pgrep -cf '^gz sim ' 2>/dev/null)" || still=0
  echo "stopped; ${still} simulator(s) from any campaign still up"
  exit 130
}
trap shutdown INT TERM

# Slots are claimed with mkdir, which is atomic, so two trials dispatched at
# the same moment cannot be handed the same cores.
SLOT_DIR="${CAMPAIGN_DIR}/.slots"
rm -rf "${SLOT_DIR}"
mkdir -p "${SLOT_DIR}"

claim_slot() {
  local index
  while true; do
    for ((index = 0; index < JOBS; index++)); do
      if mkdir "${SLOT_DIR}/${index}" 2>/dev/null; then
        echo "${index}"
        return 0
      fi
    done
    sleep 0.2
  done
}

run_trial() {
  local trial="$1" route="$2" direction="$3" seed="$4" veg_root="$5"
  local run_dir="${CAMPAIGN_DIR}/${trial}"
  mkdir -p "${run_dir}"
  local slot="" first=0
  if ((CPUS_PER_TRIAL > 0)); then
    slot="$(claim_slot)"
    first=$((CPU_BASE + slot * CPUS_PER_TRIAL))
    export TRIAL_CPUS="${first}-$((first + CPUS_PER_TRIAL - 1))"
    echo "$(date +%H:%M:%S) start ${trial} on cores ${TRIAL_CPUS}"
  else
    echo "$(date +%H:%M:%S) start ${trial}"
  fi
  if "${PROJECT_ROOT}/scripts/run_experiment.sh" \
      --run-id "${CAMPAIGN}/${trial}" --rtf "${RTF}" --veg-root "${veg_root}" \
      --routes "${route}" --directions "${direction}" \
      --cycles 1 --seed "${seed}" --veg-seed "${seed}" \
      --assistance "${ASSISTANCE}" --uav-uncertainty-threshold "${UAV_THRESHOLD}" \
      --mapping-uncertainty-maturity "${MAPPING_MATURITY}" \
      ${LOCKSTEP_ARGS[@]+"${LOCKSTEP_ARGS[@]}"} \
      >"${run_dir}/campaign.log" 2>&1; then
    echo "$(date +%H:%M:%S) done  ${trial}"
  else
    echo "$(date +%H:%M:%S) FAILED ${trial} (see ${run_dir}/campaign.log)"
    echo "${trial}" >>"${FAILURES}"
  fi
  [[ -n "${slot}" ]] && rmdir "${SLOT_DIR}/${slot}" 2>/dev/null
  return 0
}

for ((round = 0; round < ROUNDS; round++)); do
  seed=$((SEED + round))
  for route in "${ROUTES[@]}"; do
    for direction in "${DIRECTIONS[@]}"; do
      while (($(jobs -pr | wc -l) >= JOBS)); do
        wait -n
      done
      run_trial "s${seed}_${route}_${direction}" \
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
  "${CAMPAIGN_DIR}"/*/campaign.csv

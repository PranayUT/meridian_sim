#!/usr/bin/env bash
# Source this after activating the rugged-ugv Conda environment.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export GZ_SIM_RESOURCE_PATH="${PROJECT_ROOT}/models${GZ_SIM_RESOURCE_PATH:+:${GZ_SIM_RESOURCE_PATH}}"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
# Reliable same-host discovery. Override before sourcing for distributed simulation.
export GZ_IP="${GZ_IP:-127.0.0.1}"
# The planner tick, the lidar callback, and the mapping worker all contend for
# one GIL, so a trial is single-thread bound. Letting each trial's BLAS open a
# pool per core buys it nothing and, in a campaign running many simulators at
# once, costs the machine a large amount of context switching. Pinning to one
# thread also removes threaded-reduction rounding, so a seeded run is
# reproducible.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

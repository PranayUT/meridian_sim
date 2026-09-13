#!/usr/bin/env bash
# Source this after activating the rugged-ugv Conda environment.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export GZ_SIM_RESOURCE_PATH="${PROJECT_ROOT}/models${GZ_SIM_RESOURCE_PATH:+:${GZ_SIM_RESOURCE_PATH}}"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
# Reliable same-host discovery. Override before sourcing for distributed simulation.
export GZ_IP="${GZ_IP:-127.0.0.1}"

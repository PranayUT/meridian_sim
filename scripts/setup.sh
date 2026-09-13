#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

PACKAGES=(
  "python=3.12"
  "gz-sim8=8.10.*"
  "gz-launch7=7.1.*"
  "gz-transport13=13.5.*"
  "gz-msgs10=10.3.*"
  "numpy=2.*"
  "pillow>=11"
  "matplotlib=3.*"
)

# --override-channels avoids any configured Anaconda defaults; all packages are
# resolved as one compatible transaction from conda-forge.
if conda env list | awk '{print $1}' | grep -qx rugged-ugv; then
  conda install --name rugged-ugv --override-channels --channel conda-forge --yes "${PACKAGES[@]}"
else
  conda create --name rugged-ugv --override-channels --channel conda-forge --yes "${PACKAGES[@]}"
fi
if [[ ! -f "${PROJECT_ROOT}/models/hill_terrain/meshes/terrain.tif" ]]; then
  conda run --name rugged-ugv python tools/generate_terrain.py
fi

echo "Environment ready. Run:"
echo "  conda activate rugged-ugv"
echo "  source scripts/env.sh"
echo "  scripts/run_sim.sh"

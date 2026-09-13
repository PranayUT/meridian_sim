#!/usr/bin/env python3
"""Bake one vegetation variant from the painted masks under a chosen seed.

The painted regions in maps/vegetation_paint.npz stay fixed. The seed only
drives per-plant jitter, scale, and yaw, so a round gets the same corridors in
the same places with the obstacles nudged around inside them: between two seeds
the bushes and trees keep their count and move a median 0.30 m.

The output is a Gazebo resource root, so putting it ahead of models/ on
GZ_SIM_RESOURCE_PATH overrides the committed painted_vegetation model without
editing the world file, and concurrent campaigns can each use their own.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from vegetation import KINDS, generate_model, load_elevation, plants_from_masks

MASKS = ROOT / "maps" / "vegetation_paint.npz"
DEM = ROOT / "models" / "hill_terrain" / "meshes" / "terrain.tif"


def build(seed: int, out: Path, masks_path: Path = MASKS, dem: Path = DEM,
          density: float | None = None, force: bool = False) -> tuple[Path, dict | None]:
    """Write <out>/painted_vegetation for one seed. Returns (model dir, counts).

    Counts are None when an up-to-date variant was already on disk, so a
    campaign that reuses a round's seed does not rebuild 160 MB of meshes.
    """
    model_dir = out / "painted_vegetation"
    baked = model_dir / "meshes" / "collision.obj"
    if not force and baked.is_file() and baked.stat().st_mtime >= masks_path.stat().st_mtime:
        return model_dir, None
    if not masks_path.is_file():
        raise SystemExit(f"no painted masks at {masks_path}; run scripts/paint_vegetation.sh first")
    with np.load(masks_path, allow_pickle=False) as saved:
        masks = {kind: saved[kind] for kind in KINDS}
        painted_density = float(saved["density"])
    plants = plants_from_masks(masks, load_elevation(dem), density or painted_density, seed)
    return model_dir, generate_model(plants, model_dir, seed)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True,
                        help="resource root; painted_vegetation/ is created inside it")
    parser.add_argument("--masks", type=Path, default=MASKS)
    parser.add_argument("--dem", type=Path, default=DEM)
    parser.add_argument("--density", type=float, default=None,
                        help="overrides the density saved with the masks")
    parser.add_argument("--force", action="store_true", help="rebuild even if up to date")
    args = parser.parse_args()

    start = time.monotonic()
    model_dir, counts = build(args.seed, args.out, args.masks, args.dem, args.density, args.force)
    if counts is None:
        print(f"vegetation seed {args.seed}: reusing {model_dir}")
    else:
        print(f"vegetation seed {args.seed}: {counts['grass']:,} grass, {counts['bush']:,} bushes, "
              f"{counts['tree']:,} trees in {time.monotonic() - start:.1f}s -> {model_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

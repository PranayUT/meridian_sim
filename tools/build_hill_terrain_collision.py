#!/usr/bin/env python3
"""Regenerate hill_terrain's collision mesh from the real DEM crop.

meshes/terrain.tif is a 513x513, 1 m/pixel elevation crop (EPSG:6343) covering
the map_bounds.txt survey area, streamed from the USGS 3DEP one-meter lidar
mosaic (maps/TX_Central_B1_2017.vrt). The Conda Gazebo build loads that GeoTIFF
directly as the visual heightmap (gz-common's geospatial DEM loader), but its
default physics backend does not reliably build heightmap collision, so this
script derives a matching decimated collision mesh from the same elevations
(same subsampling and coordinate convention as the original synthetic
generate_terrain.py, just with real data).
"""

from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DEM = ROOT / "models" / "hill_terrain" / "meshes" / "terrain.tif"
COLLISION_OUTPUT = ROOT / "models" / "hill_terrain" / "meshes" / "collision.obj"

SIZE_XY = 512.0  # meters; matches <size> in model.sdf (513 samples @ 1 m/px)


def main() -> None:
    elevation = np.array(Image.open(DEM), dtype=np.float64)
    min_elev, max_elev = float(elevation.min()), float(elevation.max())
    relief = max_elev - min_elev

    collision = elevation[::4, ::4] - min_elev  # meters above the crop's low point
    rows, cols = collision.shape
    dz_dy, dz_dx = np.gradient(
        collision, SIZE_XY / (rows - 1), SIZE_XY / (cols - 1)
    )
    normals = np.stack((-dz_dx, dz_dy, np.ones_like(collision)), axis=-1)
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True)

    with COLLISION_OUTPUT.open("w", encoding="ascii") as mesh:
        mesh.write("# hill_terrain collision mesh, decimated from meshes/terrain.tif\n")
        for row in range(rows):
            world_y = SIZE_XY / 2 - SIZE_XY * row / (rows - 1)
            for col in range(cols):
                world_x = -SIZE_XY / 2 + SIZE_XY * col / (cols - 1)
                world_z = collision[row, col]
                mesh.write(f"v {world_x:.5f} {world_y:.5f} {world_z:.5f}\n")
        for normal_x, normal_y, normal_z in normals.reshape(-1, 3):
            mesh.write(f"vn {normal_x:.6f} {normal_y:.6f} {normal_z:.6f}\n")
        for row in range(rows - 1):
            for col in range(cols - 1):
                a = row * cols + col + 1
                b, c, d = a + 1, a + cols, a + cols + 1
                mesh.write(f"f {a}//{a} {c}//{c} {b}//{b}\n")
                mesh.write(f"f {b}//{b} {c}//{c} {d}//{d}\n")

    print(f"min elevation {min_elev:.3f} m, max elevation {max_elev:.3f} m, relief {relief:.3f} m")
    print(f"Wrote {COLLISION_OUTPUT} ({rows}x{cols} collision grid)")


if __name__ == "__main__":
    main()

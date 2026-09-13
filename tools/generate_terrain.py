#!/usr/bin/env python3
"""Generate a deterministic 513x513, 16-bit hill-country heightmap.

hill_terrain's collision.obj is currently derived from a real DEM crop
(see tools/build_hill_terrain_collision.py) rather than from this synthetic
generator. Re-running this script overwrites that real collision mesh with
the synthetic demo one; regenerate it afterward with
build_hill_terrain_collision.py if you need the real terrain back.
"""

from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "models" / "hill_terrain" / "meshes" / "terrain.png"
COLLISION_OUTPUT = ROOT / "models" / "hill_terrain" / "meshes" / "collision.obj"
MATERIALS = ROOT / "models" / "hill_terrain" / "materials"


def main() -> None:
    size = 513  # Heightmaps should be 2^n + 1 samples per side.
    axis = np.linspace(-1.0, 1.0, size)
    x, y = np.meshgrid(axis, axis)
    rng = np.random.default_rng(4207)

    # Broad ridges plus small-scale ground roughness. The smooth center is the start pad.
    height = (
        0.55 * np.sin(2.3 * x + 0.7 * y)
        + 0.38 * np.cos(3.7 * y - 0.4 * x)
        + 0.24 * np.sin(7.0 * (x + y))
        + 0.16 * np.cos(12.0 * x) * np.sin(9.0 * y)
    )
    for _ in range(14):
        cx, cy = rng.uniform(-0.9, 0.9, 2)
        sigma = rng.uniform(0.07, 0.24)
        amplitude = rng.uniform(-0.35, 0.55)
        height += amplitude * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma**2))

    height = (height - height.min()) / (height.max() - height.min())
    radius = np.hypot(x, y)
    pad_blend = np.clip((radius - 0.035) / 0.055, 0.0, 1.0)
    height = pad_blend * height + (1.0 - pad_blend) * 0.40

    pixels = np.round(height * 65535).astype(np.uint16)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels).save(OUTPUT)
    MATERIALS.mkdir(parents=True, exist_ok=True)
    texture_rng = np.random.default_rng(37)
    grass = np.empty((256, 256, 3), dtype=np.uint8)
    base = texture_rng.normal(0.0, 9.0, (256, 256))
    grass[..., 0] = np.clip(79 + base, 0, 255)
    grass[..., 1] = np.clip(105 + base, 0, 255)
    grass[..., 2] = np.clip(52 + base * 0.6, 0, 255)
    Image.fromarray(grass).save(MATERIALS / "grass.png")
    normal = np.zeros((8, 8, 3), dtype=np.uint8)
    normal[...] = (128, 128, 255)
    Image.fromarray(normal).save(MATERIALS / "flat_normal.png")
    # Gazebo's Conda-packaged DART backend does not construct an SDF heightmap
    # collision reliably. A 129x129 mesh is light enough for physics and follows
    # the exact same elevation source as the detailed visual heightmap.
    collision = height[::4, ::4]
    rows, cols = collision.shape
    dz_dy, dz_dx = np.gradient(collision * 22.0, 200.0 / (rows - 1), 200.0 / (cols - 1))
    normals = np.stack((-dz_dx, dz_dy, np.ones_like(collision)), axis=-1)
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
    with COLLISION_OUTPUT.open("w", encoding="ascii") as mesh:
        mesh.write("# Generated coarse hill-country collision mesh\n")
        for row in range(rows):
            world_y = 100.0 - 200.0 * row / (rows - 1)
            for col in range(cols):
                world_x = -100.0 + 200.0 * col / (cols - 1)
                world_z = 22.0 * collision[row, col] - 7.0
                mesh.write(f"v {world_x:.5f} {world_y:.5f} {world_z:.5f}\n")
        for normal_x, normal_y, normal_z in normals.reshape(-1, 3):
            mesh.write(f"vn {normal_x:.6f} {normal_y:.6f} {normal_z:.6f}\n")
        for row in range(rows - 1):
            for col in range(cols - 1):
                a = row * cols + col + 1
                b, c, d = a + 1, a + cols, a + cols + 1
                mesh.write(f"f {a}//{a} {c}//{c} {b}//{b}\n")
                mesh.write(f"f {b}//{b} {c}//{c} {d}//{d}\n")
    print(f"Wrote {OUTPUT} ({size}x{size}, 16-bit)")
    print(f"Wrote {COLLISION_OUTPUT} ({rows}x{cols} collision grid)")


if __name__ == "__main__":
    main()

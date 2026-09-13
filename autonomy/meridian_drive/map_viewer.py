"""Live value and uncertainty viewer for Meridian-compatible onboard maps."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch


class MapViewer:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.figure, self.axes = plt.subplots(2, 2, figsize=(12, 10), num="UGV live maps")
        self.figure.suptitle("Waiting for UGV map data…")
        self.last_timestamp = -1.0
        self.colorbars_created = False
        self.timer = self.figure.canvas.new_timer(interval=250)
        self.timer.add_callback(self.refresh)
        self.timer.start()

    @staticmethod
    def _extent(origin: np.ndarray, shape: tuple[int, int], resolution: float) -> tuple[float, ...]:
        return (
            float(origin[0]),
            float(origin[0] + shape[1] * resolution),
            float(origin[1]),
            float(origin[1] + shape[0] * resolution),
        )

    @staticmethod
    def _pose(axis: object, pose: np.ndarray, camera: bool = False, hfov: float = 0.0, distance: float = 0.0) -> None:
        x, y, yaw = (float(value) for value in pose[:3])
        axis.arrow(x, y, 1.0 * math.cos(yaw), 1.0 * math.sin(yaw), width=0.08, color="#1967d2", zorder=5)
        axis.plot(x, y, marker="o", color="#1967d2", markersize=5, zorder=6)
        if camera:
            for angle in (yaw - hfov / 2.0, yaw + hfov / 2.0):
                axis.plot([x, x + distance * math.cos(angle)], [y, y + distance * math.sin(angle)], color="#00bcd4", linewidth=1.2)

    def refresh(self) -> None:
        try:
            with np.load(self.path, allow_pickle=False) as archive:
                timestamp = float(np.asarray(archive["timestamp"]).item())
                if timestamp <= self.last_timestamp:
                    return
                occupancy = np.asarray(archive["occupancy"])
                occupancy_variance = np.asarray(archive["occupancy_variance"])
                occupancy_origin = np.asarray(archive["occupancy_origin"])
                semantics = np.asarray(archive["semantic_cost"])
                semantic_uncertainty = np.asarray(archive["semantic_uncertainty"])
                semantic_obstacles = np.asarray(archive["semantic_obstacle_probability"])
                semantic_origin = np.asarray(archive["semantic_origin"])
                resolution = float(np.asarray(archive["resolution"]).item())
                pose = np.asarray(archive["pose"])
                hfov = float(np.asarray(archive["camera_hfov"]).item())
                camera_range = float(np.asarray(archive["camera_range"]).item())
        except (FileNotFoundError, KeyError, OSError, ValueError):
            return
        self.last_timestamp = timestamp
        occupancy_axis, occupancy_uncertainty_axis = self.axes[0]
        semantic_axis, semantic_uncertainty_axis = self.axes[1]
        for axis in self.axes.flat:
            axis.clear()

        occupancy_display = np.zeros_like(occupancy, dtype=np.uint8)
        occupancy_display[occupancy == 0] = 1
        occupancy_display[occupancy == 50] = 2
        occupancy_display[occupancy == 100] = 3
        colors = ListedColormap(("#767676", "#f5f5f5", "#ffb347", "#d62828"))
        occupancy_axis.imshow(
            occupancy_display,
            origin="lower",
            extent=self._extent(occupancy_origin, occupancy.shape, resolution),
            cmap=colors,
            norm=BoundaryNorm((-0.5, 0.5, 1.5, 2.5, 3.5), colors.N),
            interpolation="nearest",
        )
        self._pose(occupancy_axis, pose)
        occupancy_axis.set_title("Lidar occupancy")
        occupancy_axis.legend(
            handles=[Patch(color=color, label=label) for color, label in zip(colors.colors, ("unknown", "ground", "tall / porous", "solid"))],
            loc="upper right",
            fontsize=8,
        )

        uncertainty_cmap = plt.get_cmap("magma").copy()
        uncertainty_cmap.set_bad("#767676")
        occupancy_uncertainty_image = occupancy_uncertainty_axis.imshow(
            occupancy_variance,
            origin="lower",
            extent=self._extent(occupancy_origin, occupancy_variance.shape, resolution),
            cmap=uncertainty_cmap,
            vmin=0.0,
            vmax=0.25,
            interpolation="nearest",
        )
        self._pose(occupancy_uncertainty_axis, pose)
        occupancy_uncertainty_axis.set_title("Lidar occupancy variance")

        semantic_cmap = plt.get_cmap("RdYlGn_r").copy()
        semantic_cmap.set_bad("#767676")
        semantic_image = semantic_axis.imshow(
            semantics,
            origin="lower",
            extent=self._extent(semantic_origin, semantics.shape, resolution),
            cmap=semantic_cmap,
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
        )
        self._pose(semantic_axis, pose, True, hfov, camera_range)
        semantic_axis.set_title("Front RGB-D semantic cost")

        # Make collision-bearing semantic cells unambiguous. One-cell dilation
        # displays the rover-footprint margin used by the planner.
        obstacle_mask = np.isfinite(semantic_obstacles) & (semantic_obstacles >= 0.45)
        padded = np.pad(obstacle_mask, 1)
        inflated = np.zeros_like(obstacle_mask)
        for row_offset, col_offset in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)):
            inflated |= padded[
                1 + row_offset : 1 + row_offset + obstacle_mask.shape[0],
                1 + col_offset : 1 + col_offset + obstacle_mask.shape[1],
            ]
        semantic_axis.imshow(
            np.ma.masked_where(~inflated, inflated),
            origin="lower",
            extent=self._extent(semantic_origin, semantics.shape, resolution),
            cmap=ListedColormap(("#d00000",)),
            vmin=0.0,
            vmax=1.0,
            alpha=0.78,
            interpolation="nearest",
        )
        if np.any(inflated):
            semantic_axis.legend(
                handles=[Patch(color="#d00000", label="semantic collision")],
                loc="upper right",
                fontsize=8,
            )

        semantic_uncertainty_image = semantic_uncertainty_axis.imshow(
            semantic_uncertainty,
            origin="lower",
            extent=self._extent(semantic_origin, semantic_uncertainty.shape, resolution),
            cmap=uncertainty_cmap,
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
        )
        self._pose(semantic_uncertainty_axis, pose, True, hfov, camera_range)
        semantic_uncertainty_axis.set_title("Semantic uncertainty")

        if not self.colorbars_created:
            self.figure.colorbar(
                occupancy_uncertainty_image,
                ax=occupancy_uncertainty_axis,
                fraction=0.046,
                pad=0.04,
                label="posterior variance",
            )
            self.figure.colorbar(
                semantic_image,
                ax=semantic_axis,
                fraction=0.046,
                pad=0.04,
                label="semantic cost",
            )
            self.figure.colorbar(
                semantic_uncertainty_image,
                ax=semantic_uncertainty_axis,
                fraction=0.046,
                pad=0.04,
                label="uncertainty",
            )
            self.colorbars_created = True

        for axis in self.axes.flat:
            axis.set_xlabel("world X [m]")
            axis.set_ylabel("world Y [m]")
            axis.set_aspect("equal")
            axis.grid(color="black", alpha=0.12, linewidth=0.4)
        self.figure.suptitle("Meridian Drive onboard mapping and uncertainty — 0.25 m cells")
        self.figure.canvas.draw_idle()


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", type=Path, default=root / "runtime" / "ground_maps.npz")
    viewer = MapViewer(parser.parse_args().map)
    plt.show()


if __name__ == "__main__":
    main()

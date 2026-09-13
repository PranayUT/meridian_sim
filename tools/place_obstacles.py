#!/usr/bin/env python3
"""Paint terrain-covering vegetation regions on the simulator orthophoto."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QMouseEvent, QPainter, QPen, QPixmap, QWheelEvent
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QHBoxLayout, QLabel,
    QMainWindow, QMessageBox, QPushButton, QSlider, QSpinBox, QVBoxLayout, QWidget,
)

from vegetation import (
    BASE_SPACING, KINDS, TERRAIN_SIZE, generate_model, install_world_include,
    load_elevation, plants_from_masks, remove_legacy_vegetation,
)

ROOT = Path(__file__).resolve().parents[1]
WORLD = ROOT / "worlds" / "hill_country.sdf"
DEM = ROOT / "models" / "hill_terrain" / "meshes" / "terrain.tif"
ORTHOPHOTO = ROOT / "models" / "hill_terrain" / "materials" / "orthophoto.png"
STATE = ROOT / "maps" / "vegetation_paint.npz"
MODEL_DIR = ROOT / "models" / "painted_vegetation"
MASK_SIZE = 512
COLORS = {"grass": (100, 240, 60, 105), "bush": (20, 150, 45, 135), "tree": (5, 75, 20, 165)}


class PaintCanvas(QWidget):
    changed = Signal()

    def __init__(self, image_path: Path) -> None:
        super().__init__()
        self.background = QPixmap(str(image_path))
        self.setMinimumSize(700, 700)
        self.setMouseTracking(True)
        self.masks = {kind: np.zeros((MASK_SIZE, MASK_SIZE), dtype=np.uint8) for kind in KINDS}
        self.kind, self.radius_m = "bush", 5.0
        self.painting = self.erasing = False
        self.panning = False
        self.pan_start: QPointF | None = None
        self.zoom = 1.0
        self.center_u = self.center_v = 0.5
        self.undo_stack: list[dict[str, np.ndarray]] = []
        self.overlay: QImage | None = None
        self.cursor: QPointF | None = None
        self._rebuild_overlay()

    def load_masks(self, masks: dict[str, np.ndarray]) -> None:
        for kind in KINDS:
            value = np.asarray(masks[kind], dtype=np.uint8)
            if value.shape != (MASK_SIZE, MASK_SIZE):
                raise ValueError(f"saved {kind} mask has the wrong size")
            self.masks[kind] = value.copy()
        self._rebuild_overlay()

    def _square(self) -> tuple[float, float, float]:
        side = float(min(self.width(), self.height()))
        return (self.width() - side) / 2, (self.height() - side) / 2, side

    def _mask_point(self, position: QPointF) -> tuple[int, int] | None:
        left, top, side = self._square()
        screen_u, screen_v = (position.x() - left) / side, (position.y() - top) / side
        if not (0 <= screen_u < 1 and 0 <= screen_v < 1): return None
        span = 1.0 / self.zoom
        u = self.center_u - span / 2 + screen_u * span
        v = self.center_v - span / 2 + screen_v * span
        return (int(v * MASK_SIZE), int(u * MASK_SIZE)) if 0 <= u < 1 and 0 <= v < 1 else None

    def _source_rect(self, width: int, height: int) -> QRectF:
        span = 1.0 / self.zoom
        return QRectF(
            (self.center_u - span / 2) * width, (self.center_v - span / 2) * height,
            span * width, span * height,
        )

    def _clamp_view(self) -> None:
        half = 0.5 / self.zoom
        self.center_u = float(np.clip(self.center_u, half, 1.0 - half))
        self.center_v = float(np.clip(self.center_v, half, 1.0 - half))

    def set_zoom(self, value: float, focal: QPointF | None = None) -> None:
        old_zoom = self.zoom
        new_zoom = float(np.clip(value, 1.0, 32.0))
        if focal is not None:
            left, top, side = self._square()
            su, sv = (focal.x() - left) / side, (focal.y() - top) / side
            if 0 <= su <= 1 and 0 <= sv <= 1:
                map_u = self.center_u + (su - .5) / old_zoom
                map_v = self.center_v + (sv - .5) / old_zoom
                self.center_u = map_u - (su - .5) / new_zoom
                self.center_v = map_v - (sv - .5) / new_zoom
        self.zoom = new_zoom; self._clamp_view(); self.changed.emit(); self.update()

    def fit_view(self) -> None:
        self.zoom = 1.0; self.center_u = self.center_v = .5; self.changed.emit(); self.update()

    def _push_undo(self) -> None:
        self.undo_stack.append({kind: mask.copy() for kind, mask in self.masks.items()})
        self.undo_stack = self.undo_stack[-20:]

    def _paint(self, position: QPointF) -> None:
        center = self._mask_point(position)
        if center is None:
            return
        row, col = center
        radius = max(1, int(round(self.radius_m * MASK_SIZE / TERRAIN_SIZE)))
        r0, r1 = max(0, row - radius), min(MASK_SIZE, row + radius + 1)
        c0, c1 = max(0, col - radius), min(MASK_SIZE, col + radius + 1)
        yy, xx = np.ogrid[r0:r1, c0:c1]
        circle = (yy - row) ** 2 + (xx - col) ** 2 <= radius * radius
        for kind, mask in self.masks.items():
            region = mask[r0:r1, c0:c1]
            region[circle] = 0 if self.erasing else int(kind == self.kind)
        self._rebuild_overlay(); self.changed.emit()

    def _rebuild_overlay(self) -> None:
        rgba = np.zeros((MASK_SIZE, MASK_SIZE, 4), dtype=np.uint8)
        for kind in KINDS:
            rgba[self.masks[kind].astype(bool)] = COLORS[kind]
        self.overlay = QImage(rgba.data, MASK_SIZE, MASK_SIZE, MASK_SIZE * 4, QImage.Format.Format_RGBA8888).copy()
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self); painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        left, top, side = self._square(); target = QRectF(left, top, side, side)
        painter.drawPixmap(target, self.background, self._source_rect(self.background.width(), self.background.height()))
        if self.overlay is not None:
            painter.drawImage(target, self.overlay, self._source_rect(self.overlay.width(), self.overlay.height()))
        if self.cursor is not None:
            radius = self.radius_m / TERRAIN_SIZE * side * self.zoom
            painter.setPen(QPen(QColor("white"), 2, Qt.PenStyle.DashLine)); painter.drawEllipse(self.cursor, radius, radius)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.MiddleButton:
            self.panning = True; self.pan_start = event.position(); return
        if event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton):
            self._push_undo(); self.painting = True
            self.erasing = event.button() == Qt.MouseButton.RightButton; self._paint(event.position())

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        self.cursor = event.position()
        if self.panning and self.pan_start is not None:
            _, _, side = self._square(); delta = event.position() - self.pan_start
            self.center_u -= delta.x() / side / self.zoom
            self.center_v -= delta.y() / side / self.zoom
            self._clamp_view(); self.pan_start = event.position(); self.update()
        elif self.painting: self._paint(event.position())
        else: self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.MiddleButton:
            self.panning = False; self.pan_start = None
        else: self.painting = False

    def leaveEvent(self, _event) -> None:
        self.cursor = None; self.update()

    def wheelEvent(self, event: QWheelEvent) -> None:
        direction = math.copysign(1, event.angleDelta().y())
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.radius_m = float(np.clip(self.radius_m + direction, 1, 40))
            self.changed.emit(); self.update()
        else:
            self.set_zoom(self.zoom * (1.25 if direction > 0 else .8), event.position())

    def undo(self) -> None:
        if self.undo_stack:
            self.masks = self.undo_stack.pop(); self._rebuild_overlay(); self.changed.emit()

    def clear(self, kind: str | None) -> None:
        self._push_undo()
        for target in KINDS if kind is None else (kind,): self.masks[target].fill(0)
        self._rebuild_overlay(); self.changed.emit()


class Editor(QMainWindow):
    def __init__(self, state_path: Path, world_path: Path) -> None:
        super().__init__(); self.state_path, self.world_path = state_path, world_path
        self.setWindowTitle("Hill Country Vegetation Painter")
        self.canvas = PaintCanvas(ORTHOPHOTO)
        self.kind = QComboBox(); self.kind.addItems([kind.title() for kind in KINDS]); self.kind.setCurrentText("Bush")
        self.brush = QSlider(Qt.Orientation.Horizontal); self.brush.setRange(1, 40); self.brush.setValue(5)
        self.density = QDoubleSpinBox(); self.density.setRange(0.2, 3); self.density.setSingleStep(.1); self.density.setValue(1)
        self.seed = QSpinBox(); self.seed.setRange(0, 2_147_483_647); self.seed.setValue(4207)
        self.replace_legacy = QCheckBox("Remove old individual grass, bush, and tree includes when saving")
        self.status = QLabel(); undo = QPushButton("Undo"); clear = QPushButton("Clear selected")
        clear_all = QPushButton("Clear all"); zoom_in = QPushButton("Zoom +")
        zoom_out = QPushButton("Zoom −"); fit = QPushButton("Fit"); save = QPushButton("Save vegetation to Gazebo")
        controls = QHBoxLayout()
        for label, widget in (("Plant", self.kind), ("Brush radius", self.brush), ("Density", self.density), ("Seed", self.seed)):
            controls.addWidget(QLabel(label)); controls.addWidget(widget)
        controls.addWidget(undo); controls.addWidget(clear); controls.addWidget(clear_all)
        controls.addWidget(zoom_out); controls.addWidget(zoom_in); controls.addWidget(fit)
        layout = QVBoxLayout(); layout.addLayout(controls); layout.addWidget(self.canvas, 1)
        layout.addWidget(QLabel("Left-drag paints · Right-drag erases · Wheel zooms · Middle-drag pans · Ctrl+wheel changes brush radius"))
        layout.addWidget(self.replace_legacy); layout.addWidget(self.status); layout.addWidget(save)
        central = QWidget(); central.setLayout(layout); self.setCentralWidget(central)
        self.kind.currentTextChanged.connect(self._kind_changed); self.brush.valueChanged.connect(self._brush_changed)
        self.canvas.changed.connect(self._update_status); self.density.valueChanged.connect(self._update_status)
        undo.clicked.connect(self.canvas.undo); clear.clicked.connect(lambda: self.canvas.clear(self.canvas.kind))
        clear_all.clicked.connect(lambda: self.canvas.clear(None)); save.clicked.connect(self.save)
        zoom_in.clicked.connect(lambda: self.canvas.set_zoom(self.canvas.zoom * 1.5))
        zoom_out.clicked.connect(lambda: self.canvas.set_zoom(self.canvas.zoom / 1.5))
        fit.clicked.connect(self.canvas.fit_view)
        self._load_state(); self._update_status(); self.resize(1050, 900)

    def _kind_changed(self, text: str) -> None:
        self.canvas.kind = text.lower(); self._update_status()

    def _brush_changed(self, value: int) -> None:
        self.canvas.radius_m = float(value); self._update_status(); self.canvas.update()

    def _load_state(self) -> None:
        if not self.state_path.exists(): return
        try:
            with np.load(self.state_path, allow_pickle=False) as saved:
                self.canvas.load_masks({kind: saved[kind] for kind in KINDS})
                self.density.setValue(float(saved["density"])); self.seed.setValue(int(saved["seed"]))
        except Exception as error: QMessageBox.warning(self, "Could not load paint", str(error))

    def _update_status(self) -> None:
        estimates = []
        for kind in KINDS:
            area = int(np.count_nonzero(self.canvas.masks[kind]))
            cell_area = BASE_SPACING[kind] ** 2 * math.sqrt(3) / 2 / self.density.value()
            estimates.append(f"{kind}: ~{round(area / cell_area):,}")
        self.status.setText(f"Zoom {self.canvas.zoom:.1f}×  |  Brush {self.canvas.radius_m:.0f} m  |  " + "  |  ".join(estimates))

    def save(self) -> None:
        try:
            plants = plants_from_masks(self.canvas.masks, load_elevation(DEM), self.density.value(), self.seed.value())
            counts = generate_model(plants, MODEL_DIR, self.seed.value())
            removed = remove_legacy_vegetation(self.world_path) if self.replace_legacy.isChecked() else 0
            install_world_include(self.world_path); self.state_path.parent.mkdir(parents=True, exist_ok=True)
            with self.state_path.open("wb") as output:
                np.savez_compressed(output, **self.canvas.masks, density=self.density.value(), seed=self.seed.value())
        except Exception as error:
            QMessageBox.critical(self, "Save failed", str(error)); return
        QMessageBox.information(self, "Vegetation saved", f"Generated {counts['grass']:,} grass clumps, {counts['bush']:,} bushes, and {counts['tree']:,} trees. Removed {removed} legacy includes. Restart Gazebo to load them.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, default=STATE); parser.add_argument("--world", type=Path, default=WORLD)
    args = parser.parse_args(); app = QApplication(sys.argv); editor = Editor(args.state.resolve(), args.world.resolve())
    editor.show(); raise SystemExit(app.exec())


if __name__ == "__main__": main()

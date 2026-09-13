from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.vegetation import (
    BEGIN, END, KINDS, generate_model, install_world_include, plants_from_masks,
    remove_legacy_vegetation,
)


class VegetationTests(unittest.TestCase):
    def test_painted_area_is_covered_deterministically(self) -> None:
        masks = {kind: np.zeros((512, 512), dtype=np.uint8) for kind in KINDS}
        masks["bush"][240:260, 240:260] = 1
        elevation = np.tile(np.arange(512, dtype=float)[:, None], (1, 512))
        first = plants_from_masks(masks, elevation, seed=12)
        second = plants_from_masks(masks, elevation, seed=12)
        self.assertGreater(len(first), 100)
        self.assertEqual(first, second)
        self.assertTrue(all(plant.kind == "bush" for plant in first))

    def test_generated_model_batches_visual_and_collision_meshes(self) -> None:
        masks = {kind: np.zeros((512, 512), dtype=np.uint8) for kind in KINDS}
        masks["grass"][255:258, 255:258] = 1
        masks["tree"][250:262, 250:262] = 1
        plants = plants_from_masks(masks, np.zeros((513, 513)), seed=3)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            counts = generate_model(plants, root)
            self.assertGreater(counts["grass"], 0)
            self.assertGreater(counts["tree"], 0)
            sdf = (root / "model.sdf").read_text()
            self.assertIn("vegetation_collision", sdf)
            self.assertIn("<label>50</label>", sdf)
            self.assertIn("<label>28</label>", sdf)
            for name in ("grass.obj", "tree.obj", "collision.obj", "vegetation.mtl"):
                self.assertTrue((root / "meshes" / name).is_file())

    def test_world_install_is_idempotent_and_can_remove_legacy(self) -> None:
        original = '<world>\n<include><name>bush_01</name><uri>model://bush</uri><pose>0 0 0 0 0 0</pose></include>\n  </world>'
        with tempfile.TemporaryDirectory() as directory:
            world = Path(directory) / "world.sdf"; world.write_text(original)
            self.assertEqual(remove_legacy_vegetation(world), 1)
            install_world_include(world); install_world_include(world)
            result = world.read_text()
            self.assertEqual(result.count(BEGIN), 1)
            self.assertEqual(result.count(END), 1)


if __name__ == "__main__":
    unittest.main()

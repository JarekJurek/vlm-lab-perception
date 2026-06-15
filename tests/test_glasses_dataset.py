from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Dict


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT / "src"))


def _require_runtime_deps() -> None:
    missing = []
    for module in ("numpy", "PIL", "yaml"):
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    if missing:
        raise unittest.SkipTest(f"Missing runtime dependencies: {', '.join(missing)}")


class GlassesDatasetTest(unittest.TestCase):
    def test_empty_threshold_applies_to_numeric_fill_percentage(self) -> None:
        _require_runtime_deps()

        from PIL import Image

        from datasets.glassesinthewild import GlassesInTheWildDataset

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            images_dir = root / "images"
            images_dir.mkdir()
            Image.new("RGB", (100, 100), color=(255, 255, 255)).save(images_dir / "below.png")
            Image.new("RGB", (100, 100), color=(255, 255, 255)).save(images_dir / "above.png")

            payload = {
                "categories": [
                    {
                        "id": 1,
                        "name": "glass",
                        "keypoints": ["bottom_front", "top_left", "top_right", "top_front", "fluid_level"],
                    }
                ],
                "images": [
                    {"id": 1, "width": 100, "height": 100, "file_name": "below.png"},
                    {"id": 2, "width": 100, "height": 100, "file_name": "above.png"},
                ],
                "annotations": [
                    {"image_id": 1, "category_id": 1, "num_keypoints": 5, "keypoints": _keypoints(fluid_y=80.0)},
                    {"image_id": 2, "category_id": 1, "num_keypoints": 5, "keypoints": _keypoints(fluid_y=78.0)},
                ],
            }
            (images_dir / "annotations.json").write_text(json.dumps(payload), encoding="utf-8")

            dataset = GlassesInTheWildDataset(root)

            below = _annotations_by_task(dataset[0])
            above = _annotations_by_task(dataset[1])

            self.assertEqual(below["fill_percentage"].answer, 0.0)
            self.assertEqual(below["fill_percentage"].metadata["raw_derived_fill_percentage"], 20.0)
            self.assertEqual(below["fill_percentage"].metadata["derived_fill_percentage"], 0.0)
            self.assertEqual(below["fill_level_ordinal"].answer, "0_percent")

            self.assertEqual(above["fill_percentage"].answer, 22.0)
            self.assertEqual(above["fill_percentage"].metadata["raw_derived_fill_percentage"], 22.0)
            self.assertEqual(above["fill_percentage"].metadata["derived_fill_percentage"], 22.0)
            self.assertEqual(above["fill_level_ordinal"].answer, "33_percent")


def _keypoints(*, fluid_y: float) -> list[float]:
    return [
        50.0,
        100.0,
        2.0,
        0.0,
        0.0,
        2.0,
        100.0,
        0.0,
        2.0,
        50.0,
        0.0,
        2.0,
        50.0,
        fluid_y,
        2.0,
    ]


def _annotations_by_task(sample: Dict[str, object]) -> Dict[str, object]:
    return {ann.metadata["task"]: ann for ann in sample["annotations"]}  # type: ignore[index, union-attr]


if __name__ == "__main__":
    unittest.main()

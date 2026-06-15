from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from datasets.labliquidvision import LabLiquidVisionDataset
from tlg import PromptBank


class LabLiquidVisionContainerCountTest(unittest.TestCase):
    def test_single_sample_emits_container_count_one(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            sample_dir = Path(tmpdir) / "processed" / "sample_001"
            sample_dir.mkdir(parents=True)
            (sample_dir / "Input_vessel.txt").write_text("Tube_50mL\n", encoding="utf-8")
            (sample_dir / "Input_vol_liquid.txt").write_text("10\n", encoding="utf-8")
            (sample_dir / "Input_vol_vessel.txt").write_text("55\n", encoding="utf-8")
            (sample_dir / "Input_color_liquid.txt").write_text("red\n", encoding="utf-8")

            dataset = LabLiquidVisionDataset(
                tmpdir,
                rgb_only=False,
                prompt_bank=PromptBank.default(),
            )
            sample = dataset[0]

        annotations = sample["annotations"]
        tasks = [ann.metadata["task"] for ann in annotations]
        count_ann = annotations[tasks.index("container_count")]
        capacity_ml_ann = annotations[tasks.index("container_capacity_ml")]
        capacity_bin_ann = annotations[tasks.index("container_capacity_bin")]

        self.assertEqual(count_ann.answer, 1)
        self.assertEqual(count_ann.metadata["source"], "single_vessel_sample")
        self.assertEqual(capacity_ml_ann.answer, 55.0)
        self.assertEqual(capacity_ml_ann.metadata["vol_vessel"], 55.0)
        self.assertEqual(capacity_ml_ann.metadata["source"], "Input_vol_vessel.txt")
        self.assertEqual(capacity_bin_ann.answer, "50_100_ml")
        self.assertEqual(capacity_bin_ann.metadata["vol_vessel"], 55.0)
        self.assertEqual(capacity_bin_ann.metadata["source"], "Input_vol_vessel.txt")


if __name__ == "__main__":
    unittest.main()

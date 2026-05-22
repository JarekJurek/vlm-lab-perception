from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from datasets.ot2 import OT2Dataset


def _task_names(sample: dict[str, object]) -> list[str]:
    return [ann.metadata["task"] for ann in sample["annotations"]]


class OT2NoTipsTest(unittest.TestCase):
    def test_no_tips_images_emit_only_tip_presence(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            labeled_dir = root / "Training Data" / "Images_with_Labels_tips_n_Liquid"
            raw_dir = root / "Training Data" / "Images"
            labeled_dir.mkdir(parents=True)
            raw_dir.mkdir(parents=True)

            (labeled_dir / "Tips_0.jpg").touch()
            (labeled_dir / "Tips_0.txt").write_text("0 0.5 0.5 0.2 0.4\n", encoding="utf-8")
            (raw_dir / "NoTips_0.jpg").touch()

            dataset = OT2Dataset(root)
            sample = dataset[1]

            self.assertEqual(sample["id"], "NoTips_0")
            self.assertEqual(_task_names(sample), ["tip_presence"])
            self.assertEqual(sample["annotations"][0].answer, "NO")

    def test_flat_id_resolution_can_find_no_tips_images(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            labeled_dir = root / "Training Data" / "Images_with_Labels_tips_n_Liquid"
            raw_dir = root / "Training Data" / "Images"
            labeled_dir.mkdir(parents=True)
            raw_dir.mkdir(parents=True)

            (labeled_dir / "Tips_0.jpg").touch()
            (labeled_dir / "Tips_0.txt").write_text("0 0.5 0.5 0.2 0.4\n", encoding="utf-8")
            (raw_dir / "NoTips_12.jpg").touch()

            dataset = OT2Dataset(root, ids=["NoTips_12"])
            sample = dataset[0]

            self.assertEqual(sample["id"], "NoTips_12")
            self.assertEqual(_task_names(sample), ["tip_presence"])
            self.assertEqual(sample["annotations"][0].metadata["source"], "no_tips_filename")


if __name__ == "__main__":
    unittest.main()

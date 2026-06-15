from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from datasets.coqe import COQEDataset  # noqa: E402
from tlg import PromptBank  # noqa: E402
from tlg.evaluation import primary_metric_for_task, score_prediction  # noqa: E402


class COQEDatasetTest(unittest.TestCase):
    def test_coqe_tlg_dataset_emits_native_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = _make_coqe_root(Path(tmpdir))
            dataset = COQEDataset(root, prompt_bank=PromptBank.default())

            self.assertEqual(len(dataset), 8)
            self.assertEqual(
                dataset.task_counts(),
                {
                    "container_capacity_ml": 2,
                    "container_capacity_bin": 2,
                    "fill_level_ordinal": 2,
                    "liquid_transfer_feasibility": 2,
                },
            )

            samples = {}
            for sample in dataset:
                samples.setdefault(sample["primary_task"], sample)
            capacity = samples["container_capacity_ml"]["annotations"][0]
            self.assertEqual(capacity.answer, 235.0)
            self.assertEqual(capacity.metadata["bbox_xyxy"], [10, 20, 50, 80])

            transfer = samples["liquid_transfer_feasibility"]["annotations"][0]
            self.assertEqual(transfer.answer, "yes")
            self.assertEqual(transfer.choices, ["yes", "no", "cant_tell"])
            self.assertEqual(len(transfer.metadata["bbox_boxes"]), 2)

    def test_coqe_tasks_use_expected_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = _make_coqe_root(Path(tmpdir))
            dataset = COQEDataset(root, prompt_bank=PromptBank.default())
            annotations = {}
            for sample in dataset:
                annotations.setdefault(sample["primary_task"], sample["annotations"][0])

            numeric = score_prediction(annotations["container_capacity_ml"], "240")
            self.assertTrue(numeric["valid"])
            self.assertAlmostEqual(numeric["abs_error"], 5.0)
            self.assertEqual(primary_metric_for_task("container_capacity_ml"), "mae")

            ordinal = score_prediction(annotations["fill_level_ordinal"], "33_percent")
            self.assertTrue(ordinal["valid"])
            self.assertTrue(ordinal["correct"])
            self.assertEqual(ordinal["ordinal_abs_error"], 0)
            self.assertEqual(primary_metric_for_task("fill_level_ordinal"), "ordinal_mae")


def _make_coqe_root(base: Path) -> Path:
    root = base / "COQE_dataset"
    images = root / "images"
    images.mkdir(parents=True)
    rows = [
        {
            "image_id": "00001.png",
            "x1": 10,
            "y1": 20,
            "x2": 50,
            "y2": 80,
            "cad_model": 1,
            "category": "bottle",
            "comparative_vol": 1,
            "content": 2,
            "volume_continuous": 235,
            "volume_discrete": 4,
        },
        {
            "image_id": "00002.png",
            "x1": 60,
            "y1": 25,
            "x2": 100,
            "y2": 90,
            "cad_model": 2,
            "category": "glass",
            "comparative_vol": 2,
            "content": 5,
            "volume_continuous": 500,
            "volume_discrete": 6,
        },
    ]
    for split in ("train", "val", "test"):
        (root / f"{split}.json").write_text(json.dumps(rows), encoding="utf-8")
        (root / f"{split}_pour_prediction.json").write_text("[]", encoding="utf-8")
    return root


if __name__ == "__main__":
    unittest.main()

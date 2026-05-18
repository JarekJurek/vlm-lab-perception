from __future__ import annotations

import sys
import unittest
from pathlib import Path


sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from tlg.annotations import TLGAnnotation, simplify_container_type
from tlg.evaluation import labels_equal, parse_task_prediction


class ContainerTypeSimpleTest(unittest.TestCase):
    def test_simplify_container_type_removes_volume_suffix(self) -> None:
        self.assertEqual(simplify_container_type("Tube_50mL"), "Tube")
        self.assertEqual(simplify_container_type("StorageBottle_250mL"), "StorageBottle")
        self.assertEqual(simplify_container_type("Cell_Flask_160mL"), "Flask")
        self.assertEqual(simplify_container_type("Gibco_500mL"), "Gibco")
        self.assertEqual(simplify_container_type("T25 Flask"), "Flask")

    def test_container_type_simple_prediction_accepts_full_label_alias(self) -> None:
        ann = TLGAnnotation(
            image_id="sample",
            dataset="labliquidvision",
            task_type="Container Type Recognition (classification, no volume)",
            question="Identify the container type.",
            choices=["Tube", "Pyrex", "StorageBottle", "Duran", "Flask", "Gibco", "Tip"],
            answer="Tube",
            metadata={"task": "container_type_simple"},
        )

        pred, valid = parse_task_prediction(ann, "Tube_50mL")

        self.assertTrue(valid)
        self.assertEqual(pred, "Tube")
        self.assertTrue(labels_equal("container_type_simple", "50 mL tube", "Tube"))
        self.assertTrue(labels_equal("container_type_simple", "T75 Flask", "Flask"))


if __name__ == "__main__":
    unittest.main()

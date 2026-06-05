from __future__ import annotations

import sys
import unittest
from pathlib import Path


sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from tlg.annotations import TLGAnnotation
from tlg.evaluation import TaskStats, score_prediction


def _ann(task: str, answer: object, choices: list[str] | None = None) -> TLGAnnotation:
    return TLGAnnotation(
        image_id="sample",
        dataset="unit",
        task_type=task,
        question="question?",
        choices=choices,
        answer=answer,
        metadata={"task": task},
    )


class TaskMetricsTest(unittest.TestCase):
    def test_binary_classification_reports_balanced_accuracy_and_macro_f1(self) -> None:
        stats = TaskStats()
        ann_yes = _ann("tip_presence", "YES", ["YES", "NO"])
        ann_no = _ann("tip_presence", "NO", ["YES", "NO"])
        for ann, pred in (
            (ann_yes, "YES"),
            (ann_yes, "NO"),
            (ann_no, "NO"),
            (ann_no, "NO"),
        ):
            stats.update_from_record(score_prediction(ann, pred))

        out = stats.to_dict()
        self.assertAlmostEqual(out["accuracy"], 0.75)
        self.assertAlmostEqual(out["balanced_accuracy"], 0.75)
        self.assertAlmostEqual(out["macro_f1"], (2.0 / 3.0 + 0.8) / 2.0)
        self.assertEqual(out["primary_metric"], "balanced_accuracy")

    def test_fill_percentage_reports_error_distribution(self) -> None:
        stats = TaskStats()
        for gt, pred in ((50.0, "45"), (80.0, "100")):
            stats.update_from_record(score_prediction(_ann("fill_percentage", gt), pred))

        out = stats.to_dict()
        self.assertAlmostEqual(out["mae"], 12.5)
        self.assertAlmostEqual(out["rmse"], (212.5) ** 0.5)
        self.assertAlmostEqual(out["signed_bias"], 7.5)
        self.assertAlmostEqual(out["within_5"], 0.5)
        self.assertAlmostEqual(out["within_10"], 0.5)
        self.assertEqual(out["primary_metric"], "mae")

    def test_container_count_reports_exact_and_off_by_one(self) -> None:
        stats = TaskStats()
        for gt, pred in ((3, "3"), (3, "4"), (3, "6")):
            stats.update_from_record(score_prediction(_ann("container_count", gt), pred))

        out = stats.to_dict()
        self.assertAlmostEqual(out["accuracy"], 1.0 / 3.0)
        self.assertAlmostEqual(out["mae"], 4.0 / 3.0)
        self.assertAlmostEqual(out["within_1"], 2.0 / 3.0)

    def test_ordinal_reports_distance_metrics(self) -> None:
        stats = TaskStats()
        choices = ["EMPTY", "LOW", "MID", "HIGH"]
        for gt, pred in (("MID", "HIGH"), ("HIGH", "LOW")):
            stats.update_from_record(score_prediction(_ann("fill_level_ordinal", gt, choices), pred))

        out = stats.to_dict()
        self.assertAlmostEqual(out["accuracy"], 0.0)
        self.assertAlmostEqual(out["ordinal_mae"], 1.5)
        self.assertAlmostEqual(out["ordinal_within_1"], 0.5)
        self.assertIn("quadratic_weighted_kappa", out)
        self.assertEqual(out["primary_metric"], "ordinal_mae")

    def test_grounding_reports_global_detection_f1_and_box_count_mae(self) -> None:
        stats = TaskStats()
        gt = [
            {"bbox_2d": [0, 0, 100, 100], "label": "Liquid"},
            {"bbox_2d": [200, 200, 300, 300], "label": "Liquid"},
        ]
        pred = '[{"bbox_2d":[0,0,100,100],"label":"Liquid"}]'

        stats.update_from_record(score_prediction(_ann("liquid_grounding", gt), pred))

        out = stats.to_dict()
        self.assertAlmostEqual(out["precision"], 1.0)
        self.assertAlmostEqual(out["recall"], 0.5)
        self.assertAlmostEqual(out["f1"], 2.0 / 3.0)
        self.assertAlmostEqual(out["micro_f1"], 2.0 / 3.0)
        self.assertAlmostEqual(out["box_count_mae"], 1.0)
        self.assertEqual(out["primary_metric"], "micro_f1")


if __name__ == "__main__":
    unittest.main()

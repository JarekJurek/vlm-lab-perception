from __future__ import annotations

import sys
import unittest
from pathlib import Path


sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from tlg.annotations import TLGAnnotation
from tlg.evaluation import TaskStats, score_prediction


def _material_annotation(answer: str) -> TLGAnnotation:
    return TLGAnnotation(
        image_id="sample",
        dataset="labpicsv1",
        task_type="Material Type Presence (multi-label)",
        question="Which material types are present?",
        answer=answer,
        metadata={"task": "material_types_present"},
    )


class MaterialTypeMetricsTest(unittest.TestCase):
    def test_material_type_scoring_adds_sample_set_metrics(self) -> None:
        rec = score_prediction(_material_annotation("LIQUID, SOLID"), "LIQUID, FOAM")

        self.assertTrue(rec["valid"])
        self.assertFalse(rec["correct"])
        self.assertFalse(rec["exact_match"])
        self.assertEqual(rec["n_gt_labels"], 2)
        self.assertEqual(rec["n_pred_labels"], 2)
        self.assertEqual(rec["n_matched_labels"], 1)
        self.assertAlmostEqual(rec["sample_precision"], 0.5)
        self.assertAlmostEqual(rec["sample_recall"], 0.5)
        self.assertAlmostEqual(rec["sample_f1"], 0.5)

    def test_task_stats_aggregates_material_type_metrics(self) -> None:
        stats = TaskStats()
        for gt, pred in (
            ("LIQUID, SOLID", "LIQUID, FOAM"),
            ("LIQUID", "LIQUID"),
        ):
            stats.update_from_record(score_prediction(_material_annotation(gt), pred))

        out = stats.to_dict()
        self.assertAlmostEqual(out["accuracy"], 0.5)
        self.assertAlmostEqual(out["exact_match"], 0.5)
        self.assertAlmostEqual(out["sample_precision"], 0.75)
        self.assertAlmostEqual(out["sample_recall"], 0.75)
        self.assertAlmostEqual(out["sample_f1"], 0.75)


if __name__ == "__main__":
    unittest.main()

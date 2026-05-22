from __future__ import annotations

import sys
import unittest
from pathlib import Path


sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from tlg.annotations import TLGAnnotation
from tlg.evaluation import TaskStats, score_prediction


def _ordering_annotation(answer: str) -> TLGAnnotation:
    return TLGAnnotation(
        image_id="sample",
        dataset="pouring",
        task_type="Water Pouring Frame Ordering",
        question="Order frames A, B, C, D.",
        answer=answer,
        metadata={"task": "pouring_frame_ordering"},
    )


class PouringOrderingMetricsTest(unittest.TestCase):
    def test_adjacent_swap_gets_partial_pairwise_credit(self) -> None:
        rec = score_prediction(_ordering_annotation("C, D, A, B"), "C, A, D, B")

        self.assertTrue(rec["valid"])
        self.assertFalse(rec["correct"])
        self.assertEqual(rec["position_correct"], 2)
        self.assertEqual(rec["position_total"], 4)
        self.assertAlmostEqual(rec["position_accuracy"], 0.5)
        self.assertEqual(rec["pairwise_correct"], 5)
        self.assertEqual(rec["pairwise_total"], 6)
        self.assertAlmostEqual(rec["pairwise_accuracy"], 5.0 / 6.0)
        self.assertAlmostEqual(rec["kendall_tau"], 2.0 / 3.0)

    def test_task_stats_aggregates_pairwise_ordering_accuracy(self) -> None:
        stats = TaskStats()
        for gt, pred in (
            ("C, D, A, B", "C, A, D, B"),
            ("B, C, D, A", "B, C, D, A"),
        ):
            stats.update_from_record(score_prediction(_ordering_annotation(gt), pred))

        out = stats.to_dict()
        self.assertEqual(out["accuracy"], 0.5)
        self.assertEqual(out["position_correct"], 6)
        self.assertEqual(out["position_total"], 8)
        self.assertAlmostEqual(out["position_accuracy"], 0.75)
        self.assertEqual(out["pairwise_correct"], 11)
        self.assertEqual(out["pairwise_total"], 12)
        self.assertAlmostEqual(out["pairwise_accuracy"], 11.0 / 12.0)
        self.assertAlmostEqual(out["kendall_tau"], 5.0 / 6.0)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import sys
import unittest
from pathlib import Path


sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from tlg.annotations import discretize_fill_level_ordinal


class FillLevelOrdinalTest(unittest.TestCase):
    def test_default_thresholds_use_percent_bins(self) -> None:
        cases = [
            (0.0, "0_percent"),
            (1.0, "0_percent"),
            (1.1, "33_percent"),
            (41.4, "33_percent"),
            (41.5, "50_percent"),
            (57.9, "50_percent"),
            (58.0, "66_percent"),
            (82.9, "66_percent"),
            (83.0, "100_percent"),
            (100.0, "100_percent"),
        ]

        for fill_percentage, expected in cases:
            with self.subTest(fill_percentage=fill_percentage):
                self.assertEqual(discretize_fill_level_ordinal(fill_percentage), expected)

    def test_glasses_empty_threshold_can_match_dataset_semantics(self) -> None:
        self.assertEqual(
            discretize_fill_level_ordinal(24.2, empty_threshold=25.0),
            "0_percent",
        )
        self.assertEqual(
            discretize_fill_level_ordinal(25.1, empty_threshold=25.0),
            "33_percent",
        )


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import sys
import unittest
from pathlib import Path


sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from tlg.annotations import discretize_fill_level_ordinal


class FillLevelOrdinalTest(unittest.TestCase):
    def test_default_thresholds_use_four_classes(self) -> None:
        cases = [
            (0.0, "EMPTY"),
            (1.0, "EMPTY"),
            (1.1, "LOW"),
            (49.9, "LOW"),
            (50.0, "MID"),
            (74.9, "MID"),
            (75.0, "HIGH"),
            (100.0, "HIGH"),
        ]

        for fill_percentage, expected in cases:
            with self.subTest(fill_percentage=fill_percentage):
                self.assertEqual(discretize_fill_level_ordinal(fill_percentage), expected)

    def test_glasses_empty_threshold_can_match_dataset_semantics(self) -> None:
        self.assertEqual(
            discretize_fill_level_ordinal(24.2, empty_threshold=25.0),
            "EMPTY",
        )
        self.assertEqual(
            discretize_fill_level_ordinal(25.1, empty_threshold=25.0),
            "LOW",
        )


if __name__ == "__main__":
    unittest.main()

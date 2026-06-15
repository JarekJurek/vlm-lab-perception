from __future__ import annotations

import sys
import unittest
from pathlib import Path


sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from tlg.annotations import discretize_container_capacity_ml, discretize_fill_level_ordinal


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


class ContainerCapacityBinTest(unittest.TestCase):
    def test_capacity_bin_boundaries(self) -> None:
        cases = [
            (0.0, "0_50_ml"),
            (49.999, "0_50_ml"),
            (50.0, "50_100_ml"),
            (100.0, "100_200_ml"),
            (3000.0, "3000_inf_ml"),
        ]

        for capacity_ml, expected in cases:
            with self.subTest(capacity_ml=capacity_ml):
                self.assertEqual(discretize_container_capacity_ml(capacity_ml), expected)

    def test_capacity_bins_cover_labliquidvision_observed_values(self) -> None:
        cases = [
            (16.0, "0_50_ml"),
            (55.0, "50_100_ml"),
            (755.0, "750_1000_ml"),
        ]

        for capacity_ml, expected in cases:
            with self.subTest(capacity_ml=capacity_ml):
                self.assertEqual(discretize_container_capacity_ml(capacity_ml), expected)


if __name__ == "__main__":
    unittest.main()

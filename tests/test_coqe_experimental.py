from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from datasets.coqe import (  # noqa: E402
    COQE_COMPARATIVE_LABELS,
    COQE_CONTENT_LABELS,
    COQE_POUR_LABELS,
    COQE_VOLUME_LABELS,
    COQEDataset,
)
from experimental.coqe.metrics import aggregate_records, edit_distance, score_prediction  # noqa: E402
from experimental.coqe.parsing import parse_prediction  # noqa: E402
from experimental.coqe.prompts import build_prompt  # noqa: E402
from experimental.coqe.rescore_predictions import rescore_predictions  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
COQE_ROOT = ROOT / "data" / "COQE_dataset"


class COQEExperimentalTest(unittest.TestCase):
    def test_label_mappings_match_readme(self) -> None:
        self.assertEqual(COQE_VOLUME_LABELS[1], "0_50_ml")
        self.assertEqual(COQE_VOLUME_LABELS[2], "50_100_ml")
        self.assertEqual(COQE_VOLUME_LABELS[10], "3000_inf_ml")
        self.assertEqual(COQE_CONTENT_LABELS[1], "0_percent")
        self.assertEqual(COQE_CONTENT_LABELS[6], "opaque")
        self.assertEqual(COQE_COMPARATIVE_LABELS[1], "yes")
        self.assertEqual(COQE_COMPARATIVE_LABELS[3], "cant_tell")
        self.assertEqual(COQE_POUR_LABELS[1], 0.0)
        self.assertEqual(COQE_POUR_LABELS[11], 1.0)
        self.assertEqual(COQE_POUR_LABELS[12], "opaque")

    def test_test_split_counts_and_pouring_distribution(self) -> None:
        dataset = COQEDataset(COQE_ROOT, split="test")
        self.assertEqual(len(dataset.records), 3000)
        self.assertEqual(len(dataset.pour_records), 15000)
        self.assertEqual(dataset.task_counts(), {
            "volume": 3000,
            "content": 3000,
            "comparative": 3000,
            "pouring": 15000,
        })
        self.assertEqual(dataset.pouring_records_per_image_distribution(), {5: 3000})

    def test_comparative_pairing_is_directional(self) -> None:
        dataset = COQEDataset(COQE_ROOT, split="test")
        first = dataset.records[10]
        second = dataset.pairs_by_image_id[first.image_id]
        self.assertEqual(first.image_id, "07606.png")
        self.assertEqual(second.image_id, "07607.png")
        self.assertEqual(first.comparative_label, "yes")
        self.assertEqual(second.comparative_label, "no")

        examples = list(dataset.iter_examples(["comparative"]))
        ex = next(e for e in examples if e.image_id == "07606.png")
        self.assertEqual(ex.source.image_id, "07606.png")
        self.assertEqual(ex.target.image_id, "07607.png")
        self.assertEqual(ex.answer, "yes")

    def test_prompts_include_corrections(self) -> None:
        dataset = COQEDataset(COQE_ROOT, split="test")
        volume = next(dataset.iter_examples(["volume"]))
        volume_prompt = build_prompt(volume)
        self.assertIn("0 mL <= volume < 50 mL", volume_prompt)
        self.assertIn("3000 mL <= volume", volume_prompt)
        self.assertIn("Do not use metadata, annotations, filenames, or ground-truth labels", volume_prompt)

        pouring = next(dataset.iter_examples(["pouring"]))
        pouring_prompt = build_prompt(pouring)
        self.assertIn("exactly five timesteps", pouring_prompt)
        self.assertIn('["opaque", "opaque", "opaque", "opaque", "opaque"]', pouring_prompt)

    def test_parser_accepts_strict_valid_json(self) -> None:
        self.assertEqual(parse_prediction("volume", '{"volume_bin": "50_100_ml"}'), ("50_100_ml", True, None))
        self.assertEqual(parse_prediction("content", '{"fill_level": "opaque"}'), ("opaque", True, None))
        self.assertEqual(parse_prediction("comparative", '{"can_pour": "cant_tell"}'), ("cant_tell", True, None))
        self.assertEqual(
            parse_prediction("pouring", '{"remaining_sequence": [0.0, 0.1, 0.2, 1.0, "opaque"]}'),
            ([0.0, 0.1, 0.2, 1.0, "opaque"], True, None),
        )

    def test_parser_accepts_qwen_assistant_prefix(self) -> None:
        self.assertEqual(
            parse_prediction("volume", 'assistant\n{"volume_bin": "50_100_ml"}'),
            ("50_100_ml", True, None),
        )

    def test_parser_rejects_invalid_outputs(self) -> None:
        cases = [
            ("volume", '```json\n{"volume_bin": "50_100_ml"}\n```'),
            ("volume", '{"volume_bin": "bad"}'),
            ("volume", '{"volume_bin": "50_100_ml", "extra": 1}'),
            ("content", '{"fill_level": 0}'),
            ("comparative", '{"can_pour": "maybe"}'),
            ("pouring", '{"remaining_sequence": [0.0]}'),
            ("pouring", '{"remaining_sequence": [0.0, 0.1, 0.2, 0.3, 0.35]}'),
        ]
        for task, response in cases:
            with self.subTest(task=task, response=response):
                parsed, valid, error = parse_prediction(task, response)
                self.assertIsNone(parsed)
                self.assertFalse(valid)
                self.assertIsNotNone(error)

    def test_pouring_metrics(self) -> None:
        self.assertEqual(edit_distance([0.0, 0.1, 0.2], [0.0, 0.2]), 1)
        score = score_prediction("pouring", [0.0, 0.1, 0.2, 0.3, 0.4], [0.0, 0.1, 0.9, 0.3, 0.4], True)
        self.assertFalse(score["correct"])
        self.assertEqual(score["edit_distance"], 1)
        self.assertTrue(score["edit_distance_le_1"])

    def test_average_per_class_accuracy(self) -> None:
        records = [
            {"task": "comparative", "ground_truth": "yes", "valid": True, "correct": True},
            {"task": "comparative", "ground_truth": "yes", "valid": True, "correct": False},
            {"task": "comparative", "ground_truth": "no", "valid": True, "correct": True},
            {"task": "comparative", "ground_truth": "cant_tell", "valid": False, "correct": False},
        ]
        metrics = aggregate_records(records)["comparative"]
        self.assertAlmostEqual(metrics["per_class_accuracy"]["yes"], 0.5)
        self.assertAlmostEqual(metrics["per_class_accuracy"]["no"], 1.0)
        self.assertAlmostEqual(metrics["per_class_accuracy"]["cant_tell"], 0.0)
        self.assertAlmostEqual(metrics["average_per_class_accuracy"], 0.5)
        self.assertEqual(metrics["paper_metrics"]["average_per_class_accuracy"], metrics["average_per_class_accuracy"])

    def test_pouring_paper_metrics(self) -> None:
        records = [
            {
                "task": "pouring",
                "ground_truth": [0.0, 0.1, 0.2, 0.3, 0.4],
                "valid": True,
                "correct": True,
                "edit_distance_le_0": True,
                "edit_distance_le_1": True,
                "edit_distance_le_2": True,
                "edit_distance_le_3": True,
                "edit_distance_le_4": True,
            },
            {
                "task": "pouring",
                "ground_truth": [0.0, 0.1, 0.2, 0.3, 0.4],
                "valid": True,
                "correct": False,
                "edit_distance_le_0": False,
                "edit_distance_le_1": True,
                "edit_distance_le_2": True,
                "edit_distance_le_3": True,
                "edit_distance_le_4": True,
            },
        ]
        metrics = aggregate_records(records)["pouring"]
        self.assertAlmostEqual(metrics["paper_metrics"]["edit_distance_accuracy"]["0"], 0.5)
        self.assertAlmostEqual(metrics["paper_metrics"]["edit_distance_accuracy"]["1"], 1.0)

    def test_rescore_predictions_refreshes_parse_fields(self) -> None:
        path = ROOT / "results" / "tmp_coqe_rescore_test.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "sample_id": "x",
            "task": "volume",
            "raw_response": 'assistant\n{"volume_bin": "50_100_ml"}',
            "ground_truth": "50_100_ml",
            "valid": False,
            "correct": False,
        }
        path.write_text(json.dumps(row) + "\n", encoding="utf-8")
        try:
            rows = rescore_predictions(path)
        finally:
            path.unlink(missing_ok=True)

        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0]["valid"])
        self.assertTrue(rows[0]["correct"])
        self.assertEqual(rows[0]["parsed_prediction"], "50_100_ml")


if __name__ == "__main__":
    unittest.main()

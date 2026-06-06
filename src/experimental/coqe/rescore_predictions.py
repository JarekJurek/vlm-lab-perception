#!/usr/bin/env python3
"""Reparse COQE prediction JSONL and write paper-style metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from experimental.coqe.metrics import aggregate_records, score_prediction  # noqa: E402
from experimental.coqe.parsing import parse_prediction  # noqa: E402


def main() -> None:
    args = parse_args()
    rows = rescore_predictions(Path(args.predictions_in))
    metrics = {
        "source_predictions": str(args.predictions_in),
        "n_predictions": len(rows),
        "metrics": aggregate_records(rows),
    }

    if args.rescored_out:
        out_path = Path(args.rescored_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    if args.metrics_out:
        out_path = Path(args.metrics_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    else:
        print(json.dumps(metrics, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rescore COQE prediction JSONL with the current parser and paper metrics.")
    parser.add_argument("--predictions-in", required=True, help="Input COQE predictions JSONL.")
    parser.add_argument("--rescored-out", default="", help="Optional output JSONL with refreshed parsed fields.")
    parser.add_argument("--metrics-out", default="", help="Optional output JSON file with aggregated metrics.")
    return parser.parse_args()


def rescore_predictions(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            task = str(row["task"])
            raw_response = str(row.get("raw_response", ""))
            ground_truth = row.get("ground_truth")
            prediction, valid, parse_error = parse_prediction(task, raw_response)
            score = score_prediction(task, prediction, ground_truth, valid)

            updated = dict(row)
            updated["parsed_prediction"] = prediction
            updated["valid"] = bool(valid)
            updated["parse_error"] = parse_error
            updated["correct"] = bool(score.get("correct"))
            updated["rescore_line_no"] = line_no
            for key, value in score.items():
                if key != "correct":
                    updated[key] = value
            rows.append(updated)
    return rows


if __name__ == "__main__":
    main()

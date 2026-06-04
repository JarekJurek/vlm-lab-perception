#!/usr/bin/env python3
"""Calculate sklearn confusion matrices from prediction files.

The script scans a prediction root for `preds.jsonl` and `preds.json` files.
Rows are expected to look like the records written by `test_qwen_vl_tasks.py`
with at least `task`, `gt`, and `pred` fields. `dataset` and `model` are used
for grouping when present, and inferred from the file path otherwise.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import yaml


INVALID_LABEL = "__INVALID__"
SINGLE_LABEL_TASKS_WITHOUT_FIXED_CHOICES = {"liquid_color"}
EXCLUDED_CHOICE_TASKS = {"material_types_present"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate sklearn confusion matrices from prediction files."
    )
    parser.add_argument(
        "--pred-root",
        type=Path,
        required=True,
        help="Directory to scan recursively for preds.jsonl and preds.json files.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <pred-root>/confusion_matrices.",
    )
    parser.add_argument(
        "--prompt-bank",
        type=Path,
        default=Path("src/tlg/prompt_bank.yaml"),
        help="Prompt bank YAML used to identify classification choices.",
    )
    parser.add_argument(
        "--dataset-prompts",
        type=Path,
        default=Path("src/tlg/dataset_prompts.yaml"),
        help="Dataset-to-task assignment YAML used to filter configured tasks.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Optional dataset names to include.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="Optional task names to include.",
    )
    parser.add_argument(
        "--include-invalid",
        action="store_true",
        help=f"Include invalid/unparsed predictions as {INVALID_LABEL}. Default: skip them.",
    )
    parser.add_argument(
        "--no-aggregate",
        action="store_true",
        help="Only write per dataset/model/task matrices, not all-dataset task matrices.",
    )
    return parser.parse_args()


def load_prediction_rows(pred_root: Path) -> Iterable[Dict[str, Any]]:
    paths = sorted({*pred_root.rglob("preds.jsonl"), *pred_root.rglob("preds.json")})
    for path in paths:
        if path.suffix == ".jsonl":
            yield from load_jsonl_rows(path, pred_root)
        elif path.suffix == ".json":
            yield from load_json_rows(path, pred_root)


def load_jsonl_rows(path: Path, pred_root: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_no}") from exc
            if isinstance(row, dict):
                yield with_path_defaults(row, path, pred_root)


def load_json_rows(path: Path, pred_root: Path) -> Iterable[Dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}") from exc

    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        rows = payload.get("predictions", payload.get("rows", []))
        if isinstance(rows, dict):
            rows = list(rows.values())
    else:
        rows = []

    if not isinstance(rows, list):
        raise ValueError(f"Expected list-like predictions in {path}")

    for row in rows:
        if isinstance(row, dict):
            yield with_path_defaults(row, path, pred_root)


def with_path_defaults(row: Dict[str, Any], path: Path, pred_root: Path) -> Dict[str, Any]:
    out = dict(row)
    rel_parts = path.relative_to(pred_root).parts
    if not out.get("dataset") and len(rel_parts) >= 2:
        out["dataset"] = rel_parts[-2]
    if not out.get("model") and len(rel_parts) >= 3:
        out["model"] = rel_parts[-3]
    if not out.get("model"):
        out["model"] = "unknown"
    out["_pred_file"] = str(path)
    return out


def load_task_choices(path: Path) -> Dict[str, List[str]]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError(f"Expected mapping at top level in {path}")

    choices: Dict[str, List[str]] = {}
    for task, spec in raw.items():
        if not isinstance(spec, Mapping):
            continue
        task_name = str(task)
        raw_choices = spec.get("choices")
        task_type = str(spec.get("task_type", ""))
        if isinstance(raw_choices, list) and task_name not in EXCLUDED_CHOICE_TASKS:
            choices[task_name] = [str(label) for label in raw_choices]
        elif "classification" in task_type.lower() and task_name in SINGLE_LABEL_TASKS_WITHOUT_FIXED_CHOICES:
            choices[task_name] = []
    return choices


def load_dataset_tasks(path: Path) -> Dict[str, List[str]]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError(f"Expected mapping at top level in {path}")

    out: Dict[str, List[str]] = {}
    for dataset, spec in raw.items():
        if isinstance(spec, list):
            out[str(dataset)] = [str(item) for item in spec]
        elif isinstance(spec, Mapping):
            include = spec.get("include", [])
            if isinstance(include, list):
                out[str(dataset)] = [str(item) for item in include]
    return out


def label_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return str(value).strip()


def matrix_labels(task: str, rows: Sequence[Dict[str, Any]], choices: Mapping[str, Sequence[str]]) -> List[str]:
    fixed = [str(label) for label in choices.get(task, [])]
    if fixed:
        extras = sorted(
            {
                label
                for row in rows
                for label in (label_text(row.get("gt")), label_text(row.get("pred")))
                if label and label not in fixed
            }
        )
        return fixed + extras

    labels = {
        label
        for row in rows
        for label in (label_text(row.get("gt")), label_text(row.get("pred")))
        if label
    }
    return sorted(labels)


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(value))


def write_matrix_files(
    *,
    out_dir: Path,
    group_name: str,
    dataset: str,
    model: str,
    task: str,
    rows: Sequence[Dict[str, Any]],
    labels: Sequence[str],
) -> None:
    from sklearn.metrics import confusion_matrix

    y_true = [label_text(row.get("gt")) for row in rows]
    y_pred = [label_text(row.get("pred")) for row in rows]
    matrix = confusion_matrix(y_true, y_pred, labels=list(labels))

    stem = "__".join(safe_name(part) for part in (group_name, model, dataset, task))
    json_path = out_dir / f"{stem}.json"
    csv_path = out_dir / f"{stem}.csv"

    payload = {
        "group": group_name,
        "model": model,
        "dataset": dataset,
        "task": task,
        "labels": list(labels),
        "matrix": matrix.tolist(),
        "n": len(rows),
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["gt\\pred", *labels])
        for label, values in zip(labels, matrix.tolist()):
            writer.writerow([label, *values])


def write_summary(path: Path, summaries: Sequence[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["group", "model", "dataset", "task", "n", "labels", "json", "csv"],
        )
        writer.writeheader()
        for row in summaries:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    pred_root = args.pred_root
    if not pred_root.exists():
        raise FileNotFoundError(f"Prediction root does not exist: {pred_root}")

    out_dir = args.out_dir or pred_root / "confusion_matrices"
    out_dir.mkdir(parents=True, exist_ok=True)

    choices = load_task_choices(args.prompt_bank)
    dataset_tasks = load_dataset_tasks(args.dataset_prompts)
    selected_datasets = set(str(dataset) for dataset in args.datasets) if args.datasets else None
    selected_tasks = set(str(task) for task in args.tasks) if args.tasks else None

    configured_single_label_tasks = {
        (dataset, task)
        for dataset, tasks in dataset_tasks.items()
        for task in tasks
        if task in choices
    }

    grouped: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    aggregate: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    skipped_invalid = 0
    skipped_non_classification = 0

    for row in load_prediction_rows(pred_root):
        dataset = str(row.get("dataset", ""))
        model = str(row.get("model", "unknown"))
        task = str(row.get("task", ""))
        if selected_datasets is not None and dataset not in selected_datasets:
            continue
        if selected_tasks is not None and task not in selected_tasks:
            continue
        if (dataset, task) not in configured_single_label_tasks:
            skipped_non_classification += 1
            continue
        if not row.get("valid", True):
            if not args.include_invalid:
                skipped_invalid += 1
                continue
            row = {**row, "pred": INVALID_LABEL}

        gt = label_text(row.get("gt"))
        pred = label_text(row.get("pred"))
        if not gt or not pred:
            skipped_invalid += 1
            continue

        grouped[(model, dataset, task)].append(row)
        aggregate[(model, task)].append(row)

    summaries: List[Dict[str, Any]] = []

    for (model, dataset, task), rows in sorted(grouped.items()):
        labels = matrix_labels(task, rows, choices)
        if args.include_invalid and INVALID_LABEL not in labels:
            labels = [*labels, INVALID_LABEL]
        write_matrix_files(
            out_dir=out_dir,
            group_name="dataset",
            dataset=dataset,
            model=model,
            task=task,
            rows=rows,
            labels=labels,
        )
        stem = "__".join(safe_name(part) for part in ("dataset", model, dataset, task))
        summaries.append(
            {
                "group": "dataset",
                "model": model,
                "dataset": dataset,
                "task": task,
                "n": len(rows),
                "labels": "|".join(labels),
                "json": f"{stem}.json",
                "csv": f"{stem}.csv",
            }
        )

    if not args.no_aggregate:
        for (model, task), rows in sorted(aggregate.items()):
            labels = matrix_labels(task, rows, choices)
            if args.include_invalid and INVALID_LABEL not in labels:
                labels = [*labels, INVALID_LABEL]
            write_matrix_files(
                out_dir=out_dir,
                group_name="task",
                dataset="all",
                model=model,
                task=task,
                rows=rows,
                labels=labels,
            )
            stem = "__".join(safe_name(part) for part in ("task", model, "all", task))
            summaries.append(
                {
                    "group": "task",
                    "model": model,
                    "dataset": "all",
                    "task": task,
                    "n": len(rows),
                    "labels": "|".join(labels),
                    "json": f"{stem}.json",
                    "csv": f"{stem}.csv",
                }
            )

    write_summary(out_dir / "summary.csv", summaries)
    print(
        f"Wrote {len(summaries)} confusion matrix summaries to {out_dir}. "
        f"Skipped {skipped_invalid} invalid rows and {skipped_non_classification} non-classification rows."
    )


if __name__ == "__main__":
    try:
        main()
    except ModuleNotFoundError as exc:
        if exc.name == "sklearn":
            print("scikit-learn is required. Install it with: pip install scikit-learn", file=sys.stderr)
            raise SystemExit(2) from exc
        raise

#!/usr/bin/env python3
"""Count TLG datapoints enabled by the current dataset prompt assignments.

A datapoint is one standardized TLGAnnotation produced by a dataset loader for
one enabled task. This matches the evaluation unit in run_vlm_tasks.py: a single
sample can produce multiple datapoints when it has multiple task annotations,
and a sample can produce zero datapoints when loader-side filtering removes all
enabled annotations.

The script loads src/tlg/dataset_prompts.yaml through PromptBank, builds the
same dataset loaders used by evaluation, iterates every sample, validates the
standardized annotations, and aggregates counts from annotation metadata. Dataset
totals are therefore sums of task annotations, not just raw image or sample
counts.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from datasets.coqe import COQEDataset
from tlg import TLGAnnotation
from tlg.cli import parse_data_root_overrides
from tlg.dataset_factory import DATA_ROOTS, build_dataset
from tlg.prompts import PromptBank
from tlg.reporting import validate_annotations


DATASET_ORDER = (
    "labliquidvision",
    "glasses",
    "dtld",
    "labpics_chemistry",
    "ot2",
    "pouring",
    "coqe",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Count standardized TLG datapoints from dataset_prompts.yaml. "
            "One datapoint is one TLGAnnotation, matching the unit evaluated by run_vlm_tasks.py."
        )
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Dataset names to count. Default: all datasets in dataset_prompts.yaml.",
    )
    parser.add_argument(
        "--data-root",
        action="append",
        default=[],
        help=(
            "Override a dataset root, e.g. --data-root labpics_chemistry='data/LabPics_Chemistry'. "
            "Can be provided multiple times."
        ),
    )
    parser.add_argument(
        "--coqe-split",
        choices=["train", "val", "test"],
        default="test",
        help="COQE split to count. Default: test.",
    )
    parser.add_argument(
        "--labpics-chemistry-split",
        choices=["train", "test", "all"],
        default="all",
        help="LabPics Chemistry split to count. Default: all.",
    )
    parser.add_argument(
        "--dtld-no-occlusion-filter",
        action="store_true",
        help="Disable DTLD occlusion filtering, matching run_vlm_tasks.py.",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip datasets whose resolved root does not exist instead of failing.",
    )
    parser.add_argument(
        "--json-out",
        type=str,
        default="",
        help="Optional path to write the full count summary as JSON.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1000,
        help="Print progress every N samples while scanning a dataset. Default: 1000.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Do not print per-dataset progress to stderr.",
    )
    return parser.parse_args()


def active_tasks(prompt_bank: PromptBank, dataset_name: str) -> List[str]:
    assignment = prompt_bank.dataset_assignments.get(dataset_name, {})
    include = list(assignment.get("include", []))
    overrides = list(assignment.get("overrides", {}).keys())
    return sorted(set(include) | set(overrides))


def active_dataset_names(prompt_bank: PromptBank, requested: Optional[Sequence[str]]) -> List[str]:
    configured = set(prompt_bank.dataset_assignments)
    if requested:
        unknown = sorted(set(requested) - configured)
        if unknown:
            raise ValueError(f"Dataset(s) not present in dataset_prompts.yaml: {', '.join(unknown)}")
        return [str(name) for name in requested]

    ordered = [name for name in DATASET_ORDER if name in configured]
    ordered.extend(sorted(configured - set(ordered)))
    return ordered


def build_count_dataset(
    dataset_name: str,
    data_root: Path,
    *,
    prompt_bank: PromptBank,
    coqe_split: str,
    labpics_chemistry_split: str,
    dtld_no_occlusion_filter: bool,
) -> Any:
    # COQE is task-expanded internally, so pass the YAML-enabled tasks explicitly.
    if dataset_name == "coqe":
        return COQEDataset(
            data_root,
            split=coqe_split,
            prompt_bank=prompt_bank,
            tasks=active_tasks(prompt_bank, dataset_name),
        )

    return build_dataset(
        dataset_name,
        data_root,
        prompt_bank=prompt_bank,
        include_depth=False,
        shuffle=False,
        seed=None,
        debug_dtld_boxes=False,
        dtld_no_occlusion_filter=dtld_no_occlusion_filter,
        coqe_split=coqe_split,
        labpics_chemistry_split=labpics_chemistry_split,
    )


def annotation_task(ann: TLGAnnotation) -> str:
    return str(ann.metadata.get("task", ann.task_type))


def count_dataset(
    dataset: Any,
    *,
    dataset_name: str,
    expected_tasks: Iterable[str],
    quiet: bool,
    progress_every: int,
) -> Dict[str, Any]:
    expected = list(expected_tasks)
    task_counts: Counter[str] = Counter({task: 0 for task in expected})
    sample_count = len(dataset)
    samples_with_datapoints = 0
    empty_samples = 0
    validation_errors: List[Dict[str, object]] = []

    progress_every = max(1, int(progress_every))
    for idx in range(sample_count):
        if not quiet and idx > 0 and idx % progress_every == 0:
            print(f"  {dataset_name}: scanned {idx}/{sample_count} samples", file=sys.stderr)

        sample = dataset[idx]
        err = validate_annotations(sample)
        if err is not None:
            validation_errors.append(
                {
                    "index": idx,
                    "sample_id": str(sample.get("id", "unknown")) if isinstance(sample, Mapping) else "unknown",
                    "error": err,
                }
            )
            continue

        annotations = sample.get("annotations") if isinstance(sample, Mapping) else None
        if not isinstance(annotations, list) or not annotations:
            empty_samples += 1
            continue

        samples_with_datapoints += 1
        for ann in annotations:
            if isinstance(ann, TLGAnnotation):
                task_counts[annotation_task(ann)] += 1

    return {
        "samples": sample_count,
        "samples_with_datapoints": samples_with_datapoints,
        "empty_samples": empty_samples,
        "datapoints": sum(task_counts.values()),
        "tasks": dict(sorted(task_counts.items())),
        "validation_errors": validation_errors,
    }


def print_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> None:
    widths = [len(str(h)) for h in headers]
    for row in rows:
        widths = [max(width, len(str(value))) for width, value in zip(widths, row)]

    def fmt(row: Sequence[object]) -> str:
        return "  ".join(str(value).ljust(width) for value, width in zip(row, widths))

    print(fmt(headers))
    print(fmt(["-" * width for width in widths]))
    for row in rows:
        print(fmt(row))


def write_json_summary(path: str, results: Mapping[str, Any]) -> None:
    if not path:
        return
    json_path = Path(path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    prompt_bank = PromptBank.default()
    dataset_names = active_dataset_names(prompt_bank, args.datasets)
    data_root_overrides = parse_data_root_overrides(args.data_root)

    results: Dict[str, Any] = {
        "prompt_assignments": str(PromptBank.default_assignments_path()),
        "coqe_split": str(args.coqe_split),
        "labpics_chemistry_split": str(args.labpics_chemistry_split),
        "dtld_no_occlusion_filter": bool(args.dtld_no_occlusion_filter),
        "datasets": {},
        "task_totals": {},
    }

    missing_roots: List[str] = []
    for dataset_name in dataset_names:
        root = data_root_overrides.get(dataset_name, DATA_ROOTS.get(dataset_name))
        if root is None:
            raise ValueError(f"No default data root configured for dataset: {dataset_name}")
        root = Path(root)

        if not args.quiet:
            print(f"Counting {dataset_name} from {root}", file=sys.stderr)

        if not root.exists():
            message = f"{dataset_name}: {root}"
            missing_roots.append(message)
            if args.skip_missing:
                results["datasets"][dataset_name] = {
                    "root": str(root),
                    "status": "missing_root",
                    "configured_tasks": active_tasks(prompt_bank, dataset_name),
                }
                continue
            continue

        dataset = build_count_dataset(
            dataset_name,
            root,
            prompt_bank=prompt_bank,
            coqe_split=str(args.coqe_split),
            labpics_chemistry_split=str(args.labpics_chemistry_split),
            dtld_no_occlusion_filter=bool(args.dtld_no_occlusion_filter),
        )
        counts = count_dataset(
            dataset,
            dataset_name=dataset_name,
            expected_tasks=active_tasks(prompt_bank, dataset_name),
            quiet=bool(args.quiet),
            progress_every=int(args.progress_every),
        )
        counts["root"] = str(root)
        counts["status"] = "ok"
        counts["configured_tasks"] = active_tasks(prompt_bank, dataset_name)
        results["datasets"][dataset_name] = counts
        write_json_summary(args.json_out, results)

    if missing_roots and not args.skip_missing:
        print("Missing dataset root(s):", file=sys.stderr)
        for item in missing_roots:
            print(f"  {item}", file=sys.stderr)
        print("Use --data-root dataset=/path or --skip-missing.", file=sys.stderr)
        return 2

    task_totals: Counter[str] = Counter()
    for info in results["datasets"].values():
        if not isinstance(info, Mapping) or info.get("status") != "ok":
            continue
        for task, count in dict(info.get("tasks", {})).items():
            task_totals[str(task)] += int(count)
    results["task_totals"] = dict(sorted(task_totals.items()))

    dataset_rows = []
    dataset_task_rows = []
    for dataset_name in dataset_names:
        info = results["datasets"].get(dataset_name)
        if not isinstance(info, Mapping):
            continue
        status = str(info.get("status", "unknown"))
        if status != "ok":
            dataset_rows.append([dataset_name, status, 0, 0, 0, 0])
            continue
        dataset_rows.append(
            [
                dataset_name,
                status,
                info.get("samples", 0),
                info.get("samples_with_datapoints", 0),
                info.get("empty_samples", 0),
                info.get("datapoints", 0),
            ]
        )
        for task, count in dict(info.get("tasks", {})).items():
            dataset_task_rows.append([dataset_name, task, count])

    print("\nDatapoints per dataset")
    print_table(
        ["dataset", "status", "samples", "samples_with_datapoints", "empty_samples", "datapoints"],
        dataset_rows,
    )

    print("\nDatapoints per dataset per task")
    print_table(["dataset", "task", "datapoints"], dataset_task_rows)

    print("\nDatapoints per task")
    print_table(["task", "datapoints"], [[task, count] for task, count in results["task_totals"].items()])

    validation_error_count = sum(
        len(info.get("validation_errors", []))
        for info in results["datasets"].values()
        if isinstance(info, Mapping)
    )
    if validation_error_count:
        print(f"\nValidation errors: {validation_error_count}", file=sys.stderr)

    if args.json_out:
        write_json_summary(args.json_out, results)
        print(f"\nWrote JSON summary: {Path(args.json_out)}")

    return 1 if validation_error_count else 0


if __name__ == "__main__":
    raise SystemExit(main())

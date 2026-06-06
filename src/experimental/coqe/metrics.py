from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from datasets.coqe import COQE_COMPARATIVE_LABELS, COQE_CONTENT_LABELS, COQE_VOLUME_LABELS


def score_prediction(task: str, prediction: Any, ground_truth: Any, valid: bool) -> Dict[str, Any]:
    correct = bool(valid and prediction == ground_truth)
    row: Dict[str, Any] = {"correct": correct}
    if task == "pouring":
        if valid and isinstance(prediction, list) and isinstance(ground_truth, list):
            distance = edit_distance(prediction, ground_truth)
            row["edit_distance"] = distance
            for threshold in range(5):
                row[f"edit_distance_le_{threshold}"] = distance <= threshold
        else:
            row["edit_distance"] = None
            for threshold in range(5):
                row[f"edit_distance_le_{threshold}"] = False
    return row


def aggregate_records(records: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    by_task: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        by_task[str(record["task"])].append(record)

    out: Dict[str, Any] = {}
    for task, rows in sorted(by_task.items()):
        out[task] = aggregate_task(task, rows)
    return out


def aggregate_task(task: str, rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    total = len(rows)
    valid = sum(1 for row in rows if bool(row.get("valid")))
    correct = sum(1 for row in rows if bool(row.get("correct")))
    out: Dict[str, Any] = {
        "n": total,
        "valid": valid,
        "invalid": total - valid,
        "accuracy": _safe_div(correct, total),
        "valid_accuracy": _safe_div(correct, valid),
    }

    if task in {"volume", "content", "comparative"}:
        labels = _labels_for_task(task)
        avg_per_class = average_per_class_accuracy(rows, labels)
        out["average_per_class_accuracy"] = avg_per_class
        out["per_class_accuracy"] = per_class_accuracy(rows, labels)
        out["primary_metric"] = "average_per_class_accuracy"
        out["paper_metrics"] = {
            "average_per_class_accuracy": avg_per_class,
        }
    elif task == "pouring":
        exact = sum(1 for row in rows if bool(row.get("correct")))
        out["exact_sequence_match"] = _safe_div(exact, total)
        edit_distance_acc: Dict[str, Optional[float]] = {}
        for threshold in range(5):
            count = sum(1 for row in rows if bool(row.get(f"edit_distance_le_{threshold}")))
            value = _safe_div(count, total)
            out[f"edit_distance_le_{threshold}"] = value
            edit_distance_acc[str(threshold)] = value
        out["primary_metric"] = "exact_sequence_match"
        out["paper_metrics"] = {
            "edit_distance_accuracy": edit_distance_acc,
        }
    else:
        out["primary_metric"] = "accuracy"
    return out


def per_class_accuracy(rows: Sequence[Mapping[str, Any]], labels: Sequence[str]) -> Dict[str, Optional[float]]:
    counts: Counter[str] = Counter()
    correct: Counter[str] = Counter()
    for row in rows:
        gt = str(row.get("ground_truth"))
        counts[gt] += 1
        if bool(row.get("correct")):
            correct[gt] += 1
    return {
        label: (_safe_div(correct[label], counts[label]) if counts[label] else None)
        for label in labels
    }


def average_per_class_accuracy(rows: Sequence[Mapping[str, Any]], labels: Sequence[str]) -> Optional[float]:
    values = [v for v in per_class_accuracy(rows, labels).values() if v is not None]
    if not values:
        return None
    return sum(values) / len(values)


def edit_distance(a: Sequence[Any], b: Sequence[Any]) -> int:
    prev = list(range(len(b) + 1))
    for i, av in enumerate(a, start=1):
        cur = [i] + [0] * len(b)
        for j, bv in enumerate(b, start=1):
            cost = 0 if av == bv else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[-1]


def _labels_for_task(task: str) -> List[str]:
    if task == "volume":
        return [COQE_VOLUME_LABELS[i] for i in sorted(COQE_VOLUME_LABELS)]
    if task == "content":
        return [COQE_CONTENT_LABELS[i] for i in sorted(COQE_CONTENT_LABELS)]
    if task == "comparative":
        return [COQE_COMPARATIVE_LABELS[i] for i in sorted(COQE_COMPARATIVE_LABELS)]
    raise ValueError(f"No class labels for task: {task!r}")


def _safe_div(num: int, den: int) -> Optional[float]:
    if den == 0:
        return None
    return float(num) / float(den)

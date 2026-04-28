"""Reporting helpers for Qwen-VL evaluation."""

from __future__ import annotations

from typing import Dict, Optional

from tlg.annotations import TLGAnnotation
from tlg.evaluation import TaskStats


def print_ground_truth(sample: Dict[str, object]) -> None:
    """Print standardized annotations for a dataset sample."""
    print("\nGround Truth (standardized annotations):")

    annotations = sample.get("annotations")
    if not isinstance(annotations, list) or not annotations:
        print("  (no standardized annotations)")
        return

    for ann in annotations:
        if not isinstance(ann, TLGAnnotation):
            continue
        task_key = str(ann.metadata.get("task", ann.task_type))
        print(f"  - {task_key}: {ann.answer}")


def validate_annotations(sample: Dict[str, object]) -> Optional[str]:
    """Validate standardized annotations for a dataset sample.

    Args:
        sample: Dataset sample with an annotations field.

    Returns:
        Error message if validation fails, otherwise None.
    """
    annotations = sample.get("annotations")
    if not isinstance(annotations, list):
        return "sample['annotations'] is not a list"
    for i, ann in enumerate(annotations):
        if not isinstance(ann, TLGAnnotation):
            return f"annotations[{i}] is not a TLGAnnotation"
        if not isinstance(ann.question, str) or not ann.question.strip():
            return f"annotations[{i}] has empty question"
        if not isinstance(ann.task_type, str) or not ann.task_type.strip():
            return f"annotations[{i}] has empty task_type"
        task = ann.metadata.get("task")
        if not isinstance(task, str) or not task.strip():
            return f"annotations[{i}] missing metadata['task']"
    return None


def print_stats(title: str, stats: Dict[str, TaskStats]) -> None:
    """Print aggregated task stats.

    Args:
        title: Section title.
        stats: TaskStats mapping keyed by task name.
    """
    if not stats:
        return
    print("\n" + "=" * 60)
    print(title)
    for task in sorted(stats.keys()):
        d = stats[task].to_dict()
        n_valid = d.get("n_valid", 0)
        n = d.get("n", 0)
        acc = d.get("accuracy")
        mae = d.get("mae")
        within = d.get("within_tol")
        parts = [f"n={n}", f"valid={n_valid}"]
        if isinstance(acc, float):
            parts.append(f"acc={acc:.3f}")
        if isinstance(mae, float):
            parts.append(f"mae={mae:.3f}")
        if isinstance(within, float):
            parts.append(f"within_tol={within:.3f}")
        print(f"  - {task}: " + ", ".join(parts))

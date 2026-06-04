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
        primary_metric = d.get("primary_metric")
        primary_score = d.get("primary_score")
        acc = d.get("accuracy")
        balanced_acc = d.get("balanced_accuracy")
        macro_f1 = d.get("macro_f1")
        micro_f1 = d.get("micro_f1")
        positive_f1 = d.get("positive_f1")
        exact_match = d.get("exact_match")
        sample_precision = d.get("sample_precision")
        sample_recall = d.get("sample_recall")
        sample_f1 = d.get("sample_f1")
        mae = d.get("mae")
        rmse = d.get("rmse")
        signed_bias = d.get("signed_bias")
        ordinal_mae = d.get("ordinal_mae")
        ordinal_within_1 = d.get("ordinal_within_1")
        qwk = d.get("quadratic_weighted_kappa")
        within = d.get("within_tol")
        within_1 = d.get("within_1")
        within_5 = d.get("within_5")
        within_10 = d.get("within_10")
        precision = d.get("precision")
        recall = d.get("recall")
        f1 = d.get("f1")
        mean_iou = d.get("mean_iou")
        pairwise_acc = d.get("pairwise_accuracy")
        parts = [f"n={n}", f"valid={n_valid}"]
        if isinstance(primary_metric, str) and isinstance(primary_score, float):
            parts.append(f"primary={primary_metric}:{primary_score:.3f}")
        if isinstance(acc, float):
            parts.append(f"acc={acc:.3f}")
        if isinstance(balanced_acc, float):
            parts.append(f"balanced_acc={balanced_acc:.3f}")
        if isinstance(macro_f1, float):
            parts.append(f"macro_f1={macro_f1:.3f}")
        if isinstance(micro_f1, float):
            parts.append(f"micro_f1={micro_f1:.3f}")
        if isinstance(positive_f1, float):
            parts.append(f"positive_f1={positive_f1:.3f}")
        if isinstance(exact_match, float):
            parts.append(f"exact_match={exact_match:.3f}")
        if isinstance(sample_precision, float):
            parts.append(f"sample_precision={sample_precision:.3f}")
        if isinstance(sample_recall, float):
            parts.append(f"sample_recall={sample_recall:.3f}")
        if isinstance(sample_f1, float):
            parts.append(f"sample_f1={sample_f1:.3f}")
        if isinstance(mae, float):
            parts.append(f"mae={mae:.3f}")
        if isinstance(rmse, float):
            parts.append(f"rmse={rmse:.3f}")
        if isinstance(signed_bias, float):
            parts.append(f"bias={signed_bias:.3f}")
        if isinstance(ordinal_mae, float):
            parts.append(f"ordinal_mae={ordinal_mae:.3f}")
        if isinstance(ordinal_within_1, float):
            parts.append(f"ordinal_within_1={ordinal_within_1:.3f}")
        if isinstance(qwk, float):
            parts.append(f"qwk={qwk:.3f}")
        if isinstance(within, float):
            parts.append(f"within_tol={within:.3f}")
        if isinstance(within_1, float):
            parts.append(f"within_1={within_1:.3f}")
        if isinstance(within_5, float):
            parts.append(f"within_5={within_5:.3f}")
        if isinstance(within_10, float):
            parts.append(f"within_10={within_10:.3f}")
        if isinstance(precision, float):
            parts.append(f"precision={precision:.3f}")
        if isinstance(recall, float):
            parts.append(f"recall={recall:.3f}")
        if isinstance(f1, float):
            parts.append(f"f1={f1:.3f}")
        if isinstance(mean_iou, float):
            parts.append(f"mean_iou={mean_iou:.3f}")
        if isinstance(pairwise_acc, float):
            parts.append(f"pairwise_acc={pairwise_acc:.3f}")
        print(f"  - {task}: " + ", ".join(parts))

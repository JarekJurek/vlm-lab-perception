from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple

from .annotations import TLGAnnotation


_NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def norm_text(s: object) -> str:
    if s is None:
        return ""
    return " ".join(str(s).strip().split())


def norm_label(s: object) -> str:
    return norm_text(s).lower()


def extract_first_number(text: str) -> Optional[float]:
    text = norm_text(text)
    if not text:
        return None
    m = _NUM_RE.search(text)
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def parse_yes_no(text: str) -> Optional[str]:
    t = norm_label(text)
    if re.search(r"\byes\b", t):
        return "YES"
    if re.search(r"\bno\b", t):
        return "NO"
    return None


def parse_choice(text: str, choices: Iterable[str]) -> Optional[str]:
    t = norm_label(text)
    if not t:
        return None

    for c in choices:
        if norm_label(c) == t:
            return str(c)

    hits = []
    for c in choices:
        cn = norm_label(c)
        if cn and re.search(rf"\b{re.escape(cn)}\b", t):
            hits.append(str(c))
    if len(hits) == 1:
        return hits[0]
    return None


def parse_ordinal_fill_level(text: str) -> Optional[str]:
    t = norm_label(text)
    for lvl in ["EMPTY", "LOW", "MID", "HIGH", "FULL"]:
        if re.search(rf"\b{lvl.lower()}\b", t):
            return lvl
    return None


def parse_task_prediction(ann: TLGAnnotation, response: str) -> Tuple[Optional[object], bool]:
    """Parse model response into a canonical prediction for the given task.

    Returns (prediction, is_valid).
    """

    task = str(ann.metadata.get("task", ann.task_type))

    if ann.choices:
        pred = parse_choice(response, ann.choices)
        return pred, pred is not None

    if task in {"liquid_presence", "tip_presence"}:
        pred = parse_yes_no(response)
        return pred, pred is not None

    if task == "fill_level_ordinal":
        pred = parse_ordinal_fill_level(response)
        return pred, pred is not None

    if task in {"fill_percentage", "fill_height_mm"}:
        pred = extract_first_number(response)
        return pred, pred is not None

    if task in {"tip_count", "liquid_count"}:
        n = extract_first_number(response)
        if n is None:
            return None, False
        n_int = int(round(n))
        if abs(n - n_int) > 1e-6:
            return None, False
        if n_int < 0:
            return None, False
        return n_int, True

    if task in {"container_type", "liquid_color"}:
        pred = norm_text(response)
        return pred, bool(pred)

    pred = norm_text(response)
    return pred, bool(pred)


def labels_equal(pred: object, gt: object) -> bool:
    return norm_label(pred) == norm_label(gt)


def default_tolerance(task: str) -> Optional[float]:
    if task == "fill_percentage":
        return 5.0
    if task == "fill_height_mm":
        return 5.0
    return None


def score_prediction(
    ann: TLGAnnotation,
    response: str,
    *,
    tol_overrides: Optional[Dict[str, float]] = None,
) -> Dict[str, object]:
    task = str(ann.metadata.get("task", ann.task_type))
    gt = ann.answer
    pred, valid = parse_task_prediction(ann, response)

    record: Dict[str, object] = {
        "task": task,
        "task_type": ann.task_type,
        "question": ann.question,
        "gt": gt,
        "pred_raw": response,
        "pred": pred,
        "valid": bool(valid),
    }

    if not valid:
        record["correct"] = None
        return record

    if task in {"fill_percentage", "fill_height_mm"}:
        gt_f = float(gt) if isinstance(gt, (int, float)) else None
        pred_f = float(pred) if isinstance(pred, (int, float)) else None
        if gt_f is None or pred_f is None:
            record["correct"] = None
            return record
        abs_err = abs(pred_f - gt_f)
        record["abs_error"] = abs_err

        if tol_overrides and task in tol_overrides:
            tol = float(tol_overrides[task])
        else:
            tol = default_tolerance(task)

        if tol is not None:
            record["within_tol"] = bool(abs_err <= float(tol))
            record["correct"] = record["within_tol"]
        else:
            record["correct"] = None

        return record

    if task in {"tip_count", "liquid_count"}:
        record["correct"] = bool(int(pred) == int(gt)) if isinstance(gt, (int, float)) else None
        return record

    record["correct"] = labels_equal(pred, gt)
    return record


@dataclass
class TaskStats:
    n: int = 0
    n_valid: int = 0
    n_correct: int = 0
    abs_error_sum: float = 0.0
    n_abs_error: int = 0
    n_within_tol: int = 0

    def update_from_record(self, rec: Dict[str, object]) -> None:
        self.n += 1
        if rec.get("valid"):
            self.n_valid += 1
            if rec.get("correct") is True:
                self.n_correct += 1
            abs_error = rec.get("abs_error")
            if isinstance(abs_error, (int, float)):
                self.abs_error_sum += float(abs_error)
                self.n_abs_error += 1
            if rec.get("within_tol") is True:
                self.n_within_tol += 1

    def to_dict(self) -> Dict[str, object]:
        out: Dict[str, object] = {
            "n": int(self.n),
            "n_valid": int(self.n_valid),
        }
        if self.n_valid > 0:
            out["accuracy"] = float(self.n_correct) / float(self.n_valid)
        if self.n_abs_error > 0:
            out["mae"] = float(self.abs_error_sum) / float(self.n_abs_error)
        if self.n_abs_error > 0 or self.n_within_tol > 0:
            out["within_tol"] = float(self.n_within_tol) / float(self.n_valid) if self.n_valid else None
        return out

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

from .annotations import TLGAnnotation


_NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

_COLOR_CANONICALS = (
    "red", "blue", "green", "yellow", "orange", "brown",
    "black", "white", "pink", "purple", "clear", "transparent",
)

_CONTAINER_TYPE_ALIASES = {
    "tube50ml": "Tube_50mL",
    "50mltube": "Tube_50mL",
    "tube15ml": "Tube_15mL",
    "15mltube": "Tube_15mL",
    "pyrex100ml": "Pyrex_100mL",
    "pyrex100mlbeaker": "Pyrex_100mL",
    "storagebottle250ml": "StorageBottle_250mL",
    "250mlstoragebottle": "StorageBottle_250mL",
    "storagebottle500ml": "StorageBottle_500mL",
    "500mlstoragebottle": "StorageBottle_500mL",
    "duran500ml": "Duran_500mL",
    "duran250ml": "Duran_250mL",
    "duran100ml": "Duran_100mL",
    "cellflask160ml": "Cell_Flask_160mL",
    "cellflask400ml": "Cell_Flask_400mL",
    "gibco125ml": "Gibco_125mL",
    "gibco500ml": "Gibco_500mL",
    "t25flask": "T25 Flask",
    "t75flask": "T75 Flask",
    "t175flask": "T175 Flask",
    "grex": "G-rex",
    "grexflask": "G-rex",
    "tip": "Tip",
    "pipettetip": "Tip",
}

_YES_NO_TASKS = {"liquid_presence", "tip_presence"}
_COUNT_TASKS = {"container_count"}
_DEFAULT_TOLERANCES = {"fill_percentage": 5.0, "fill_height_mm": 5.0}
_NUMERIC_RANGES = {"fill_percentage": (0.0, 100.0), "fill_height_mm": (0.0, None)}
_MATERIAL_TYPE_CHOICES = ("LIQUID", "FOAM", "GEL", "SOLID", "POWDER", "GRANULAR", "VAPOR")
_COLOR_VARIANTS = {
    "reddish": "red", "bluish": "blue", "greenish": "green",
    "yellowish": "yellow", "orangish": "orange", "orangeish": "orange",
    "brownish": "brown", "colourless": "clear", "colorless": "clear",
}


def norm_text(s: object) -> str:
    if s is None:
        return ""
    return " ".join(str(s).strip().split())


def norm_label(s: object) -> str:
    return norm_text(s).lower()


def _compact_alnum(s: object) -> str:
    return "".join(ch for ch in norm_label(s) if ch.isalnum())


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

    choice_list = list(choices)
    for c in choice_list:
        if norm_label(c) == t:
            return str(c)

    hits = []
    for c in choice_list:
        cn = norm_label(c)
        if cn and re.search(rf"\b{re.escape(cn)}\b", t):
            hits.append(str(c))
    if len(hits) == 1:
        return hits[0]

    t_compact = _compact_alnum(t)
    for c in choice_list:
        cn = norm_label(c)
        cc = _compact_alnum(c)
        if cc and cc != cn and cc in t_compact:
            hits.append(str(c))
    if len(hits) == 1:
        return hits[0]
    return None


def canonicalize_liquid_color(s: object) -> Optional[str]:
    t = norm_label(s)
    if not t:
        return None

    for color in _COLOR_CANONICALS:
        if re.search(rf"\b{re.escape(color)}\b", t):
            return color

    for variant, canonical in _COLOR_VARIANTS.items():
        if variant in t:
            return canonical

    return None


def canonicalize_container_type(s: object) -> Optional[str]:
    t = norm_text(s)
    if not t:
        return None

    compact = _compact_alnum(t)
    if compact in _CONTAINER_TYPE_ALIASES:
        return _CONTAINER_TYPE_ALIASES[compact]

    return t


def _task_name(ann: TLGAnnotation) -> str:
    return str(ann.metadata.get("task", ann.task_type))


def _parse_count(response: str) -> Tuple[Optional[int], bool]:
    n = extract_first_number(response)
    if n is None:
        return None, False

    n_int = int(round(n))
    if abs(n - n_int) > 1e-6 or n_int < 0:
        return None, False
    return n_int, True


def parse_material_types(text: object) -> Optional[str]:
    t = norm_label(text)
    if not t:
        return None

    hits = []
    for label in _MATERIAL_TYPE_CHOICES:
        if re.search(rf"\b{re.escape(label.lower())}\b", t):
            hits.append(label)

    has_none = bool(re.search(r"\bnone\b", t)) or bool(re.search(r"\bno materials?\b", t))
    if has_none and hits:
        return None
    if has_none:
        return "NONE"
    if not hits:
        return None
    return ", ".join(label for label in _MATERIAL_TYPE_CHOICES if label in set(hits))


def parse_task_prediction(ann: TLGAnnotation, response: str) -> Tuple[Optional[object], bool]:
    """Parse model response into (canonical_prediction, is_valid)."""
    task = _task_name(ann)

    if task == "material_types_present":
        pred = parse_material_types(response)
        return pred, pred is not None

    if ann.choices:
        pred = parse_choice(response, ann.choices)
        if pred is None and task == "container_type":
            pred = canonicalize_container_type(response)
            if pred not in set(str(c) for c in ann.choices):
                pred = None
        return pred, pred is not None

    if task in _YES_NO_TASKS:
        pred = parse_yes_no(response)
        return pred, pred is not None

    if task in _NUMERIC_RANGES:
        pred = extract_first_number(response)
        if pred is None:
            return None, False
        lo, hi = _NUMERIC_RANGES[task]
        return pred, (lo is None or pred >= lo) and (hi is None or pred <= hi)

    if task in _COUNT_TASKS:
        return _parse_count(response)

    if task == "liquid_color":
        pred = canonicalize_liquid_color(response) or norm_text(response)
        return pred, bool(pred)

    if task == "container_type":
        pred = canonicalize_container_type(response) or norm_text(response)
        return pred, bool(pred)

    pred = norm_text(response)
    return pred, bool(pred)


def labels_equal(task: str, pred: object, gt: object) -> bool:
    if task == "liquid_color":
        pred_c = canonicalize_liquid_color(pred)
        gt_c = canonicalize_liquid_color(gt)
        if pred_c is not None and gt_c is not None:
            return pred_c == gt_c
        return norm_label(pred) == norm_label(gt)

    if task == "container_type":
        pred_c = canonicalize_container_type(pred)
        gt_c = canonicalize_container_type(gt)
        if pred_c is not None and gt_c is not None:
            return _compact_alnum(pred_c) == _compact_alnum(gt_c)
        return norm_label(pred) == norm_label(gt)

    if task == "material_types_present":
        pred_c = parse_material_types(pred)
        gt_c = parse_material_types(gt)
        return pred_c is not None and gt_c is not None and pred_c == gt_c

    return norm_label(pred) == norm_label(gt)


def default_tolerance(task: str) -> Optional[float]:
    return _DEFAULT_TOLERANCES.get(task)


def _as_float(value: object) -> Optional[float]:
    return float(value) if isinstance(value, (int, float)) else None


def _numeric_tolerance(task: str, tol_overrides: Optional[Dict[str, float]]) -> Optional[float]:
    if tol_overrides and task in tol_overrides:
        return float(tol_overrides[task])
    return default_tolerance(task)


def score_prediction(
    ann: TLGAnnotation,
    response: str,
    *,
    tol_overrides: Optional[Dict[str, float]] = None,
) -> Dict[str, object]:
    task = _task_name(ann)
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

    if task in _NUMERIC_RANGES:
        gt_f = _as_float(gt)
        pred_f = _as_float(pred)
        if gt_f is None or pred_f is None:
            record["correct"] = None
            return record

        abs_err = abs(pred_f - gt_f)
        record["abs_error"] = abs_err

        tol = _numeric_tolerance(task, tol_overrides)
        if tol is not None:
            record["within_tol"] = bool(abs_err <= tol)
            record["correct"] = record["within_tol"]
        else:
            record["correct"] = None

        return record

    if task in _COUNT_TASKS:
        record["correct"] = bool(int(pred) == int(gt)) if isinstance(gt, (int, float)) else None
        return record

    record["correct"] = labels_equal(task, pred, gt)
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
        if not rec.get("valid"):
            return

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
            "parse_rate": (float(self.n_valid) / float(self.n)) if self.n > 0 else None,
            "accuracy_all": (float(self.n_correct) / float(self.n)) if self.n > 0 else None,
        }
        if self.n_valid > 0:
            out["accuracy"] = float(self.n_correct) / float(self.n_valid)
        if self.n_abs_error > 0:
            out["mae"] = float(self.abs_error_sum) / float(self.n_abs_error)
        if self.n_abs_error > 0 or self.n_within_tol > 0:
            out["within_tol"] = float(self.n_within_tol) / float(self.n_valid) if self.n_valid else None
        return out

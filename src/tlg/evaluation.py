from __future__ import annotations

import ast
import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .annotations import TLGAnnotation
from .annotations import simplify_container_type


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

_YES_NO_TASKS = {"tip_presence"}
_COUNT_TASKS = {"container_count"}
_GROUNDING_TASKS = {"liquid_grounding"}
_DEFAULT_TOLERANCES = {"fill_percentage": 5.0, "fill_height_mm": 5.0}
_NUMERIC_RANGES = {"fill_percentage": (0.0, 100.0), "fill_height_mm": (0.0, None)}
_ORDINAL_TASKS = {"fill_level_ordinal"}
_ORDINAL_VALUE = {"EMPTY": 0, "LOW": 1, "MID": 2, "HIGH": 3, "FULL": 4}
_CLASSIFICATION_TASKS = {
    "tip_presence",
    "container_type",
    "container_type_simple",
    "highlighted_region_type",
    "liquid_color",
    "multiphase_material_presence",
}
_POSITIVE_CLASS_TASKS = {"multiphase_material_presence": "YES"}
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


def canonicalize_container_type_simple(s: object) -> Optional[str]:
    t = norm_text(s)
    if not t:
        return None

    canonical_full = canonicalize_container_type(t)
    simple = simplify_container_type(canonical_full)
    if simple != canonical_full:
        return simple

    compact = _compact_alnum(t)
    aliases = {
        "tube": "Tube",
        "pyrex": "Pyrex",
        "storagebottle": "StorageBottle",
        "storage": "StorageBottle",
        "bottle": "StorageBottle",
        "duran": "Duran",
        "cellflask": "Flask",
        "cellcultureflask": "Flask",
        "flask": "Flask",
        "gibco": "Gibco",
        "t25flask": "Flask",
        "t75flask": "Flask",
        "t175flask": "Flask",
        "grex": "G-rex",
        "grexflask": "G-rex",
        "tip": "Tip",
        "pipettetip": "Tip",
    }
    return aliases.get(compact, t)


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


def _material_type_set(text: object) -> Optional[set[str]]:
    parsed = parse_material_types(text)
    if parsed is None:
        return None
    if parsed == "NONE":
        return set()
    return {label.strip() for label in parsed.split(",") if label.strip()}


def parse_frame_order_labels(text: object) -> Optional[str]:
    labels = re.findall(r"\b[A-D]\b", norm_text(text).upper())
    if len(labels) not in (3, 4):
        return None
    if len(set(labels)) != len(labels):
        return None
    return ", ".join(labels)


def _frame_order_label_list(text: object) -> Optional[List[str]]:
    parsed = parse_frame_order_labels(text)
    if parsed is None:
        return None
    return [label.strip() for label in parsed.split(",")]


def _score_frame_ordering(pred: object, gt: object) -> Dict[str, object]:
    pred_labels = _frame_order_label_list(pred)
    gt_labels = _frame_order_label_list(gt)
    if pred_labels is None or gt_labels is None:
        return {}

    position_total = len(gt_labels)
    position_correct = sum(
        1
        for pred_label, gt_label in zip(pred_labels, gt_labels)
        if pred_label == gt_label
    )

    pred_pos = {label: i for i, label in enumerate(pred_labels)}
    total = 0
    pairwise_correct = 0
    for i, earlier in enumerate(gt_labels):
        for later in gt_labels[i + 1:]:
            total += 1
            if (
                earlier in pred_pos
                and later in pred_pos
                and pred_pos[earlier] < pred_pos[later]
            ):
                pairwise_correct += 1

    pairwise_accuracy = (float(pairwise_correct) / float(total)) if total > 0 else None
    kendall_tau = None
    if pairwise_accuracy is not None:
        kendall_tau = (2.0 * pairwise_accuracy) - 1.0

    return {
        "position_correct": position_correct,
        "position_total": position_total,
        "position_accuracy": (
            float(position_correct) / float(position_total)
        ) if position_total > 0 else None,
        "pairwise_correct": pairwise_correct,
        "pairwise_total": total,
        "pairwise_accuracy": pairwise_accuracy,
        "kendall_tau": kendall_tau,
    }


def _score_material_types(pred: object, gt: object) -> Dict[str, object]:
    pred_set = _material_type_set(pred)
    gt_set = _material_type_set(gt)
    if pred_set is None or gt_set is None:
        return {"correct": None}

    n_overlap = len(pred_set & gt_set)
    exact_match = pred_set == gt_set
    sample_precision = (
        float(n_overlap) / float(len(pred_set))
        if pred_set
        else (1.0 if not gt_set else 0.0)
    )
    sample_recall = (
        float(n_overlap) / float(len(gt_set))
        if gt_set
        else (1.0 if not pred_set else 0.0)
    )
    sample_f1 = (
        2.0 * sample_precision * sample_recall / (sample_precision + sample_recall)
        if (sample_precision + sample_recall)
        else 0.0
    )

    return {
        "correct": exact_match,
        "exact_match": exact_match,
        "n_gt_labels": len(gt_set),
        "n_pred_labels": len(pred_set),
        "n_matched_labels": n_overlap,
        "sample_precision": sample_precision,
        "sample_recall": sample_recall,
        "sample_f1": sample_f1,
    }


def parse_task_prediction(ann: TLGAnnotation, response: str) -> Tuple[Optional[object], bool]:
    """Parse model response into (canonical_prediction, is_valid)."""
    task = _task_name(ann)

    if task in _GROUNDING_TASKS:
        pred = parse_grounding_boxes(response)
        return pred, pred is not None

    if task == "material_types_present":
        pred = parse_material_types(response)
        return pred, pred is not None

    if task == "pouring_frame_ordering":
        pred = parse_frame_order_labels(response)
        return pred, pred is not None

    if ann.choices:
        pred = parse_choice(response, ann.choices)
        if pred is None and task == "container_type":
            pred = canonicalize_container_type(response)
            if pred not in set(str(c) for c in ann.choices):
                pred = None
        if pred is None and task == "container_type_simple":
            pred = canonicalize_container_type_simple(response)
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

    if task == "container_type_simple":
        pred = canonicalize_container_type_simple(response) or norm_text(response)
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

    if task == "container_type_simple":
        pred_c = canonicalize_container_type_simple(pred)
        gt_c = canonicalize_container_type_simple(gt)
        if pred_c is not None and gt_c is not None:
            return _compact_alnum(pred_c) == _compact_alnum(gt_c)
        return norm_label(pred) == norm_label(gt)

    if task == "material_types_present":
        pred_c = parse_material_types(pred)
        gt_c = parse_material_types(gt)
        return pred_c is not None and gt_c is not None and pred_c == gt_c

    if task == "pouring_frame_ordering":
        pred_c = parse_frame_order_labels(pred)
        gt_c = parse_frame_order_labels(gt)
        return pred_c is not None and gt_c is not None and pred_c == gt_c

    return norm_label(pred) == norm_label(gt)


def default_tolerance(task: str) -> Optional[float]:
    return _DEFAULT_TOLERANCES.get(task)


def _as_float(value: object) -> Optional[float]:
    return float(value) if isinstance(value, (int, float)) else None


def _as_int(value: object) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _numeric_tolerance(task: str, tol_overrides: Optional[Dict[str, float]]) -> Optional[float]:
    if tol_overrides and task in tol_overrides:
        return float(tol_overrides[task])
    return default_tolerance(task)


def _score_numeric_error(pred_f: float, gt_f: float) -> Dict[str, object]:
    signed_error = float(pred_f - gt_f)
    abs_error = abs(signed_error)
    return {
        "signed_error": signed_error,
        "abs_error": abs_error,
        "squared_error": signed_error * signed_error,
    }


def _score_ordinal(pred: object, gt: object) -> Dict[str, object]:
    pred_value = _ORDINAL_VALUE.get(norm_text(pred).upper())
    gt_value = _ORDINAL_VALUE.get(norm_text(gt).upper())
    if pred_value is None or gt_value is None:
        return {}

    abs_error = abs(pred_value - gt_value)
    return {
        "ordinal_error": pred_value - gt_value,
        "ordinal_abs_error": abs_error,
        "ordinal_within_1": bool(abs_error <= 1),
    }


def parse_grounding_boxes(text: str) -> Optional[List[Dict[str, object]]]:
    payload = _load_json_like(text)
    if payload is None:
        return None

    if isinstance(payload, dict):
        for key in ("objects", "detections", "boxes", "regions", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                payload = value
                break

    if not isinstance(payload, list):
        return None

    out: List[Dict[str, object]] = []
    for item in payload:
        if not isinstance(item, dict):
            return None
        label = _canonical_grounding_label(item.get("label"))
        bbox = item.get("bbox_2d", item.get("bbox"))
        if label is None or not isinstance(bbox, list) or len(bbox) != 4:
            return None
        try:
            x1, y1, x2, y2 = [float(v) for v in bbox]
        except (TypeError, ValueError):
            return None
        if x2 <= x1 or y2 <= y1:
            return None
        out.append(
            {
                "bbox_2d": [
                    _clamp_coord(x1),
                    _clamp_coord(y1),
                    _clamp_coord(x2),
                    _clamp_coord(y2),
                ],
                "label": label,
            }
        )
    return out


def _load_json_like(text: str) -> Optional[Any]:
    s = norm_text(text)
    if not s:
        return None

    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", s, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        s = fenced.group(1).strip()

    for candidate in _json_candidates(s):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
        try:
            return ast.literal_eval(candidate)
        except (SyntaxError, ValueError):
            pass
    return None


def _json_candidates(s: str) -> Iterable[str]:
    yield s
    start_list = s.find("[")
    end_list = s.rfind("]")
    if 0 <= start_list < end_list:
        yield s[start_list : end_list + 1]
    start_obj = s.find("{")
    end_obj = s.rfind("}")
    if 0 <= start_obj < end_obj:
        yield s[start_obj : end_obj + 1]


def _canonical_grounding_label(value: object) -> Optional[str]:
    compact = _compact_alnum(value)
    if compact in {"tip", "pipettetip"}:
        return "Tip"
    if compact == "liquid":
        return "Liquid"
    return None


def _clamp_coord(value: float) -> float:
    return max(0.0, min(1000.0, float(value)))


def _score_grounding(pred: object, gt: object, *, iou_threshold: float = 0.5) -> Dict[str, object]:
    pred_boxes = _coerce_grounding_targets(pred)
    gt_boxes = _coerce_grounding_targets(gt)
    if pred_boxes is None or gt_boxes is None:
        return {"correct": None}

    matched_pred: set[int] = set()
    matches = 0
    iou_sum = 0.0

    for gt_item in gt_boxes:
        best_idx: Optional[int] = None
        best_iou = 0.0
        gt_label = str(gt_item["label"])
        gt_box = gt_item["bbox_2d"]
        for idx, pred_item in enumerate(pred_boxes):
            if idx in matched_pred:
                continue
            if str(pred_item["label"]) != gt_label:
                continue
            iou = _bbox_iou(gt_box, pred_item["bbox_2d"])
            if iou > best_iou:
                best_iou = iou
                best_idx = idx
        if best_idx is not None and best_iou >= iou_threshold:
            matched_pred.add(best_idx)
            matches += 1
            iou_sum += best_iou

    n_gt = len(gt_boxes)
    n_pred = len(pred_boxes)
    precision = float(matches) / float(n_pred) if n_pred else (1.0 if n_gt == 0 else 0.0)
    recall = float(matches) / float(n_gt) if n_gt else (1.0 if n_pred == 0 else 0.0)
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "correct": bool(matches == n_gt and matches == n_pred),
        "n_gt_boxes": n_gt,
        "n_pred_boxes": n_pred,
        "n_matched_boxes": matches,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_iou": (iou_sum / float(matches)) if matches else 0.0,
        "iou_threshold": float(iou_threshold),
    }


def _coerce_grounding_targets(value: object) -> Optional[List[Dict[str, Any]]]:
    if not isinstance(value, list):
        return None
    out: List[Dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            return None
        label = _canonical_grounding_label(item.get("label"))
        bbox = item.get("bbox_2d")
        if label is None or not isinstance(bbox, list) or len(bbox) != 4:
            return None
        try:
            x1, y1, x2, y2 = [float(v) for v in bbox]
        except (TypeError, ValueError):
            return None
        if x2 <= x1 or y2 <= y1:
            return None
        out.append(
            {
                "label": label,
                "bbox_2d": [_clamp_coord(x1), _clamp_coord(y1), _clamp_coord(x2), _clamp_coord(y2)],
            }
        )
    return out


def _bbox_iou(a: List[float], b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


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

        record.update(_score_numeric_error(pred_f, gt_f))
        abs_err = float(record["abs_error"])

        tol = _numeric_tolerance(task, tol_overrides)
        if tol is not None:
            record["within_tol"] = bool(abs_err <= tol)
            record["correct"] = record["within_tol"]
        else:
            record["correct"] = None

        for tol_value in (5.0, 10.0):
            record[f"within_{int(tol_value)}"] = bool(abs_err <= tol_value)
        return record

    if task in _COUNT_TASKS:
        gt_int = _as_int(gt)
        pred_int = _as_int(pred)
        if gt_int is None or pred_int is None:
            record["correct"] = None
            return record

        record.update(_score_numeric_error(float(pred_int), float(gt_int)))
        record["within_1"] = bool(abs(int(record["signed_error"])) <= 1)
        record["correct"] = bool(pred_int == gt_int)
        return record

    if task in _GROUNDING_TASKS:
        record.update(_score_grounding(pred, gt))
        return record

    if task == "material_types_present":
        record.update(_score_material_types(pred, gt))
        return record

    record["correct"] = labels_equal(task, pred, gt)
    if task in _ORDINAL_TASKS:
        record.update(_score_ordinal(pred, gt))
    if task == "pouring_frame_ordering":
        record.update(_score_frame_ordering(pred, gt))
    return record


@dataclass
class TaskStats:
    task_name: Optional[str] = None
    n: int = 0
    n_valid: int = 0
    n_correct: int = 0
    abs_error_sum: float = 0.0
    squared_error_sum: float = 0.0
    signed_error_sum: float = 0.0
    n_abs_error: int = 0
    n_within_tol: int = 0
    n_within_1: int = 0
    n_within_5: int = 0
    n_within_10: int = 0
    abs_errors: List[float] = field(default_factory=list)
    precision_sum: float = 0.0
    recall_sum: float = 0.0
    f1_sum: float = 0.0
    mean_iou_sum: float = 0.0
    n_grounding: int = 0
    grounding_gt_boxes: int = 0
    grounding_pred_boxes: int = 0
    grounding_matched_boxes: int = 0
    grounding_count_abs_error_sum: float = 0.0
    position_correct: int = 0
    position_total: int = 0
    pairwise_correct: int = 0
    pairwise_total: int = 0
    ordinal_abs_error_sum: float = 0.0
    ordinal_total: int = 0
    ordinal_within_1: int = 0
    ordinal_pairs: List[Tuple[int, int]] = field(default_factory=list)
    exact_match_sum: int = 0
    n_exact_match: int = 0
    sample_precision_sum: float = 0.0
    sample_recall_sum: float = 0.0
    sample_f1_sum: float = 0.0
    n_sample_set: int = 0
    multilabel_gt_labels: int = 0
    multilabel_pred_labels: int = 0
    multilabel_matched_labels: int = 0
    class_counts: Dict[str, Dict[str, int]] = field(default_factory=dict)

    def update_from_record(self, rec: Dict[str, object]) -> None:
        self.n += 1
        task = str(rec.get("task", ""))
        if task and self.task_name is None:
            self.task_name = task
        if not rec.get("valid"):
            return

        self.n_valid += 1
        if rec.get("correct") is True:
            self.n_correct += 1

        abs_error = rec.get("abs_error")
        if isinstance(abs_error, (int, float)):
            abs_error_f = float(abs_error)
            self.abs_error_sum += abs_error_f
            self.abs_errors.append(abs_error_f)
            self.n_abs_error += 1
        squared_error = rec.get("squared_error")
        if isinstance(squared_error, (int, float)):
            self.squared_error_sum += float(squared_error)
        signed_error = rec.get("signed_error")
        if isinstance(signed_error, (int, float)):
            self.signed_error_sum += float(signed_error)
        if rec.get("within_tol") is True:
            self.n_within_tol += 1
        if rec.get("within_1") is True:
            self.n_within_1 += 1
        if rec.get("within_5") is True:
            self.n_within_5 += 1
        if rec.get("within_10") is True:
            self.n_within_10 += 1

        if task in _CLASSIFICATION_TASKS or task in _ORDINAL_TASKS:
            self._update_classification_counts(task, rec.get("pred"), rec.get("gt"))

        if rec.get("task") == "material_types_present" and "sample_precision" not in rec:
            rec = {**rec, **_score_material_types(rec.get("pred"), rec.get("gt"))}

        exact_match = rec.get("exact_match")
        if isinstance(exact_match, bool):
            self.exact_match_sum += int(exact_match)
            self.n_exact_match += 1

        sample_precision = rec.get("sample_precision")
        sample_recall = rec.get("sample_recall")
        sample_f1 = rec.get("sample_f1")
        if all(isinstance(v, (int, float)) for v in (sample_precision, sample_recall, sample_f1)):
            self.sample_precision_sum += float(sample_precision)
            self.sample_recall_sum += float(sample_recall)
            self.sample_f1_sum += float(sample_f1)
            self.n_sample_set += 1
        n_gt_labels = rec.get("n_gt_labels")
        n_pred_labels = rec.get("n_pred_labels")
        n_matched_labels = rec.get("n_matched_labels")
        if all(isinstance(v, int) for v in (n_gt_labels, n_pred_labels, n_matched_labels)):
            self.multilabel_gt_labels += int(n_gt_labels)
            self.multilabel_pred_labels += int(n_pred_labels)
            self.multilabel_matched_labels += int(n_matched_labels)

        precision = rec.get("precision")
        recall = rec.get("recall")
        f1 = rec.get("f1")
        mean_iou = rec.get("mean_iou")
        if all(isinstance(v, (int, float)) for v in (precision, recall, f1, mean_iou)):
            self.precision_sum += float(precision)
            self.recall_sum += float(recall)
            self.f1_sum += float(f1)
            self.mean_iou_sum += float(mean_iou)
            self.n_grounding += 1
        n_gt_boxes = rec.get("n_gt_boxes")
        n_pred_boxes = rec.get("n_pred_boxes")
        n_matched_boxes = rec.get("n_matched_boxes")
        if all(isinstance(v, int) for v in (n_gt_boxes, n_pred_boxes, n_matched_boxes)):
            n_gt_boxes_i = int(n_gt_boxes)
            n_pred_boxes_i = int(n_pred_boxes)
            self.grounding_gt_boxes += n_gt_boxes_i
            self.grounding_pred_boxes += n_pred_boxes_i
            self.grounding_matched_boxes += int(n_matched_boxes)
            self.grounding_count_abs_error_sum += abs(n_pred_boxes_i - n_gt_boxes_i)

        pairwise_correct = rec.get("pairwise_correct")
        pairwise_total = rec.get("pairwise_total")
        if isinstance(pairwise_correct, int) and isinstance(pairwise_total, int):
            self.pairwise_correct += pairwise_correct
            self.pairwise_total += pairwise_total

        position_correct = rec.get("position_correct")
        position_total = rec.get("position_total")
        if isinstance(position_correct, int) and isinstance(position_total, int):
            self.position_correct += position_correct
            self.position_total += position_total

        ordinal_abs_error = rec.get("ordinal_abs_error")
        if isinstance(ordinal_abs_error, (int, float)):
            self.ordinal_abs_error_sum += float(ordinal_abs_error)
            self.ordinal_total += 1
        if rec.get("ordinal_within_1") is True:
            self.ordinal_within_1 += 1
        if task in _ORDINAL_TASKS:
            pred_value = _ORDINAL_VALUE.get(norm_text(rec.get("pred")).upper())
            gt_value = _ORDINAL_VALUE.get(norm_text(rec.get("gt")).upper())
            if pred_value is not None and gt_value is not None:
                self.ordinal_pairs.append((pred_value, gt_value))

    def _update_classification_counts(self, task: str, pred: object, gt: object) -> None:
        pred_label = _metric_label(task, pred)
        gt_label = _metric_label(task, gt)
        if not pred_label or not gt_label:
            return
        self._class_count(gt_label)["support"] += 1
        if pred_label == gt_label:
            self._class_count(gt_label)["tp"] += 1
        else:
            self._class_count(gt_label)["fn"] += 1
            self._class_count(pred_label)["fp"] += 1

    def _class_count(self, label: str) -> Dict[str, int]:
        return self.class_counts.setdefault(label, {"tp": 0, "fp": 0, "fn": 0, "support": 0})

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
            out["rmse"] = math.sqrt(float(self.squared_error_sum) / float(self.n_abs_error))
            out["signed_bias"] = float(self.signed_error_sum) / float(self.n_abs_error)
            out["median_abs_error"] = _median(self.abs_errors)
        if self.n_abs_error > 0 or self.n_within_tol > 0:
            out["within_tol"] = float(self.n_within_tol) / float(self.n_valid) if self.n_valid else None
        if self.n_abs_error > 0 or self.n_within_1 > 0:
            out["within_1"] = float(self.n_within_1) / float(self.n_valid) if self.n_valid else None
        if self.n_abs_error > 0 or self.n_within_5 > 0:
            out["within_5"] = float(self.n_within_5) / float(self.n_valid) if self.n_valid else None
        if self.n_abs_error > 0 or self.n_within_10 > 0:
            out["within_10"] = float(self.n_within_10) / float(self.n_valid) if self.n_valid else None
        if self.n_exact_match > 0:
            out["exact_match"] = float(self.exact_match_sum) / float(self.n_exact_match)
        if self.n_sample_set > 0:
            out["sample_precision"] = self.sample_precision_sum / float(self.n_sample_set)
            out["sample_recall"] = self.sample_recall_sum / float(self.n_sample_set)
            out["sample_f1"] = self.sample_f1_sum / float(self.n_sample_set)
        if self.n_sample_set > 0:
            micro_precision = (
                float(self.multilabel_matched_labels) / float(self.multilabel_pred_labels)
                if self.multilabel_pred_labels
                else (1.0 if self.multilabel_gt_labels == 0 else 0.0)
            )
            micro_recall = (
                float(self.multilabel_matched_labels) / float(self.multilabel_gt_labels)
                if self.multilabel_gt_labels
                else (1.0 if self.multilabel_pred_labels == 0 else 0.0)
            )
            out["micro_precision"] = micro_precision
            out["micro_recall"] = micro_recall
            out["micro_f1"] = _f1(micro_precision, micro_recall)
        if self.n_grounding > 0:
            out["precision"] = self.precision_sum / float(self.n_grounding)
            out["recall"] = self.recall_sum / float(self.n_grounding)
            out["f1"] = self.f1_sum / float(self.n_grounding)
            out["mean_iou"] = self.mean_iou_sum / float(self.n_grounding)
        if self.n_grounding > 0:
            micro_precision = (
                float(self.grounding_matched_boxes) / float(self.grounding_pred_boxes)
                if self.grounding_pred_boxes
                else (1.0 if self.grounding_gt_boxes == 0 else 0.0)
            )
            micro_recall = (
                float(self.grounding_matched_boxes) / float(self.grounding_gt_boxes)
                if self.grounding_gt_boxes
                else (1.0 if self.grounding_pred_boxes == 0 else 0.0)
            )
            out["micro_precision"] = micro_precision
            out["micro_recall"] = micro_recall
            out["micro_f1"] = _f1(micro_precision, micro_recall)
            out["box_count_mae"] = (
                self.grounding_count_abs_error_sum / float(self.n_grounding)
                if self.n_grounding
                else None
            )
        if self.class_counts:
            out.update(_classification_summary(self.class_counts))
            positive_label = _POSITIVE_CLASS_TASKS.get(self.task_name or "")
            per_class = out.get("per_class")
            if positive_label and isinstance(per_class, dict):
                positive_metrics = per_class.get(positive_label)
                if isinstance(positive_metrics, dict):
                    out["positive_precision"] = positive_metrics.get("precision")
                    out["positive_recall"] = positive_metrics.get("recall")
                    out["positive_f1"] = positive_metrics.get("f1")
                else:
                    out["positive_precision"] = 0.0
                    out["positive_recall"] = 0.0
                    out["positive_f1"] = 0.0
        if self.position_total > 0:
            out["position_correct"] = int(self.position_correct)
            out["position_total"] = int(self.position_total)
            out["position_accuracy"] = float(self.position_correct) / float(self.position_total)
        if self.pairwise_total > 0:
            out["pairwise_correct"] = int(self.pairwise_correct)
            out["pairwise_total"] = int(self.pairwise_total)
            out["pairwise_accuracy"] = float(self.pairwise_correct) / float(self.pairwise_total)
            out["kendall_tau"] = (2.0 * out["pairwise_accuracy"]) - 1.0
        if self.ordinal_total > 0:
            out["ordinal_mae"] = self.ordinal_abs_error_sum / float(self.ordinal_total)
            out["ordinal_within_1"] = float(self.ordinal_within_1) / float(self.ordinal_total)
            out["quadratic_weighted_kappa"] = _quadratic_weighted_kappa(self.ordinal_pairs)
        primary_metric = primary_metric_for_task(self.task_name or "")
        if primary_metric in out:
            out["primary_metric"] = primary_metric
            out["primary_score"] = out[primary_metric]
        return out


def _metric_label(task: str, value: object) -> str:
    if task == "liquid_color":
        return canonicalize_liquid_color(value) or norm_text(value)
    if task == "container_type":
        return canonicalize_container_type(value) or norm_text(value)
    if task == "container_type_simple":
        return canonicalize_container_type_simple(value) or norm_text(value)
    return norm_text(value)


def _f1(precision: float, recall: float) -> float:
    return (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )


def _median(values: List[float]) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _classification_summary(class_counts: Dict[str, Dict[str, int]]) -> Dict[str, object]:
    labels = sorted(class_counts)
    per_class: Dict[str, Dict[str, float]] = {}
    precisions: List[float] = []
    recalls: List[float] = []
    f1s: List[float] = []
    support_recalls: List[float] = []
    total_tp = total_fp = total_fn = 0

    for label in labels:
        counts = class_counts[label]
        tp = int(counts.get("tp", 0))
        fp = int(counts.get("fp", 0))
        fn = int(counts.get("fn", 0))
        support = int(counts.get("support", 0))
        precision = float(tp) / float(tp + fp) if (tp + fp) else 0.0
        recall = float(tp) / float(tp + fn) if (tp + fn) else 0.0
        f1 = _f1(precision, recall)
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": float(support),
        }
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        if support > 0:
            support_recalls.append(recall)
        total_tp += tp
        total_fp += fp
        total_fn += fn

    micro_precision = float(total_tp) / float(total_tp + total_fp) if (total_tp + total_fp) else 0.0
    micro_recall = float(total_tp) / float(total_tp + total_fn) if (total_tp + total_fn) else 0.0
    return {
        "macro_precision": sum(precisions) / float(len(precisions)) if precisions else None,
        "macro_recall": sum(recalls) / float(len(recalls)) if recalls else None,
        "macro_f1": sum(f1s) / float(len(f1s)) if f1s else None,
        "balanced_accuracy": (
            sum(support_recalls) / float(len(support_recalls))
            if support_recalls
            else None
        ),
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": _f1(micro_precision, micro_recall),
        "per_class": per_class,
    }


def _quadratic_weighted_kappa(pairs: List[Tuple[int, int]]) -> Optional[float]:
    if not pairs:
        return None
    n_classes = max(_ORDINAL_VALUE.values()) + 1
    observed = [[0.0 for _ in range(n_classes)] for _ in range(n_classes)]
    pred_hist = [0.0 for _ in range(n_classes)]
    gt_hist = [0.0 for _ in range(n_classes)]
    for pred, gt in pairs:
        observed[pred][gt] += 1.0
        pred_hist[pred] += 1.0
        gt_hist[gt] += 1.0

    n = float(len(pairs))
    observed_weighted = 0.0
    expected_weighted = 0.0
    denom = float((n_classes - 1) ** 2)
    for pred in range(n_classes):
        for gt in range(n_classes):
            weight = float((pred - gt) ** 2) / denom
            observed_weighted += weight * observed[pred][gt]
            expected_weighted += weight * (pred_hist[pred] * gt_hist[gt] / n)
    if expected_weighted == 0.0:
        return 1.0 if observed_weighted == 0.0 else 0.0
    return 1.0 - (observed_weighted / expected_weighted)


def primary_metric_for_task(task: str) -> str:
    if task == "tip_presence":
        return "balanced_accuracy"
    if task in {"container_type", "container_type_simple", "highlighted_region_type", "liquid_color"}:
        return "macro_f1"
    if task == "multiphase_material_presence":
        return "positive_f1"
    if task == "material_types_present":
        return "micro_f1"
    if task in {"fill_percentage", "fill_height_mm", "container_count"}:
        return "mae"
    if task == "fill_level_ordinal":
        return "ordinal_mae"
    if task == "liquid_grounding":
        return "micro_f1"
    if task == "pouring_frame_ordering":
        return "pairwise_accuracy"
    return "accuracy"

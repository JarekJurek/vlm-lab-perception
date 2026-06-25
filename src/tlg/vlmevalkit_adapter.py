from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union

from tlg.annotations import TLGAnnotation
from tlg.evaluation import TaskStats, score_prediction
from tlg.manifest import TLGManifestDataset
from tlg.model_runner import make_labeled_image_strip
from tlg.prompts import PromptBank
from tlg.target_overlay import is_valid_bbox_xywh, make_target_box_overlay, should_mark_target_box


_PREDICTION_EXTRA_KEYS = (
    "abs_error",
    "squared_error",
    "signed_error",
    "within_1",
    "within_5",
    "within_10",
    "within_tol",
    "ordinal_error",
    "ordinal_abs_error",
    "ordinal_within_1",
    "exact_match",
    "n_gt_labels",
    "n_pred_labels",
    "n_matched_labels",
    "sample_precision",
    "sample_recall",
    "sample_f1",
    "pairwise_correct",
    "pairwise_total",
    "pairwise_accuracy",
    "position_correct",
    "position_total",
    "position_accuracy",
    "kendall_tau",
    "n_gt_boxes",
    "n_pred_boxes",
    "n_matched_boxes",
    "precision",
    "recall",
    "f1",
    "mean_iou",
    "iou_threshold",
)


class TLGVLMEvalDataset:
    """VLMEvalKit-compatible view of a TLG manifest.

    The class intentionally does not subclass VLMEvalKit at import time. It exposes
    the methods VLMEvalKit needs (`__len__`, `__getitem__`, `build_prompt`, and
    `evaluate`) while keeping the rest of this package usable without installing
    VLMEvalKit.
    """

    def __init__(
        self,
        manifest_path: Union[str, Path],
        *,
        split: str = "test",
        datasets: Optional[Iterable[str]] = None,
        tasks: Optional[Iterable[str]] = None,
        image_cache_dir: Optional[Union[str, Path]] = None,
        prompt_bank: Optional[PromptBank] = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.split = str(split)
        self.dataset_names = None if datasets is None else {str(d) for d in datasets}
        self.tasks = None if tasks is None else {str(t) for t in tasks}
        self.image_cache_dir = Path(image_cache_dir) if image_cache_dir is not None else None
        self.prompt_bank = prompt_bank or PromptBank.default()
        self.data: List[Dict[str, Any]] = manifest_to_vlmeval_rows(
            self.manifest_path,
            split=self.split,
            datasets=self.dataset_names,
            tasks=self.tasks,
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return dict(self.data[index])

    def build_prompt(self, line: Union[int, Mapping[str, Any]]) -> List[Dict[str, str]]:
        if isinstance(line, int):
            line = self.data[line]
        return build_vlmeval_prompt(
            line,
            image_cache_dir=self.image_cache_dir,
            prompt_bank=self.prompt_bank,
        )

    def evaluate(self, eval_file: Union[str, Path], **judge_kwargs: Any) -> Dict[str, Any]:
        del judge_kwargs
        rows = load_prediction_table(eval_file)
        return score_vlmeval_predictions(rows)["metrics"]


def manifest_to_vlmeval_rows(
    manifest_path: Union[str, Path],
    *,
    split: str = "test",
    datasets: Optional[Iterable[str]] = None,
    tasks: Optional[Iterable[str]] = None,
) -> List[Dict[str, Any]]:
    task_filter = None if tasks is None else {str(t) for t in tasks}
    ds = TLGManifestDataset(manifest_path, split=split, datasets=datasets)
    rows: List[Dict[str, Any]] = []
    for sample_idx, sample in enumerate(ds):
        annotations = sample.get("annotations")
        if not isinstance(annotations, list):
            continue
        for ann_idx, ann in enumerate(annotations):
            if not isinstance(ann, TLGAnnotation):
                continue
            task = str(ann.metadata.get("task", ann.task_type))
            if task_filter is not None and task not in task_filter:
                continue
            rows.append(_row_from_sample(sample, ann, sample_idx=sample_idx, ann_idx=ann_idx, task=task))
    return rows


def build_vlmeval_prompt(
    line: Mapping[str, Any],
    *,
    image_cache_dir: Optional[Union[str, Path]] = None,
    prompt_bank: Optional[PromptBank] = None,
) -> List[Dict[str, str]]:
    """Return VLMEvalKit interleaved image/text messages for one flattened task row."""
    from PIL import Image, UnidentifiedImageError

    prompt_bank = prompt_bank or PromptBank.default()
    cache_dir = Path(image_cache_dir) if image_cache_dir is not None else None

    def open_rgb(path: object, *, label: str) -> Any:
        try:
            return Image.open(path).convert("RGB")
        except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
            raise OSError(f"Failed to open {label}: {path} ({exc})") from exc

    task = str(line.get("task", ""))
    dataset_name = str(line.get("dataset", ""))
    sample_id = str(line.get("sample_id", line.get("index", "sample")))
    question = str(line.get("question", ""))

    rgb_path = line.get("rgb_image")
    if not rgb_path:
        raise FileNotFoundError(f"VLMEval row is missing rgb_image: index={line.get('index')!r}")
    rgb_image = open_rgb(rgb_path, label="RGB image")

    task_image = rgb_image
    prompt = question
    extra_image_paths: List[str] = []
    derived_image = False

    region_box_image = _make_annotation_box_image(rgb_image, _json_obj(line.get("annotation_metadata")))
    target_box_image = None
    bbox = line.get("bbox_visib") or line.get("bbox_obj")
    if is_valid_bbox_xywh(bbox):
        target_box_image = make_target_box_overlay(rgb_image, rgb_image, bbox=bbox)

    use_region_box = region_box_image is not None
    use_target_box = (
        target_box_image is not None
        and not use_region_box
        and should_mark_target_box(dataset_name, task)
    )
    if use_region_box or use_target_box:
        task_image = region_box_image if use_region_box else target_box_image
        derived_image = True
        target_region_prompt = prompt_bank.task_prompt_prefix(dataset_name, task)
        target_region_prompt = target_region_prompt or prompt_bank.target_region_bbox_prompt()
        prompt = f"{target_region_prompt.rstrip()}\n\n{prompt}"

    comparison_path = line.get("comparison_rgb_image")
    if comparison_path:
        prompt = (
            "The first image is the first frame. "
            "The second image is the second frame.\n\n"
            f"{prompt}"
        )
        extra_image_paths.append(str(comparison_path))

    ordered_paths = _json_list(line.get("ordered_rgb_images"))
    if ordered_paths:
        ordered_images = [
            open_rgb(path, label=f"ordered RGB image {i + 1}")
            for i, path in enumerate(ordered_paths)
        ]
        labels = _json_list(line.get("ordered_image_labels"))
        if len(labels) != len(ordered_images):
            labels = [chr(ord("A") + i) for i in range(len(ordered_images))]
        task_image = make_labeled_image_strip(ordered_images, labels)
        prompt = (
            "The input image is a horizontal strip of frames. "
            "The frames are provided from left to right with labels: "
            f"{', '.join(str(label) for label in labels)}. "
            "Use these labels in your answer.\n\n"
            f"{prompt}"
        )
        extra_image_paths = []
        derived_image = True

    primary_image_path = str(rgb_path)
    if derived_image:
        if cache_dir is None:
            raise ValueError("image_cache_dir is required for rows that need derived task images")
        primary_image_path = str(_save_task_image(task_image, cache_dir, sample_id=sample_id, task=task))

    messages = [{"type": "image", "value": primary_image_path}]
    messages.extend({"type": "image", "value": path} for path in extra_image_paths)
    messages.append({"type": "text", "value": prompt})
    return messages


def score_vlmeval_predictions(
    rows: Iterable[Mapping[str, Any]],
    *,
    tol_overrides: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    pred_rows: List[Dict[str, Any]] = []
    overall_stats: Dict[str, TaskStats] = {}
    per_dataset_stats: Dict[str, Dict[str, TaskStats]] = {}

    for row in rows:
        ann = annotation_from_vlmeval_row(row)
        task = str(row.get("task", ann.metadata.get("task", ann.task_type)))
        dataset_name = str(row.get("dataset", ann.dataset))
        response = str(row.get("prediction", row.get("pred_raw", "")))
        rec = score_prediction(ann, response, tol_overrides=tol_overrides)

        for stats_map in (overall_stats, per_dataset_stats.setdefault(dataset_name, {})):
            st = stats_map.setdefault(task, TaskStats())
            st.update_from_record(rec)

        out = {
            "model": row.get("model", ""),
            "dataset": dataset_name,
            "sample_id": row.get("sample_id", ""),
            "image_id": row.get("image_id", ann.image_id),
            "annotation_id": row.get("annotation_id", ann.image_id),
            "task": rec.get("task"),
            "task_type": rec.get("task_type"),
            "question": rec.get("question"),
            "gt": rec.get("gt"),
            "pred": rec.get("pred"),
            "pred_raw": rec.get("pred_raw"),
            "valid": rec.get("valid"),
            "correct": rec.get("correct"),
        }
        for key in _PREDICTION_EXTRA_KEYS:
            if key in rec:
                out[key] = rec[key]
        pred_rows.append(out)

    return {
        "predictions": pred_rows,
        "metrics": {
            "overall": {task: stats.to_dict() for task, stats in overall_stats.items()},
            "per_dataset": {
                dataset: {task: stats.to_dict() for task, stats in stats_map.items()}
                for dataset, stats_map in per_dataset_stats.items()
            },
        },
    }


def annotation_from_vlmeval_row(row: Mapping[str, Any]) -> TLGAnnotation:
    metadata = _json_obj(row.get("annotation_metadata"))
    if "task" not in metadata and row.get("task") is not None:
        metadata["task"] = str(row.get("task"))
    return TLGAnnotation(
        image_id=str(row.get("image_id", row.get("sample_id", ""))),
        dataset=str(row.get("dataset", "")),
        task_type=str(row.get("task_type", "")),
        question=str(row.get("question", "")),
        choices=[str(c) for c in _json_list(row.get("choices"))] or None,
        answer=_json_value(row.get("answer")),
        metadata=metadata,
    )


def load_prediction_table(path: Union[str, Path]) -> List[Dict[str, Any]]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return [dict(row) for row in payload if isinstance(row, Mapping)]
        if isinstance(payload, Mapping) and isinstance(payload.get("data"), list):
            return [dict(row) for row in payload["data"] if isinstance(row, Mapping)]
        raise ValueError(f"Unsupported JSON prediction shape: {path}")

    import pandas as pd

    if suffix in {".xlsx", ".xls"}:
        frame = pd.read_excel(path)
    elif suffix in {".csv", ".tsv"}:
        frame = pd.read_csv(path, sep="\t" if suffix == ".tsv" else ",")
    else:
        raise ValueError(f"Unsupported prediction file type: {path}")
    return [dict(row) for row in frame.to_dict(orient="records")]


def _row_from_sample(
    sample: Mapping[str, Any],
    ann: TLGAnnotation,
    *,
    sample_idx: int,
    ann_idx: int,
    task: str,
) -> Dict[str, Any]:
    sample_id = str(sample.get("id", ann.image_id or sample_idx))
    return {
        "index": f"{sample_idx}:{ann_idx}",
        "sample_index": sample_idx,
        "annotation_index": ann_idx,
        "sample_id": sample_id,
        "image_id": ann.image_id,
        "annotation_id": f"{sample_id}:{task}:{ann_idx}",
        "dataset": str(sample.get("dataset", ann.dataset)),
        "split": str(sample.get("split", "")),
        "task": task,
        "task_type": ann.task_type,
        "question": ann.question,
        "answer": json.dumps(ann.answer, ensure_ascii=False),
        "choices": json.dumps(list(ann.choices or []), ensure_ascii=False),
        "annotation_metadata": json.dumps(ann.metadata, ensure_ascii=False),
        "rgb_image": _path_or_empty(sample.get("rgb_image")),
        "comparison_rgb_image": _path_or_empty(sample.get("comparison_rgb_image")),
        "ordered_rgb_images": json.dumps(_string_list(sample.get("ordered_rgb_images")), ensure_ascii=False),
        "ordered_image_labels": json.dumps(_string_list(sample.get("ordered_image_labels")), ensure_ascii=False),
        "bbox_visib": sample.get("bbox_visib"),
        "bbox_obj": sample.get("bbox_obj"),
    }


def _save_task_image(image: Any, cache_dir: Path, *, sample_id: str, task: str) -> Path:
    safe_sample = _safe_path_part(sample_id)
    safe_task = _safe_path_part(task)
    path = cache_dir / safe_sample / f"{safe_task}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path


def _make_annotation_box_image(image: Any, metadata: Mapping[str, Any]) -> Optional[Any]:
    boxes = metadata.get("bbox_boxes")
    if isinstance(boxes, (list, tuple)) and boxes:
        parsed_boxes = []
        for item in boxes:
            if not isinstance(item, Mapping):
                return None
            bbox = item.get("bbox_xyxy")
            if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
                return None
            try:
                x0, y0, x1, y1 = [float(v) for v in bbox[:4]]
            except (TypeError, ValueError):
                return None
            if x1 <= x0 or y1 <= y0:
                return None
            parsed_boxes.append(((x0, y0, x1, y1), _box_color(item.get("color")), str(item.get("label", "")).strip()))
        return _draw_annotation_boxes(image, parsed_boxes)

    bbox = metadata.get("bbox_xyxy")
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return None
    try:
        x0, y0, x1, y1 = [float(v) for v in bbox[:4]]
    except (TypeError, ValueError):
        return None
    if x1 <= x0 or y1 <= y0:
        return None
    return make_target_box_overlay(image, image, bbox=[x0, y0, x1 - x0, y1 - y0])


def _box_color(value: object) -> tuple[int, int, int]:
    name = str(value).strip().lower()
    if name == "green":
        return (0, 180, 70)
    if name == "yellow":
        return (255, 215, 0)
    return (255, 0, 0)


def _draw_annotation_boxes(
    image: Any,
    boxes: Sequence[tuple[Sequence[float], tuple[int, int, int], str]],
    *,
    width: int = 6,
    padding: int = 4,
) -> Any:
    from PIL import ImageDraw, ImageFont

    marked = image.convert("RGB").copy()
    draw = ImageDraw.Draw(marked)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 16)
    except OSError:
        font = ImageFont.load_default()

    for bbox, color, label in boxes:
        x0, y0, x1, y1 = [int(round(v)) for v in bbox[:4]]
        x0 = max(0, x0 - padding)
        y0 = max(0, y0 - padding)
        x1 = min(marked.width - 1, x1 + padding)
        y1 = min(marked.height - 1, y1 + padding)
        draw.rectangle((x0, y0, x1, y1), outline=color, width=max(1, int(width)))
        if label:
            _draw_box_label(draw, label, x0, y0, color, font)
    return marked


def _draw_box_label(draw: Any, label: str, x: int, y: int, color: tuple[int, int, int], font: object) -> None:
    try:
        left, top, right, bottom = draw.textbbox((x, y), label, font=font)
        text_w = right - left
        text_h = bottom - top
    except Exception:
        text_w = max(36, len(label) * 8)
        text_h = 14
    label_y = max(0, y - text_h - 6)
    draw.rectangle((x, label_y, x + text_w + 8, label_y + text_h + 6), fill=color)
    draw.text((x + 4, label_y + 3), label, fill=(0, 0, 0), font=font)


def _json_obj(value: object) -> Dict[str, Any]:
    parsed = _json_value(value)
    return dict(parsed) if isinstance(parsed, Mapping) else {}


def _json_list(value: object) -> List[Any]:
    parsed = _json_value(value)
    return list(parsed) if isinstance(parsed, list) else []


def _json_value(value: object) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return ""
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            return value
    return value


def _path_or_empty(value: object) -> str:
    return "" if value is None else str(value)


def _string_list(value: object) -> List[str]:
    if not isinstance(value, (list, tuple)):
        return []
    return [str(item) for item in value]


def _safe_path_part(value: object) -> str:
    text = str(value).strip() or "item"
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in text)

#!/usr/bin/env python3
"""Draw OT2 visual-grounding predictions on dataset images.

Example:
    python src/datasets/ot2/visualize_grounding_predictions.py \
        --predictions bakery/qwen_VG_test/preds.jsonl \
        --data-root data/OT2-Computer-Vision \
        --out-dir bakery/qwen_VG_test/boxes

Output:
    By default, writes one GT image and one prediction image per sample:
        out_dir/gt/<sample_id>.png
        out_dir/pred/<sample_id>.png
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from datasets.ot2 import OT2Dataset  # noqa: E402


TASK_NAMES = {"liquid_grounding", "tip_liquid_grounding"}
GT_COLORS = {"Tip": (0, 180, 80), "Liquid": (255, 150, 0)}
PRED_COLORS = {"Tip": (0, 190, 255), "Liquid": (230, 70, 230)}


def main() -> None:
    args = parse_args()
    predictions_path = Path(args.predictions)
    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gt_dir = out_dir / "gt"
    pred_dir = out_dir / "pred"
    if not args.combined:
        gt_dir.mkdir(parents=True, exist_ok=True)
        pred_dir.mkdir(parents=True, exist_ok=True)

    sample_index = _build_sample_index(data_root)

    written = 0
    skipped = 0
    for row in _iter_jsonl(predictions_path):
        if str(row.get("task", "")) not in TASK_NAMES:
            continue
        if args.model and str(row.get("model", "")) != args.model:
            continue

        sample_id = str(row.get("sample_id", ""))
        sample = sample_index.get(sample_id)
        if sample is None:
            skipped += 1
            print(f"Skipping unresolved sample_id={sample_id!r}")
            continue

        pred_boxes = _prediction_boxes(row)
        if pred_boxes is None:
            skipped += 1
            print(f"Skipping unparsable prediction for sample_id={sample_id!r}")
            continue

        gt_boxes = row.get("gt")
        if not isinstance(gt_boxes, list):
            gt_boxes = []

        image_path = Path(str(sample["rgb_image"]))
        with Image.open(image_path) as image:
            base_image = image.convert("RGB")

        sample_name = _safe_name(sample_id)
        if args.combined:
            canvas = base_image.copy()
            draw = ImageDraw.Draw(canvas)
            if not args.no_gt:
                _draw_boxes(draw, gt_boxes, canvas.size, prefix="GT", colors=GT_COLORS, width=2)
            _draw_boxes(draw, pred_boxes, canvas.size, prefix="PRED", colors=PRED_COLORS, width=4)
            canvas.save(out_dir / f"{sample_name}.png")
        else:
            if not args.no_gt:
                gt_canvas = base_image.copy()
                _draw_boxes(
                    ImageDraw.Draw(gt_canvas),
                    gt_boxes,
                    gt_canvas.size,
                    prefix="GT",
                    colors=GT_COLORS,
                    width=3,
                )
                gt_canvas.save(gt_dir / f"{sample_name}.png")

            pred_canvas = base_image.copy()
            _draw_boxes(
                ImageDraw.Draw(pred_canvas),
                pred_boxes,
                pred_canvas.size,
                prefix="PRED",
                colors=PRED_COLORS,
                width=3,
            )
            pred_canvas.save(pred_dir / f"{sample_name}.png")
        written += 1
        if args.limit is not None and written >= args.limit:
            break

    if args.combined:
        print(f"Wrote {written} combined visualizations to {out_dir}")
    elif args.no_gt:
        print(f"Wrote {written} prediction visualizations to {pred_dir}")
    else:
        print(f"Wrote {written} GT visualizations to {gt_dir}")
        print(f"Wrote {written} prediction visualizations to {pred_dir}")
    if skipped:
        print(f"Skipped {skipped} rows")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize OT2 tip/liquid grounding predictions.")
    parser.add_argument("--predictions", required=True, help="Path to predictions JSONL from test_qwen_vl_tasks.py.")
    parser.add_argument("--data-root", default="data/OT2-Computer-Vision", help="OT2 dataset root.")
    parser.add_argument("--out-dir", required=True, help="Directory for rendered PNG files.")
    parser.add_argument("--model", default="", help="Optional model key to filter, e.g. qwen or sdvlm.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of images to write.")
    parser.add_argument("--no-gt", action="store_true", help="Only draw predictions, not ground truth.")
    parser.add_argument("--combined", action="store_true", help="Draw GT and predictions on one image.")
    return parser.parse_args()


def _build_sample_index(data_root: Path) -> Dict[str, Dict[str, object]]:
    dataset = OT2Dataset(data_root)
    index: Dict[str, Dict[str, object]] = {}
    for i in range(len(dataset)):
        sample = dataset[i]
        index[str(sample["id"])] = sample
    return index


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"Skipping invalid JSON at {path}:{line_no}: {exc}")
                continue
            if isinstance(row, dict):
                yield row


def _prediction_boxes(row: Dict[str, Any]) -> Optional[List[Dict[str, object]]]:
    pred = row.get("pred")
    if isinstance(pred, list):
        return _normalize_boxes(pred)

    raw = row.get("pred_raw")
    if isinstance(raw, str):
        return parse_grounding_boxes(raw)

    return None


def parse_grounding_boxes(text: str) -> Optional[List[Dict[str, object]]]:
    payload = _load_json_like(text)
    if payload is not None:
        if isinstance(payload, dict):
            for key in ("objects", "detections", "boxes", "regions", "items"):
                value = payload.get(key)
                if isinstance(value, list):
                    payload = value
                    break

        if isinstance(payload, list):
            return _normalize_boxes(payload)

    # Some runs truncate the model output before the closing JSON bracket.
    # For visualization, recover any complete bbox objects that are still present.
    return _extract_complete_box_objects(text)


def _load_json_like(text: str) -> Optional[Any]:
    s = " ".join(str(text).strip().split())
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


def _normalize_boxes(items: List[Any]) -> Optional[List[Dict[str, object]]]:
    out: List[Dict[str, object]] = []
    for item in items:
        if not isinstance(item, dict):
            return None
        label = _canonical_label(item.get("label"))
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


def _extract_complete_box_objects(text: str) -> Optional[List[Dict[str, object]]]:
    pattern = re.compile(
        r"\{\s*['\"]bbox_2d['\"]\s*:\s*\[\s*"
        r"([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*,\s*"
        r"([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\]\s*,\s*"
        r"['\"]label['\"]\s*:\s*['\"]([^'\"]+)['\"]\s*\}",
        flags=re.IGNORECASE,
    )
    out: List[Dict[str, object]] = []
    for match in pattern.finditer(str(text)):
        label = _canonical_label(match.group(5))
        if label is None:
            continue
        x1, y1, x2, y2 = [float(match.group(i)) for i in range(1, 5)]
        if x2 <= x1 or y2 <= y1:
            continue
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
    return out or None


def _canonical_label(value: object) -> Optional[str]:
    compact = "".join(ch for ch in str(value).strip().lower() if ch.isalnum())
    if compact in {"tip", "pipettetip"}:
        return "Tip"
    if compact == "liquid":
        return "Liquid"
    return None


def _clamp_coord(value: float) -> float:
    return max(0.0, min(1000.0, float(value)))


def _draw_boxes(
    draw: ImageDraw.ImageDraw,
    boxes: List[Dict[str, object]],
    image_size: tuple[int, int],
    *,
    prefix: str,
    colors: Dict[str, tuple[int, int, int]],
    width: int,
) -> None:
    font = ImageFont.load_default()
    for item in boxes:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label", ""))
        bbox = item.get("bbox_2d")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        try:
            x1, y1, x2, y2 = _scale_bbox([float(v) for v in bbox], image_size)
        except (TypeError, ValueError):
            continue
        color = colors.get(label, (255, 255, 255))
        draw.rectangle((x1, y1, x2, y2), outline=color, width=width)
        _draw_label(draw, (x1, y1), f"{prefix} {label}", color, font)


def _scale_bbox(bbox: List[float], image_size: tuple[int, int]) -> tuple[int, int, int, int]:
    width, height = image_size
    x1, y1, x2, y2 = bbox
    return (
        int(round(max(0.0, min(1000.0, x1)) * width / 1000.0)),
        int(round(max(0.0, min(1000.0, y1)) * height / 1000.0)),
        int(round(max(0.0, min(1000.0, x2)) * width / 1000.0)),
        int(round(max(0.0, min(1000.0, y2)) * height / 1000.0)),
    )


def _draw_label(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    color: tuple[int, int, int],
    font: ImageFont.ImageFont,
) -> None:
    x, y = xy
    bbox = draw.textbbox((x, y), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    y0 = max(0, y - text_h - 4)
    draw.rectangle((x, y0, x + text_w + 4, y0 + text_h + 4), fill=color)
    draw.text((x + 2, y0 + 2), text, fill=(0, 0, 0), font=font)


def _safe_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return safe or "sample"


if __name__ == "__main__":
    main()

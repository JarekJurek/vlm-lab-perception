"""Utilities for marking target objects in RGB images."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Optional, Tuple


DTLD_TARGET_BOX_TASKS = frozenset(
    {
        "container_type",
        "container_type_simple",
        "fill_height_mm",
        "liquid_presence",
    }
)


def should_mark_target_box(dataset_name: str, task_name: str) -> bool:
    """Return whether this task should receive a marked target image."""

    return dataset_name == "dtld" and task_name in DTLD_TARGET_BOX_TASKS


def make_target_box_overlay(
    image: Any,
    mask: Any,
    *,
    bbox: Optional[Sequence[float]] = None,
    color: Tuple[int, int, int] = (255, 0, 0),
    width: int = 6,
    padding: int = 4,
) -> Any:
    """Return an RGB copy of ``image`` with the target object's bbox drawn.

    ``bbox`` is expected in BOP/COCO-style ``[x, y, width, height]`` format.
    When absent, the bounding box is derived from non-zero mask pixels.
    """

    from PIL import Image, ImageDraw

    base = image.convert("RGB")
    box = _xyxy_from_bbox(bbox) if bbox is not None else None
    if box is None:
        resampling = getattr(getattr(Image, "Resampling", Image), "NEAREST")
        target = mask.convert("L").resize(base.size, resampling)
        box = target.point(lambda p: 255 if p > 0 else 0).getbbox()
    if box is None:
        return base

    x0, y0, x1, y1 = box
    pad = max(0, int(padding))
    x0 = max(0, int(round(x0)) - pad)
    y0 = max(0, int(round(y0)) - pad)
    x1 = min(base.width - 1, int(round(x1)) + pad)
    y1 = min(base.height - 1, int(round(y1)) + pad)

    marked = base.copy()
    draw = ImageDraw.Draw(marked)
    draw.rectangle((x0, y0, x1, y1), outline=color, width=max(1, int(width)))
    return marked


def is_valid_bbox_xywh(bbox: object) -> bool:
    """Return whether ``bbox`` is a valid ``[x, y, width, height]`` box."""

    if not isinstance(bbox, Sequence) or isinstance(bbox, (str, bytes)):
        return False
    return _xyxy_from_bbox(bbox) is not None


def _xyxy_from_bbox(bbox: Sequence[float]) -> Optional[Tuple[float, float, float, float]]:
    if len(bbox) < 4:
        return None
    try:
        x, y, w, h = [float(v) for v in bbox[:4]]
    except (TypeError, ValueError):
        return None
    if w <= 0 or h <= 0:
        return None
    return x, y, x + w, y + h

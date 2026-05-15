"""Debug helpers for inspecting model input images."""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Optional


def debug_dtld_box_state(
    *,
    prefix: str,
    sample: dict[str, object],
    task_name: str,
    use_target_box: bool,
    use_region_box: bool,
    target_box_image: Optional[Any],
) -> None:
    """Print enough state to verify whether DTLD target boxes are available."""

    bbox_visib = sample.get("bbox_visib")
    bbox_obj = sample.get("bbox_obj")
    mask_path = sample.get("mask_image")
    selected_bbox = bbox_visib or bbox_obj
    print(
        f"  [{prefix}][DTLD bbox debug] task={task_name} "
        f"mask_image={mask_path!s} "
        f"bbox_visib={bbox_visib!r} valid_visib={_valid_xywh(bbox_visib)} "
        f"bbox_obj={bbox_obj!r} valid_obj={_valid_xywh(bbox_obj)} "
        f"selected_bbox={selected_bbox!r} "
        f"target_box_ready={target_box_image is not None} "
        f"use_region_box={use_region_box} use_target_box={use_target_box}"
    )


def debug_dtld_sample_state(*, sample: dict[str, object]) -> None:
    """Print DTLD sample fields produced by the dataloader."""

    bbox_visib = sample.get("bbox_visib")
    bbox_obj = sample.get("bbox_obj")
    print(
        "  [DTLD dataloader debug] "
        f"id={sample.get('id', 'unknown')!s} "
        f"rgb_image={sample.get('rgb_image')!s} "
        f"mask_image={sample.get('mask_image')!s} "
        f"mask_visib_image={sample.get('mask_visib_image')!s} "
        f"mask_full_image={sample.get('mask_full_image')!s} "
        f"bbox_visib={bbox_visib!r} valid_visib={_valid_xywh(bbox_visib)} "
        f"bbox_obj={bbox_obj!r} valid_obj={_valid_xywh(bbox_obj)} "
        f"visib_fract={sample.get('visib_fract')!r}"
    )


def save_dtld_box_debug_images(
    *,
    out_dir: Path,
    model_key: str,
    sample_id: str,
    task_name: str,
    rgb_image: Any,
    mask_image: Optional[Any],
    target_box_image: Optional[Any],
    task_image: Any,
) -> None:
    """Save raw and selected DTLD images for visual inspection."""

    task_dir = out_dir / model_key / "dtld" / _safe_relpath(sample_id) / _safe_name(task_name)
    task_dir.mkdir(parents=True, exist_ok=True)
    rgb_image.save(task_dir / "01_rgb.png")
    if mask_image is not None:
        mask_image.save(task_dir / "02_mask.png")
    if target_box_image is not None:
        target_box_image.save(task_dir / "03_target_box.png")
    task_image.save(task_dir / "04_model_input.png")
    print(f"  [{model_key}][DTLD bbox debug] wrote images: {task_dir}")


def _valid_xywh(value: object) -> bool:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) < 4:
        return False
    try:
        _x, _y, w, h = [float(v) for v in value[:4]]
    except (TypeError, ValueError):
        return False
    return w > 0 and h > 0


def _safe_relpath(value: str) -> Path:
    parts = [part for part in str(value).split("/") if part not in {"", ".", ".."}]
    return Path(*[_safe_name(part) for part in parts]) if parts else Path("unknown")


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return cleaned or "unknown"

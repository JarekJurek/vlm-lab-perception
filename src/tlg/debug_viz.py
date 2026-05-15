"""Debug helpers for inspecting model input images."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional

from tlg.target_overlay import is_valid_bbox_xywh


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
    selected_bbox = bbox_visib or bbox_obj
    print(
        f"  [{prefix}][DTLD bbox debug] task={task_name} "
        f"bbox_visib={bbox_visib!r} valid_visib={is_valid_bbox_xywh(bbox_visib)} "
        f"bbox_obj={bbox_obj!r} valid_obj={is_valid_bbox_xywh(bbox_obj)} "
        f"selected_bbox={selected_bbox!r} "
        f"target_box_ready={target_box_image is not None} "
        f"use_region_box={use_region_box} use_target_box={use_target_box}"
    )


def save_dtld_box_debug_images(
    *,
    out_dir: Path,
    model_key: str,
    sample_id: str,
    task_name: str,
    rgb_image: Any,
    target_box_image: Optional[Any],
    task_image: Any,
) -> None:
    """Save raw and selected DTLD images for visual inspection."""

    task_dir = out_dir / model_key / "dtld" / _safe_relpath(sample_id) / _safe_name(task_name)
    task_dir.mkdir(parents=True, exist_ok=True)
    rgb_image.save(task_dir / "01_rgb.png")
    if target_box_image is not None:
        target_box_image.save(task_dir / "02_target_box.png")
    task_image.save(task_dir / "03_model_input.png")
    print(f"  [{model_key}][DTLD bbox debug] wrote images: {task_dir}")


def _safe_relpath(value: str) -> Path:
    parts = [part for part in str(value).split("/") if part not in {"", ".", ".."}]
    return Path(*[_safe_name(part) for part in parts]) if parts else Path("unknown")


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return cleaned or "unknown"

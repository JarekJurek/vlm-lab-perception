from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

from datasets.coqe import COQETaskExample


YELLOW = (255, 215, 0)
GREEN = (0, 180, 70)


def render_example_image(example: COQETaskExample) -> Image.Image:
    image = Image.open(example.image_path).convert("RGB")
    if example.task == "comparative":
        if example.target is None:
            raise ValueError("Comparative examples require a target record.")
        return draw_boxes(
            image,
            [
                (example.source.bbox_xyxy, YELLOW, "SOURCE"),
                (example.target.bbox_xyxy, GREEN, "TARGET"),
            ],
        )
    return draw_boxes(image, [(example.source.bbox_xyxy, YELLOW, "QUERY")])


def draw_boxes(
    image: Image.Image,
    boxes: Sequence[Tuple[Sequence[int], Tuple[int, int, int], str]],
    *,
    width: int = 6,
    padding: int = 4,
) -> Image.Image:
    marked = image.convert("RGB").copy()
    draw = ImageDraw.Draw(marked)
    font = _load_font()
    for bbox, color, label in boxes:
        x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(marked.width - 1, x2 + padding)
        y2 = min(marked.height - 1, y2 + padding)
        draw.rectangle((x1, y1, x2, y2), outline=color, width=max(1, int(width)))
        _draw_label(draw, label, x1, y1, color, font)
    return marked


def maybe_save_input_image(image: Image.Image, out_dir: Optional[Path], example: COQETaskExample) -> None:
    if out_dir is None:
        return
    task_dir = out_dir / example.task
    task_dir.mkdir(parents=True, exist_ok=True)
    safe_id = example.id.replace("/", "_").replace(":", "_")
    image.save(task_dir / f"{safe_id}.png")


def _draw_label(draw: ImageDraw.ImageDraw, label: str, x: int, y: int, color: Tuple[int, int, int], font: object) -> None:
    text = str(label)
    try:
        left, top, right, bottom = draw.textbbox((x, y), text, font=font)
        text_w = right - left
        text_h = bottom - top
    except Exception:
        text_w = max(36, len(text) * 8)
        text_h = 14
    label_y = max(0, y - text_h - 6)
    draw.rectangle((x, label_y, x + text_w + 8, label_y + text_h + 6), fill=color)
    draw.text((x + 4, label_y + 3), text, fill=(0, 0, 0), font=font)


def _load_font() -> object:
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", 16)
    except OSError:
        return ImageFont.load_default()

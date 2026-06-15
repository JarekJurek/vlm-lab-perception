#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image, ImageDraw, ImageFont


KEYPOINT_COLORS = {
    "top_front": (40, 170, 255),
    "bottom_front": (255, 170, 40),
    "fluid_level": (40, 220, 100),
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a visual audit sheet for low fill-percentage glasses examples."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/glasses_in_the_wild/glasses_in_the_wild_1000"),
        help="Glasses in the Wild dataset root.",
    )
    parser.add_argument("--min-fill", type=float, default=0.0)
    parser.add_argument("--max-fill", type=float, default=21.0)
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/glasses_low_fill_0_21_audit.png"),
        help="Output PNG contact sheet.",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=Path("results/glasses_low_fill_0_21_audit.csv"),
        help="Output CSV with selected examples.",
    )
    args = parser.parse_args()

    examples = load_examples(args.root)
    low = [ex for ex in examples if args.min_fill <= ex["fill_percentage"] <= args.max_fill]
    selected = select_evenly(low, args.count)

    if not selected:
        raise SystemExit(f"No examples found in [{args.min_fill}, {args.max_fill}]")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.csv_out.parent.mkdir(parents=True, exist_ok=True)

    write_csv(args.csv_out, selected)
    make_sheet(args.root / "images", selected, args.out, args.min_fill, args.max_fill)

    print(f"total_examples={len(examples)}")
    print(f"selected_band={len(low)} ({len(low) / len(examples) * 100:.1f}%)")
    print(f"selected_for_sheet={len(selected)}")
    print(f"fill_min={min(ex['fill_percentage'] for ex in examples):.3f}")
    print(f"fill_max={max(ex['fill_percentage'] for ex in examples):.3f}")
    print(f"sheet={args.out}")
    print(f"csv={args.csv_out}")


def load_examples(root: Path) -> List[Dict[str, Any]]:
    annotations_path = root / "images" / "annotations.json"
    with annotations_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    names = data["categories"][0]["keypoints"]
    images = {int(img["id"]): img for img in data["images"]}
    examples: List[Dict[str, Any]] = []

    for ann in data["annotations"]:
        image_id = int(ann["image_id"])
        image = images.get(image_id)
        if image is None:
            continue

        keypoints = named_keypoints(ann.get("keypoints", []), names)
        fill = estimate_fill_percentage(keypoints)
        if fill is None:
            continue

        examples.append(
            {
                "image_id": image_id,
                "file_name": image["file_name"],
                "fill_percentage": fill,
                "ordinal_with_21_threshold": ordinal(fill, empty_threshold=21.0),
                "ordinal_with_default_threshold": ordinal(fill, empty_threshold=1.0),
                "keypoints": keypoints,
            }
        )

    examples.sort(key=lambda ex: (ex["fill_percentage"], ex["image_id"]))
    return examples


def named_keypoints(raw: List[float], names: List[str]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for i, name in enumerate(names):
        base = i * 3
        if base + 2 >= len(raw):
            break
        out[name] = {
            "x": float(raw[base]),
            "y": float(raw[base + 1]),
            "v": float(raw[base + 2]),
        }
    return out


def estimate_fill_percentage(keypoints: Dict[str, Dict[str, float]]) -> Optional[float]:
    bottom = keypoints.get("bottom_front")
    top = keypoints.get("top_front")
    fluid = keypoints.get("fluid_level")
    if bottom is None or top is None or fluid is None:
        return None
    if bottom["v"] <= 0 or top["v"] <= 0 or fluid["v"] <= 0:
        return None

    height = bottom["y"] - top["y"]
    if abs(height) < 1e-6:
        return None
    level = (bottom["y"] - fluid["y"]) / height
    return max(0.0, min(1.0, level)) * 100.0


def ordinal(fill: float, *, empty_threshold: float) -> str:
    p = max(0.0, min(100.0, float(fill)))
    if p <= empty_threshold:
        return "EMPTY"
    if p < 50.0:
        return "LOW"
    if p < 75.0:
        return "MID"
    return "HIGH"


def select_evenly(examples: List[Dict[str, Any]], count: int) -> List[Dict[str, Any]]:
    if len(examples) <= count:
        return examples
    if count <= 1:
        return examples[:count]

    selected: List[Dict[str, Any]] = []
    seen = set()
    for i in range(count):
        idx = round(i * (len(examples) - 1) / (count - 1))
        while idx in seen and idx + 1 < len(examples):
            idx += 1
        seen.add(idx)
        selected.append(examples[idx])
    return selected


def write_csv(path: Path, examples: List[Dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_id",
                "file_name",
                "fill_percentage",
                "ordinal_with_21_threshold",
                "ordinal_with_default_threshold",
            ],
        )
        writer.writeheader()
        for ex in examples:
            writer.writerow(
                {
                    "image_id": ex["image_id"],
                    "file_name": ex["file_name"],
                    "fill_percentage": f"{ex['fill_percentage']:.6f}",
                    "ordinal_with_21_threshold": ex["ordinal_with_21_threshold"],
                    "ordinal_with_default_threshold": ex["ordinal_with_default_threshold"],
                }
            )


def make_sheet(
    images_dir: Path,
    examples: List[Dict[str, Any]],
    out: Path,
    min_fill: float,
    max_fill: float,
) -> None:
    cols = 5
    rows = (len(examples) + cols - 1) // cols
    thumb = 256
    label_h = 74
    pad = 12
    cell_w = thumb + pad * 2
    cell_h = thumb + label_h + pad * 2
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h + 44), (245, 245, 242))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    title = f"Glasses low-fill audit: {len(examples)} examples in {min_fill:g}-{max_fill:g}%"
    draw.text((pad, 12), title, fill=(20, 20, 20), font=font)
    draw.text((pad, 28), "cyan=top_front, orange=bottom_front, green=fluid_level", fill=(50, 50, 50), font=font)

    for i, ex in enumerate(examples):
        row, col = divmod(i, cols)
        x0 = col * cell_w + pad
        y0 = row * cell_h + pad + 44
        image_path = images_dir / ex["file_name"]
        with Image.open(image_path) as im:
            im = im.convert("RGB").resize((thumb, thumb))
        overlay = im.copy()
        annotate_image(overlay, ex["keypoints"])
        sheet.paste(overlay, (x0, y0))

        text_y = y0 + thumb + 4
        draw.text((x0, text_y), f"#{ex['image_id']}  {ex['fill_percentage']:.1f}%", fill=(10, 10, 10), font=font)
        draw.text((x0, text_y + 14), f"21% ordinal: {ex['ordinal_with_21_threshold']}", fill=(10, 10, 10), font=font)
        draw.text((x0, text_y + 28), f"default ordinal: {ex['ordinal_with_default_threshold']}", fill=(10, 10, 10), font=font)
        draw.text((x0, text_y + 42), truncate(ex["file_name"], 38), fill=(70, 70, 70), font=font)

    sheet.save(out)


def annotate_image(image: Image.Image, keypoints: Dict[str, Dict[str, float]]) -> None:
    draw = ImageDraw.Draw(image, "RGBA")
    width, _ = image.size
    for name in ("top_front", "bottom_front", "fluid_level"):
        kp = keypoints.get(name)
        if not kp or kp["v"] <= 0:
            continue
        x, y = kp["x"], kp["y"]
        color = KEYPOINT_COLORS[name]
        draw.line([(0, y), (width, y)], fill=(*color, 140), width=2)
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=(*color, 230), outline=(0, 0, 0, 220), width=1)

    top = keypoints.get("top_front")
    bottom = keypoints.get("bottom_front")
    fluid = keypoints.get("fluid_level")
    if top and bottom and fluid:
        x = max(8.0, min(image.size[0] - 8.0, (top["x"] + bottom["x"] + fluid["x"]) / 3.0))
        draw.line([(x, top["y"]), (x, bottom["y"])], fill=(255, 255, 255, 170), width=2)
        draw.line([(x - 6, fluid["y"]), (x + 6, fluid["y"])], fill=(40, 220, 100, 255), width=3)


def truncate(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


if __name__ == "__main__":
    main()

from __future__ import annotations

import hashlib
import json
import random
from functools import partial
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union, cast

from PIL import Image

from tlg import PromptBank, TLGAnnotation, make_tlg_annotation


_IMG_EXTS: Tuple[str, ...] = (".jpg", ".jpeg", ".png")
_REGION_CHOICES = {
    "VESSEL",
    "LIQUID",
    "FOAM",
    "GEL",
    "SOLID",
    "POWDER",
    "GRANULAR",
    "VAPOR",
    "BACKGROUND",
}
_MATERIAL_ORDER = ("LIQUID", "FOAM", "GEL", "SOLID", "POWDER", "GRANULAR", "VAPOR")
_MATERIAL_LABEL_MAP = {
    "Liquid GENERAL": "LIQUID",
    "Liquid Suspension": "LIQUID",
    "Foam": "FOAM",
    "Gel": "GEL",
    "Solid GENERAL": "SOLID",
    "Solid Bulk": "SOLID",
    "Powder": "POWDER",
    "Granular": "GRANULAR",
    "Vapor": "VAPOR",
}
_SEMANTIC_LABEL_MAP = {
    "1_Vessel": "VESSEL",
    "6_Liquid GENERAL": "LIQUID",
    "7_Liquid Suspension": "LIQUID",
    "8_Foam": "FOAM",
    "9_Gel": "GEL",
    "10_Solid GENERAL": "SOLID",
    "11_Granular": "GRANULAR",
    "12_Powder": "POWDER",
    "13_Solid Bulk": "SOLID",
    "14_Vapor": "VAPOR",
}


class LabPicsV1Dataset:
    """Minimal LabPicsV1 Simple-split loader for image and boxed-region VQA."""

    def __init__(
        self,
        root: Union[str, Path],
        ids: Optional[Iterable[str]] = None,
        split: str = "train",
        prompt_bank: Optional[PromptBank] = None,
        shuffle: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        root_path = Path(root)
        if not root_path.exists():
            raise FileNotFoundError(f"Dataset root not found: {root_path}")

        self.root = _resolve_simple_root(root_path)
        self.default_split = _normalize_split(split)
        self._prompt_bank = prompt_bank or PromptBank.default()
        self._inst_category_cache: Dict[str, Dict[str, Dict[str, object]]] = {}

        if ids is None:
            self._entries = self._discover_entries(self.default_split)
        else:
            self._entries = self._resolve_ids(list(ids), default_split=self.default_split)

        if shuffle:
            rng = random.Random(seed)
            rng.shuffle(self._entries)

    def _discover_entries(self, split: str) -> List[Tuple[str, str]]:
        image_dir = self._split_dir(split) / "Image"
        if not image_dir.is_dir():
            return []
        stems = sorted({p.stem for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in _IMG_EXTS})
        return [(split, stem) for stem in stems]

    def _resolve_ids(self, ids: List[str], default_split: str) -> List[Tuple[str, str]]:
        entries: List[Tuple[str, str]] = []
        for raw in ids:
            s = str(raw).strip()
            if not s:
                continue

            split = default_split
            sample_id = s
            if "/" in s:
                maybe_split, rest = s.split("/", 1)
                split = _normalize_split(maybe_split)
                sample_id = rest.strip()

            if not sample_id:
                raise ValueError(f"Invalid LabPicsV1 id: {raw!r}")
            if self._find_image_path(split, sample_id) is None:
                raise FileNotFoundError(f"Could not resolve LabPicsV1 id={raw!r} under split={split!r}")
            entries.append((split, sample_id))
        return entries

    def _split_dir(self, split: str) -> Path:
        if split == "train":
            return self.root / "Train"
        if split == "test":
            return self.root / "Test"
        raise ValueError(f"Unsupported split: {split!r}")

    def _find_image_path(self, split: str, stem: str) -> Optional[Path]:
        image_dir = self._split_dir(split) / "Image"
        for ext in _IMG_EXTS:
            p = image_dir / f"{stem}{ext}"
            if p.exists():
                return p
        return None

    def _load_inst_category(self, split: str) -> Dict[str, Dict[str, object]]:
        if split in self._inst_category_cache:
            return self._inst_category_cache[split]

        path = self._split_dir(split) / "InstCategory.json"
        if not path.exists():
            self._inst_category_cache[split] = {}
            return self._inst_category_cache[split]

        with path.open("r", encoding="utf-8") as f:
            data = cast(Dict[str, Dict[str, object]], json.load(f))
        self._inst_category_cache[split] = data
        return data

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]

    def __getitem__(self, index: int) -> Dict[str, object]:
        split, stem = self._entries[index]
        split_dir = self._split_dir(split)
        image_path = self._find_image_path(split, stem)
        if image_path is None:
            raise FileNotFoundError(f"RGB image not found for LabPicsV1 sample: split={split}, id={stem}")

        instance_path = split_dir / "Instance" / f"{stem}.png"
        inst_category = self._load_inst_category(split).get(stem, {})
        material_cats = _as_dict_of_str_list(inst_category.get("MaterialCats"))
        part_cats = _as_dict_of_str_list(inst_category.get("PartCats"))
        multiphase_material = _as_int_list(inst_category.get("MultiPhaseMaterial"))
        multiphase_vessels = _as_int_list(inst_category.get("MultiPhaseVessels"))

        sample: Dict[str, object] = {
            "id": f"simple/{split}/{stem}",
            "dataset_variant": "simple",
            "split": split,
            "image_name": stem,
            "path": str(image_path),
            "rgb_image": image_path,
            "instance_mask_path": str(instance_path) if instance_path.exists() else None,
            "semantic_root": str(split_dir / "Semantic"),
            "material_categories": material_cats,
            "part_categories": part_cats,
            "multi_phase_material": multiphase_material,
            "multi_phase_vessels": multiphase_vessels,
        }
        sample["annotations"] = self._build_annotations(sample)
        return sample

    def _build_annotations(self, sample: Dict[str, object]) -> List[TLGAnnotation]:
        dataset = "labpicsv1"
        image_id = str(sample.get("id", ""))
        add = partial(make_tlg_annotation, self._prompt_bank, dataset=dataset, image_id=image_id)

        material_types = _material_types_present(_as_label_map(sample.get("material_categories")))
        material_answer = "NONE" if not material_types else ", ".join(material_types)
        multiphase_answer = "YES" if sample.get("multi_phase_material") else "NO"
        region = _select_region_candidate(sample)
        needs_liquid_targets = (
            self._prompt_bank.supports(dataset, "liquid_presence")
            or self._prompt_bank.supports(dataset, "liquid_grounding")
        )
        liquid_targets = _liquid_grounding_targets(sample) if needs_liquid_targets else []

        annotations: List[TLGAnnotation] = []
        if region is not None and self._prompt_bank.supports(dataset, "highlighted_region_type"):
            annotations.append(
                add(
                    task="highlighted_region_type",
                    answer=region["label"],
                    metadata={
                        "bbox_xyxy": region["bbox_xyxy"],
                        "target_box_label": region["label"],
                        "region_source": region["source"],
                        "region_instance_id": region.get("instance_id"),
                    },
                )
            )
        if self._prompt_bank.supports(dataset, "material_types_present"):
            annotations.append(
                add(
                    task="material_types_present",
                    answer=material_answer,
                    metadata={"material_types": material_types or ["NONE"]},
                )
            )
        if self._prompt_bank.supports(dataset, "liquid_presence"):
            annotations.append(
                add(
                    task="liquid_presence",
                    answer="YES" if liquid_targets else "NO",
                    metadata={
                        "source": "instance_or_semantic_liquid_mask",
                        "liquid_region_count": len(liquid_targets),
                    },
                )
            )
        if self._prompt_bank.supports(dataset, "multiphase_material_presence"):
            annotations.append(
                add(
                    task="multiphase_material_presence",
                    answer=multiphase_answer,
                    metadata={
                        "multi_phase_material": sample.get("multi_phase_material", []),
                        "multi_phase_vessels": sample.get("multi_phase_vessels", []),
                    },
                )
            )
        if self._prompt_bank.supports(dataset, "liquid_grounding"):
            annotations.append(
                add(
                    task="liquid_grounding",
                    answer=liquid_targets,
                    metadata={
                        "coordinate_scale": 1000,
                        "source": "instance_or_semantic_liquid_mask",
                        "liquid_region_count": len(liquid_targets),
                    },
                )
            )
        return annotations


def _select_region_candidate(sample: Dict[str, object]) -> Optional[Dict[str, object]]:
    mask_path = sample.get("instance_mask_path")
    if not mask_path:
        return None

    path = Path(str(mask_path))
    if not path.exists():
        return None

    with Image.open(path) as img:
        mask = img.convert("RGB")
        boxes = _instance_boxes(mask)

    material_cats = _as_label_map(sample.get("material_categories"))
    candidates: List[Dict[str, object]] = []

    for inst_id, bbox in sorted(boxes["vessel"].items()):
        candidates.append({"label": "VESSEL", "bbox_xyxy": bbox, "source": "vessel", "instance_id": inst_id})

    for inst_id, bbox in sorted(boxes["material"].items()):
        labels = material_cats.get(str(inst_id), [])
        mapped = sorted({_MATERIAL_LABEL_MAP[label] for label in labels if label in _MATERIAL_LABEL_MAP})
        if len(mapped) == 1:
            candidates.append(
                {"label": mapped[0], "bbox_xyxy": bbox, "source": "material", "instance_id": inst_id}
            )

    candidates = [c for c in candidates if str(c["label"]) in _REGION_CHOICES]

    key = str(sample.get("id", ""))
    idx = int(hashlib.sha256(key.encode("utf-8")).hexdigest(), 16) % (len(candidates) + 1)
    if idx == len(candidates):
        with Image.open(path) as img:
            bg_box = _find_background_box(img.convert("RGB"))
        if bg_box is not None:
            return {"label": "BACKGROUND", "bbox_xyxy": bg_box, "source": "background"}

    if not candidates:
        candidates = _semantic_region_candidates(sample)
        if not candidates:
            return None
        idx = int(hashlib.sha256(key.encode("utf-8")).hexdigest(), 16) % len(candidates)
        return candidates[idx]
    idx = idx % len(candidates)
    return candidates[idx]


def _semantic_region_candidates(sample: Dict[str, object]) -> List[Dict[str, object]]:
    semantic_root = sample.get("semantic_root")
    stem = sample.get("image_name")
    if not semantic_root or not stem:
        return []

    root = Path(str(semantic_root))
    out: List[Dict[str, object]] = []
    for class_dir_name, label in sorted(_SEMANTIC_LABEL_MAP.items()):
        path = root / class_dir_name / f"{stem}.png"
        if not path.exists():
            continue
        with Image.open(path) as img:
            mask = img.convert("L")
            lut = [255 if value not in (0, 254) else 0 for value in range(256)]
            bbox = mask.point(lut).getbbox()
        if bbox is None:
            continue
        x0, y0, x1, y1 = bbox
        out.append(
            {
                "label": label,
                "bbox_xyxy": [x0, y0, x1 - 1, y1 - 1],
                "source": "semantic",
                "semantic_class": class_dir_name,
            }
        )
    return out


def _liquid_grounding_targets(sample: Dict[str, object]) -> List[Dict[str, object]]:
    mask_path = sample.get("instance_mask_path")
    material_cats = _as_label_map(sample.get("material_categories"))
    targets: List[Dict[str, object]] = []

    if mask_path and Path(str(mask_path)).exists():
        with Image.open(Path(str(mask_path))) as img:
            mask = img.convert("RGB")
            width, height = mask.size
            material_channel = mask.split()[2]

        liquid_instance_ids = {
            int(inst_id)
            for inst_id, labels in material_cats.items()
            if any(_MATERIAL_LABEL_MAP.get(label) == "LIQUID" for label in labels)
        }
        if liquid_instance_ids:
            lut = [255 if value in liquid_instance_ids else 0 for value in range(256)]
            liquid_mask = material_channel.point(lut)
            if liquid_mask.getbbox() is not None:
                for bbox in _connected_component_boxes(liquid_mask):
                    targets.append({"bbox_2d": _xyxy_to_qwen_bbox(bbox, width, height), "label": "Liquid"})

    if targets:
        return targets

    semantic_root = sample.get("semantic_root")
    stem = sample.get("image_name")
    if not semantic_root or not stem:
        return []

    root = Path(str(semantic_root))
    for class_dir_name, label in sorted(_SEMANTIC_LABEL_MAP.items()):
        if label != "LIQUID":
            continue
        path = root / class_dir_name / f"{stem}.png"
        if not path.exists():
            continue
        with Image.open(path) as img:
            mask = img.convert("L")
            lut = [255 if value not in (0, 254) else 0 for value in range(256)]
            binary = mask.point(lut)
            if binary.getbbox() is None:
                continue
            for bbox in _connected_component_boxes(binary):
                targets.append({"bbox_2d": _xyxy_to_qwen_bbox(bbox, mask.width, mask.height), "label": "Liquid"})

    return targets


def _instance_boxes(mask: Image.Image) -> Dict[str, Dict[int, List[int]]]:
    boxes: Dict[str, Dict[int, List[int]]] = {"vessel": {}, "part": {}, "material": {}}
    channel_names = ("vessel", "part", "material")
    for channel, name in zip(mask.split(), channel_names):
        colors = channel.getcolors(maxcolors=256) or []
        for _count, inst_id_raw in colors:
            inst_id = int(inst_id_raw)
            if inst_id in (0, 254):
                continue
            lut = [255 if value == inst_id else 0 for value in range(256)]
            bbox = channel.point(lut).getbbox()
            if bbox is None:
                continue
            x0, y0, x1, y1 = bbox
            boxes[name][inst_id] = [x0, y0, x1 - 1, y1 - 1]
    return boxes


def _find_background_box(mask: Image.Image) -> Optional[List[int]]:
    width, height = mask.size
    patch = max(24, min(width, height) // 8)
    patch = min(patch, width, height)
    if patch <= 0:
        return None

    integral = [[0] * (width + 1)]
    data_iter = iter(mask.getdata())
    for _y in range(height):
        row = [0]
        running = 0
        for _x in range(width):
            pixel = next(data_iter)
            running += 0 if pixel == (0, 0, 0) else 1
            row.append(integral[-1][len(row)] + running)
        integral.append(row)

    step = max(1, patch // 2)
    for y0 in range(0, height - patch + 1, step):
        y1 = y0 + patch
        for x0 in range(0, width - patch + 1, step):
            x1 = x0 + patch
            total = integral[y1][x1] - integral[y0][x1] - integral[y1][x0] + integral[y0][x0]
            if total == 0:
                return [x0, y0, x1 - 1, y1 - 1]
    return None


def _connected_component_boxes(mask: Image.Image) -> List[List[int]]:
    try:
        import cv2  # type: ignore
        import numpy as np

        binary_arr = np.asarray(mask.convert("L")) > 0
        if not bool(binary_arr.any()):
            return []
        n_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
            binary_arr.astype("uint8"),
            connectivity=8,
        )
        boxes: List[List[int]] = []
        for label_idx in range(1, int(n_labels)):
            x0, y0, w, h, _area = [int(v) for v in stats[label_idx]]
            if w <= 0 or h <= 0:
                continue
            boxes.append([x0, y0, x0 + w - 1, y0 + h - 1])
        return boxes
    except Exception:
        pass

    binary = mask.convert("1")
    width, height = binary.size
    pixels = binary.load()
    visited = set()
    boxes: List[List[int]] = []

    for y in range(height):
        for x in range(width):
            if (x, y) in visited or not pixels[x, y]:
                continue

            stack = [(x, y)]
            visited.add((x, y))
            x0 = x1 = x
            y0 = y1 = y

            while stack:
                cx, cy = stack.pop()
                x0 = min(x0, cx)
                x1 = max(x1, cx)
                y0 = min(y0, cy)
                y1 = max(y1, cy)

                for nx, ny in (
                    (cx - 1, cy - 1),
                    (cx, cy - 1),
                    (cx + 1, cy - 1),
                    (cx - 1, cy),
                    (cx + 1, cy),
                    (cx - 1, cy + 1),
                    (cx, cy + 1),
                    (cx + 1, cy + 1),
                ):
                    if nx < 0 or ny < 0 or nx >= width or ny >= height:
                        continue
                    if (nx, ny) in visited or not pixels[nx, ny]:
                        continue
                    visited.add((nx, ny))
                    stack.append((nx, ny))

            boxes.append([x0, y0, x1, y1])

    return boxes


def _xyxy_to_qwen_bbox(bbox: List[int], width: int, height: int) -> List[int]:
    x0, y0, x1, y1 = [float(v) for v in bbox[:4]]
    return [
        _clamp_qwen_coord(1000.0 * x0 / max(1, width - 1)),
        _clamp_qwen_coord(1000.0 * y0 / max(1, height - 1)),
        _clamp_qwen_coord(1000.0 * x1 / max(1, width - 1)),
        _clamp_qwen_coord(1000.0 * y1 / max(1, height - 1)),
    ]


def _clamp_qwen_coord(value: float) -> int:
    return int(round(max(0.0, min(1000.0, float(value)))))


def _material_types_present(material_cats: Dict[str, List[str]]) -> List[str]:
    found = set()
    for labels in material_cats.values():
        for label in labels:
            mapped = _MATERIAL_LABEL_MAP.get(label)
            if mapped is not None:
                found.add(mapped)
    return [label for label in _MATERIAL_ORDER if label in found]


def _resolve_simple_root(root: Path) -> Path:
    candidate = root / "Simple"
    if candidate.is_dir():
        return candidate
    if root.name.lower() == "simple" and (root / "Train").is_dir() and (root / "Test").is_dir():
        return root
    if (root / "Train").is_dir() and (root / "Test").is_dir():
        return root
    raise FileNotFoundError(
        f"Could not resolve LabPicsV1 Simple root from: {root}. "
        "Expected dataset root containing Simple/ or a direct Simple root."
    )


def _normalize_split(split: str) -> str:
    s = str(split).strip().lower()
    if s in {"train", "test"}:
        return s
    raise ValueError(f"split must be 'train' or 'test'. Got: {split!r}")


def _as_dict_of_str_list(value: object) -> Dict[str, List[str]]:
    if not isinstance(value, dict):
        return {}
    out: Dict[str, List[str]] = {}
    for k, v in value.items():
        if not isinstance(v, list):
            continue
        labels = [str(x) for x in v if str(x).strip()]
        if labels:
            out[str(k)] = labels
    return out


def _as_label_map(value: object) -> Dict[str, List[str]]:
    return value if isinstance(value, dict) else {}


def _as_int_list(value: object) -> List[int]:
    if not isinstance(value, list):
        return []
    out: List[int] = []
    for item in value:
        try:
            out.append(int(item))
        except (TypeError, ValueError):
            continue
    return out

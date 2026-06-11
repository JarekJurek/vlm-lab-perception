from __future__ import annotations

import hashlib
import json
import os
import random
from functools import partial
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union, cast

from PIL import Image

from tlg import PromptBank, TLGAnnotation, make_tlg_annotation


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
    "Liquid": "LIQUID",
    "Suspension": "LIQUID",
    "Foam": "FOAM",
    "Gel": "GEL",
    "Solid": "SOLID",
    "SolidLargChunk": "SOLID",
    "Powder": "POWDER",
    "Granular": "GRANULAR",
    "Vapor": "VAPOR",
}
_MULTIPHASE_LABELS = {"Suspension", "Foam", "Gel", "Granular", "Powder", "Vapor"}
_VESSEL_TYPE_CHOICES = (
    "Beaker",
    "Flask",
    "RoundFlask",
    "Erlenmeyer",
    "Vial",
    "Tube",
    "Bottle",
    "Jar",
    "Cylinder",
    "SeparatoryFunnel",
    "Funnel",
    "Condenser",
    "Connector",
    "Dish",
    "Bowl",
    "HeatingVessel",
    "Syringe",
    "Pippete",
    "Burete",
    "ChromatographyColumn",
)


def _sample_names(split_dir: Path, *, require_files: bool) -> List[str]:
    names: List[str] = []
    with os.scandir(split_dir) as split_entries:
        for entry in split_entries:
            if not entry.is_dir():
                continue
            if not require_files:
                names.append(entry.name)
                continue
            with os.scandir(entry.path) as sample_entries:
                filenames = {child.name for child in sample_entries}
            if "Image.jpg" in filenames and "Data.json" in filenames:
                names.append(entry.name)
    return sorted(names)


class LabPicsChemistryDataset:
    """LabPics Chemistry loader for TLG VQA-style annotations."""

    def __init__(
        self,
        root: Union[str, Path],
        ids: Optional[Iterable[str]] = None,
        split: str = "all",
        prompt_bank: Optional[PromptBank] = None,
        shuffle: bool = False,
        seed: Optional[int] = None,
        validate_files: bool = True,
    ) -> None:
        root_path = Path(root)
        if not root_path.exists():
            raise FileNotFoundError(f"Dataset root not found: {root_path}")

        self.root = root_path
        self.default_split = _normalize_split(split)
        self._prompt_bank = prompt_bank or PromptBank.default()
        self._validate_files = validate_files

        if ids is None:
            self._entries = self._discover_entries(
                self.default_split,
                validate_files=self._validate_files,
            )
        else:
            self._entries = self._resolve_ids(list(ids), default_split=self.default_split)

        if shuffle:
            rng = random.Random(seed)
            rng.shuffle(self._entries)

    def _discover_entries(self, split: str, *, validate_files: bool = True) -> List[Tuple[str, str]]:
        splits = ("train", "test") if split == "all" else (split,)
        entries: List[Tuple[str, str]] = []
        for split_name in splits:
            split_dir = self._split_dir(split_name)
            if not split_dir.is_dir():
                continue
            entries.extend(
                (split_name, name)
                for name in _sample_names(split_dir, require_files=validate_files)
            )
        return entries

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
                raise ValueError(f"Invalid LabPics Chemistry id: {raw!r}")

            candidate_splits = ("train", "test") if split == "all" else (split,)
            matches = [
                (split_name, sample_id)
                for split_name in candidate_splits
                if self._sample_dir(split_name, sample_id).is_dir()
                and (self._sample_dir(split_name, sample_id) / "Image.jpg").exists()
            ]
            if not matches:
                raise FileNotFoundError(
                    f"Could not resolve LabPics Chemistry id={raw!r} under split={split!r}"
                )
            entries.append(matches[0])
        return entries

    def _split_dir(self, split: str) -> Path:
        if split == "train":
            return self.root / "Train"
        if split == "test":
            return self.root / "Test"
        raise ValueError(f"Unsupported split: {split!r}")

    def _sample_dir(self, split: str, sample_id: str) -> Path:
        return self._split_dir(split) / sample_id

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]

    def __getitem__(self, index: int) -> Dict[str, object]:
        split, sample_name = self._entries[index]
        sample_dir = self._sample_dir(split, sample_name)
        image_path = sample_dir / "Image.jpg"
        if not image_path.exists():
            raise FileNotFoundError(f"RGB image not found for LabPics Chemistry sample: {sample_dir}")

        data = _read_json(sample_dir / "Data.json")
        source_path = sample_dir / "Source.txt"
        sample: Dict[str, object] = {
            "id": f"labpics_chemistry/{split}/{sample_name}",
            "split": split,
            "image_name": sample_name,
            "sample_dir": str(sample_dir),
            "path": str(image_path),
            "rgb_image": image_path,
            "source_text": source_path.read_text(encoding="utf-8", errors="ignore") if source_path.exists() else "",
            "ignore_mask_path": str(sample_dir / "Ignore.png") if (sample_dir / "Ignore.png").exists() else None,
            "vessels": cast(Dict[str, object], data.get("Vessels", {})),
            "materials_parts": cast(Dict[str, object], data.get("MaterialsAndParts", {})),
        }
        sample["annotations"] = self._build_annotations(sample)
        return sample

    def _build_annotations(self, sample: Dict[str, object]) -> List[TLGAnnotation]:
        dataset = "labpics_chemistry"
        image_id = str(sample.get("id", ""))
        add = partial(make_tlg_annotation, self._prompt_bank, dataset=dataset, image_id=image_id)

        vessels = _as_entry_map(sample.get("vessels"))
        materials_parts = _as_entry_map(sample.get("materials_parts"))
        material_types = _material_types_present(sample)
        material_answer = "NONE" if not material_types else ", ".join(material_types)
        multiphase_answer = "YES" if _has_multiphase_material(materials_parts) else "NO"
        region = _select_region_candidate(sample)
        vessel_region = _select_vessel_type_candidate(sample)
        liquid_targets = (
            _liquid_grounding_targets(sample)
            if self._prompt_bank.supports(dataset, "liquid_grounding")
            else []
        )

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
        if self._prompt_bank.supports(dataset, "multiphase_material_presence"):
            annotations.append(
                add(
                    task="multiphase_material_presence",
                    answer=multiphase_answer,
                    metadata={
                        "source": "material_classes_and_relations",
                        "positive_labels": sorted(_MULTIPHASE_LABELS),
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
                        "source": "material_instance_or_semantic_liquid_mask",
                        "liquid_region_count": len(liquid_targets),
                    },
                )
            )
        if self._prompt_bank.supports(dataset, "container_count"):
            annotations.append(
                add(
                    task="container_count",
                    answer=len(vessels),
                    metadata={"vessel_count": len(vessels)},
                )
            )
        if vessel_region is not None and self._prompt_bank.supports(dataset, "container_type"):
            annotations.append(
                add(
                    task="container_type",
                    answer=vessel_region["label"],
                    metadata={
                        "bbox_xyxy": vessel_region["bbox_xyxy"],
                        "target_box_label": vessel_region["label"],
                        "region_source": "vessel",
                        "region_instance_id": vessel_region.get("instance_id"),
                    },
                )
            )
        return annotations


def _read_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        return cast(Dict[str, object], json.load(f))


def _select_region_candidate(sample: Dict[str, object]) -> Optional[Dict[str, object]]:
    candidates: List[Dict[str, object]] = []
    for inst_id, entry in sorted(_as_entry_map(sample.get("vessels")).items()):
        bbox = _entry_bbox(sample, entry)
        if bbox is not None:
            candidates.append({"label": "VESSEL", "bbox_xyxy": bbox, "source": "vessel", "instance_id": inst_id})

    for inst_id, entry in sorted(_as_entry_map(sample.get("materials_parts")).items()):
        bbox = _entry_bbox(sample, entry)
        if bbox is None:
            continue
        label = _region_label_for_material_or_part(entry)
        if label in _REGION_CHOICES:
            candidates.append({"label": label, "bbox_xyxy": bbox, "source": "material_part", "instance_id": inst_id})

    key = str(sample.get("id", ""))
    idx = _stable_index(key, len(candidates) + 1)
    if idx == len(candidates):
        bg_box = _find_background_box(sample)
        if bg_box is not None:
            return {"label": "BACKGROUND", "bbox_xyxy": bg_box, "source": "background"}

    if not candidates:
        candidates = _semantic_region_candidates(sample)
        if not candidates:
            return None
        return candidates[_stable_index(key, len(candidates))]
    return candidates[idx % len(candidates)]


def _select_vessel_type_candidate(sample: Dict[str, object]) -> Optional[Dict[str, object]]:
    candidates: List[Dict[str, object]] = []
    for inst_id, entry in sorted(_as_entry_map(sample.get("vessels")).items()):
        bbox = _entry_bbox(sample, entry)
        label = _vessel_type_label(entry)
        if bbox is not None and label is not None:
            candidates.append({"label": label, "bbox_xyxy": bbox, "instance_id": inst_id})
    if not candidates:
        return None
    return candidates[_stable_index(str(sample.get("id", "")) + ":container_type", len(candidates))]


def _entry_bbox(sample: Dict[str, object], entry: Dict[str, object]) -> Optional[List[int]]:
    sample_dir = Path(str(sample.get("sample_dir", "")))
    mask_rel = str(entry.get("MaskFilePath", "")).strip()
    if not mask_rel:
        return None
    mask_path = sample_dir / mask_rel.lstrip("/")
    if not mask_path.exists():
        return None
    with Image.open(mask_path) as img:
        bbox = _binary_bbox(img.convert("L"))
    return bbox


def _binary_bbox(mask: Image.Image) -> Optional[List[int]]:
    binary = mask.point(lambda p: 255 if p not in (0, 254) else 0)
    bbox = binary.getbbox()
    if bbox is None:
        return None
    x0, y0, x1, y1 = bbox
    return [x0, y0, x1 - 1, y1 - 1]


def _region_label_for_material_or_part(entry: Dict[str, object]) -> str:
    if bool(entry.get("IsPart")):
        return "SOLID"
    labels = _as_str_list(entry.get("MaterialType_ClassNames")) + _as_str_list(entry.get("All_ClassNames"))
    for desired in _MATERIAL_ORDER:
        for label in labels:
            if _MATERIAL_LABEL_MAP.get(label) == desired:
                return desired
    return "SOLID"


def _material_types_present(sample: Dict[str, object]) -> List[str]:
    found = set()
    for entry in _as_entry_map(sample.get("materials_parts")).values():
        if bool(entry.get("IsPart")):
            continue
        for label in _as_str_list(entry.get("MaterialType_ClassNames")) + _as_str_list(entry.get("All_ClassNames")):
            mapped = _MATERIAL_LABEL_MAP.get(label)
            if mapped is not None:
                found.add(mapped)

    sample_dir = Path(str(sample.get("sample_dir", "")))
    sem_root = sample_dir / "SemanticMaps" / "FullImage"
    if sem_root.is_dir():
        for raw_label, mapped in _MATERIAL_LABEL_MAP.items():
            if (sem_root / f"{raw_label}.png").exists():
                found.add(mapped)

    return [label for label in _MATERIAL_ORDER if label in found]


def _has_multiphase_material(materials_parts: Dict[str, Dict[str, object]]) -> bool:
    for entry in materials_parts.values():
        if bool(entry.get("IsPart")):
            continue
        labels = set(_as_str_list(entry.get("MaterialType_ClassNames")) + _as_str_list(entry.get("All_ClassNames")))
        if labels & _MULTIPHASE_LABELS:
            return True
        if _as_int_list(entry.get("ContainMaterials_Indx")) or _as_int_list(entry.get("InsideOfMaterials_Indx")):
            return True
    return False


def _liquid_grounding_targets(sample: Dict[str, object]) -> List[Dict[str, object]]:
    sample_dir = Path(str(sample.get("sample_dir", "")))
    targets: List[Dict[str, object]] = []
    image_size = _image_size(sample)

    for entry in _as_entry_map(sample.get("materials_parts")).values():
        if bool(entry.get("IsPart")):
            continue
        labels = _as_str_list(entry.get("MaterialType_ClassNames")) + _as_str_list(entry.get("All_ClassNames"))
        if not any(_MATERIAL_LABEL_MAP.get(label) == "LIQUID" for label in labels):
            continue
        mask_rel = str(entry.get("MaskFilePath", "")).strip()
        mask_path = sample_dir / mask_rel.lstrip("/")
        if not mask_path.exists():
            continue
        with Image.open(mask_path) as img:
            binary = img.convert("L").point(lambda p: 255 if p not in (0, 254) else 0)
            for bbox in _connected_component_boxes(binary):
                targets.append({"bbox_2d": _xyxy_to_qwen_bbox(bbox, binary.width, binary.height), "label": "Liquid"})

    if targets:
        return targets

    semantic_path = sample_dir / "SemanticMaps" / "FullImage" / "Liquid.png"
    if semantic_path.exists():
        with Image.open(semantic_path) as img:
            binary = img.convert("L").point(lambda p: 255 if p not in (0, 254) else 0)
            for bbox in _connected_component_boxes(binary):
                targets.append({"bbox_2d": _xyxy_to_qwen_bbox(bbox, binary.width, binary.height), "label": "Liquid"})

    if targets or image_size is None:
        return targets
    return []


def _semantic_region_candidates(sample: Dict[str, object]) -> List[Dict[str, object]]:
    sample_dir = Path(str(sample.get("sample_dir", "")))
    sem_root = sample_dir / "SemanticMaps" / "FullImage"
    if not sem_root.is_dir():
        return []

    class_to_region = {
        "Vessel": "VESSEL",
        "Liquid": "LIQUID",
        "Foam": "FOAM",
        "Gel": "GEL",
        "Solid": "SOLID",
        "SolidLargChunk": "SOLID",
        "Powder": "POWDER",
        "Granular": "GRANULAR",
        "Vapor": "VAPOR",
    }
    out: List[Dict[str, object]] = []
    for class_name, label in sorted(class_to_region.items()):
        path = sem_root / f"{class_name}.png"
        if not path.exists():
            continue
        with Image.open(path) as img:
            bbox = _binary_bbox(img.convert("L"))
        if bbox is not None:
            out.append({"label": label, "bbox_xyxy": bbox, "source": "semantic", "semantic_class": class_name})
    return out


def _find_background_box(sample: Dict[str, object]) -> Optional[List[int]]:
    image_path = sample.get("rgb_image")
    if not image_path:
        return None
    with Image.open(Path(str(image_path))) as img:
        width, height = img.size

    occupied_boxes = [
        bbox
        for entry in list(_as_entry_map(sample.get("vessels")).values()) + list(_as_entry_map(sample.get("materials_parts")).values())
        for bbox in [_entry_bbox(sample, entry)]
        if bbox is not None
    ]

    patch = max(24, min(width, height) // 8)
    patch = min(patch, width, height)
    if patch <= 0:
        return None
    step = max(1, patch // 2)
    for y0 in range(0, height - patch + 1, step):
        for x0 in range(0, width - patch + 1, step):
            candidate = [x0, y0, x0 + patch - 1, y0 + patch - 1]
            if not any(_boxes_overlap(candidate, occupied) for occupied in occupied_boxes):
                return candidate
    return None


def _boxes_overlap(a: List[int], b: List[int]) -> bool:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return ax0 <= bx1 and ax1 >= bx0 and ay0 <= by1 and ay1 >= by0


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


def _vessel_type_label(entry: Dict[str, object]) -> Optional[str]:
    labels = _as_str_list(entry.get("VesselType_ClassNames")) + _as_str_list(entry.get("All_ClassNames"))
    for label in _VESSEL_TYPE_CHOICES:
        if label in labels:
            return label
    return None


def _image_size(sample: Dict[str, object]) -> Optional[Tuple[int, int]]:
    path = sample.get("rgb_image")
    if not path:
        return None
    with Image.open(Path(str(path))) as img:
        return img.size


def _stable_index(key: str, length: int) -> int:
    if length <= 0:
        return 0
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest(), 16) % length


def _normalize_split(split: str) -> str:
    s = str(split).strip().lower()
    if s in {"train", "test", "all"}:
        return s
    raise ValueError(f"split must be 'train', 'test', or 'all'. Got: {split!r}")


def _as_entry_map(value: object) -> Dict[str, Dict[str, object]]:
    if not isinstance(value, dict):
        return {}
    return {str(k): cast(Dict[str, object], v) for k, v in value.items() if isinstance(v, dict)}


def _as_str_list(value: object) -> List[str]:
    if not isinstance(value, list):
        return []
    return [str(x) for x in value if str(x).strip()]


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

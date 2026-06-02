from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

import numpy as np
from PIL import Image

from functools import partial

from tlg import PromptBank, TLGAnnotation, make_tlg_annotation, simplify_container_type
from tlg.annotations import discretize_fill_level_ordinal


_TEXT_KEYS = {
    "color_liquid": "Input_color_liquid.txt",
    "vessel": "Input_vessel.txt",
    "vol_liquid": "Input_vol_liquid.txt",
    "vol_vessel": "Input_vol_vessel.txt",
}

_ARRAY_KEYS = {
    "rgb": "Input_RGBImage.npy",
    "depth_map": "Input_DepthMap.npy",
    "content_depth": "Input_ContentDepth.npy",
    "content_depth_segmented": "Input_ContentDepth_segmented.npy",
    "content_mask_clean": "Input_ContentMaskClean.npy",
    "empty_vessel_depth": "Input_EmptyVessel_Depth.npy",
    "empty_vessel_depth_segmented": "Input_EmptyVessel_Depth_segmented.npy",
    "vessel_mask": "Input_VesselMask.npy",
}

_IMAGE_KEYS = {
    "rgb_image": "Input_RGBImage.png",
}


class LabLiquidVisionDataset:
    """Minimal folder-based dataset loader for LabLiquidVision."""

    def __init__(
        self,
        root: Union[str, Path],
        ids: Optional[Iterable[str]] = None,
        rgb_only: bool = False,
        prompt_bank: Optional[PromptBank] = None,
        shuffle: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        root_path = Path(root)
        if not root_path.exists():
            raise FileNotFoundError(f"Dataset root not found: {root_path}")

        # Accept both:
        # - base folder:   .../LabLiquidVision/ (contains `processed/`)
        # - processed dir: .../LabLiquidVision/processed/
        processed = root_path / "processed"
        if root_path.name != "processed" and processed.is_dir():
            self.root = processed
        else:
            self.root = root_path

        self._rgb_only = rgb_only
        self._prompt_bank = prompt_bank or PromptBank.default()
        self._seed = seed

        ids_list = list(ids) if ids is not None else self._discover_ids()
        if shuffle:
            rng = np.random.default_rng(seed)
            ids_list = list(rng.permutation(ids_list))

        self._ids = ids_list

    def _discover_ids(self) -> List[str]:
        return sorted([p.name for p in self.root.iterdir() if p.is_dir()])

    def __len__(self) -> int:
        return len(self._ids)

    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]

    def __getitem__(self, index: int) -> Dict[str, object]:
        sample_id = self._ids[index]
        sample_dir = self.root / sample_id
        if not sample_dir.is_dir():
            raise FileNotFoundError(f"Sample directory not found: {sample_dir}")

        sample: Dict[str, object] = {
            "id": sample_id,
            "path": str(sample_dir),
        }

        for key, filename in _TEXT_KEYS.items():
            file_path = sample_dir / filename
            if file_path.exists():
                sample[key] = _read_text_value(file_path)

        sample["annotations"] = self._build_annotations(sample_id, sample)

        if self._rgb_only:
            rgb_path = sample_dir / _IMAGE_KEYS["rgb_image"]
            if rgb_path.exists():
                sample["rgb_image"] = rgb_path
                return sample

            rgb_npy_path = sample_dir / _ARRAY_KEYS["rgb"]
            if rgb_npy_path.exists():
                sample["rgb"] = np.load(rgb_npy_path)
                return sample

            raise FileNotFoundError(
                f"Required RGB image not found: {rgb_path} or {rgb_npy_path}"
            )

        for key, filename in _ARRAY_KEYS.items():
            file_path = sample_dir / filename
            if file_path.exists():
                sample[key] = np.load(file_path)

        for key, filename in _IMAGE_KEYS.items():
            file_path = sample_dir / filename
            if file_path.exists():
                sample[key] = file_path

        return sample

    def _build_annotations(self, image_id: str, sample: Dict[str, object]) -> List[TLGAnnotation]:
        dataset = "labliquidvision"
        annotations: List[TLGAnnotation] = []

        add = partial(
            make_tlg_annotation,
            self._prompt_bank,
            dataset=dataset,
            image_id=image_id,
        )

        vessel = sample.get("vessel")
        color_liquid = sample.get("color_liquid")
        vol_liquid = _as_float(sample.get("vol_liquid"))
        vol_vessel = _as_float(sample.get("vol_vessel"))

        if self._prompt_bank.supports(dataset, "container_count"):
            annotations.append(
                add(
                    task="container_count",
                    answer=1,
                    metadata={"container_count": 1, "source": "single_vessel_sample"},
                )
            )

        if vol_liquid is not None and vol_vessel is not None and vol_vessel > 0:
            fill_percentage = (vol_liquid / vol_vessel) * 100.0
            fill_percentage = float(max(0.0, min(100.0, fill_percentage)))
            if self._prompt_bank.supports(dataset, "fill_percentage"):
                annotations.append(
                    add(
                        task="fill_percentage",
                        answer=round(fill_percentage, 1),
                        metadata={
                            "vol_liquid": vol_liquid,
                            "vol_vessel": vol_vessel,
                            "derived_fill_percentage": round(fill_percentage, 6),
                        },
                    )
                )
            if self._prompt_bank.supports(dataset, "fill_level_ordinal"):
                annotations.append(
                    add(
                        task="fill_level_ordinal",
                        answer=discretize_fill_level_ordinal(fill_percentage),
                        metadata={"derived_fill_percentage": round(fill_percentage, 6)},
                    )
                )

        if isinstance(vessel, str) and vessel.strip():
            vessel_label = vessel.strip()
            if self._prompt_bank.supports(dataset, "container_type"):
                annotations.append(add(task="container_type", answer=vessel_label, metadata={"vessel": vessel_label}))
            if self._prompt_bank.supports(dataset, "container_type_simple"):
                annotations.append(
                    add(
                        task="container_type_simple",
                        answer=simplify_container_type(vessel_label),
                        metadata={"vessel": vessel_label, "source_task": "container_type"},
                    )
                )

        if (
            isinstance(color_liquid, str)
            and color_liquid.strip()
            and self._prompt_bank.supports(dataset, "liquid_color")
        ):
            annotations.append(
                add(task="liquid_color", answer=color_liquid.strip(), metadata={"color_liquid": color_liquid.strip()})
            )

        if self._prompt_bank.supports(dataset, "liquid_grounding"):
            targets = _liquid_mask_grounding_targets(Path(str(sample["path"])))
            if targets is not None:
                annotations.append(
                    add(
                        task="liquid_grounding",
                        answer=targets,
                        metadata={
                            "coordinate_scale": 1000,
                            "source": "content_mask_clean",
                            "liquid_region_count": len(targets),
                        },
                    )
                )

        return annotations


def _read_text_value(path: Path) -> Union[str, float]:
    raw = path.read_text(encoding="utf-8").strip()
    try:
        return float(raw)
    except ValueError:
        return raw


def _as_float(value: object) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _liquid_mask_grounding_targets(sample_dir: Path) -> Optional[List[Dict[str, object]]]:
    bbox = _mask_bbox(sample_dir / "Input_ContentMaskClean.png")
    if bbox is None:
        bbox = _mask_bbox(sample_dir / "Input_ContentMaskClean.npy")
    if bbox is None:
        return None
    if not bbox:
        return []
    return [{"bbox_2d": bbox, "label": "Liquid"}]


def _mask_bbox(path: Path) -> Optional[List[int]]:
    if not path.exists():
        return None

    if path.suffix.lower() == ".npy":
        arr = np.load(path)
        mask = np.asarray(arr) > 0
        if mask.ndim > 2:
            mask = np.any(mask, axis=-1)
        ys, xs = np.nonzero(mask)
        if len(xs) == 0 or len(ys) == 0:
            return []
        height, width = mask.shape[:2]
        return _xyxy_to_qwen_bbox(float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()), width, height)

    with Image.open(path) as img:
        mask_img = img.convert("L")
        bbox = mask_img.point(lambda p: 255 if p > 0 else 0).getbbox()
        if bbox is None:
            return []
        x0, y0, x1, y1 = bbox
        return _xyxy_to_qwen_bbox(float(x0), float(y0), float(x1 - 1), float(y1 - 1), mask_img.width, mask_img.height)


def _xyxy_to_qwen_bbox(x0: float, y0: float, x1: float, y1: float, width: int, height: int) -> List[int]:
    return [
        _clamp_qwen_coord(1000.0 * x0 / max(1, width - 1)),
        _clamp_qwen_coord(1000.0 * y0 / max(1, height - 1)),
        _clamp_qwen_coord(1000.0 * x1 / max(1, width - 1)),
        _clamp_qwen_coord(1000.0 * y1 / max(1, height - 1)),
    ]


def _clamp_qwen_coord(value: float) -> int:
    return int(round(max(0.0, min(1000.0, float(value)))))

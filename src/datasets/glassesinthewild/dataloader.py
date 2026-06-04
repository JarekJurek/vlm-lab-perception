from __future__ import annotations

import json
from functools import partial
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union, cast

import numpy as np

from tlg import PromptBank, TLGAnnotation, make_tlg_annotation
from tlg.annotations import discretize_fill_level_ordinal


class GlassesInTheWildDataset:
    """Minimal loader for the Glasses in the Wild keypoint dataset."""

    def __init__(
        self,
        root: Union[str, Path],
        ids: Optional[Iterable[int]] = None,
        prompt_bank: Optional[PromptBank] = None,
        shuffle: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.root}")

        self._prompt_bank = prompt_bank or PromptBank.default()

        self.images_dir = self.root / "images"
        if not self.images_dir.is_dir():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")

        annotations_path = self.images_dir / "annotations.json"
        if not annotations_path.exists():
            raise FileNotFoundError(f"Annotation file not found: {annotations_path}")

        with annotations_path.open("r", encoding="utf-8") as f:
            annotations_data = cast(Dict[str, Any], json.load(f))

        self._keypoint_names = self._extract_keypoint_names(annotations_data)

        images = cast(List[Dict[str, Any]], annotations_data.get("images", []))
        images_by_id = {int(img["id"]): img for img in images}
        annotations = cast(List[Dict[str, Any]], annotations_data.get("annotations", []))

        if ids is not None:
            selected_ids = {int(i) for i in ids}
            annotations = [a for a in annotations if int(a.get("image_id", -1)) in selected_ids]

        self._samples: List[Dict[str, object]] = []
        for ann in annotations:
            image_id = int(ann["image_id"])
            image_info = images_by_id.get(image_id)
            if image_info is None:
                continue

            file_name = image_info["file_name"]
            image_path = self.images_dir / file_name
            if not image_path.exists():
                continue

            keypoints = ann.get("keypoints", [])
            named_keypoints = self._named_keypoints(keypoints)

            sample: Dict[str, object] = {
                "id": str(image_id),
                "path": str(image_path),
                "rgb_image": image_path,
                "image_name": file_name,
                "width": int(image_info.get("width", 0)),
                "height": int(image_info.get("height", 0)),
                "category_id": int(ann.get("category_id", 0)),
                "num_keypoints": int(ann.get("num_keypoints", 0)),
                "keypoints": named_keypoints,
            }

            fill_percentage = _estimate_fill_percentage(named_keypoints)
            if fill_percentage is not None:
                sample["fill_percentage"] = fill_percentage

            sample["annotations"] = self._build_annotations(str(image_id), sample)

            self._samples.append(sample)
            
            if shuffle:
                rng = np.random.default_rng(seed)
                self._samples = [
                    self._samples[i]
                    for i in rng.permutation(len(self._samples))
                ]

    def _build_annotations(self, image_id: str, sample: Dict[str, object]) -> List[TLGAnnotation]:
        dataset = "glasses"
        annotations: List[TLGAnnotation] = []

        add = partial(
            make_tlg_annotation,
            self._prompt_bank,
            dataset=dataset,
            image_id=image_id,
        )

        fill_percentage = sample.get("fill_percentage")
        fill = float(fill_percentage) if isinstance(fill_percentage, (int, float)) else None

        if fill is not None:
            fill_clamped = float(max(0.0, min(100.0, fill)))
            if self._prompt_bank.supports(dataset, "fill_percentage"):
                annotations.append(
                    add(
                        task="fill_percentage",
                        answer=round(fill_clamped, 1),
                        metadata={"derived_fill_percentage": round(fill_clamped, 6)},
                    )
                )
            if self._prompt_bank.supports(dataset, "fill_level_ordinal"):
                annotations.append(
                    add(
                        task="fill_level_ordinal",
                        answer=discretize_fill_level_ordinal(fill_clamped),
                        metadata={"derived_fill_percentage": round(fill_clamped, 6)},
                    )
                )
        return annotations

    def _extract_keypoint_names(self, annotations_data: Dict[str, Any]) -> List[str]:
        categories = cast(List[Dict[str, Any]], annotations_data.get("categories", []))
        if not categories:
            return []
        first_category = categories[0]
        names = first_category.get("keypoints", [])
        if not isinstance(names, list):
            return []
        return [str(name) for name in names]

    def _named_keypoints(self, raw_keypoints: List[float]) -> Dict[str, Dict[str, float]]:
        named: Dict[str, Dict[str, float]] = {}
        for i, name in enumerate(self._keypoint_names):
            base = i * 3
            if base + 2 >= len(raw_keypoints):
                break
            named[name] = {
                "x": float(raw_keypoints[base]),
                "y": float(raw_keypoints[base + 1]),
                "v": float(raw_keypoints[base + 2]),
            }
        return named

    def __len__(self) -> int:
        return len(self._samples)

    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]

    def __getitem__(self, index: int) -> Dict[str, object]:
        return self._samples[index]


def _estimate_fill_percentage(keypoints: Dict[str, Dict[str, float]]) -> Optional[float]:
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
    level = max(0.0, min(1.0, level))
    return level * 100.0

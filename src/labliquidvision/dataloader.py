from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

import numpy as np


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
    ) -> None:
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.root}")
        self._ids = list(ids) if ids is not None else self._discover_ids()
        self._rgb_only = rgb_only

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


def _read_text_value(path: Path) -> Union[str, float]:
    raw = path.read_text(encoding="utf-8").strip()
    try:
        return float(raw)
    except ValueError:
        return raw

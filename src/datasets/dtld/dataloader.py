from __future__ import annotations

import json
from functools import partial
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union, cast

from tlg import PromptBank, TLGAnnotation, make_tlg_annotation


_DTLD_OBJECT_NAMES = {
    15: "T25 Flask",
    16: "T75 Flask",
    17: "T175 Flask",
    19: "G-rex",
}


class DTLDDataset:
    """Minimal folder-based dataset loader for DTLD."""

    def __init__(
        self,
        root: Union[str, Path],
        scene_ids: Optional[Iterable[str]] = None,
        rgb_only: bool = True,
        prompt_bank: Optional[PromptBank] = None,
    ) -> None:
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.root}")

        self._scene_ids = list(scene_ids) if scene_ids is not None else self._discover_scene_ids()
        self._rgb_only = rgb_only
        self._prompt_bank = prompt_bank or PromptBank.default()
        self._scene_cache: Dict[str, Dict[str, object]] = {}
        self._entries = self._build_entries()

    def _discover_scene_ids(self) -> List[str]:
        return sorted([p.name for p in self.root.iterdir() if p.is_dir()])

    def _load_scene_annotations(self, scene_id: str) -> Dict[str, object]:
        if scene_id in self._scene_cache:
            return self._scene_cache[scene_id]

        scene_dir = self.root / scene_id
        liquid_path = scene_dir / "scene_gt_liquid.json"
        info_path = scene_dir / "scene_gt_info.json"

        if not liquid_path.exists():
            raise FileNotFoundError(f"Missing annotation file: {liquid_path}")

        with liquid_path.open("r", encoding="utf-8") as f:
            liquid = cast(Dict[str, List[Dict[str, Any]]], json.load(f))

        info = None
        if info_path.exists():
            try:
                with info_path.open("r", encoding="utf-8") as f:
                    info = cast(Dict[str, List[Dict[str, Any]]], json.load(f))
            except json.JSONDecodeError:
                info = None

        scene_data = {"liquid": liquid, "info": info}
        self._scene_cache[scene_id] = scene_data
        return scene_data

    def _build_entries(self) -> List[Tuple[str, int, int]]:
        entries: List[Tuple[str, int, int]] = []
        for scene_id in self._scene_ids:
            scene_dir = self.root / scene_id
            liquid_path = scene_dir / "scene_gt_liquid.json"
            if not liquid_path.exists():
                continue

            scene_data = self._load_scene_annotations(scene_id)
            liquid = cast(Dict[str, List[Dict[str, Any]]], scene_data["liquid"])
            for frame_key, objects in liquid.items():
                frame_id = int(frame_key)
                for obj_idx in range(len(objects)):
                    entries.append((scene_id, frame_id, obj_idx))
        return entries

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]

    def __getitem__(self, index: int) -> Dict[str, object]:
        scene_id, frame_id, obj_idx = self._entries[index]
        scene_dir = self.root / scene_id

        rgb_path = scene_dir / "rgb" / f"{frame_id:06d}.png"
        if not rgb_path.exists():
            raise FileNotFoundError(f"RGB frame not found: {rgb_path}")

        sample: Dict[str, object] = {
            "id": f"{scene_id}/{frame_id:06d}/{obj_idx:06d}",
            "scene_id": scene_id,
            "frame_id": frame_id,
            "obj_idx": obj_idx,
            "rgb_image": rgb_path,
            "path": str(scene_dir),
        }

        scene_data = self._load_scene_annotations(scene_id)
        liquid = cast(Dict[str, List[Dict[str, Any]]], scene_data["liquid"])
        frame_liquid = liquid[str(frame_id)][obj_idx]

        sample["obj_id"] = int(frame_liquid.get("obj_id", -1))
        if "liquid_h" in frame_liquid:
            sample["liquid_h"] = frame_liquid["liquid_h"]
        if "liquid_label" in frame_liquid:
            sample["liquid_label"] = frame_liquid["liquid_label"]

        info = cast(Optional[Dict[str, List[Dict[str, Any]]]], scene_data.get("info"))
        if info is not None and str(frame_id) in info and obj_idx < len(info[str(frame_id)]):
            frame_info = info[str(frame_id)][obj_idx]
            if "bbox_obj" in frame_info:
                sample["bbox_obj"] = frame_info["bbox_obj"]
            if "bbox_visib" in frame_info:
                sample["bbox_visib"] = frame_info["bbox_visib"]
            if "visib_fract" in frame_info:
                sample["visib_fract"] = frame_info["visib_fract"]

        mask_visib_path = scene_dir / "mask_visib" / f"{frame_id:06d}_{obj_idx:06d}.png"
        if mask_visib_path.exists():
            sample["mask_image"] = mask_visib_path
            sample["mask_visib_image"] = mask_visib_path

        mask_full_path = scene_dir / "mask" / f"{frame_id:06d}_{obj_idx:06d}.png"
        if mask_full_path.exists():
            sample["mask_full_image"] = mask_full_path

        liquid_label_path = scene_dir / "liquid_label" / f"{frame_id:06d}_{obj_idx:06d}.png"
        if liquid_label_path.exists():
            sample["liquid_label_image"] = liquid_label_path

        mask_visib_liquid_path = scene_dir / "mask_visib_liquid" / f"{frame_id:06d}_{obj_idx:06d}.png"
        if mask_visib_liquid_path.exists():
            sample["mask_visib_liquid_image"] = mask_visib_liquid_path

        if not self._rgb_only:
            depth_path = scene_dir / "depth" / f"{frame_id:06d}.png"
            if depth_path.exists():
                sample["depth_image"] = depth_path

        sample["annotations"] = self._build_annotations(sample)

        return sample

    def _build_annotations(self, sample: Dict[str, object]) -> List[TLGAnnotation]:
        dataset = "dtld"
        image_id = str(sample.get("id", ""))
        annotations: List[TLGAnnotation] = []

        add = partial(
            make_tlg_annotation,
            self._prompt_bank,
            dataset=dataset,
            image_id=image_id,
        )

        obj_id = sample.get("obj_id")
        obj_id_int = int(obj_id) if isinstance(obj_id, int) or (isinstance(obj_id, float) and obj_id.is_integer()) else None
        liquid_h = sample.get("liquid_h")
        liquid_h_f = float(liquid_h) if isinstance(liquid_h, (int, float)) else None

        if obj_id_int is not None and obj_id_int in _DTLD_OBJECT_NAMES:
            annotations.append(
                add(task="container_type", answer=_DTLD_OBJECT_NAMES[obj_id_int], metadata={"obj_id": obj_id_int})
            )

        if liquid_h_f is not None:
            annotations.append(
                add(task="fill_height_mm", answer=round(liquid_h_f, 2), metadata={"liquid_h": liquid_h_f})
            )
            annotations.append(
                add(
                    task="liquid_presence",
                    answer="YES" if liquid_h_f > 0 else "NO",
                    metadata={"liquid_h": liquid_h_f},
                )
            )

        return annotations

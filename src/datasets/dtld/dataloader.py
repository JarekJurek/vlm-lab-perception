from __future__ import annotations

import json
import warnings
from functools import partial
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union, cast

import numpy as np

from tlg import PromptBank, TLGAnnotation, make_tlg_annotation, simplify_container_type
from tlg.target_overlay import is_valid_bbox_xywh


_DTLD_OBJECT_NAMES = {
    15: "T25 Flask",
    16: "T75 Flask",
    17: "T175 Flask",
    19: "G-rex",
}

# Treat a DTLD target as occluded when its box is substantially on top of another
# annotated object box. The overlap is measured against the smaller box.
_DEFAULT_OCCLUDED_BBOX_OVERLAP_THRESHOLD = 0.6


def _as_int(value: object) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _has_visible_area(info: Dict[str, Any]) -> bool:
    visib_fract = info.get("visib_fract")
    if isinstance(visib_fract, (int, float)):
        return float(visib_fract) > 0.0

    bbox = info.get("bbox_visib")
    if isinstance(bbox, list) and len(bbox) >= 4:
        w = bbox[2]
        h = bbox[3]
        if isinstance(w, (int, float)) and isinstance(h, (int, float)):
            return float(w) > 0.0 and float(h) > 0.0

    return True


def _bbox_xywh(bbox: object) -> Optional[Tuple[float, float, float, float]]:
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return None
    try:
        x, y, w, h = [float(v) for v in bbox[:4]]
    except (TypeError, ValueError):
        return None
    if w <= 0.0 or h <= 0.0:
        return None
    return x, y, w, h


def _bbox_area(box: Tuple[float, float, float, float]) -> float:
    return box[2] * box[3]


def _bbox_intersection_area(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
) -> float:
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ax1 = ax0 + aw
    ay1 = ay0 + ah
    bx1 = bx0 + bw
    by1 = by0 + bh
    return max(0.0, min(ax1, bx1) - max(ax0, bx0)) * max(0.0, min(ay1, by1) - max(ay0, by0))


def _max_bbox_overlap_fraction(
    target_bbox: object,
    infos: Optional[List[Dict[str, Any]]],
    *,
    target_idx: int,
    bbox_key: str,
) -> float:
    target = _bbox_xywh(target_bbox)
    if target is None or infos is None:
        return 0.0

    target_area = _bbox_area(target)
    max_overlap = 0.0
    for other_idx, other_info in enumerate(infos):
        if other_idx == target_idx:
            continue
        other = _bbox_xywh(other_info.get(bbox_key))
        if other is None:
            continue
        denom = min(target_area, _bbox_area(other))
        if denom <= 0.0:
            continue
        max_overlap = max(max_overlap, _bbox_intersection_area(target, other) / denom)
    return max_overlap


def _is_dtld_target_occluded(
    info: Optional[Dict[str, Any]],
    infos: Optional[List[Dict[str, Any]]],
    *,
    target_idx: int,
    bbox_overlap_threshold: float,
) -> Tuple[bool, Dict[str, object]]:
    metadata: Dict[str, object] = {
        "occlusion_bbox_overlap_threshold": bbox_overlap_threshold,
    }
    if info is None:
        metadata["occlusion_reason"] = "missing_info"
        return False, metadata

    max_visible_overlap = _max_bbox_overlap_fraction(
        info.get("bbox_visib"),
        infos,
        target_idx=target_idx,
        bbox_key="bbox_visib",
    )
    max_full_overlap = _max_bbox_overlap_fraction(
        info.get("bbox_obj"),
        infos,
        target_idx=target_idx,
        bbox_key="bbox_obj",
    )
    metadata["max_bbox_visib_overlap_fraction"] = round(max_visible_overlap, 6)
    metadata["max_bbox_obj_overlap_fraction"] = round(max_full_overlap, 6)

    if max_visible_overlap >= bbox_overlap_threshold or max_full_overlap >= bbox_overlap_threshold:
        metadata["occlusion_reason"] = "bbox_overlap"
        return True, metadata

    metadata["occlusion_reason"] = None
    return False, metadata


def _count_dtld_visible_containers(
    objects: List[Dict[str, Any]],
    infos: Optional[List[Dict[str, Any]]],
) -> int:
    count = 0
    for idx, obj in enumerate(objects):
        obj_id = _as_int(obj.get("obj_id"))
        if obj_id not in _DTLD_OBJECT_NAMES:
            continue
        if infos is not None and idx < len(infos) and not _has_visible_area(infos[idx]):
            continue
        count += 1
    return count


def _has_occluded_dtld_visible_container(
    objects: List[Dict[str, Any]],
    infos: Optional[List[Dict[str, Any]]],
    *,
    bbox_overlap_threshold: float,
) -> bool:
    if infos is None:
        return False

    for idx, obj in enumerate(objects):
        obj_id = _as_int(obj.get("obj_id"))
        if obj_id not in _DTLD_OBJECT_NAMES:
            continue
        if idx >= len(infos) or not _has_visible_area(infos[idx]):
            continue

        occluded, _metadata = _is_dtld_target_occluded(
            infos[idx],
            infos,
            target_idx=idx,
            bbox_overlap_threshold=bbox_overlap_threshold,
        )
        if occluded:
            return True
    return False


class DTLDDataset:
    """Minimal folder-based dataset loader for DTLD."""

    def __init__(
        self,
        root: Union[str, Path],
        scene_ids: Optional[Iterable[str]] = None,
        rgb_only: bool = True,
        skip_bad_scenes: bool = True,
        prompt_bank: Optional[PromptBank] = None,
        shuffle: bool = False,
        seed: Optional[int] = None,
        debug_boxes: bool = False,
        skip_occluded_target_questions: bool = True,
        occluded_bbox_overlap_threshold: float = _DEFAULT_OCCLUDED_BBOX_OVERLAP_THRESHOLD,
    ) -> None:
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.root}")

        self._scene_ids = list(scene_ids) if scene_ids is not None else self._discover_scene_ids()
        self._rgb_only = rgb_only
        self._skip_bad_scenes = bool(skip_bad_scenes)
        self._prompt_bank = prompt_bank or PromptBank.default()
        self._seed = seed
        self._debug_boxes = bool(debug_boxes)
        self._skip_occluded_target_questions = bool(skip_occluded_target_questions)
        self._occluded_bbox_overlap_threshold = float(occluded_bbox_overlap_threshold)
        self._scene_cache: Dict[str, Dict[str, object]] = {}
        self._entries = self._build_entries()

        if shuffle:
            self._entries = self._balance_entries_by_scene(self._entries, seed=seed)

    @staticmethod
    def _balance_entries_by_scene(
        entries: List[Tuple[str, int, int]],
        *,
        seed: Optional[int],
    ) -> List[Tuple[str, int, int]]:
        grouped: Dict[str, List[Tuple[str, int, int]]] = {}
        for entry in entries:
            grouped.setdefault(entry[0], []).append(entry)

        scene_ids = list(grouped)
        rng = np.random.default_rng(seed)
        scene_ids = [scene_ids[i] for i in rng.permutation(len(scene_ids))]

        balanced: List[Tuple[str, int, int]] = []
        remaining = True
        round_idx = 0
        while remaining:
            remaining = False
            for scene_id in scene_ids:
                scene_entries = grouped[scene_id]
                if round_idx < len(scene_entries):
                    balanced.append(scene_entries[round_idx])
                    remaining = True
            round_idx += 1
        return balanced

    def balanced_indices_by_scene(
        self,
        *,
        max_samples: int,
        shuffle: bool = False,
        seed: Optional[int] = None,
    ) -> List[int]:
        grouped: Dict[str, List[int]] = {}
        for idx, (scene_id, _frame_id, _obj_idx) in enumerate(self._entries):
            grouped.setdefault(scene_id, []).append(idx)

        scene_ids = list(grouped)
        if shuffle:
            rng = np.random.default_rng(seed)
            scene_ids = [scene_ids[i] for i in rng.permutation(len(scene_ids))]

        selected: List[int] = []
        round_idx = 0
        while len(selected) < max_samples:
            added = False
            for scene_id in scene_ids:
                scene_indices = grouped[scene_id]
                if round_idx < len(scene_indices):
                    selected.append(scene_indices[round_idx])
                    added = True
                    if len(selected) >= max_samples:
                        break
            if not added:
                break
            round_idx += 1
        return selected

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

        try:
            with liquid_path.open("r", encoding="utf-8") as f:
                liquid = cast(Dict[str, List[Dict[str, Any]]], json.load(f))
        except json.JSONDecodeError as e:
            size = None
            try:
                size = liquid_path.stat().st_size
            except OSError:
                size = None
            raise json.JSONDecodeError(
                f"{e.msg} (file={liquid_path}, size_bytes={size})",
                e.doc,
                e.pos,
            ) from e

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

            try:
                scene_data = self._load_scene_annotations(scene_id)
            except json.JSONDecodeError as e:
                if not self._skip_bad_scenes:
                    raise
                warnings.warn(
                    f"Skipping DTLD scene_id={scene_id!r}: failed to parse {liquid_path} ({e})",
                    RuntimeWarning,
                )
                continue
            liquid = cast(Dict[str, List[Dict[str, Any]]], scene_data["liquid"])
            for frame_key, objects in sorted(liquid.items(), key=lambda item: int(item[0])):
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
        frame_liquid_objects = liquid[str(frame_id)]
        frame_liquid = frame_liquid_objects[obj_idx]

        obj_id = _as_int(frame_liquid.get("obj_id"))
        sample["obj_id"] = obj_id if obj_id is not None else -1
        if "liquid_h" in frame_liquid:
            sample["liquid_h"] = frame_liquid["liquid_h"]
        if "liquid_label" in frame_liquid:
            sample["liquid_label"] = frame_liquid["liquid_label"]

        info = cast(Optional[Dict[str, List[Dict[str, Any]]]], scene_data.get("info"))
        frame_infos = info.get(str(frame_id)) if info is not None and str(frame_id) in info else None
        sample["container_count"] = _count_dtld_visible_containers(frame_liquid_objects, frame_infos)
        sample["container_count_ambiguous"] = _has_occluded_dtld_visible_container(
            frame_liquid_objects,
            frame_infos,
            bbox_overlap_threshold=self._occluded_bbox_overlap_threshold,
        )
        frame_info = None
        if info is not None and str(frame_id) in info and obj_idx < len(info[str(frame_id)]):
            frame_info = info[str(frame_id)][obj_idx]
            if "bbox_obj" in frame_info:
                sample["bbox_obj"] = frame_info["bbox_obj"]
            if "bbox_visib" in frame_info:
                sample["bbox_visib"] = frame_info["bbox_visib"]
            if "visib_fract" in frame_info:
                sample["visib_fract"] = frame_info["visib_fract"]

        target_occluded, occlusion_metadata = _is_dtld_target_occluded(
            frame_info,
            frame_infos,
            target_idx=obj_idx,
            bbox_overlap_threshold=self._occluded_bbox_overlap_threshold,
        )
        sample["target_occluded"] = target_occluded
        sample["target_occlusion"] = occlusion_metadata

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

        if self._debug_boxes:
            self._print_box_debug(
                sample=sample,
                info_loaded=info is not None,
                frame_info_count=len(frame_infos) if frame_infos is not None else None,
                frame_info=frame_info,
            )

        if not self._rgb_only:
            depth_path = scene_dir / "depth" / f"{frame_id:06d}.png"
            if depth_path.exists():
                sample["depth_image"] = depth_path

        sample["annotations"] = self._build_annotations(sample)

        return sample

    def _print_box_debug(
        self,
        *,
        sample: Dict[str, object],
        info_loaded: bool,
        frame_info_count: Optional[int],
        frame_info: Optional[Dict[str, Any]],
    ) -> None:
        bbox_visib = sample.get("bbox_visib")
        bbox_obj = sample.get("bbox_obj")
        print(
            "  [DTLD dataloader bbox debug] "
            f"id={sample.get('id')} "
            f"info_loaded={info_loaded} "
            f"frame_info_count={frame_info_count} "
            f"frame_info_for_obj={frame_info is not None} "
            f"bbox_visib={bbox_visib!r} valid_visib={is_valid_bbox_xywh(bbox_visib)} "
            f"bbox_obj={bbox_obj!r} valid_obj={is_valid_bbox_xywh(bbox_obj)} "
            f"target_occluded={sample.get('target_occluded')} "
            f"target_occlusion={sample.get('target_occlusion')!r}"
        )

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
        obj_idx = _as_int(sample.get("obj_idx"))
        target_is_usable = not (
            self._skip_occluded_target_questions and bool(sample.get("target_occluded"))
        )

        def add_target_task(task: str, answer: object, metadata: Optional[Dict[str, object]] = None) -> None:
            if not self._prompt_bank.supports(dataset, task):
                return
            if not target_is_usable:
                return
            task_metadata = dict(metadata or {})
            task_metadata["target_occluded"] = bool(sample.get("target_occluded"))
            annotations.append(add(task=task, answer=answer, metadata=task_metadata))

        if obj_id_int is not None and obj_id_int in _DTLD_OBJECT_NAMES:
            container_type = _DTLD_OBJECT_NAMES[obj_id_int]
            add_target_task("container_type", container_type, {"obj_id": obj_id_int})
            if self._prompt_bank.supports(dataset, "container_type_simple"):
                add_target_task(
                    "container_type_simple",
                    simplify_container_type(container_type),
                    {"obj_id": obj_id_int, "source_task": "container_type"},
                )

        container_count = sample.get("container_count")
        if (
            obj_idx == 0
            and self._prompt_bank.supports(dataset, "container_count")
            and isinstance(container_count, int)
            and not bool(sample.get("container_count_ambiguous"))
        ):
            annotations.append(
                add(
                    task="container_count",
                    answer=container_count,
                    metadata={
                        "container_count": container_count,
                        "container_count_ambiguous": False,
                        "frame_level": True,
                    },
                )
            )

        if liquid_h_f is not None:
            add_target_task("fill_height_mm", round(liquid_h_f, 2), {"liquid_h": liquid_h_f})
            add_target_task(
                "liquid_presence",
                "YES" if liquid_h_f > 0 else "NO",
                {"liquid_h": liquid_h_f},
            )

        return annotations

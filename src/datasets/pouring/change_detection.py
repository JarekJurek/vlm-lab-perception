"""
PouringChangeDetectionDataset and helpers for two-frame change detection on Pouring test split.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, cast

import numpy as np

from tlg import PromptBank, TLGAnnotation, make_tlg_annotation

from .dataloader import PouringDataset


def _label_change(area_a: float, area_b: float, eps: float) -> str:
    """Label a frame pair based on liquid area change.
    
    Returns one of: NO_CHANGE, LIQUID_INCREASED, LIQUID_DECREASED.
    All valid pairs are labeled; no None filtering.
    """
    if area_a == 0 and area_b == 0:
        # Both empty
        return "NO_CHANGE"
    if area_a > 0 and area_b > area_a + eps:
        # Liquid increased
        return "LIQUID_INCREASED"
    if area_a == 0 and area_b > area_a + eps:
        # Treat liquid appearing from empty as an increase.
        return "LIQUID_INCREASED"
    if area_a > 0 and area_b < area_a - eps:
        # Liquid decreased
        return "LIQUID_DECREASED"
    # All other cases (small change, area_b == 0 when area_a > 0 but within eps)
    return "NO_CHANGE"


class PouringChangeDetectionDataset:
    """Pairs of frames for change detection in Pouring test split."""

    def __init__(
        self,
        root: Union[str, Path],
        prompt_bank: Optional[PromptBank] = None,
        shuffle: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        self.root = Path(root)
        self.prompt_bank = prompt_bank or PromptBank.default()
        self.split = "test"
        self._seed = None if seed is None else int(seed)
        self._dataset = PouringDataset(root, splits=(self.split,), prompt_bank=self.prompt_bank)

        samples = self._build_samples()
        if shuffle:
            rng = np.random.default_rng(seed)
            samples = [samples[i] for i in rng.permutation(len(samples))]

        self.samples = samples

    def _build_samples(self) -> List[Dict[str, Any]]:
        samples: List[Dict[str, Any]] = []
        frames_by_video: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        for frame in self._dataset:
            video_id = str(cast(Any, frame["video_id"]))
            liquid_stats = frame.get("liquid_stats", {})
            if not isinstance(liquid_stats, dict):
                liquid_stats = {}

            frame_index = int(cast(Any, frame["frame_index"]))
            img_path = Path(str(cast(Any, frame["path"])))
            area = float(liquid_stats.get("total_area", 0.0))

            frames_by_video[video_id].append(
                {
                    "frame_index": frame_index,
                    "img_path": img_path,
                    "area": area,
                }
            )

        has_liquid_presence = self.prompt_bank.supports("pouring", "liquid_presence")
        has_change_detection = self.prompt_bank.supports("pouring", "pouring_change_detection")

        if has_liquid_presence:
            for video_id, frame_infos in sorted(frames_by_video.items()):
                for frame_info in sorted(frame_infos, key=lambda item: item["frame_index"]):
                    self._append_single_frame_sample(samples, video_id, frame_info)

        if not has_change_detection:
            return samples

        pair_samples: List[Dict[str, Any]] = []
        for video_id, frame_infos in sorted(frames_by_video.items()):
            frame_infos = sorted(frame_infos, key=lambda item: item["frame_index"])
            # Generate both frame orders, including same-frame pairs. The
            # same-frame pairs provide additional NO_CHANGE examples for the
            # balanced subset.
            for i, a in enumerate(frame_infos):
                for j, b in enumerate(frame_infos):
                    self._append_pair_sample(pair_samples, video_id, a, b)
        samples.extend(self._balance_pair_samples(pair_samples))
        return samples

    def _balance_pair_samples(self, pair_samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        by_label: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for sample in pair_samples:
            by_label[str(sample["label"])].append(sample)

        if len(by_label) <= 1:
            return pair_samples

        target_count = min(len(group) for group in by_label.values())
        rng = np.random.default_rng(self._seed)
        balanced: List[Dict[str, Any]] = []
        for label in sorted(by_label):
            group = by_label[label]
            if len(group) > target_count:
                chosen = sorted(rng.choice(len(group), size=target_count, replace=False).tolist())
                balanced.extend(group[i] for i in chosen)
            else:
                balanced.extend(group)
        return balanced

    def _append_pair_sample(
        self,
        samples: List[Dict[str, Any]],
        video_id: str,
        a: Dict[str, Any],
        b: Dict[str, Any],
    ) -> None:
        eps = max(20.0, 0.01 * a["area"])
        label = _label_change(a["area"], b["area"], eps)

        sample_id = f"{self.split}/{video_id}/rgb_{a['frame_index']}__rgb_{b['frame_index']}"

        annotations = [self._build_change_annotation(sample_id, a, b, label)]

        samples.append(
            {
                "id": sample_id,
                "primary_task": "pouring_change_detection",
                "split": self.split,
                "video_id": video_id,
                "frame_a_index": a["frame_index"],
                "frame_a_path": str(a["img_path"]),
                "rgb_image": a["img_path"],
                "area_a": a["area"],
                "frame_b_index": b["frame_index"],
                "frame_b_path": str(b["img_path"]),
                "comparison_rgb_image": b["img_path"],
                "area_b": b["area"],
                "label": label,
                "annotations": annotations,
            }
        )

    def _append_single_frame_sample(
        self,
        samples: List[Dict[str, Any]],
        video_id: str,
        a: Dict[str, Any],
    ) -> None:
        sample_id = f"{self.split}/{video_id}/rgb_{a['frame_index']}"
        annotations: List[TLGAnnotation] = []

        if self.prompt_bank.supports("pouring", "liquid_presence"):
            annotations.append(self._build_liquid_presence_annotation(sample_id, a))

        if not annotations:
            return

        samples.append(
            {
                "id": sample_id,
                "primary_task": "liquid_presence",
                "split": self.split,
                "video_id": video_id,
                "frame_a_index": a["frame_index"],
                "frame_a_path": str(a["img_path"]),
                "rgb_image": a["img_path"],
                "area_a": a["area"],
                "annotations": annotations,
            }
        )

    def _build_liquid_presence_annotation(self, sample_id: str, a: dict) -> TLGAnnotation:
        return make_tlg_annotation(
            self.prompt_bank,
            dataset="pouring",
            task="liquid_presence",
            image_id=sample_id,
            answer="YES" if a["area"] > 0 else "NO",
            metadata={
                "frame_index": a["frame_index"],
                "area": a["area"],
            },
        )

    def _build_change_annotation(self, sample_id: str, a: dict, b: dict, label: str) -> TLGAnnotation:
        return make_tlg_annotation(
            self.prompt_bank,
            dataset="pouring",
            task="pouring_change_detection",
            image_id=sample_id,
            answer=label,
            metadata={
                "frame_a_index": a["frame_index"],
                "frame_b_index": b["frame_index"],
                "area_a": a["area"],
                "area_b": b["area"],
            },
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

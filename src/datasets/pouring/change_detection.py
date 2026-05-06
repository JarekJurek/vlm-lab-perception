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


DEFAULT_RANDOM_PAIR_COUNT = 20


def _label_change(area_a: float, area_b: float, eps: float) -> str:
    """Label a frame pair based on liquid area change.
    
    Returns one of: NO_CHANGE, LIQUID_APPEARED, LIQUID_INCREASED, LIQUID_DECREASED.
    All valid pairs are labeled; no None filtering.
    """
    if area_a == 0 and area_b == 0:
        # Both empty
        return "NO_CHANGE"
    if area_a == 0 and area_b > 0:
        # Liquid appeared
        return "LIQUID_APPEARED"
    if area_a > 0 and area_b > area_a + eps:
        # Liquid increased
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
        max_pair_distance: Optional[int] = None,
        shuffle: bool = False,
        seed: Optional[int] = None,
        *,
        sample_two_random_frames_per_video: bool = True,
        random_pairs: bool = False,
        num_pairs: Optional[int] = None,
    ) -> None:
        self.root = Path(root)
        self.prompt_bank = prompt_bank or PromptBank.default()
        self.max_pair_distance = max_pair_distance
        self.split = "test"
        self._dataset = PouringDataset(root, splits=(self.split,), prompt_bank=self.prompt_bank)
        self._sample_two_random_frames_per_video = bool(sample_two_random_frames_per_video)
        self._random_pairs = bool(random_pairs)
        self._num_pairs = None if num_pairs is None else int(num_pairs)
        self._seed = None if seed is None else int(seed)

        samples = self._build_samples()
        # Optionally subsample / randomize the pair list in a seedable way.
        if self._random_pairs:
            rng = np.random.default_rng(self._seed)
            if self._num_pairs is None:
                # Shuffle all pairs
                samples = [samples[i] for i in rng.permutation(len(samples))]
            else:
                k = min(self._num_pairs, len(samples))
                perm = rng.permutation(len(samples))[:k]
                samples = [samples[i] for i in perm]
        else:
            if shuffle:
                rng = np.random.default_rng(seed)
                samples = [samples[i] for i in rng.permutation(len(samples))]

        self.samples = samples

    def _build_samples(self) -> List[Dict[str, Any]]:
        samples: List[Dict[str, Any]] = []
        candidate_pairs: List[tuple[str, Dict[str, Any], Dict[str, Any]]] = []
        frames_by_video: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        rng = np.random.default_rng(self._seed)

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

        for video_id, frame_infos in sorted(frames_by_video.items()):
            frame_infos = sorted(frame_infos, key=lambda item: item["frame_index"])
            if self._sample_two_random_frames_per_video:
                candidate_pairs.extend((video_id, a, b) for a, b in self._candidate_pairs(frame_infos))
            else:
                # Generate ordered pairs (a, b) with a < b.
                for i, a in enumerate(frame_infos):
                    for j in range(i + 1, len(frame_infos)):
                        b = frame_infos[j]
                        if self.max_pair_distance is not None and (b["frame_index"] - a["frame_index"]) > self.max_pair_distance:
                            continue
                        self._append_sample(samples, video_id, a, b)

        if self._sample_two_random_frames_per_video:
            if not candidate_pairs:
                return samples

            pair_count = self._num_pairs if self._num_pairs is not None else DEFAULT_RANDOM_PAIR_COUNT
            k = min(int(pair_count), len(candidate_pairs))
            chosen = rng.permutation(len(candidate_pairs))[:k]
            for idx in chosen:
                video_id, a, b = candidate_pairs[int(idx)]
                if rng.random() < 0.5:
                    a, b = b, a
                self._append_sample(samples, video_id, a, b)
        return samples

    def _candidate_pairs(
        self, frame_infos: List[Dict[str, Any]]
    ) -> List[tuple[Dict[str, Any], Dict[str, Any]]]:
        candidates: List[tuple[Dict[str, Any], Dict[str, Any]]] = []
        for i, a in enumerate(frame_infos):
            for j in range(i + 1, len(frame_infos)):
                b = frame_infos[j]
                pair_distance = b["frame_index"] - a["frame_index"]
                if self.max_pair_distance is not None and pair_distance > self.max_pair_distance:
                    continue
                candidates.append((a, b))
        return candidates

    def _append_sample(
        self,
        samples: List[Dict[str, Any]],
        video_id: str,
        a: Dict[str, Any],
        b: Dict[str, Any],
    ) -> None:
        eps = max(20.0, 0.01 * a["area"])
        label = _label_change(a["area"], b["area"], eps)
        sample_id = f"{self.split}/{video_id}/rgb_{a['frame_index']}__rgb_{b['frame_index']}"

        # Only build annotations that the PromptBank says are enabled
        annotations: List[TLGAnnotation] = []
        if self.prompt_bank.supports("pouring", "liquid_presence"):
            annotations.append(self._build_liquid_presence_annotation(sample_id, a))
        if self.prompt_bank.supports("pouring", "pouring_change_detection"):
            annotations.append(self._build_change_annotation(sample_id, a, b, label))

        # Only include the sample if at least one enabled annotation exists.
        if annotations:
            samples.append(
                {
                    "id": sample_id,
                    "split": self.split,
                    "video_id": video_id,
                    "frame_a_index": a["frame_index"],
                    "frame_b_index": b["frame_index"],
                    "frame_a_path": str(a["img_path"]),
                    "frame_b_path": str(b["img_path"]),
                    "rgb_image": a["img_path"],
                    "comparison_rgb_image": b["img_path"],
                    "area_a": a["area"],
                    "area_b": b["area"],
                    "label": label,
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

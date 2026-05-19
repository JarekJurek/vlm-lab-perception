"""Frame-ordering samples for the Pouring test split."""
from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from math import comb
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, cast

import numpy as np

from tlg import PromptBank, TLGAnnotation, make_tlg_annotation

from .dataloader import PouringDataset


_ORDER_LABELS = ("A", "B", "C", "D")
_DEFAULT_MAX_ORDERING_SAMPLES_PER_VIDEO = 128


class PouringFrameOrderingDataset:
    """Three- or four-frame ordering tasks for Pouring test videos."""

    def __init__(
        self,
        root: Union[str, Path],
        prompt_bank: Optional[PromptBank] = None,
        shuffle: bool = False,
        seed: Optional[int] = None,
        max_ordering_samples_per_video: int = _DEFAULT_MAX_ORDERING_SAMPLES_PER_VIDEO,
    ) -> None:
        self.root = Path(root)
        self.prompt_bank = prompt_bank or PromptBank.default()
        self.split = "test"
        self._seed = None if seed is None else int(seed)
        self.max_ordering_samples_per_video = int(max_ordering_samples_per_video)
        if self.max_ordering_samples_per_video <= 0:
            raise ValueError("max_ordering_samples_per_video must be positive")
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
        has_frame_ordering = self.prompt_bank.supports("pouring", "pouring_frame_ordering")

        if has_liquid_presence:
            for video_id, frame_infos in sorted(frames_by_video.items()):
                for frame_info in sorted(frame_infos, key=lambda item: item["frame_index"]):
                    self._append_single_frame_sample(samples, video_id, frame_info)

        if not has_frame_ordering:
            return samples

        for video_id, frame_infos in sorted(frames_by_video.items()):
            frame_infos = sorted(frame_infos, key=lambda item: item["frame_index"])
            if len(frame_infos) >= 3:
                self._append_ordering_samples(samples, video_id, frame_infos)
        return samples

    def _append_ordering_samples(
        self,
        samples: List[Dict[str, Any]],
        video_id: str,
        frame_infos: Sequence[Dict[str, Any]],
    ) -> None:
        frame_count = 4 if len(frame_infos) >= 4 else 3
        for chosen_indices in _sample_frame_combinations(
            len(frame_infos),
            frame_count,
            max_samples=self.max_ordering_samples_per_video,
            seed=_stable_seed(self._seed, video_id),
        ):
            self._append_ordering_sample(samples, video_id, frame_infos, chosen_indices)

    def _append_ordering_sample(
        self,
        samples: List[Dict[str, Any]],
        video_id: str,
        frame_infos: Sequence[Dict[str, Any]],
        chosen_indices: Sequence[int],
    ) -> None:
        chronological_frames = [frame_infos[i] for i in chosen_indices]
        frame_count = len(chronological_frames)

        combo_key = "__".join(str(frame["frame_index"]) for frame in chronological_frames)
        rng = np.random.default_rng(_stable_seed(self._seed, f"{video_id}:{combo_key}"))
        display_order = rng.permutation(frame_count).tolist()
        if display_order == list(range(frame_count)):
            display_order = list(reversed(display_order))

        display_frames = [chronological_frames[i] for i in display_order]
        display_labels = list(_ORDER_LABELS[:frame_count])
        chronological_positions = [display_order.index(i) for i in range(frame_count)]
        answer_labels = [display_labels[i] for i in chronological_positions]
        answer = ", ".join(answer_labels)

        chronological_stems = "__".join(f"rgb_{f['frame_index']}" for f in chronological_frames)
        display_stems = "__".join(f"{label}_rgb_{frame['frame_index']}" for label, frame in zip(display_labels, display_frames))
        sample_id = f"{self.split}/{video_id}/order_{chronological_stems}/display_{display_stems}"

        annotations = [self._build_ordering_annotation(sample_id, chronological_frames, display_frames, answer)]

        samples.append(
            {
                "id": sample_id,
                "primary_task": "pouring_frame_ordering",
                "split": self.split,
                "video_id": video_id,
                "frame_count": frame_count,
                "ordering_frame_positions": list(chosen_indices),
                "rgb_image": display_frames[0]["img_path"],
                "ordered_rgb_images": [frame["img_path"] for frame in display_frames],
                "ordered_image_labels": display_labels,
                "display_frame_indices": [frame["frame_index"] for frame in display_frames],
                "chronological_frame_indices": [frame["frame_index"] for frame in chronological_frames],
                "answer_order_labels": answer_labels,
                "label": answer,
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

    def _build_ordering_annotation(
        self,
        sample_id: str,
        chronological_frames: Sequence[dict],
        display_frames: Sequence[dict],
        answer: str,
    ) -> TLGAnnotation:
        return make_tlg_annotation(
            self.prompt_bank,
            dataset="pouring",
            task="pouring_frame_ordering",
            image_id=sample_id,
            answer=answer,
            metadata={
                "display_labels": list(_ORDER_LABELS[: len(display_frames)]),
                "display_frame_indices": [frame["frame_index"] for frame in display_frames],
                "chronological_frame_indices": [frame["frame_index"] for frame in chronological_frames],
            },
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


def _sample_frame_combinations(
    n_frames: int,
    frame_count: int,
    *,
    max_samples: int,
    seed: int,
) -> List[Tuple[int, ...]]:
    total = comb(n_frames, frame_count)
    if total <= max_samples:
        return list(combinations(range(n_frames), frame_count))

    rng = np.random.default_rng(seed)
    selected = set()
    while len(selected) < max_samples:
        combo = tuple(sorted(int(i) for i in rng.choice(n_frames, size=frame_count, replace=False)))
        selected.add(combo)
    return sorted(selected)


def _stable_seed(seed: Optional[int], video_id: str) -> int:
    base = 0 if seed is None else int(seed)
    for char in video_id:
        base = (base * 33 + ord(char)) % (2**32)
    return base


PouringChangeDetectionDataset = PouringFrameOrderingDataset

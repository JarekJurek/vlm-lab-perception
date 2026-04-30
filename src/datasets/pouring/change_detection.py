"""
PouringChangeDetectionDataset and helpers for two-frame change detection on Pouring test split.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from tlg import PromptBank, TLGAnnotation, make_tlg_annotation

from .dataloader import PouringDataset


def _label_change(area_a: float, area_b: float, eps: float) -> Optional[str]:
    if area_a == 0 and area_b == 0:
        return "NO_CHANGE"
    if area_a == 0 and area_b > 0:
        return "LIQUID_APPEARED"
    if area_a > 0 and area_b > area_a + eps:
        return "LIQUID_INCREASED"
    return None


class PouringChangeDetectionDataset:
    """Pairs of frames for change detection in Pouring test split."""

    def __init__(
        self,
        root: Union[str, Path],
        prompt_bank: Optional[PromptBank] = None,
        max_pair_distance: Optional[int] = None,
    ) -> None:
        self.root = Path(root)
        self.prompt_bank = prompt_bank or PromptBank.default()
        self.max_pair_distance = max_pair_distance
        self.split = "test"
        self._dataset = PouringDataset(root, splits=(self.split,), prompt_bank=self.prompt_bank)
        self.samples = self._build_samples()

    def _build_samples(self) -> List[Dict[str, Any]]:
        samples: List[Dict[str, Any]] = []
        frames_by_video: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        for frame in self._dataset:
            video_id = str(frame["video_id"])
            liquid_stats = frame.get("liquid_stats", {})
            if not isinstance(liquid_stats, dict):
                liquid_stats = {}

            frames_by_video[video_id].append(
                {
                    "frame_index": int(frame["frame_index"]),
                    "img_path": Path(str(frame["path"])),
                    "area": float(liquid_stats.get("total_area", 0.0)),
                }
            )

        for video_id, frame_infos in sorted(frames_by_video.items()):
            frame_infos = sorted(frame_infos, key=lambda item: item["frame_index"])
            # Generate ordered pairs (a, b) with a < b
            for i, a in enumerate(frame_infos):
                for j in range(i + 1, len(frame_infos)):
                    b = frame_infos[j]
                    pair_distance = b["frame_index"] - a["frame_index"]
                    if self.max_pair_distance is not None and pair_distance > self.max_pair_distance:
                        continue
                    eps = max(20.0, 0.01 * a["area"])
                    label = _label_change(a["area"], b["area"], eps)
                    if label is None:
                        continue
                    sample_id = f"{self.split}/{video_id}/rgb_{a['frame_index']}__rgb_{b['frame_index']}"
                    ann = self._build_annotation(sample_id, a, b, label)
                    samples.append(
                        {
                            "id": sample_id,
                            "split": self.split,
                            "video_id": video_id,
                            "frame_a_index": a["frame_index"],
                            "frame_b_index": b["frame_index"],
                            "frame_a_path": str(a["img_path"]),
                            "frame_b_path": str(b["img_path"]),
                            "area_a": a["area"],
                            "area_b": b["area"],
                            "label": label,
                            "annotations": [ann],
                        }
                    )
        return samples

    def _build_annotation(self, sample_id: str, a: dict, b: dict, label: str) -> TLGAnnotation:
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

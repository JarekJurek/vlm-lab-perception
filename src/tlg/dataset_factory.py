"""Dataset factory helpers for Qwen-VL evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
from datasets.dtld import DTLDDataset
from datasets.glassesinthewild import GlassesInTheWildDataset
from datasets.labliquidvision import LabLiquidVisionDataset
from datasets.ot2 import OT2Dataset
from datasets.pouring import PouringChangeDetectionDataset, PouringDataset
from tlg.prompts import PromptBank


DATA_ROOTS = {
    "labliquidvision": Path("data") / "LabLiquidVision",
    "glasses": Path("data") / "glasses_in_the_wild" / "glasses_in_the_wild_1000",
    "dtld": Path("data") / "DTLD_dataset",
    "pouring": Path("data") / "pouring",
    "ot2": Path("data") / "OT2-Computer-Vision",
}


class _CombinedDataset:
    def __init__(self, datasets: Sequence[Any], *, shuffle: bool = False, seed: Optional[int] = None) -> None:
        self._datasets = list(datasets)
        self._index: List[Tuple[int, int]] = [
            (dataset_idx, sample_idx)
            for dataset_idx, dataset in enumerate(self._datasets)
            for sample_idx in range(len(dataset))
        ]
        if shuffle:
            rng = np.random.default_rng(seed)
            self._index = [self._index[i] for i in rng.permutation(len(self._index))]

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, index: int) -> Any:
        dataset_idx, sample_idx = self._index[index]
        return self._datasets[dataset_idx][sample_idx]

    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]


def build_dataset(dataset_name: str, data_root: Path, *, prompt_bank: PromptBank, include_depth: bool = False, shuffle: bool = False, seed: Optional[int] = None) -> Any:
    """Construct a dataset instance for a given dataset name.

    Args:
        dataset_name: Dataset identifier.
        data_root: Path to the dataset root.
        prompt_bank: Prompt bank used by dataset loaders.
        include_depth: When *True*, depth arrays and other non-RGB modalities
            are loaded alongside the RGB image.  Currently only affects
            ``labliquidvision`` (which stores depth maps as ``.npy`` files).
            Set this to *True* when using SD-VLM with ground-truth depth.

    Returns:
        Dataset instance.
    """
    if dataset_name == "labliquidvision":
        return LabLiquidVisionDataset(data_root, rgb_only=not include_depth, prompt_bank=prompt_bank, shuffle=shuffle, seed=seed)
    if dataset_name == "glasses":
        return GlassesInTheWildDataset(data_root, prompt_bank=prompt_bank, shuffle=shuffle, seed=seed)
    if dataset_name == "dtld":
        return DTLDDataset(data_root, rgb_only=True, prompt_bank=prompt_bank, shuffle=shuffle, seed=seed)
    if dataset_name == "pouring":
        pouring_datasets: List[Any] = []
        if prompt_bank.supports("pouring", "liquid_presence"):
            pouring_datasets.append(
                PouringDataset(data_root, splits=("test",), prompt_bank=prompt_bank, shuffle=shuffle, seed=seed)
            )
        if prompt_bank.supports("pouring", "pouring_change_detection"):
            pouring_datasets.append(
                PouringChangeDetectionDataset(
                    data_root, prompt_bank=prompt_bank, shuffle=shuffle, seed=seed, random_pairs=True
                )
            )
        return _CombinedDataset(pouring_datasets, shuffle=shuffle, seed=seed)
    if dataset_name == "ot2":
        return OT2Dataset(data_root, prompt_bank=prompt_bank, shuffle=shuffle, seed=seed)
    raise ValueError(f"Unsupported dataset: {dataset_name}")

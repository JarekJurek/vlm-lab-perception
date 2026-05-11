"""Dataset factory helpers for Qwen-VL evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from datasets.dtld import DTLDDataset
from datasets.glassesinthewild import GlassesInTheWildDataset
from datasets.labliquidvision import LabLiquidVisionDataset
from datasets.labpicsv1 import LabPicsV1Dataset
from datasets.ot2 import OT2Dataset
from datasets.pouring import PouringChangeDetectionDataset
from tlg.prompts import PromptBank


DATA_ROOTS = {
    "labliquidvision": Path("data") / "LabLiquidVision",
    "glasses": Path("data") / "glasses_in_the_wild" / "glasses_in_the_wild_1000",
    "dtld": Path("data") / "DTLD_dataset",
    "labpicsv1": Path("data") / "LabPicsV1",
    "pouring": Path("data") / "pouring",
    "ot2": Path("data") / "OT2-Computer-Vision",
}
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
    if dataset_name == "labpicsv1":
        return LabPicsV1Dataset(data_root, split="train", prompt_bank=prompt_bank, shuffle=shuffle, seed=seed)
    if dataset_name == "pouring":
        return PouringChangeDetectionDataset(data_root, prompt_bank=prompt_bank, shuffle=shuffle, seed=seed)
    if dataset_name == "ot2":
        return OT2Dataset(data_root, prompt_bank=prompt_bank, shuffle=shuffle, seed=seed)
    raise ValueError(f"Unsupported dataset: {dataset_name}")

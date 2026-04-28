"""Dataset factory helpers for Qwen-VL evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from datasets.dtld import DTLDDataset
from datasets.glassesinthewild import GlassesInTheWildDataset
from datasets.labliquidvision import LabLiquidVisionDataset
from datasets.ot2 import OT2Dataset
from datasets.pouring import PouringDataset
from tlg.prompts import PromptBank


DATA_ROOTS = {
    "labliquidvision": Path("data") / "LabLiquidVision",
    "glasses": Path("data") / "glasses_in_the_wild" / "glasses_in_the_wild_1000",
    "dtld": Path("data") / "DTLD_dataset",
    # These defaults may vary depending on how you lay out `data/`.
    "pouring": Path("data") / "pouring",
    "ot2": Path("data") / "OT2-Computer-Vision",
}


def build_dataset(dataset_name: str, data_root: Path, *, prompt_bank: PromptBank) -> Any:
    """Construct a dataset instance for a given dataset name.

    Args:
        dataset_name: Dataset identifier.
        data_root: Path to the dataset root.
        prompt_bank: Prompt bank used by dataset loaders.

    Returns:
        Dataset instance.
    """
    if dataset_name == "labliquidvision":
        return LabLiquidVisionDataset(data_root, rgb_only=True, prompt_bank=prompt_bank)
    if dataset_name == "glasses":
        return GlassesInTheWildDataset(data_root, prompt_bank=prompt_bank)
    if dataset_name == "dtld":
        return DTLDDataset(data_root, rgb_only=True, prompt_bank=prompt_bank)
    if dataset_name == "pouring":
        return PouringDataset(data_root, splits=("test",), prompt_bank=prompt_bank)
    if dataset_name == "ot2":
        return OT2Dataset(data_root, prompt_bank=prompt_bank)
    raise ValueError(f"Unsupported dataset: {dataset_name}")

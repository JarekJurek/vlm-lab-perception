"""Dataset factory helpers for Qwen-VL evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from datasets.coqe import COQEDataset
from datasets.dtld import DTLDDataset
from datasets.glassesinthewild import GlassesInTheWildDataset
from datasets.labliquidvision import LabLiquidVisionDataset
from datasets.labpics_chemistry import LabPicsChemistryDataset
from datasets.ot2 import OT2Dataset
from datasets.pouring import PouringFrameOrderingDataset
from tlg.prompts import PromptBank


DATA_ROOTS = {
    "labliquidvision": Path("data") / "LabLiquidVision",
    "glasses": Path("data") / "glasses_in_the_wild" / "glasses_in_the_wild_1000",
    "dtld": Path("data") / "DTLD_dataset",
    "labpics_chemistry": Path("data") / "LabPics Chemistry",
    "pouring": Path("data") / "pouring",
    "ot2": Path("data") / "OT2-Computer-Vision",
    "coqe": Path("data") / "COQE_dataset",
}
def build_dataset(
    dataset_name: str,
    data_root: Path,
    *,
    prompt_bank: PromptBank,
    include_depth: bool = False,
    shuffle: bool = False,
    seed: Optional[int] = None,
    debug_dtld_boxes: bool = False,
    dtld_no_occlusion_filter: bool = False,
    coqe_split: str = "test",
    labpics_chemistry_split: str = "all",
) -> Any:
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
        dtld_kwargs = {}
        if dtld_no_occlusion_filter:
            dtld_kwargs = {
                "skip_occluded_target_questions": False,
                "occluded_bbox_overlap_threshold": 2.0,
            }
        return DTLDDataset(
            data_root,
            rgb_only=True,
            prompt_bank=prompt_bank,
            shuffle=shuffle,
            seed=seed,
            debug_boxes=debug_dtld_boxes,
            **dtld_kwargs,
        )
    if dataset_name == "labpics_chemistry":
        return LabPicsChemistryDataset(
            data_root,
            split=labpics_chemistry_split,
            prompt_bank=prompt_bank,
            shuffle=shuffle,
            seed=seed,
        )
    if dataset_name == "pouring":
        return PouringFrameOrderingDataset(data_root, prompt_bank=prompt_bank, shuffle=shuffle, seed=seed)
    if dataset_name == "ot2":
        return OT2Dataset(data_root, prompt_bank=prompt_bank, shuffle=shuffle, seed=seed)
    if dataset_name == "coqe":
        return COQEDataset(data_root, split=coqe_split, prompt_bank=prompt_bank, shuffle=shuffle, seed=seed)
    raise ValueError(f"Unsupported dataset: {dataset_name}")

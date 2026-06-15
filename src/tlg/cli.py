"""Command-line interface helpers for VLM task evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the VLM evaluation script."""
    parser = argparse.ArgumentParser(description="Run VLM tasks on liquid datasets")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["labliquidvision", "glasses", "dtld", "labpics_chemistry", "pouring", "ot2", "coqe"],
        default=None,
        help=(
            "One or more datasets to evaluate. Default: labliquidvision for dataset loaders, "
            "or all datasets present when --manifest is set."
        ),
    )
    parser.add_argument(
        "--model",
        choices=[
            "qwen",
            "sdvlm",
            "internvl35_8b",
            "llavaov2_8b",
            "kimi_vl_a3b",
        ],
        default="qwen",
        help=(
            "Which VLM to run. "
            "'qwen' (default): Qwen3-VL only. "
            "'sdvlm': SD-VLM only (requires the llava package from SD-VLM). "
            "'internvl35_8b', 'llavaov2_8b', and 'kimi_vl_a3b': run one additional "
            "open HF VLM."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate dataset loading and standardized annotations without loading a model.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Max samples per dataset/task to process (default: all available).",
    )
    parser.add_argument(
        "--data-root",
        action="append",
        default=[],
        help=(
            "Override a dataset root, e.g. --data-root labliquidvision=/path/to/root. "
            "Can be provided multiple times."
        ),
    )
    parser.add_argument(
        "--prompt-bank",
        type=str,
        default=str(Path(__file__).with_name("prompt_bank.yaml")),
        help="Prompt definition YAML to use for this run. Default: src/tlg/prompt_bank.yaml.",
    )
    parser.add_argument(
        "--dataset-prompts",
        type=str,
        default=str(Path(__file__).with_name("dataset_prompts.yaml")),
        help=(
            "Dataset-to-task prompt assignment YAML to use for this run. "
            "Default: src/tlg/dataset_prompts.yaml."
        ),
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="",
        help="Optional exported TLG manifest JSON. When set, dataset loaders are bypassed.",
    )
    parser.add_argument(
        "--manifest-split",
        choices=["train", "test", "all"],
        default="test",
        help="Generated manifest split to evaluate when --manifest is set. Default: test.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle samples before applying --max-samples.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed used with --shuffle (default: 0).",
    )
    parser.add_argument(
        "--predictions-out",
        type=str,
        default="",
        help="Optional path to write per-task JSONL predictions with ground truth.",
    )
    parser.add_argument(
        "--metrics-out",
        type=str,
        default="",
        help="Optional path to write aggregated metrics as JSON.",
    )
    parser.add_argument(
        "--viz-attn",
        action="store_true",
        help=(
            "If set, extract attention weights and write blended heatmap overlays (PNG). "
            "This is a qualitative debug aid and may be slower / more memory intensive."
        ),
    )
    parser.add_argument(
        "--viz-attn-mode",
        choices=["all", "first_answer_token"],
        default="all",
        help="Which generated tokens to use for attention visualization.",
    )
    parser.add_argument(
        "--viz-out",
        type=str,
        default="",
        help=(
            "Output directory for attention visualizations (used with --viz-attn). "
            "Files are written under {viz_out}/{dataset}/{sample_id}/{task}.png"
        ),
    )
    parser.add_argument(
        "--debug-dtld-boxes",
        action="store_true",
        help=(
            "Print DTLD bbox state and save raw, boxed, and final model-input "
            "images for checking whether target boxes are visible."
        ),
    )
    parser.add_argument(
        "--debug-image-out",
        type=str,
        default="",
        help=(
            "Output directory for --debug-dtld-boxes images. Defaults to "
            "{viz_out}/debug_dtld_boxes when --viz-out is set, otherwise results/debug_dtld_boxes."
        ),
    )
    parser.add_argument(
        "--dtld-no-occlusion-filter",
        action="store_true",
        help="Disable DTLD occlusion filtering for target-object and container-count tasks.",
    )
    parser.add_argument(
        "--coqe-split",
        choices=["train", "val", "test"],
        default="test",
        help="COQE split to evaluate when using the COQE dataset loader. Default: test.",
    )
    parser.add_argument(
        "--labpics-chemistry-split",
        choices=["train", "test", "all"],
        default="all",
        help="LabPics Chemistry split to evaluate when using the LabPics Chemistry dataset loader. Default: all.",
    )
    parser.add_argument(
        "--tol",
        action="append",
        default=[],
        help=(
            "Override numeric tolerances for within-tolerance scoring, e.g. "
            "--tol fill_percentage=2.5 --tol fill_height_mm=10."
        ),
    )
    # --- SD-VLM specific arguments ---
    parser.add_argument(
        "--sdvlm-model-path",
        type=str,
        default="cpystan/SD-VLM-7B",
        help="HuggingFace model ID or local path for SD-VLM (default: cpystan/SD-VLM-7B).",
    )
    parser.add_argument(
        "--sdvlm-vision-tower",
        type=str,
        default="openai/clip-vit-large-patch14-336",
        help="Vision tower identifier for SD-VLM (default: openai/clip-vit-large-patch14-336).",
    )
    parser.add_argument(
        "--sdvlm-no-depth",
        action="store_true",
        help="Disable depth encoding in SD-VLM (sets use_depth=False).",
    )
    parser.add_argument(
        "--sdvlm-gt-depth",
        action="store_true",
        help=(
            "Use ground-truth depth maps from the dataset instead of estimating depth "
            "internally (only applicable for datasets that provide depth, e.g. labliquidvision)."
        ),
    )
    parser.add_argument(
        "--sdvlm-depth-dir",
        type=str,
        default="",
        help=(
            "Directory containing precomputed .npy depth maps to load as SD-VLM "
            "GT-style depth. Does not run a depth model."
        ),
    )
    return parser.parse_args()


def parse_tol_overrides(items: List[str]) -> Dict[str, float]:
    """Parse task-level tolerance overrides from CLI arguments.

    Args:
        items: List of task=value strings.

    Returns:
        Mapping from task name to tolerance value.
    """
    out: Dict[str, float] = {}
    for raw in items:
        s = str(raw).strip()
        if not s or "=" not in s:
            raise ValueError(f"--tol must be task=value. Got: {raw!r}")
        task, val = s.split("=", 1)
        task = task.strip()
        val = val.strip()
        if not task or not val:
            raise ValueError(f"Invalid --tol value: {raw!r}")
        try:
            out[task] = float(val)
        except ValueError as exc:
            raise ValueError(f"Invalid tolerance for task={task!r}: {val!r}") from exc
    return out


def parse_data_root_overrides(items: List[str]) -> Dict[str, Path]:
    """Parse dataset root overrides from CLI arguments.

    Args:
        items: List of dataset=/path strings.

    Returns:
        Mapping from dataset name to overridden path.
    """
    out: Dict[str, Path] = {}
    for raw in items:
        s = str(raw).strip()
        if not s or "=" not in s:
            raise ValueError(
                "--data-root must be of the form dataset=/path. "
                f"Got: {raw!r}"
            )
        ds, path = s.split("=", 1)
        ds = ds.strip()
        path = path.strip()
        if not ds or not path:
            raise ValueError(f"Invalid --data-root value: {raw!r}")
        out[ds] = Path(path)
    return out

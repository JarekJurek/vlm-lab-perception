"""Test Qwen-VL on liquid container vision tasks.

This script evaluates Qwen-VL on visual understanding tasks for multiple
liquid-container datasets:
    - LabLiquidVision
    - Glasses in the Wild
    - DTLD
"""

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import sys

# Ensure `src/` is on PYTHONPATH so imports match README usage.
sys.path.append(str(Path(__file__).parent / "src"))

from datasets.dtld import DTLDDataset
from datasets.glassesinthewild import GlassesInTheWildDataset
from datasets.labliquidvision import LabLiquidVisionDataset
from datasets.ot2 import OT2Dataset
from datasets.pouring import PouringDataset

from tlg import PromptBank, TLGAnnotation
from tlg.evaluation import TaskStats, score_prediction


# MODEL_NAME = "Qwen/Qwen3-VL-30B-A3B-Instruct"
MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
MAX_NEW_TOKENS = 256


DATA_ROOTS = {
    # Pass the base dataset dir; loader will auto-use `processed/`.
    "labliquidvision": Path("data") / "LabLiquidVision",
    "glasses": Path("data") / "glasses_in_the_wild" / "glasses_in_the_wild_1000",
    "dtld": Path("data") / "DTLD_dataset",
    # These defaults may vary depending on how you lay out `data/`.
    "pouring": Path("data") / "pouring",
    "ot2": Path("data") / "OT2-Computer-Vision",
}


def load_model():
    """Load Qwen3-VL model and processor."""
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model: {MODEL_NAME}")
    print(f"Device: {device}")
    
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16,
        device_map="auto"
    ).eval()
    
    print("Model loaded successfully.")
    
    return model, processor


def run_task(
    model,
    processor,
    image,
    prompt: str,
    mask=None,
) -> str:
    """Run a single VLM task.
    from PIL import Image, UnidentifiedImageError
    
    Args:
        processor: Qwen3-VL processor.
        image: RGB image of the container.
        prompt: Task prompt text.
        mask: Optional vessel mask image.
    
    Returns:
        Model's text response.
    """
    import torch

    # Build content list (image placeholders) and images list
    content = [{"type": "image"}]
    images = [image]

    if mask is not None:
        content.append({"type": "image"})
        images.append(mask)
        prompt = f"The second image shows the container mask.\n\n{prompt}"

    content.append({"type": "text", "text": prompt})

    # Build messages
    messages = [{"role": "user", "content": content}]

    # Prepare inputs using chat template + images
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = processor(
        text=[text],
        images=images,
        return_tensors="pt"
    ).to(model.device)
    
    # Generate
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False
        )
    
    # Decode
    response = processor.batch_decode(
        output,
        skip_special_tokens=True
    )[0]
    
    # Extract just the answer (remove the prompt echo if present)
    if prompt in response:
        response = response.split(prompt)[-1].strip()
    
    return response


def test_sample(
    model,
    processor,
    sample: Dict[str, Any],
    annotations: List[TLGAnnotation],
) -> Dict[str, str]:
    """Test all tasks on a single sample.

    Args:
        model: Qwen3-VL model.
        processor: Qwen3-VL processor.
        sample: Dataset sample containing 'rgb_image'.

    Returns:
        Dictionary mapping task names to model responses.
    """

    from PIL import Image, UnidentifiedImageError

    rgb_path = sample.get("rgb_image")
    if rgb_path is not None:
        try:
            rgb_image = Image.open(rgb_path).convert("RGB")
        except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as e:
            raise OSError(f"Failed to open RGB image: {rgb_path} ({e})") from e
    else:
        rgb_array = sample.get("rgb")
        if rgb_array is None:
            raise FileNotFoundError("Input_RGBImage.png or Input_RGBImage.npy not found for sample.")
        if rgb_array.shape[0] in (1, 3, 4):
            rgb_array = rgb_array.transpose(1, 2, 0)
        rgb_image = Image.fromarray(rgb_array.astype("uint8"))

    mask_image = None
    mask_path = sample.get("mask_image")
    if mask_path is not None:
        try:
            mask_image = Image.open(mask_path).convert("RGB")
        except (UnidentifiedImageError, OSError, SyntaxError, ValueError):
            mask_image = None

    results: Dict[str, str] = {}
    for ann in annotations:
        task_name = str(ann.metadata.get("task", ann.task_type))
        print(f"  Running task: {task_name}...")
        response = run_task(model, processor, rgb_image, ann.question, mask=mask_image)
        results[task_name] = response

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Qwen-VL tasks on liquid datasets")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["labliquidvision", "glasses", "dtld", "pouring", "ot2"],
        default=["labliquidvision"],
        help="One or more datasets to evaluate.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate dataset loading and standardized annotations without loading a model.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=1,
        help="Max samples per dataset to process (default: 1).",
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
        "--use-mask",
        action="store_true",
        help=(
            "If a dataset sample provides a mask image, pass it as a second image. "
            "Default: off (benchmark uses RGB only)."
        ),
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle samples before selecting the first --max-samples.",
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
        "--tol",
        action="append",
        default=[],
        help=(
            "Override numeric tolerances for within-tolerance scoring, e.g. "
            "--tol fill_percentage=2.5 --tol fill_height_mm=10."
        ),
    )
    return parser.parse_args()


def _parse_tol_overrides(items: List[str]) -> Dict[str, float]:
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
        except ValueError as e:
            raise ValueError(f"Invalid tolerance for task={task!r}: {val!r}") from e
    return out


def _parse_data_root_overrides(items: List[str]) -> Dict[str, Path]:
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


def build_dataset(dataset_name: str, data_root: Path, *, prompt_bank: PromptBank):
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


def print_ground_truth(sample: Dict[str, Any]) -> None:
    print("\nGround Truth (standardized annotations):")

    annotations = sample.get("annotations")
    if not isinstance(annotations, list) or not annotations:
        print("  (no standardized annotations)")
        return

    for ann in annotations:
        if not isinstance(ann, TLGAnnotation):
            continue
        task_key = str(ann.metadata.get("task", ann.task_type))
        print(f"  - {task_key}: {ann.answer}")


def _validate_annotations(sample: Dict[str, Any]) -> Optional[str]:
    annotations = sample.get("annotations")
    if not isinstance(annotations, list):
        return "sample['annotations'] is not a list"
    for i, ann in enumerate(annotations):
        if not isinstance(ann, TLGAnnotation):
            return f"annotations[{i}] is not a TLGAnnotation"
        if not isinstance(ann.question, str) or not ann.question.strip():
            return f"annotations[{i}] has empty question"
        if not isinstance(ann.task_type, str) or not ann.task_type.strip():
            return f"annotations[{i}] has empty task_type"
        task = ann.metadata.get("task")
        if not isinstance(task, str) or not task.strip():
            return f"annotations[{i}] missing metadata['task']"
    return None


def main():
    """Run all VLM tests."""
    args = parse_args()
    dataset_names = args.datasets

    tol_overrides = _parse_tol_overrides(args.tol)

    data_root_overrides = _parse_data_root_overrides(args.data_root)
    prompt_bank = PromptBank.default()

    if args.dry_run:
        print("Dry run: validating dataset loading + standardized annotations (no model).")

    model = processor = None
    if not args.dry_run:
        model, processor = load_model()

    pred_out_path = Path(args.predictions_out) if args.predictions_out else None
    metrics_out_path = Path(args.metrics_out) if args.metrics_out else None
    pred_fh = None
    if pred_out_path is not None:
        pred_out_path.parent.mkdir(parents=True, exist_ok=True)
        pred_fh = pred_out_path.open("w", encoding="utf-8")

    overall_stats: Dict[str, TaskStats] = {}
    per_dataset_stats: Dict[str, Dict[str, TaskStats]] = {}

    for dataset_name in dataset_names:
        data_root = data_root_overrides.get(dataset_name, DATA_ROOTS[dataset_name])

        print(f"\n{'=' * 60}")
        print(f"Dataset: {dataset_name}")
        print(f"Loading dataset from: {data_root}")

        if not data_root.exists():
            print("Dataset root does not exist; skipping.")
            continue

        dataset = build_dataset(dataset_name, data_root, prompt_bank=prompt_bank)
        print(f"Dataset size: {len(dataset)}")

        if len(dataset) == 0:
            print("Error: Dataset is empty!")
            continue

        max_samples = max(1, int(args.max_samples))
        indices = list(range(len(dataset)))
        if args.shuffle:
            rng = random.Random(int(args.seed))
            rng.shuffle(indices)
        indices = indices[: min(len(indices), max_samples)]
        for j, idx0 in enumerate(indices):
            idx = j + 1
            try:
                sample = dataset[idx0]
            except (FileNotFoundError, OSError, SyntaxError) as e:
                print(f"Failed to load sample index={idx0}: {e}")
                # In dry-run we don't want a missing file to crash the whole script.
                if args.dry_run:
                    break
                raise

            print("\n" + "-" * 60)
            print(f"Sample {idx}/{len(dataset)}")
            print(f"Sample ID: {sample.get('id', 'unknown')}")

            err = _validate_annotations(sample)
            if err is not None:
                print(f"Annotation validation failed: {err}")
                continue

            print_ground_truth(sample)

            annotations = sample.get("annotations")
            if not isinstance(annotations, list) or not annotations:
                print("  No standardized annotations available for this sample; skipping.")
                continue

            if args.dry_run:
                print("\nDry run OK: annotations are well-formed.")
                continue

            # Optionally pass a mask image as a second input.
            if not args.use_mask:
                sample_no_mask = dict(sample)
                sample_no_mask.pop("mask_image", None)
                sample = sample_no_mask

            try:
                results = test_sample(model, processor, sample, annotations)
            except (FileNotFoundError, OSError, SyntaxError) as e:
                print(f"Failed to load images for sample index={idx0}: {e}")
                continue
            ds_stats = per_dataset_stats.setdefault(dataset_name, {})
            for ann in annotations:
                task = str(ann.metadata.get("task", ann.task_type))
                resp = results.get(task)
                if resp is None:
                    continue

                rec = score_prediction(ann, resp, tol_overrides=tol_overrides)

                # Update stats.
                for stats_map in (overall_stats, ds_stats):
                    st = stats_map.setdefault(task, TaskStats())
                    st.update_from_record(rec)

                # Write JSONL record.
                if pred_fh is not None:
                    row = {
                        "dataset": dataset_name,
                        "sample_id": sample.get("id", "unknown"),
                        "image_id": ann.image_id,
                        "task": rec.get("task"),
                        "task_type": rec.get("task_type"),
                        "question": rec.get("question"),
                        "gt": rec.get("gt"),
                        "pred": rec.get("pred"),
                        "pred_raw": rec.get("pred_raw"),
                        "valid": rec.get("valid"),
                        "correct": rec.get("correct"),
                    }
                    if "abs_error" in rec:
                        row["abs_error"] = rec["abs_error"]
                    if "within_tol" in rec:
                        row["within_tol"] = rec["within_tol"]
                    pred_fh.write(json.dumps(row, ensure_ascii=False) + "\n")

            print("\nPredictions:")
            for task_name, response in results.items():
                print(f"  {task_name}: {response}")

    # Close prediction writer.
    if pred_fh is not None:
        pred_fh.close()

    # Print aggregated metrics.
    def _print_stats(title: str, stats: Dict[str, TaskStats]) -> None:
        if not stats:
            return
        print("\n" + "=" * 60)
        print(title)
        for task in sorted(stats.keys()):
            d = stats[task].to_dict()
            n_valid = d.get("n_valid", 0)
            n = d.get("n", 0)
            acc = d.get("accuracy")
            mae = d.get("mae")
            within = d.get("within_tol")
            parts = [f"n={n}", f"valid={n_valid}"]
            if isinstance(acc, float):
                parts.append(f"acc={acc:.3f}")
            if isinstance(mae, float):
                parts.append(f"mae={mae:.3f}")
            if isinstance(within, float):
                parts.append(f"within_tol={within:.3f}")
            print(f"  - {task}: " + ", ".join(parts))

    _print_stats("Overall Metrics", overall_stats)
    for ds in sorted(per_dataset_stats.keys()):
        _print_stats(f"Metrics for dataset={ds}", per_dataset_stats[ds])

    if metrics_out_path is not None:
        metrics_out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_name": MODEL_NAME,
            "max_new_tokens": int(MAX_NEW_TOKENS),
            "datasets": list(dataset_names),
            "max_samples": int(args.max_samples),
            "use_mask": bool(args.use_mask),
            "overall": {k: v.to_dict() for k, v in overall_stats.items()},
            "per_dataset": {
                ds: {k: v.to_dict() for k, v in stats.items()} for ds, stats in per_dataset_stats.items()
            },
        }
        metrics_out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

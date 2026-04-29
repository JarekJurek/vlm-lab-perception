"""Test Qwen-VL on liquid container vision tasks.

This script evaluates Qwen-VL on visual understanding tasks for multiple
liquid-container datasets:
    - LabLiquidVision
    - Glasses in the Wild
    - DTLD
"""

import json
import random
from pathlib import Path
from typing import Dict

import sys

# Ensure `src/` is on PYTHONPATH so imports match README usage.
sys.path.append(str(Path(__file__).parent / "src"))

from tlg import PromptBank
from tlg.cli import parse_args, parse_data_root_overrides, parse_tol_overrides
from tlg.dataset_factory import DATA_ROOTS, build_dataset
from tlg.evaluation import TaskStats, score_prediction
from tlg.model_runner import MAX_NEW_TOKENS, MODEL_NAME, load_model, test_sample
from tlg.reporting import print_ground_truth, print_stats, validate_annotations



def main():
    """Run all VLM tests."""
    args = parse_args()
    dataset_names = args.datasets

    tol_overrides = parse_tol_overrides(args.tol)
    data_root_overrides = parse_data_root_overrides(args.data_root)
    prompt_bank = PromptBank.default()

    if args.dry_run:
        print("Dry run: validating dataset loading + standardized annotations (no model).")

    viz_attn = bool(args.viz_attn) and (not args.dry_run)
    viz_out_dir = Path(args.viz_out) if args.viz_out else None
    if viz_attn and viz_out_dir is None:
        raise ValueError("--viz-attn requires --viz-out")

    model = processor = None
    if not args.dry_run:
        model, processor = load_model(viz_attn=viz_attn)

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

            err = validate_annotations(sample)
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
                results = test_sample(
                    model,
                    processor,
                    sample,
                    annotations,
                    dataset_name=dataset_name,
                    sample_id=str(sample.get("id", "unknown")),
                    viz_attn=viz_attn,
                    viz_out_dir=viz_out_dir,
                    viz_attn_mode=str(args.viz_attn_mode),
                )
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
    print_stats("Overall Metrics", overall_stats)
    for ds in sorted(per_dataset_stats.keys()):
        print_stats(f"Metrics for dataset={ds}", per_dataset_stats[ds])

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

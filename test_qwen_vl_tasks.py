"""Test VLMs on liquid container vision tasks.

This script evaluates one or more VLMs on visual understanding tasks for
multiple liquid-container datasets:
    - LabLiquidVision
    - Glasses in the Wild
    - DTLD

Supported models (--model flag):
    qwen   – Qwen3-VL (default)
    sdvlm  – SD-VLM  (requires the llava package from cpystan/SD-VLM)
    both   – run Qwen3-VL and SD-VLM on every sample, report metrics for each
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import sys

# Ensure `src/` is on PYTHONPATH so imports match README usage.
sys.path.append(str(Path(__file__).parent / "src"))

from tlg import PromptBank
from tlg.cli import parse_args, parse_data_root_overrides, parse_tol_overrides
from tlg.dataset_factory import DATA_ROOTS, build_dataset
from tlg.evaluation import TaskStats, score_prediction
from tlg.model_runner import MAX_NEW_TOKENS, MODEL_NAME, load_model, test_sample
from tlg.sdvlm_runner import (
    MAX_NEW_TOKENS as SDVLM_MAX_NEW_TOKENS,
    load_sdvlm_model,
    test_sample_sdvlm,
)
from tlg.reporting import print_ground_truth, print_stats, validate_annotations


def main():
    """Run all VLM tests."""
    args = parse_args()
    dataset_names = args.datasets

    tol_overrides = parse_tol_overrides(args.tol)
    data_root_overrides = parse_data_root_overrides(args.data_root)
    prompt_bank = PromptBank.default()

    model_choice = args.model  # "qwen" | "sdvlm" | "both"
    run_qwen = model_choice in ("qwen", "both")
    run_sdvlm = model_choice in ("sdvlm", "both")

    precomputed_depth_dir = Path(args.sdvlm_depth_dir) if args.sdvlm_depth_dir else None
    if run_sdvlm and args.sdvlm_no_depth and precomputed_depth_dir is not None:
        raise ValueError("--sdvlm-no-depth cannot be combined with --sdvlm-depth-dir.")

    # SD-VLM needs depth arrays in the sample when external/GT depth is set.
    use_gt_depth = bool(args.sdvlm_gt_depth) or precomputed_depth_dir is not None
    include_depth = run_sdvlm and bool(args.sdvlm_gt_depth) and precomputed_depth_dir is None

    if args.dry_run:
        print("Dry run: validating dataset loading + standardized annotations (no model).")

    viz_attn = bool(args.viz_attn) and (not args.dry_run)
    viz_out_dir = Path(args.viz_out) if args.viz_out else None
    if viz_attn and viz_out_dir is None:
        raise ValueError("--viz-attn requires --viz-out")
    debug_dtld_boxes = bool(args.debug_dtld_boxes)
    debug_image_out_dir = None
    if debug_dtld_boxes:
        if args.debug_image_out:
            debug_image_out_dir = Path(args.debug_image_out)
        elif viz_out_dir is not None:
            debug_image_out_dir = viz_out_dir / "debug_dtld_boxes"
        else:
            debug_image_out_dir = Path("results") / "debug_dtld_boxes"

    # --- Load models ---
    model = processor = None
    sdvlm_model = sdvlm_tokenizer = sdvlm_image_processor = None

    if not args.dry_run:
        if run_qwen:
            model, processor = load_model(viz_attn=viz_attn)
        if run_sdvlm:
            sdvlm_model, sdvlm_tokenizer, sdvlm_image_processor = load_sdvlm_model(
                model_path=args.sdvlm_model_path,
                vision_tower=args.sdvlm_vision_tower,
                use_depth=not args.sdvlm_no_depth,
                gt_depth=use_gt_depth,
                viz_attn=viz_attn,
            )

    pred_out_path = Path(args.predictions_out) if args.predictions_out else None
    metrics_out_path = Path(args.metrics_out) if args.metrics_out else None
    pred_fh = None
    if pred_out_path is not None:
        pred_out_path.parent.mkdir(parents=True, exist_ok=True)
        pred_fh = pred_out_path.open("w", encoding="utf-8")

    # Per-model stats containers.
    overall_stats: Dict[str, Dict[str, TaskStats]] = {
        "qwen": {},
        "sdvlm": {},
    }
    per_dataset_stats: Dict[str, Dict[str, Dict[str, TaskStats]]] = {
        "qwen": {},
        "sdvlm": {},
    }

    for dataset_name in dataset_names:
        data_root = data_root_overrides.get(dataset_name, DATA_ROOTS[dataset_name])

        print(f"\n{'=' * 60}")
        print(f"Dataset: {dataset_name}")
        print(f"Loading dataset from: {data_root}")

        if not data_root.exists():
            print("Dataset root does not exist; skipping.")
            continue

        dataset = build_dataset(
            dataset_name,
            data_root,
            prompt_bank=prompt_bank,
            include_depth=include_depth,
            shuffle=args.shuffle,
            seed=int(args.seed),
            debug_dtld_boxes=debug_dtld_boxes,
        )
        print(f"Dataset size: {len(dataset)}")
        task_pool_counts = _primary_task_counts(dataset)
        if task_pool_counts:
            print(f"Task pool sizes: {_format_counts(task_pool_counts)}")

        if len(dataset) == 0:
            print("Error: Dataset is empty!")
            continue

        max_samples = len(dataset) if args.max_samples is None else max(1, int(args.max_samples))
        indices = _select_indices(
            dataset,
            dataset_name=dataset_name,
            max_samples=max_samples,
            shuffle=bool(args.shuffle),
            seed=int(args.seed),
        )
        selected_task_counts = _primary_task_counts(dataset, indices) if task_pool_counts else {}
        if selected_task_counts:
            print(f"Selected samples: {len(indices)} total ({_format_counts(selected_task_counts)})")
        else:
            print(f"Selected samples: {len(indices)}")

        seen_task_counts: Dict[str, int] = {}
        for j, idx0 in enumerate(indices):
            idx = j + 1
            try:
                sample = dataset[idx0]
            except (FileNotFoundError, OSError, SyntaxError) as e:
                print(f"Failed to load sample index={idx0}: {e}")
                if args.dry_run:
                    break
                raise

            print("\n" + "-" * 60)
            primary_task = sample.get("primary_task") if isinstance(sample, dict) else None
            if primary_task and selected_task_counts:
                task = str(primary_task)
                seen_task_counts[task] = seen_task_counts.get(task, 0) + 1
                print(
                    f"Sample {idx}/{len(indices)} "
                    f"({task} {seen_task_counts[task]}/{selected_task_counts[task]})"
                )
            else:
                print(f"Sample {idx}/{len(indices)}")
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

            # --- Qwen inference ---
            if run_qwen:
                try:
                    qwen_results = test_sample(
                        model,
                        processor,
                        sample,
                        annotations,
                        dataset_name=dataset_name,
                        sample_id=str(sample.get("id", "unknown")),
                        viz_attn=viz_attn,
                        viz_out_dir=viz_out_dir,
                        viz_attn_mode=str(args.viz_attn_mode),
                        debug_dtld_boxes=debug_dtld_boxes,
                        debug_image_out_dir=debug_image_out_dir,
                    )
                except (FileNotFoundError, OSError, SyntaxError) as e:
                    print(f"[Qwen] Failed to load images for sample index={idx0}: {e}")
                    qwen_results = {}

                _record_results(
                    model_key="qwen",
                    results=qwen_results,
                    annotations=annotations,
                    sample=sample,
                    dataset_name=dataset_name,
                    tol_overrides=tol_overrides,
                    overall_stats=overall_stats,
                    per_dataset_stats=per_dataset_stats,
                    pred_fh=pred_fh,
                )

                print("\n[Qwen] Predictions:")
                for task_name, response in qwen_results.items():
                    print(f"  {task_name}: {response}")

            # --- SD-VLM inference ---
            if run_sdvlm:
                try:
                    if precomputed_depth_dir is not None:
                        _attach_precomputed_depth_map(sample, dataset_name, precomputed_depth_dir)
                    sdvlm_results = test_sample_sdvlm(
                        sdvlm_model,
                        sdvlm_tokenizer,
                        sdvlm_image_processor,
                        sample,
                        annotations,
                        dataset_name=dataset_name,
                        sample_id=str(sample.get("id", "unknown")),
                        viz_attn=viz_attn,
                        viz_out_dir=viz_out_dir,
                        viz_attn_mode=str(args.viz_attn_mode),
                        debug_dtld_boxes=debug_dtld_boxes,
                        debug_image_out_dir=debug_image_out_dir,
                    )
                except (FileNotFoundError, OSError, SyntaxError) as e:
                    print(f"[SD-VLM] Failed to load images for sample index={idx0}: {e}")
                    sdvlm_results = {}

                _record_results(
                    model_key="sdvlm",
                    results=sdvlm_results,
                    annotations=annotations,
                    sample=sample,
                    dataset_name=dataset_name,
                    tol_overrides=tol_overrides,
                    overall_stats=overall_stats,
                    per_dataset_stats=per_dataset_stats,
                    pred_fh=pred_fh,
                )

                print("\n[SD-VLM] Predictions:")
                for task_name, response in sdvlm_results.items():
                    print(f"  {task_name}: {response}")

    # Close prediction writer.
    if pred_fh is not None:
        pred_fh.close()

    # Print aggregated metrics per model.
    active_models = {"qwen": run_qwen, "sdvlm": run_sdvlm}
    for model_key in ("qwen", "sdvlm"):
        if not active_models[model_key]:
            continue
        label = "Qwen" if model_key == "qwen" else "SD-VLM"
        print_stats(f"[{label}] Overall Metrics", overall_stats[model_key])
        for ds in sorted(per_dataset_stats[model_key].keys()):
            print_stats(f"[{label}] Metrics for dataset={ds}", per_dataset_stats[model_key][ds])

    if metrics_out_path is not None:
        metrics_out_path.parent.mkdir(parents=True, exist_ok=True)
        payload: Dict[str, Any] = {
            "datasets": list(dataset_names),
            "max_samples": "all" if args.max_samples is None else int(args.max_samples),
        }
        if run_qwen:
            payload["qwen"] = {
                "model_name": MODEL_NAME,
                "max_new_tokens": int(MAX_NEW_TOKENS),
                "overall": {k: v.to_dict() for k, v in overall_stats["qwen"].items()},
                "per_dataset": {
                    ds: {k: v.to_dict() for k, v in stats.items()}
                    for ds, stats in per_dataset_stats["qwen"].items()
                },
            }
        if run_sdvlm:
            payload["sdvlm"] = {
                "model_name": args.sdvlm_model_path,
                "max_new_tokens": int(SDVLM_MAX_NEW_TOKENS),
                "use_depth": not args.sdvlm_no_depth,
                "gt_depth": use_gt_depth,
                "precomputed_depth_dir": str(precomputed_depth_dir) if precomputed_depth_dir is not None else None,
                "overall": {k: v.to_dict() for k, v in overall_stats["sdvlm"].items()},
                "per_dataset": {
                    ds: {k: v.to_dict() for k, v in stats.items()}
                    for ds, stats in per_dataset_stats["sdvlm"].items()
                },
            }
        metrics_out_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )


def _record_results(
    *,
    model_key: str,
    results: Dict[str, str],
    annotations: list,
    sample: Dict[str, Any],
    dataset_name: str,
    tol_overrides: Dict[str, float],
    overall_stats: Dict[str, Dict[str, TaskStats]],
    per_dataset_stats: Dict[str, Dict[str, Dict[str, TaskStats]]],
    pred_fh,
    extra_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
) -> None:
    """Update stats and write JSONL records for a single sample's results."""
    ds_stats = per_dataset_stats[model_key].setdefault(dataset_name, {})
    for ann in annotations:
        task = str(ann.metadata.get("task", ann.task_type))
        resp = results.get(task)
        if resp is None:
            continue

        rec = score_prediction(ann, resp, tol_overrides=tol_overrides)

        for stats_map in (overall_stats[model_key], ds_stats):
            st = stats_map.setdefault(task, TaskStats())
            st.update_from_record(rec)

        if pred_fh is not None:
            row = {
                "model": model_key,
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
            if extra_metadata is not None:
                row.update(extra_metadata.get(task, {}))
            for key in (
                "pairwise_correct",
                "pairwise_total",
                "pairwise_accuracy",
                "n_gt_boxes",
                "n_pred_boxes",
                "n_matched_boxes",
                "precision",
                "recall",
                "f1",
                "mean_iou",
                "iou_threshold",
            ):
                if key in rec:
                    row[key] = rec[key]
            pred_fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _attach_precomputed_depth_map(sample: Dict[str, Any], dataset_name: str, depth_dir: Path) -> None:
    depth_path = _precomputed_depth_path(sample, dataset_name, depth_dir)
    if not depth_path.exists():
        raise FileNotFoundError(f"Precomputed depth map not found: {depth_path}")
    sample["depth_map"] = np.load(depth_path)
    sample["depth_map_path"] = depth_path


def _precomputed_depth_path(sample: Dict[str, Any], dataset_name: str, depth_dir: Path) -> Path:
    if dataset_name == "dtld":
        scene_id = str(sample.get("scene_id", "unknown"))
        frame_id = sample.get("frame_id", "unknown")
        if isinstance(frame_id, int):
            frame_name = f"{frame_id:06d}"
        else:
            frame_name = _safe_path_part(str(frame_id))
        return depth_dir / dataset_name / _safe_path_part(scene_id) / f"{frame_name}.npy"

    sample_id = _safe_path_part(str(sample.get("id", "unknown")))
    return depth_dir / dataset_name / f"{sample_id}.npy"


def _safe_path_part(value: str) -> str:
    return value.replace("/", "__").replace("\\", "__")


def _select_indices(
    dataset,
    *,
    dataset_name: str,
    max_samples: int,
    shuffle: bool,
    seed: int,
) -> List[int]:
    """Select sample indices from the dataset.

    Dataset-specific composition should already be handled by the dataloader.
    """

    if shuffle and dataset_name == "dtld" and hasattr(dataset, "balanced_indices_by_scene"):
        return dataset.balanced_indices_by_scene(
            max_samples=min(len(dataset), max_samples),
            shuffle=True,
            seed=seed,
        )

    task_groups: Dict[str, List[int]] = {}
    for i in range(len(dataset)):
        try:
            sample = dataset[i]
        except (FileNotFoundError, OSError, SyntaxError):
            continue
        primary_task = sample.get("primary_task") if isinstance(sample, dict) else None
        if not primary_task:
            task_groups = {}
            break
        task_groups.setdefault(str(primary_task), []).append(i)

    if task_groups:
        import random

        selected: List[int] = []
        for task in sorted(task_groups):
            group = list(task_groups[task])
            if shuffle:
                rng = random.Random(f"{seed}:{dataset_name}:{task}")
                rng.shuffle(group)
            selected.extend(group[: min(len(group), max_samples)])
        if shuffle:
            rng = random.Random(seed)
            rng.shuffle(selected)
        return selected

    indices = list(range(len(dataset)))
    if shuffle:
        import random

        rng = random.Random(seed)
        rng.shuffle(indices)
    return indices[: min(len(indices), max_samples)]


def _primary_task_counts(dataset, indices: Optional[List[int]] = None) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    selected_indices = range(len(dataset)) if indices is None else indices
    for i in selected_indices:
        try:
            sample = dataset[i]
        except (FileNotFoundError, OSError, SyntaxError):
            continue
        primary_task = sample.get("primary_task") if isinstance(sample, dict) else None
        if not primary_task:
            return {}
        task = str(primary_task)
        counts[task] = counts.get(task, 0) + 1
    return counts


def _format_counts(counts: Dict[str, int]) -> str:
    return ", ".join(f"{task}={counts[task]}" for task in sorted(counts))


if __name__ == "__main__":
    main()

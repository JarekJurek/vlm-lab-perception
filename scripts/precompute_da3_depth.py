#!/usr/bin/env python3
"""Precompute DepthAnything 3 cache entries for SD-VLM task images."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from tlg import PromptBank
from tlg.dataset_factory import DATA_ROOTS, build_dataset
from tlg.reporting import validate_annotations
from tlg.sdvlm_runner import (
    DA3_MODEL_NAME,
    DepthAnything3Provider,
    build_sdvlm_task_inputs,
)


def main() -> None:
    args = parse_args()
    data_root_overrides = _parse_data_root_overrides(args.data_root)
    prompt_bank = PromptBank.default()
    provider = DepthAnything3Provider(
        model_path=str(args.da3_model_path),
        cache_dir=Path(args.da3_cache_dir),
        device=str(args.da3_device),
    )

    manifest_path = Path(args.manifest_out) if args.manifest_out else None
    manifest_fh = None
    if manifest_path is not None:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_fh = manifest_path.open("w", encoding="utf-8")

    total_tasks = 0
    total_hits = 0
    total_misses = 0
    try:
        for dataset_name in args.datasets:
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
                include_depth=False,
                shuffle=args.shuffle,
                seed=int(args.seed),
                debug_dtld_boxes=bool(args.debug_dtld_boxes),
            )
            indices = _select_indices(
                dataset,
                dataset_name=dataset_name,
                max_samples=len(dataset) if args.max_samples is None else max(1, int(args.max_samples)),
                shuffle=bool(args.shuffle),
                seed=int(args.seed),
            )
            print(f"Dataset size: {len(dataset)}")
            print(f"Selected samples: {len(indices)}")

            for idx, dataset_index in enumerate(indices, start=1):
                sample = dataset[dataset_index]
                sample_id = str(sample.get("id", "unknown"))
                annotations = sample.get("annotations")
                if not isinstance(annotations, list) or not annotations:
                    print(f"[{idx}/{len(indices)}] {sample_id}: no annotations; skipping.")
                    continue
                err = validate_annotations(sample)
                if err is not None:
                    print(f"[{idx}/{len(indices)}] {sample_id}: invalid annotations: {err}")
                    continue

                task_inputs = build_sdvlm_task_inputs(
                    sample,
                    annotations,
                    dataset_name=dataset_name,
                    sample_id=sample_id,
                    debug_dtld_boxes=bool(args.debug_dtld_boxes),
                    debug_image_out_dir=Path(args.debug_image_out) if args.debug_image_out else None,
                )
                print(f"[{idx}/{len(indices)}] {sample_id}: {len(task_inputs)} task images")

                for task_input in task_inputs:
                    _, meta = provider.get_depth_image(
                        image=task_input.image,
                        sample=sample,
                        sample_id=sample_id,
                        task_name=task_input.task_name,
                    )
                    total_tasks += 1
                    status = str(meta.get("sdvlm_depth_cache_status", ""))
                    if status == "hit":
                        total_hits += 1
                    elif status == "miss":
                        total_misses += 1
                    if manifest_fh is not None:
                        row = {
                            "dataset": dataset_name,
                            "sample_id": sample_id,
                            "task": task_input.task_name,
                            **meta,
                        }
                        manifest_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    finally:
        if manifest_fh is not None:
            manifest_fh.close()

    print("\nDone.")
    print(f"Task images processed: {total_tasks}")
    print(f"DA3 cache misses: {total_misses}")
    print(f"DA3 cache hits: {total_hits}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute DepthAnything 3 depth cache for SD-VLM inputs.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["labliquidvision", "dtld"],
        default=["labliquidvision", "dtld"],
        help="Datasets to precompute.",
    )
    parser.add_argument(
        "--data-root",
        action="append",
        default=[],
        help="Override a dataset root, e.g. --data-root dtld=/path/to/root.",
    )
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples per dataset/task selection.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle samples before applying --max-samples.")
    parser.add_argument("--seed", type=int, default=0, help="Shuffle seed.")
    parser.add_argument("--da3-model-path", default=DA3_MODEL_NAME, help="DepthAnything 3 model ID or local path.")
    parser.add_argument(
        "--da3-cache-dir",
        default="bakery/sdvlm_da3_all/da3_depth_cache",
        help="Directory for cached DepthAnything 3 depth arrays.",
    )
    parser.add_argument("--da3-device", default="cuda", help="DepthAnything 3 inference device.")
    parser.add_argument("--manifest-out", default="", help="Optional JSONL manifest of cache entries.")
    parser.add_argument("--debug-dtld-boxes", action="store_true", help="Print DTLD bbox state.")
    parser.add_argument(
        "--debug-image-out",
        default="",
        help="Optional directory for DTLD debug images when --debug-dtld-boxes is set.",
    )
    return parser.parse_args()


def _parse_data_root_overrides(items: List[str]) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for raw in items:
        if "=" not in raw:
            raise ValueError(f"--data-root must be dataset=/path. Got: {raw!r}")
        dataset, path = raw.split("=", 1)
        out[dataset.strip()] = Path(path.strip())
    return out


def _select_indices(
    dataset: Any,
    *,
    dataset_name: str,
    max_samples: int,
    shuffle: bool,
    seed: int,
) -> List[int]:
    if shuffle and dataset_name == "dtld" and hasattr(dataset, "balanced_indices_by_scene"):
        return dataset.balanced_indices_by_scene(
            max_samples=min(len(dataset), max_samples),
            shuffle=True,
            seed=seed,
        )

    indices = list(range(len(dataset)))
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(indices)
    return indices[: min(len(indices), max_samples)]


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Export current TLG dataloader samples to a reproducible JSON manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from datasets.labpicsv1 import LabPicsV1Dataset
from tlg import PromptBank
from tlg.cli import parse_data_root_overrides
from tlg.dataset_factory import DATA_ROOTS, build_dataset
from tlg.manifest import MANIFEST_VERSION, serialize_manifest_sample


def parse_args() -> argparse.Namespace:
    prompt_bank = PromptBank.default()
    parser = argparse.ArgumentParser(description="Export TLG samples to a split JSON manifest.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=sorted(prompt_bank.dataset_assignments.keys()),
        choices=sorted(prompt_bank.dataset_assignments.keys()),
        help="Datasets to export. Default: all datasets configured in dataset_prompts.yaml.",
    )
    parser.add_argument(
        "--data-root",
        action="append",
        default=[],
        help="Override a dataset root, e.g. --data-root dtld=/path/to/root.",
    )
    parser.add_argument("--out", type=Path, required=True, help="Output manifest JSON path.")
    parser.add_argument(
        "--max-samples-per-dataset",
        type=int,
        default=None,
        help="Optional cap applied after loading valid samples and before generated train/test splitting.",
    )
    parser.add_argument("--shuffle", action="store_true", help="Shuffle each dataset before capping and splitting.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for shuffling and generated split assignment.")
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Generated train split ratio per dataset. Default: 0.8.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_samples_per_dataset is not None and args.max_samples_per_dataset <= 0:
        raise ValueError("--max-samples-per-dataset must be positive when provided.")
    if not 0.0 <= float(args.train_ratio) <= 1.0:
        raise ValueError("--train-ratio must be between 0 and 1.")

    prompt_bank = PromptBank.default()
    data_root_overrides = parse_data_root_overrides(args.data_root)

    samples_out: List[Dict[str, Any]] = []
    dataset_counts: Dict[str, Dict[str, int]] = {}
    for dataset_name in args.datasets:
        data_root = data_root_overrides.get(dataset_name, DATA_ROOTS[dataset_name])
        print(f"\nDataset: {dataset_name}")
        print(f"Loading from: {data_root}")
        if not data_root.exists():
            print("  root does not exist; skipping")
            continue

        valid_samples = list(
            _iter_valid_samples(
                dataset_name,
                data_root,
                prompt_bank=prompt_bank,
                limit=args.max_samples_per_dataset,
            )
        )
        print(f"  valid samples with annotations: {len(valid_samples)}")

        if args.shuffle:
            rng = random.Random(_stable_seed(int(args.seed), dataset_name))
            rng.shuffle(valid_samples)

        train_count = int(len(valid_samples) * float(args.train_ratio))
        split_counts = {"train": 0, "test": 0}
        for i, sample in enumerate(valid_samples):
            generated_split = "train" if i < train_count else "test"
            samples_out.append(
                serialize_manifest_sample(
                    sample,
                    dataset_name=dataset_name,
                    split=generated_split,
                )
            )
            split_counts[generated_split] += 1

        dataset_counts[dataset_name] = split_counts
        print(f"  exported: train={split_counts['train']} test={split_counts['test']}")

    payload = {
        "version": MANIFEST_VERSION,
        "created_from": {
            "prompt_bank": str(PromptBank.default_path()),
            "dataset_prompts": str(PromptBank.default_assignments_path()),
        },
        "seed": int(args.seed),
        "train_ratio": float(args.train_ratio),
        "shuffle": bool(args.shuffle),
        "max_samples_per_dataset": args.max_samples_per_dataset,
        "datasets": dataset_counts,
        "samples": samples_out,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nWrote {len(samples_out)} samples to {args.out}")


def _iter_valid_samples(
    dataset_name: str,
    data_root: Path,
    *,
    prompt_bank: PromptBank,
    limit: Optional[int] = None,
):
    yielded = 0
    for dataset in _source_datasets(dataset_name, data_root, prompt_bank=prompt_bank):
        for idx in range(len(dataset)):
            try:
                sample = dataset[idx]
            except (FileNotFoundError, OSError, SyntaxError, ValueError) as exc:
                print(f"  skipping sample index={idx}: {exc}")
                continue
            annotations = sample.get("annotations") if isinstance(sample, dict) else None
            if isinstance(annotations, list) and annotations:
                yield sample
                yielded += 1
                if limit is not None and yielded >= int(limit):
                    return


def _source_datasets(dataset_name: str, data_root: Path, *, prompt_bank: PromptBank) -> Iterable[Any]:
    if dataset_name == "labpicsv1":
        return [
            LabPicsV1Dataset(data_root, split="train", prompt_bank=prompt_bank),
            LabPicsV1Dataset(data_root, split="test", prompt_bank=prompt_bank),
        ]
    return [
        build_dataset(
            dataset_name,
            data_root,
            prompt_bank=prompt_bank,
            include_depth=False,
            shuffle=False,
            seed=None,
        )
    ]


def _stable_seed(seed: int, label: str) -> int:
    digest = hashlib.sha256(f"{seed}:{label}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


if __name__ == "__main__":
    main()

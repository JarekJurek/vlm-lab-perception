#!/usr/bin/env python3
"""Precompute Marigold-DC depth-completion maps for dataset RGB/depth pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

import numpy as np
from PIL import Image, UnidentifiedImageError

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from datasets.dtld import DTLDDataset
from tlg import PromptBank
from tlg.dataset_factory import DATA_ROOTS, build_dataset
from tlg.reporting import validate_annotations


MARIGOLD_DC_MODULE = "marigold_dc"


def main() -> None:
    args = parse_args()
    data_root_overrides = _parse_data_root_overrides(args.data_root)
    prompt_bank = PromptBank.default()

    cache_dir = Path(args.marigold_dc_cache_dir)
    sparse_cache_dir = Path(args.sparse_depth_cache_dir) if args.sparse_depth_cache_dir else None
    manifest_path = Path(args.manifest_out) if args.manifest_out else None
    manifest_fh = None
    if manifest_path is not None:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_fh = manifest_path.open("w", encoding="utf-8")

    total_images = 0
    total_hits = 0
    total_misses = 0
    total_skipped_missing_depth = 0
    seen_depth_paths: Set[Path] = set()

    try:
        for dataset_name in args.datasets:
            data_root = data_root_overrides.get(dataset_name, DATA_ROOTS[dataset_name])
            print(f"\n{'=' * 60}")
            print(f"Dataset: {dataset_name}")
            print(f"Loading dataset from: {data_root}")
            if not data_root.exists():
                print("Dataset root does not exist; skipping.")
                continue

            dataset = _build_dataset_with_depth(
                dataset_name,
                data_root,
                prompt_bank=prompt_bank,
                shuffle=bool(args.shuffle),
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

                dense_depth = _load_dataset_depth(
                    sample,
                    dataset_name=dataset_name,
                    dtld_depth_scale=float(args.dtld_depth_scale),
                    labliquidvision_depth_scale=float(args.labliquidvision_depth_scale),
                )
                if dense_depth is None:
                    total_skipped_missing_depth += 1
                    print(f"[{idx}/{len(indices)}] {sample_id}: no dataset depth; skipping.")
                    continue

                depth_path = _depth_output_path(cache_dir, dataset_name, sample)
                if depth_path in seen_depth_paths:
                    print(f"[{idx}/{len(indices)}] {sample_id}: duplicate RGB/depth frame; already queued.")
                    continue
                seen_depth_paths.add(depth_path)
                depth_path.parent.mkdir(parents=True, exist_ok=True)

                image = _load_original_rgb_image(sample)
                sparse_depth = _make_sparse_depth(
                    dense_depth,
                    max_points=int(args.max_guidance_points),
                    sample_fraction=args.guidance_sample_fraction,
                    seed=_stable_sample_seed(int(args.seed), dataset_name, sample_id),
                )
                guidance_points = int(np.count_nonzero(sparse_depth > 0.0))
                if guidance_points == 0:
                    total_skipped_missing_depth += 1
                    print(f"[{idx}/{len(indices)}] {sample_id}: no valid depth guidance; skipping.")
                    continue

                sparse_depth_path = None
                if sparse_cache_dir is not None:
                    sparse_depth_path = _depth_output_path(sparse_cache_dir, dataset_name, sample)
                    sparse_depth_path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(sparse_depth_path, sparse_depth.astype(np.float32, copy=False))

                if depth_path.exists() and not args.overwrite:
                    depth = np.load(depth_path)
                    status = "hit"
                else:
                    depth = _run_marigold_dc(
                        image=image,
                        sparse_depth=sparse_depth,
                        out_depth_path=depth_path,
                        python_executable=str(args.python_executable),
                        module=str(args.marigold_dc_module),
                        checkpoint=str(args.checkpoint) if args.checkpoint else "",
                        num_inference_steps=int(args.num_inference_steps),
                        ensemble_size=int(args.ensemble_size),
                        processing_resolution=int(args.processing_resolution),
                        use_full_precision=bool(args.use_full_precision),
                        use_tiny_vae=bool(args.use_tiny_vae),
                        extra_args=args.marigold_extra_args,
                    )
                    status = "miss" if not args.overwrite else "overwrite"

                total_images += 1
                if status == "hit":
                    total_hits += 1
                else:
                    total_misses += 1
                print(
                    f"[{idx}/{len(indices)}] {sample_id}: Marigold-DC depth cache "
                    f"{status}: {depth_path} ({guidance_points} guidance px)"
                )

                if manifest_fh is not None:
                    row = {
                        "dataset": dataset_name,
                        "sample_id": sample_id,
                        "image_path": str(sample.get("rgb_image", "")),
                        "depth_path": str(depth_path),
                        "depth_shape": list(np.asarray(depth).shape),
                        "cache_status": status,
                        "sparse_depth_path": str(sparse_depth_path) if sparse_depth_path is not None else "",
                        "guidance_points": guidance_points,
                        "max_guidance_points": int(args.max_guidance_points),
                        "guidance_sample_fraction": args.guidance_sample_fraction,
                        "checkpoint": str(args.checkpoint),
                    }
                    manifest_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    finally:
        if manifest_fh is not None:
            manifest_fh.close()

    print("\nDone.")
    print(f"Original RGB/depth pairs processed: {total_images}")
    print(f"Marigold-DC cache misses: {total_misses}")
    print(f"Marigold-DC cache hits: {total_hits}")
    print(f"Skipped samples without valid depth: {total_skipped_missing_depth}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute Marigold-DC depth-completion maps for original RGB images."
    )
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
    parser.add_argument(
        "--marigold-dc-cache-dir",
        default="bakery/sdvlm_marigold_dc_all/marigold_dc_depth_cache",
        help="Directory for cached Marigold-DC dense depth arrays.",
    )
    parser.add_argument(
        "--sparse-depth-cache-dir",
        default="",
        help="Optional directory to save the sparse guidance arrays passed to Marigold-DC.",
    )
    parser.add_argument(
        "--max-guidance-points",
        type=int,
        default=100,
        help="Maximum valid depth pixels to keep as sparse guidance. Use 0 to keep all selected pixels.",
    )
    parser.add_argument(
        "--guidance-sample-fraction",
        type=float,
        default=None,
        help=(
            "Optional fraction of valid depth pixels to sample before applying --max-guidance-points. "
            "By default, only --max-guidance-points controls sparsity."
        ),
    )
    parser.add_argument(
        "--dtld-depth-scale",
        type=float,
        default=1000.0,
        help="Divisor converting DTLD depth PNG values to meters (default assumes millimeters).",
    )
    parser.add_argument(
        "--labliquidvision-depth-scale",
        type=float,
        default=1.0,
        help="Divisor converting LabLiquidVision depth arrays to meters.",
    )
    parser.add_argument("--python-executable", default=sys.executable, help="Python executable used to run Marigold-DC.")
    parser.add_argument("--marigold-dc-module", default=MARIGOLD_DC_MODULE, help="Python module name for Marigold-DC.")
    parser.add_argument("--checkpoint", default="", help="Optional Marigold-DC checkpoint argument.")
    parser.add_argument("--num-inference-steps", type=int, default=50, help="Marigold-DC diffusion steps.")
    parser.add_argument("--ensemble-size", type=int, default=1, help="Marigold-DC ensemble size.")
    parser.add_argument("--processing-resolution", type=int, default=768, help="Marigold-DC processing resolution.")
    parser.add_argument("--use-full-precision", action="store_true", help="Pass --use_full_precision to Marigold-DC.")
    parser.add_argument("--use-tiny-vae", action="store_true", help="Pass --use_tiny_vae to Marigold-DC.")
    parser.add_argument("--overwrite", action="store_true", help="Recompute existing cache files.")
    parser.add_argument("--manifest-out", default="", help="Optional JSONL manifest of cache entries.")
    parser.add_argument("--debug-dtld-boxes", action="store_true", help="Print DTLD bbox state.")
    parser.add_argument(
        "--marigold-extra-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Additional raw arguments passed after the Marigold-DC command. Put this option last.",
    )
    return parser.parse_args()


def _build_dataset_with_depth(
    dataset_name: str,
    data_root: Path,
    *,
    prompt_bank: PromptBank,
    shuffle: bool,
    seed: int,
    debug_dtld_boxes: bool,
) -> Any:
    if dataset_name == "dtld":
        return DTLDDataset(
            data_root,
            rgb_only=False,
            prompt_bank=prompt_bank,
            shuffle=shuffle,
            seed=seed,
            debug_boxes=debug_dtld_boxes,
        )
    return build_dataset(
        dataset_name,
        data_root,
        prompt_bank=prompt_bank,
        include_depth=True,
        shuffle=shuffle,
        seed=seed,
        debug_dtld_boxes=debug_dtld_boxes,
    )


def _parse_data_root_overrides(items: List[str]) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for raw in items:
        if "=" not in raw:
            raise ValueError(f"--data-root must be dataset=/path. Got: {raw!r}")
        dataset, path = raw.split("=", 1)
        out[dataset.strip()] = Path(path.strip())
    return out


def _load_original_rgb_image(sample: Dict[str, Any]) -> Image.Image:
    rgb_path = sample.get("rgb_image")
    if rgb_path is not None:
        try:
            return Image.open(rgb_path).convert("RGB")
        except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
            raise OSError(f"Failed to open RGB image: {rgb_path} ({exc})") from exc

    rgb_array = sample.get("rgb")
    if rgb_array is None:
        raise FileNotFoundError("Sample has neither 'rgb_image' nor 'rgb'.")
    arr = np.asarray(rgb_array)
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
        arr = arr.transpose(1, 2, 0)
    return Image.fromarray(arr.astype("uint8")).convert("RGB")


def _load_dataset_depth(
    sample: Dict[str, Any],
    *,
    dataset_name: str,
    dtld_depth_scale: float,
    labliquidvision_depth_scale: float,
) -> Optional[np.ndarray]:
    if dataset_name == "labliquidvision":
        depth = sample.get("depth_map")
        if depth is None:
            return None
        return _sanitize_depth(np.asarray(depth, dtype=np.float32), scale=labliquidvision_depth_scale)

    if dataset_name == "dtld":
        depth_path = sample.get("depth_image")
        if depth_path is None:
            return None
        try:
            depth_image = Image.open(depth_path)
        except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
            raise OSError(f"Failed to open DTLD depth image: {depth_path} ({exc})") from exc
        return _sanitize_depth(np.asarray(depth_image, dtype=np.float32), scale=dtld_depth_scale)

    raise ValueError(f"Unsupported dataset: {dataset_name}")


def _sanitize_depth(depth: np.ndarray, *, scale: float) -> np.ndarray:
    arr = np.asarray(depth, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
        arr = arr[0]
    elif arr.ndim == 3 and arr.shape[-1] in (1, 3, 4):
        arr = arr[..., 0]
    if scale <= 0.0:
        raise ValueError(f"Depth scale must be positive, got {scale}.")
    arr = arr / float(scale)
    arr[~np.isfinite(arr)] = 0.0
    arr[arr < 0.0] = 0.0
    return arr.astype(np.float32, copy=False)


def _make_sparse_depth(
    dense_depth: np.ndarray,
    *,
    max_points: int,
    sample_fraction: Optional[float],
    seed: int,
) -> np.ndarray:
    arr = np.asarray(dense_depth, dtype=np.float32)
    valid_flat = np.flatnonzero(np.isfinite(arr.reshape(-1)) & (arr.reshape(-1) > 0.0))
    if valid_flat.size == 0:
        return np.zeros_like(arr, dtype=np.float32)

    target_count = int(valid_flat.size)
    if sample_fraction is not None:
        if not 0.0 < sample_fraction <= 1.0:
            raise ValueError("--guidance-sample-fraction must be in (0, 1].")
        target_count = max(1, int(round(valid_flat.size * sample_fraction)))
    if max_points > 0:
        target_count = min(target_count, int(max_points))

    rng = np.random.default_rng(seed)
    selected = valid_flat if target_count >= valid_flat.size else rng.choice(valid_flat, size=target_count, replace=False)
    sparse = np.zeros(arr.size, dtype=np.float32)
    sparse[selected] = arr.reshape(-1)[selected]
    return sparse.reshape(arr.shape)


def _run_marigold_dc(
    *,
    image: Image.Image,
    sparse_depth: np.ndarray,
    out_depth_path: Path,
    python_executable: str,
    module: str,
    checkpoint: str,
    num_inference_steps: int,
    ensemble_size: int,
    processing_resolution: int,
    use_full_precision: bool,
    use_tiny_vae: bool,
    extra_args: Sequence[str],
) -> np.ndarray:
    with tempfile.TemporaryDirectory(prefix="marigold_dc_") as tmpdir_raw:
        tmpdir = Path(tmpdir_raw)
        image_path = tmpdir / "rgb.png"
        sparse_path = tmpdir / "sparse_depth.npy"
        image.save(image_path)
        np.save(sparse_path, sparse_depth.astype(np.float32, copy=False))

        cmd = [
            python_executable,
            "-m",
            module,
            "--in-image",
            str(image_path),
            "--in-depth",
            str(sparse_path),
            "--out-depth",
            str(out_depth_path),
            "--num_inference_steps",
            str(num_inference_steps),
            "--ensemble_size",
            str(ensemble_size),
            "--processing_resolution",
            str(processing_resolution),
        ]
        if checkpoint:
            cmd.extend(["--checkpoint", checkpoint])
        if use_full_precision:
            cmd.append("--use_full_precision")
        if use_tiny_vae:
            cmd.append("--use_tiny_vae")
        cmd.extend(extra_args)

        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"Marigold-DC command failed with exit code {result.returncode}: {' '.join(cmd)}")

    if not out_depth_path.exists():
        raise RuntimeError(f"Marigold-DC finished but did not create output depth: {out_depth_path}")
    return np.load(out_depth_path)


def _depth_output_path(cache_dir: Path, dataset_name: str, sample: Dict[str, Any]) -> Path:
    if dataset_name == "dtld":
        scene_id = str(sample.get("scene_id", "unknown"))
        frame_id = sample.get("frame_id", "unknown")
        if isinstance(frame_id, int):
            frame_name = f"{frame_id:06d}"
        else:
            frame_name = _safe_path_part(str(frame_id))
        return cache_dir / dataset_name / _safe_path_part(scene_id) / f"{frame_name}.npy"

    sample_id = _safe_path_part(str(sample.get("id", "unknown")))
    return cache_dir / dataset_name / f"{sample_id}.npy"


def _safe_path_part(value: str) -> str:
    return value.replace("/", "__").replace("\\", "__")


def _stable_sample_seed(seed: int, dataset_name: str, sample_id: str) -> int:
    payload = f"{seed}:{dataset_name}:{sample_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], byteorder="big", signed=False)


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

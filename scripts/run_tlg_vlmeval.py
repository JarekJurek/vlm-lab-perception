#!/usr/bin/env python3
"""Run the TLG manifest benchmark through a VLMEvalKit-compatible path."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from tlg.vlmevalkit_adapter import (  # noqa: E402
    TLGVLMEvalDataset,
    load_prediction_table,
    score_vlmeval_predictions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate TLG manifest rows with VLMEvalKit models.")
    parser.add_argument("--manifest", required=True, help="Path to exported TLG manifest JSON.")
    parser.add_argument(
        "--manifest-split",
        choices=["train", "test", "all"],
        default="test",
        help="Generated manifest split to evaluate. Default: test.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Optional manifest dataset names to include.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="Optional task names to include, e.g. fill_percentage container_type.",
    )
    parser.add_argument("--model", default="", help="VLMEvalKit model key. Required unless --predictions-in or --dry-run is used.")
    parser.add_argument("--work-dir", default="results/tlg_vlmeval", help="Output directory.")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum flattened task rows to run.")
    parser.add_argument("--resume", action="store_true", help="Skip rows already present in the raw prediction JSONL.")
    parser.add_argument("--predictions-in", default="", help="Existing VLMEvalKit-style prediction file to rescore.")
    parser.add_argument("--predictions-out", default="", help="Compatibility JSONL output path.")
    parser.add_argument("--metrics-out", default="", help="Compatibility metrics JSON output path.")
    parser.add_argument("--raw-predictions-out", default="", help="Raw VLMEvalKit-style JSONL output path.")
    parser.add_argument("--dump-rows-out", default="", help="Optional JSONL dump of flattened VLMEvalKit rows.")
    parser.add_argument("--dry-run", action="store_true", help="Build prompts and validate conversion without model inference.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    work_dir = Path(args.work_dir)
    model_key = str(args.model).strip()
    model_slug = _safe_path_part(model_key or "predictions")
    raw_pred_path = Path(args.raw_predictions_out) if args.raw_predictions_out else work_dir / model_slug / "vlmeval_predictions.jsonl"
    pred_out_path = Path(args.predictions_out) if args.predictions_out else work_dir / model_slug / "preds.jsonl"
    metrics_out_path = Path(args.metrics_out) if args.metrics_out else work_dir / model_slug / "metrics.json"
    image_cache_dir = work_dir / "image_cache"

    dataset = TLGVLMEvalDataset(
        manifest_path,
        split=str(args.manifest_split),
        datasets=args.datasets,
        tasks=args.tasks,
        image_cache_dir=image_cache_dir,
    )
    rows = dataset.data
    if args.max_samples is not None:
        rows = rows[: max(0, int(args.max_samples))]
    print(f"Manifest: {manifest_path}")
    print(f"Split: {args.manifest_split}")
    print(f"Flattened task rows: {len(rows)}")
    if args.dump_rows_out:
        _write_jsonl(Path(args.dump_rows_out), rows)
        print(f"Wrote flattened rows: {args.dump_rows_out}")

    if args.dry_run:
        _dry_run_prompts(dataset, rows)
        return

    if args.predictions_in:
        raw_rows = load_prediction_table(args.predictions_in)
        print(f"Loaded prediction rows: {len(raw_rows)}")
    else:
        if not model_key:
            raise ValueError("--model is required unless --predictions-in or --dry-run is used.")
        model = _build_vlmeval_model(model_key)
        completed = _load_completed_indices(raw_pred_path) if args.resume else set()
        raw_rows = _run_inference(
            dataset=dataset,
            rows=rows,
            model=model,
            model_key=model_key,
            dataset_name="TLG",
            raw_pred_path=raw_pred_path,
            completed_indices=completed,
        )

    scored = score_vlmeval_predictions(raw_rows)
    pred_out_path.parent.mkdir(parents=True, exist_ok=True)
    with pred_out_path.open("w", encoding="utf-8") as fh:
        for row in scored["predictions"]:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    metrics_payload: Dict[str, Any] = {
        "manifest": str(manifest_path),
        "manifest_split": str(args.manifest_split),
        "datasets": args.datasets or "all",
        "tasks": args.tasks or "all",
        "model": model_key or None,
        "max_samples": "all" if args.max_samples is None else int(args.max_samples),
        **scored["metrics"],
    }
    metrics_out_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_out_path.write_text(json.dumps(metrics_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote predictions: {pred_out_path}")
    print(f"Wrote metrics: {metrics_out_path}")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def _dry_run_prompts(dataset: TLGVLMEvalDataset, rows: List[Mapping[str, Any]]) -> None:
    if not rows:
        print("Dry run: no rows selected.")
        return
    first = rows[0]
    messages = dataset.build_prompt(first)
    print("Dry run OK: built first VLMEvalKit prompt.")
    print(f"First row index: {first.get('index')}")
    print(f"First row task: {first.get('task')}")
    print(f"Prompt message types: {[msg.get('type') for msg in messages]}")


def _build_vlmeval_model(model_key: str) -> Any:
    try:
        config = importlib.import_module("vlmeval.config")
    except ImportError as exc:
        raise ImportError(
            "VLMEvalKit is not installed. Install it before running inference, "
            "or pass --predictions-in to rescore an existing file."
        ) from exc

    supported = _vlmeval_supported_models(config)
    if model_key not in supported:
        known = ", ".join(sorted(str(k) for k in supported.keys())[:40])
        raise ValueError(f"Unknown VLMEvalKit model key: {model_key!r}. Known examples: {known}")
    return supported[model_key]()


def _vlmeval_supported_models(config: Any) -> Dict[str, Callable[[], Any]]:
    """Return a flattened VLMEvalKit model registry across supported versions."""
    candidates: Dict[str, Callable[[], Any]] = {}
    for attr in ("supported_VLM", "supported_VLMs", "SUPPORTED_VLM", "SUPPORTED_VLMS"):
        value = getattr(config, attr, None)
        if isinstance(value, Mapping):
            candidates.update(_callable_items(value))

    for _, value in vars(config).items():
        if isinstance(value, Mapping):
            candidates.update(_callable_items(value))
    return candidates


def _callable_items(value: Mapping[Any, Any]) -> Dict[str, Callable[[], Any]]:
    out: Dict[str, Callable[[], Any]] = {}
    for key, item in value.items():
        if callable(item):
            out[str(key)] = item
        elif isinstance(item, Mapping):
            out.update(_callable_items(item))
    return out


def _run_inference(
    *,
    dataset: TLGVLMEvalDataset,
    rows: Iterable[Mapping[str, Any]],
    model: Any,
    model_key: str,
    dataset_name: str,
    raw_pred_path: Path,
    completed_indices: set[str],
) -> List[Dict[str, Any]]:
    raw_pred_path.parent.mkdir(parents=True, exist_ok=True)
    row_list = list(rows)
    selected_indices = {str(row.get("index", "")) for row in row_list}
    resuming = bool(completed_indices)
    all_rows = (
        [
            row
            for row in load_prediction_table(raw_pred_path)
            if str(row.get("index", "")) in selected_indices
        ]
        if resuming and raw_pred_path.exists()
        else []
    )
    mode = "a" if resuming and raw_pred_path.exists() else "w"
    with raw_pred_path.open(mode, encoding="utf-8") as fh:
        for i, row in enumerate(row_list, start=1):
            index = str(row.get("index", ""))
            if index in completed_indices:
                continue
            messages = dataset.build_prompt(row)
            prediction = _generate(model, messages, dataset_name=dataset_name)
            out = dict(row)
            out["model"] = model_key
            out["prediction"] = prediction
            fh.write(json.dumps(out, ensure_ascii=False) + "\n")
            fh.flush()
            all_rows.append(out)
            print(f"[{i}] {row.get('dataset')} {row.get('sample_id')} {row.get('task')}: {prediction}")
    return all_rows


def _generate(model: Any, messages: List[Dict[str, str]], *, dataset_name: str) -> str:
    if hasattr(model, "generate"):
        try:
            return str(model.generate(messages, dataset=dataset_name))
        except TypeError:
            return str(model.generate(messages))
    if hasattr(model, "generate_inner"):
        try:
            return str(model.generate_inner(messages, dataset=dataset_name))
        except TypeError:
            return str(model.generate_inner(messages))
    raise TypeError("VLMEvalKit model object must expose generate() or generate_inner().")


def _load_completed_indices(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed = set()
    for row in load_prediction_table(path):
        if row.get("prediction") is not None and row.get("index") is not None:
            completed.add(str(row["index"]))
    return completed


def _safe_path_part(value: object) -> str:
    text = str(value).strip() or "item"
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in text)


if __name__ == "__main__":
    main()

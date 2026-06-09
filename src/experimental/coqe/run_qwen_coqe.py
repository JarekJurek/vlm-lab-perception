#!/usr/bin/env python3
"""Run standalone Qwen inference and evaluation on COQE tasks."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from datasets.coqe import COQE_TASKS, COQEExpDataset, COQETaskExample  # noqa: E402
from experimental.coqe import PROMPT_VERSION  # noqa: E402
from experimental.coqe.metrics import aggregate_records, score_prediction  # noqa: E402
from experimental.coqe.overlays import maybe_save_input_image, render_example_image  # noqa: E402
from experimental.coqe.parsing import parse_prediction  # noqa: E402
from experimental.coqe.prompts import build_prompt  # noqa: E402


DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_QWEN_MAX_PIXELS = 512 * 28 * 28
DEFAULT_MAX_NEW_TOKENS = 128


def main() -> None:
    args = parse_args()
    dataset = COQEExpDataset(
        args.data_root,
        split=args.split,
        shuffle=bool(args.shuffle),
        seed=int(args.seed),
    )
    print(f"COQE root: {dataset.root}")
    print(f"Split: {dataset.split}")
    print(f"Task pool sizes: {dataset.task_counts()}")
    print(f"Pouring records per image distribution: {dataset.pouring_records_per_image_distribution()}")

    examples = select_examples(
        dataset.iter_examples(args.tasks),
        tasks=args.tasks,
        max_samples=args.max_samples,
    )
    print(f"Selected examples: {len(examples)} ({_counts_by_task(examples)})")

    if args.dry_run:
        validate_examples(examples, input_images_out=Path(args.input_images_out) if args.input_images_out else None)
        print("Dry run OK: COQE examples, prompts, labels, and rendered inputs are valid.")
        return

    model, processor = load_qwen_model(args.model)

    pred_out_path = Path(args.predictions_out) if args.predictions_out else None
    pred_fh = None
    if pred_out_path is not None:
        pred_out_path.parent.mkdir(parents=True, exist_ok=True)
        pred_fh = pred_out_path.open("w", encoding="utf-8")

    input_images_out = Path(args.input_images_out) if args.input_images_out else None
    records: List[Dict[str, Any]] = []
    try:
        for index, example in enumerate(examples, start=1):
            print(f"[{index}/{len(examples)}] {example.task}: {example.id}")
            prompt = build_prompt(example)
            image = render_example_image(example)
            maybe_save_input_image(image, input_images_out, example)
            raw_response = run_qwen_task(
                model,
                processor,
                image,
                prompt,
                max_new_tokens=int(args.max_new_tokens),
            )
            parsed, valid, parse_error = parse_prediction(example.task, raw_response)
            score = score_prediction(example.task, parsed, example.answer, valid)
            row = make_prediction_row(
                example=example,
                prompt=prompt,
                raw_response=raw_response,
                parsed=parsed,
                valid=valid,
                parse_error=parse_error,
                score=score,
                model_name=str(args.model),
            )
            records.append(row)
            if pred_fh is not None:
                pred_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                pred_fh.flush()
    finally:
        if pred_fh is not None:
            pred_fh.close()

    metrics = {
        "dataset": "coqe",
        "split": args.split,
        "model": args.model,
        "prompt_version": PROMPT_VERSION,
        "tasks": list(args.tasks),
        "max_samples": args.max_samples if args.max_samples is not None else "all",
        "metrics": aggregate_records(records),
    }
    print(json.dumps(metrics["metrics"], indent=2, ensure_ascii=False))

    if args.metrics_out:
        metrics_out_path = Path(args.metrics_out)
        metrics_out_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_out_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone Qwen evaluation for COQE.")
    parser.add_argument("--data-root", default="data/COQE_dataset", help="Path to COQE dataset root.")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test", help="COQE split to evaluate.")
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=list(COQE_TASKS),
        default=list(COQE_TASKS),
        help="COQE tasks to run.",
    )
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum examples per task.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HuggingFace Qwen model ID or local path.")
    parser.add_argument("--predictions-out", default="", help="Optional JSONL predictions output path.")
    parser.add_argument("--metrics-out", default="", help="Optional JSON metrics output path.")
    parser.add_argument("--input-images-out", default="", help="Optional directory for rendered model input images.")
    parser.add_argument("--dry-run", action="store_true", help="Validate data and prompts without loading Qwen.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle records before selecting examples.")
    parser.add_argument("--seed", type=int, default=0, help="Shuffle seed.")
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS, help="Qwen generation cap.")
    return parser.parse_args()


def select_examples(
    examples: Iterable[COQETaskExample],
    *,
    tasks: Sequence[str],
    max_samples: Optional[int],
) -> List[COQETaskExample]:
    if max_samples is None:
        return list(examples)

    limit = max(1, int(max_samples))
    wanted = set(str(task) for task in tasks)
    counts: Dict[str, int] = {task: 0 for task in wanted}
    out: List[COQETaskExample] = []
    for example in examples:
        task = example.task
        if counts.get(task, 0) >= limit:
            continue
        out.append(example)
        counts[task] = counts.get(task, 0) + 1
        if all(counts.get(task, 0) >= limit for task in wanted):
            break
    return out


def validate_examples(examples: Sequence[COQETaskExample], *, input_images_out: Optional[Path]) -> None:
    for example in examples:
        prompt = build_prompt(example)
        if not prompt.strip():
            raise ValueError(f"Empty prompt for example: {example.id}")
        image = render_example_image(example)
        if image.width <= 0 or image.height <= 0:
            raise ValueError(f"Invalid rendered image size for example: {example.id}")
        maybe_save_input_image(image, input_images_out, example)
        parsed, valid, parse_error = parse_prediction(example.task, _gold_response(example))
        if not valid:
            raise ValueError(f"Gold response failed parser for {example.id}: {parse_error}")
        if parsed != example.answer:
            raise ValueError(f"Gold response mismatch for {example.id}: parsed={parsed!r}, gt={example.answer!r}")


def load_qwen_model(model_name: str):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor_kwargs: Dict[str, object] = {}
    max_pixels = _env_int("QWEN_MAX_PIXELS", DEFAULT_QWEN_MAX_PIXELS)
    min_pixels = _env_int("QWEN_MIN_PIXELS")
    if max_pixels is not None:
        processor_kwargs["max_pixels"] = max_pixels
    if min_pixels is not None:
        processor_kwargs["min_pixels"] = min_pixels

    print(f"Loading Qwen model: {model_name}")
    processor = AutoProcessor.from_pretrained(model_name, **processor_kwargs)
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        dtype=torch.float16,
        device_map="auto",
    ).eval()
    print("Qwen model loaded.")
    return model, processor


def run_qwen_task(model: Any, processor: Any, image: Any, prompt: str, *, max_new_tokens: int) -> str:
    import torch

    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
    input_ids = inputs.get("input_ids")
    image_grid = inputs.get("image_grid_thw")
    if input_ids is not None:
        msg = f"  Qwen input tokens: {int(input_ids.shape[-1])}"
        if image_grid is not None:
            grid_values = image_grid.detach().cpu().tolist()
            visual_tokens = sum(int(t) * int(h) * int(w) for t, h, w in grid_values)
            msg += f", image_grid_thw={grid_values}, visual_tokens={visual_tokens}"
        print(msg)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=int(max_new_tokens), do_sample=False)

    response = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
    if prompt in response:
        response = response.split(prompt)[-1].strip()
    return response.strip()


def make_prediction_row(
    *,
    example: COQETaskExample,
    prompt: str,
    raw_response: str,
    parsed: Any,
    valid: bool,
    parse_error: Optional[str],
    score: Dict[str, Any],
    model_name: str,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "sample_id": example.id,
        "task": example.task,
        "split": example.split,
        "image_id": example.image_id,
        "model": model_name,
        "prompt_version": PROMPT_VERSION,
        "prompt": prompt,
        "raw_response": raw_response,
        "parsed_prediction": parsed,
        "ground_truth": example.answer,
        "valid": bool(valid),
        "parse_error": parse_error,
        "correct": bool(score.get("correct")),
        "metadata": {
            "bbox": example.bbox_metadata,
            "angle": example.angle,
            "source_category": example.source.category,
            "target_category": example.target.category if example.target is not None else None,
        },
    }
    for key, value in score.items():
        if key != "correct":
            row[key] = value
    return row


def _gold_response(example: COQETaskExample) -> str:
    if example.task == "volume":
        return json.dumps({"volume_bin": example.answer})
    if example.task == "content":
        return json.dumps({"fill_level": example.answer})
    if example.task == "comparative":
        return json.dumps({"can_pour": example.answer})
    if example.task == "pouring":
        return json.dumps({"remaining_sequence": example.answer})
    raise ValueError(f"Unknown task: {example.task!r}")


def _counts_by_task(examples: Sequence[COQETaskExample]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for example in examples:
        out[example.task] = out.get(example.task, 0) + 1
    return dict(sorted(out.items()))


def _env_int(name: str, default: Optional[int] = None) -> Optional[int]:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


if __name__ == "__main__":
    main()

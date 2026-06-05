#!/usr/bin/env python3
"""Calculate ground-truth label histograms for configured dataset tasks.

By default the script reads annotations from the current dataset loaders, which
keeps the output aligned with the implemented datasets and tasks. It can also
read evaluation JSONL files containing `dataset`, `task`, and `gt` fields with
`--source predictions`.

Multi-label material annotations are counted per individual label because that
is the useful class-balance view for that task.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import yaml

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from tlg import PromptBank


NUMERIC_BIN_TASKS = {"fill_percentage": 10.0, "fill_height_mm": 10.0}
GROUNDING_COUNT_TASKS = {"liquid_grounding"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate label distributions for configured dataset tasks."
    )
    parser.add_argument(
        "--source",
        choices=["datasets", "predictions"],
        default="datasets",
        help=(
            "Read ground truth from current dataset loaders, or from existing "
            "*/preds.jsonl files. Default: datasets."
        ),
    )
    parser.add_argument(
        "--pred-root",
        type=Path,
        default=Path("bakery/qwen_by_dataset"),
        help="Directory containing per-dataset */preds.jsonl files.",
    )
    parser.add_argument(
        "--dataset-prompts",
        type=Path,
        default=Path("src/tlg/dataset_prompts.yaml"),
        help="Dataset-to-task assignment YAML.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Optional subset of dataset names. Default: all datasets in the assignment YAML.",
    )
    parser.add_argument(
        "--data-root",
        action="append",
        default=[],
        help="Override a dataset root, e.g. --data-root dtld=/path/to/root.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional max samples per dataset when --source datasets is used.",
    )
    parser.add_argument("--shuffle", action="store_true", help="Shuffle datasets before applying --max-samples.")
    parser.add_argument("--seed", type=int, default=0, help="Shuffle seed.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("bakery/qwen_by_dataset/label_distribution"),
        help="Output directory for PNG summaries.",
    )
    parser.add_argument(
        "--png",
        action="store_true",
        help="Deprecated: PNGs are always written.",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Also write label_distribution.csv and label_distribution_by_task.csv.",
    )
    parser.add_argument(
        "--markdown",
        action="store_true",
        help="Also write label_distribution.md.",
    )
    return parser.parse_args()


def parse_data_root_overrides(items: Sequence[str]) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for raw in items:
        if "=" not in raw:
            raise ValueError(f"--data-root must be dataset=/path. Got: {raw!r}")
        dataset, path = raw.split("=", 1)
        dataset = dataset.strip()
        if not dataset:
            raise ValueError(f"--data-root dataset name is empty. Got: {raw!r}")
        out[dataset] = Path(path.strip())
    return out


def load_dataset_tasks(path: Path) -> Dict[str, List[str]]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError(f"Expected mapping at top level in {path}")

    out: Dict[str, List[str]] = {}
    for dataset, spec in raw.items():
        if isinstance(spec, list):
            out[str(dataset)] = [str(item) for item in spec]
            continue
        if isinstance(spec, Mapping):
            include = spec.get("include", [])
            if not isinstance(include, list):
                raise ValueError(f"{dataset}.include must be a list in {path}")
            out[str(dataset)] = [str(item) for item in include]
    return out


def iter_prediction_rows(pred_root: Path) -> Iterable[Dict[str, Any]]:
    for path in sorted(pred_root.glob("*/preds.jsonl")):
        with path.open("r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {path}:{line_no}") from exc
                if isinstance(row, dict):
                    yield row


def iter_dataset_annotation_rows(
    dataset_tasks: Mapping[str, Sequence[str]],
    *,
    selected_datasets: Optional[Sequence[str]],
    dataset_prompts_path: Path,
    data_root_overrides: Mapping[str, Path],
    max_samples: Optional[int],
    shuffle: bool,
    seed: int,
) -> Iterable[Dict[str, Any]]:
    from tlg.dataset_factory import DATA_ROOTS, build_dataset

    prompt_bank = PromptBank.from_files(PromptBank.default_path(), dataset_prompts_path)
    dataset_names = list(selected_datasets) if selected_datasets is not None else sorted(dataset_tasks)

    for dataset_name in dataset_names:
        if dataset_name not in dataset_tasks:
            raise ValueError(f"Dataset {dataset_name!r} is not configured in {dataset_prompts_path}")
        data_root = data_root_overrides.get(dataset_name, DATA_ROOTS.get(dataset_name))
        if data_root is None:
            print(f"Dataset {dataset_name!r} has no default data root; skipping.", file=sys.stderr)
            continue
        if not data_root.exists():
            print(f"Dataset {dataset_name!r} root does not exist ({data_root}); skipping.", file=sys.stderr)
            continue

        dataset = build_dataset(
            dataset_name,
            data_root,
            prompt_bank=prompt_bank,
            include_depth=False,
            shuffle=shuffle,
            seed=seed,
        )
        n_samples = len(dataset)
        limit = n_samples if max_samples is None else min(n_samples, max(0, int(max_samples)))
        for index in range(limit):
            try:
                sample = dataset[index]
            except (FileNotFoundError, OSError, SyntaxError) as exc:
                print(
                    f"Skipping {dataset_name} sample index={index}: {exc}",
                    file=sys.stderr,
                )
                continue

            if isinstance(sample, dict):
                sample_id = sample.get("id", sample.get("image_id", str(index)))
            else:
                sample_id = str(index)
            annotations = sample.get("annotations", []) if isinstance(sample, dict) else []
            if not isinstance(annotations, list):
                continue
            for ann in annotations:
                metadata = getattr(ann, "metadata", {})
                if isinstance(metadata, Mapping):
                    task = str(metadata.get("task", getattr(ann, "task_type", "")))
                else:
                    task = str(getattr(ann, "task_type", ""))
                yield {
                    "dataset": dataset_name,
                    "task": task,
                    "gt": getattr(ann, "answer", None),
                    "sample_id": sample_id,
                }


def labels_for_gt(task: str, gt: Any) -> List[str]:
    if task == "material_types_present":
        text = str(gt).strip()
        if not text or text.upper() == "NONE":
            return ["NONE"]
        return [part.strip() for part in text.split(",") if part.strip()]

    if task in NUMERIC_BIN_TASKS:
        value = _as_float(gt)
        if value is None:
            return [str(gt)]
        width = NUMERIC_BIN_TASKS[task]
        lo = int((value // width) * width)
        hi = int(lo + width)
        return [f"{lo}-{hi}"]

    if task in GROUNDING_COUNT_TASKS:
        if isinstance(gt, list):
            return [str(len(gt))]
        return [str(gt)]

    return [format_label(gt)]


def discretize_fill_level_ordinal(fill_percentage: float, *, empty_threshold: float = 1.0) -> str:
    p = max(0.0, min(100.0, float(fill_percentage)))
    threshold = max(0.0, min(100.0, float(empty_threshold)))
    if p <= threshold:
        return "EMPTY"
    if p < 50.0:
        return "LOW"
    if p < 75.0:
        return "MID"
    return "HIGH"


def format_label(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return str(value)


def _as_float(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def sort_key(label: str) -> Tuple[int, float, str]:
    if "-" in label:
        first = label.split("-", 1)[0]
        try:
            return (0, float(first), label)
        except ValueError:
            pass
    try:
        return (0, float(label), label)
    except ValueError:
        return (1, 0.0, label)


def write_distribution_csv(
    path: Path,
    counts: Mapping[Tuple[str, str], Counter[str]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["dataset", "task", "label", "count", "percentage"])
        for dataset, task in sorted(counts):
            counter = counts[(dataset, task)]
            total = sum(counter.values())
            for label, count in sorted(counter.items(), key=lambda item: sort_key(item[0])):
                pct = (100.0 * count / total) if total else 0.0
                writer.writerow([dataset, task, label, count, f"{pct:.2f}"])


def write_by_task_csv(path: Path, counts: Mapping[Tuple[str, str], Counter[str]]) -> None:
    by_task: Dict[str, Counter[str]] = defaultdict(Counter)
    for (_dataset, task), counter in counts.items():
        by_task[task].update(counter)

    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["task", "label", "count", "percentage"])
        for task in sorted(by_task):
            counter = by_task[task]
            total = sum(counter.values())
            for label, count in sorted(counter.items(), key=lambda item: sort_key(item[0])):
                pct = (100.0 * count / total) if total else 0.0
                writer.writerow([task, label, count, f"{pct:.2f}"])


def write_markdown(
    path: Path,
    counts: Mapping[Tuple[str, str], Counter[str]],
    dataset_tasks: Mapping[str, Sequence[str]],
    observed_task_rows: Mapping[Tuple[str, str], int],
    source_description: str,
) -> None:
    lines = [
        "# Label Distribution",
        "",
        f"Ground-truth label histograms from {source_description}, filtered to tasks enabled in `src/tlg/dataset_prompts.yaml`.",
        "",
    ]

    for dataset in sorted(dataset_tasks):
        lines.extend([f"## {dataset}", ""])
        for task in dataset_tasks[dataset]:
            key = (dataset, task)
            counter = counts.get(key, Counter())
            n_rows = observed_task_rows.get(key, 0)
            if not counter:
                lines.extend([f"### {task}", "", "_No rows found in the selected source._", ""])
                continue
            total = sum(counter.values())
            suffix = ""
            if task == "material_types_present":
                suffix = f" ({n_rows} samples; multi-label counts sum to {total})"
            elif task in NUMERIC_BIN_TASKS:
                suffix = f" ({n_rows} samples; {int(NUMERIC_BIN_TASKS[task])}-unit bins)"
            elif task in GROUNDING_COUNT_TASKS:
                suffix = f" ({n_rows} samples; number of GT regions)"
            else:
                suffix = f" ({n_rows} samples)"

            lines.extend([f"### {task}{suffix}", "", "| Label | Count | Percentage |", "| --- | ---: | ---: |"])
            for label, count in sorted(counter.items(), key=lambda item: sort_key(item[0])):
                pct = (100.0 * count / total) if total else 0.0
                lines.append(f"| {label} | {count} | {pct:.2f}% |")
            lines.append("")

    missing = []
    for dataset, tasks in sorted(dataset_tasks.items()):
        for task in tasks:
            if (dataset, task) not in observed_task_rows:
                missing.append(f"- `{dataset}` / `{task}`")
    if missing:
        lines.extend(["## Missing Configured Tasks", ""])
        lines.extend(missing)
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def write_pngs(
    out_dir: Path,
    counts: Mapping[Tuple[str, str], Counter[str]],
    dataset_tasks: Optional[Mapping[str, Sequence[str]]] = None,
) -> None:
    try:
        import matplotlib
    except ModuleNotFoundError:
        write_pngs_with_pillow(out_dir, counts, dataset_tasks)
        return

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    png_dir = out_dir
    png_dir.mkdir(parents=True, exist_ok=True)
    for filename, title, counter in _png_chart_specs(counts, dataset_tasks):
        labels = [label for label, _count in sorted(counter.items(), key=lambda item: sort_key(item[0]))]
        values = [counter[label] for label in labels]
        width = max(7.0, min(18.0, 0.45 * len(labels) + 4.0))
        plt.figure(figsize=(width, 4.8))
        if labels:
            plt.bar(labels, values, color="#3f7f93")
        else:
            plt.text(0.5, 0.5, "No rows found in the selected source.", ha="center", va="center")
            plt.xticks([])
            plt.yticks([])
        plt.title(title)
        plt.ylabel("Count")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(png_dir / filename, dpi=160)
        plt.close()


def write_pngs_with_pillow(
    out_dir: Path,
    counts: Mapping[Tuple[str, str], Counter[str]],
    dataset_tasks: Optional[Mapping[str, Sequence[str]]] = None,
) -> None:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ModuleNotFoundError:
        print("Neither matplotlib nor Pillow is installed; skipping PNG histogram output.")
        return

    font_regular = _load_font(ImageFont, size=18, bold=False)
    font_bold = _load_font(ImageFont, size=24, bold=True)
    font_small = _load_font(ImageFont, size=14, bold=False)

    png_dir = out_dir
    png_dir.mkdir(parents=True, exist_ok=True)

    for filename, title, counter in _png_chart_specs(counts, dataset_tasks):
        items = sorted(counter.items(), key=lambda item: sort_key(item[0]))
        max_count = max(counter.values()) if counter else 1
        total = sum(counter.values())

        width = 1200
        row_h = 36
        top = 96
        bottom = 48
        left_label_w = 285
        right_pad = 120
        plot_w = width - left_label_w - right_pad - 48
        height = max(260, top + bottom + row_h * len(items))

        img = Image.new("RGB", (width, height), "#ffffff")
        draw = ImageDraw.Draw(img)

        subtitle = f"n={total}"
        draw.text((32, 24), title, fill="#1f2933", font=font_bold)
        draw.text((32, 58), subtitle, fill="#52616b", font=font_regular)

        if not items:
            draw.text(
                (32, top + 12),
                "No rows found in the selected source.",
                fill="#8a4b38",
                font=font_regular,
            )
            img.save(png_dir / filename)
            continue

        axis_x = left_label_w + 32
        axis_y = top - 10
        draw.line((axis_x, axis_y, axis_x + plot_w, axis_y), fill="#d8dee4", width=1)

        for i, (label, count) in enumerate(items):
            y = top + i * row_h
            pct = (100.0 * count / total) if total else 0.0
            bar_w = int(plot_w * (count / max_count)) if max_count else 0
            label_text = _truncate_label(str(label), max_chars=30)
            value_text = f"{count} ({pct:.1f}%)"

            draw.text((32, y + 5), label_text, fill="#25313b", font=font_regular)
            draw.rounded_rectangle(
                (axis_x, y + 4, axis_x + max(2, bar_w), y + 26),
                radius=3,
                fill="#3f7f93",
            )
            draw.text((axis_x + bar_w + 8, y + 5), value_text, fill="#25313b", font=font_small)

        img.save(png_dir / filename)


def _png_chart_specs(
    counts: Mapping[Tuple[str, str], Counter[str]],
    dataset_tasks: Optional[Mapping[str, Sequence[str]]] = None,
) -> List[Tuple[str, str, Counter[str]]]:
    specs: List[Tuple[str, str, Counter[str]]] = []
    keys = set(counts.keys())
    if dataset_tasks:
        keys.update(
            (dataset, task)
            for dataset, tasks in dataset_tasks.items()
            for task in tasks
        )

    for dataset, task in sorted(keys):
        counter = Counter(counts.get((dataset, task), Counter()))
        filename = f"{_safe_png_name(dataset)}__{_safe_png_name(task)}.png"
        specs.append((filename, f"{dataset} / {task}", counter))

    task_names = sorted({task for _dataset, task in keys})
    for task in task_names:
        counter: Counter[str] = Counter()
        for (dataset, task_name), task_counter in counts.items():
            if task_name == task:
                counter.update(task_counter)
        filename = f"task__{_safe_png_name(task)}.png"
        specs.append((filename, f"All datasets / {task}", counter))

    return specs


def _safe_png_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(value))


def _load_font(image_font: Any, *, size: int, bold: bool) -> Any:
    candidates = (
        "/usr/share/fonts/truetype/croscore/Arimo-Bold.ttf",
        "/usr/share/fonts/truetype/croscore/Arimo-Regular.ttf",
        "/usr/share/fonts/truetype/ubuntu/UbuntuSans[wdth,wght].ttf",
    )
    preferred = candidates[0] if bold else candidates[1]
    for path in (preferred, *candidates):
        try:
            return image_font.truetype(path, size=size)
        except OSError:
            continue
    return image_font.load_default()


def _truncate_label(label: str, *, max_chars: int) -> str:
    if len(label) <= max_chars:
        return label
    return label[: max_chars - 3] + "..."


def main() -> None:
    args = parse_args()
    dataset_tasks = load_dataset_tasks(args.dataset_prompts)
    if args.datasets is not None:
        requested = [str(dataset) for dataset in args.datasets]
        unknown = sorted(set(requested) - set(dataset_tasks))
        if unknown:
            raise ValueError(f"Dataset(s) not configured in {args.dataset_prompts}: {', '.join(unknown)}")
        dataset_tasks = {dataset: dataset_tasks[dataset] for dataset in requested}

    allowed = {
        (dataset, task)
        for dataset, tasks in dataset_tasks.items()
        for task in tasks
    }

    counts: Dict[Tuple[str, str], Counter[str]] = defaultdict(Counter)
    observed_task_rows: Counter[Tuple[str, str]] = Counter()
    derived_ordinal_counts: Dict[str, Counter[str]] = defaultdict(Counter)
    derived_ordinal_rows: Counter[str] = Counter()
    labliquidvision_sample_ids: set[str] = set()

    if args.source == "datasets":
        rows = iter_dataset_annotation_rows(
            dataset_tasks,
            selected_datasets=list(dataset_tasks),
            dataset_prompts_path=args.dataset_prompts,
            data_root_overrides=parse_data_root_overrides(args.data_root),
            max_samples=args.max_samples,
            shuffle=bool(args.shuffle),
            seed=int(args.seed),
        )
        source_description = "current dataset-loader annotations"
    else:
        rows = iter_prediction_rows(args.pred_root)
        source_description = f"prediction JSONL rows under `{args.pred_root}`"

    for row in rows:
        dataset = str(row.get("dataset", ""))
        task = str(row.get("task", ""))
        if dataset == "labliquidvision":
            sample_id = row.get("sample_id", row.get("image_id"))
            if sample_id is not None:
                labliquidvision_sample_ids.add(str(sample_id))
        key = (dataset, task)
        if key not in allowed or "gt" not in row:
            continue
        observed_task_rows[key] += 1
        if task == "fill_percentage" and "fill_level_ordinal" in dataset_tasks.get(dataset, []):
            fill = _as_float(row["gt"])
            if fill is not None:
                empty_threshold = 21.0 if dataset == "glasses" else 1.0
                derived_ordinal_counts[dataset][
                    discretize_fill_level_ordinal(fill, empty_threshold=empty_threshold)
                ] += 1
                derived_ordinal_rows[dataset] += 1
        for label in labels_for_gt(task, row["gt"]):
            counts[key][label] += 1

    for dataset, counter in derived_ordinal_counts.items():
        key = (dataset, "fill_level_ordinal")
        if key not in observed_task_rows:
            counts[key].update(counter)
            observed_task_rows[key] = derived_ordinal_rows[dataset]

    labliquidvision_count_key = ("labliquidvision", "container_count")
    if (
        "container_count" in dataset_tasks.get("labliquidvision", [])
        and labliquidvision_count_key not in observed_task_rows
        and labliquidvision_sample_ids
    ):
        counts[labliquidvision_count_key]["1"] = len(labliquidvision_sample_ids)
        observed_task_rows[labliquidvision_count_key] = len(labliquidvision_sample_ids)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_pngs(args.out_dir, counts, dataset_tasks)
    if args.csv:
        write_distribution_csv(args.out_dir / "label_distribution.csv", counts)
        write_by_task_csv(args.out_dir / "label_distribution_by_task.csv", counts)
    if args.markdown:
        write_markdown(
            args.out_dir / "label_distribution.md",
            counts,
            dataset_tasks,
            observed_task_rows,
            source_description,
        )

    print(f"Wrote label distribution outputs to {args.out_dir}")


if __name__ == "__main__":
    main()

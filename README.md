# TLG: Transparent Liquid Gauge

TLG is a codebase for evaluating vision-language models on transparent liquid
perception tasks: fill level estimation, container recognition, counting,
liquid grounding, material reasoning, and temporal pouring order.

The project is currently organized around two practical workflows:

1. run the current dataset loaders directly with `run_vlm_tasks.py`;
2. export a reproducible JSON manifest and evaluate from that manifest.

The root runner supports Qwen3-VL, SD-VLM, and several open Hugging Face VLMs.
COQE/Qwen evaluation is kept as an experimental workflow under
`src/experimental/coqe`.

## Quick Start

Install the Python dependencies from the repository root:

```bash
pip install -r requirements.txt
```

Run a loader-only validation pass before loading a model:

```bash
python run_vlm_tasks.py --dry-run --datasets labliquidvision --max-samples 3
python run_vlm_tasks.py --dry-run --datasets pouring --max-samples 3 --shuffle --seed 42
```

Run Qwen3-VL on one dataset:

```bash
python run_vlm_tasks.py \
    --datasets labliquidvision \
    --model qwen \
    --max-samples 20 \
    --shuffle \
    --seed 42 \
    --predictions-out results/qwen_labliquidvision_preds.jsonl \
    --metrics-out results/qwen_labliquidvision_metrics.json
```

Run one of the additional open VLMs:

```bash
python run_vlm_tasks.py --datasets dtld --model internvl35_8b --max-samples 20
python run_vlm_tasks.py --datasets pouring --model llavaov2_8b --max-samples 20
python run_vlm_tasks.py --datasets ot2 --model kimi_vl_a3b --max-samples 20
```

Supported `--model` values:

| Model key | Default model ID |
| --- | --- |
| `qwen` | `Qwen/Qwen3-VL-8B-Instruct` |
| `sdvlm` | `cpystan/SD-VLM-7B` |
| `internvl35_8b` | `OpenGVLab/InternVL3_5-8B` |
| `llavaov2_8b` | `lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct` |
| `kimi_vl_a3b` | `moonshotai/Kimi-VL-A3B-Thinking-2506` |

You can override Hugging Face model IDs with environment variables:
`QWEN_MODEL_ID`, `INTERNVL35_MODEL_ID`, `LLAVAOV2_MODEL_ID`, and
`KIMI_VL_MODEL_ID`.

## Dataset Roots

By default, the runner looks for datasets under `data/`:

| Dataset key | Default root |
| --- | --- |
| `labliquidvision` | `data/LabLiquidVision` |
| `glasses` | `data/glasses_in_the_wild/glasses_in_the_wild_1000` |
| `dtld` | `data/DTLD_dataset` |
| `labpics_chemistry` | `data/LabPics_Chemistry` |
| `pouring` | `data/pouring` |
| `ot2` | `data/OT2-Computer-Vision` |

Override roots with one or more `--data-root` flags:

```bash
python run_vlm_tasks.py \
    --dry-run \
    --datasets labliquidvision dtld \
    --data-root labliquidvision=/path/to/LabLiquidVision \
    --data-root dtld=/path/to/DTLD_dataset
```

Some dataset folders include convenience sync scripts, for example
`src/datasets/labliquidvision/sync_dataset.sh`,
`src/datasets/dtld/sync_dataset.sh`, and `src/datasets/coqe/sync_dataset.sh`.
Those scripts are environment-specific helpers rather than required runtime
APIs.

## Runner Usage

`run_vlm_tasks.py` is the main evaluator. It loads standardized annotations
from the selected dataset loaders, asks the selected VLM each task prompt,
parses the response, and reports task-level metrics.

Common commands:

```bash
# Default dataset is labliquidvision when no manifest is used.
python run_vlm_tasks.py --dry-run

# Evaluate multiple datasets in one process.
python run_vlm_tasks.py --datasets labliquidvision glasses dtld pouring ot2

# Write predictions and aggregate metrics.
python run_vlm_tasks.py \
    --datasets labliquidvision \
    --predictions-out results/preds.jsonl \
    --metrics-out results/metrics.json

# Change numeric scoring tolerance for selected tasks.
python run_vlm_tasks.py \
    --datasets labliquidvision dtld \
    --tol fill_percentage=2.5 \
    --tol fill_height_mm=10
```

Useful flags:

| Flag | Description |
| --- | --- |
| `--datasets` | Dataset keys to evaluate. Choices: `labliquidvision`, `glasses`, `dtld`, `labpics_chemistry`, `pouring`, `ot2`, `coqe`. |
| `--model` | One model key to run in this process. |
| `--dry-run` | Validate loading and standardized annotations without loading a model. |
| `--max-samples N` | Cap samples per dataset, or per task for task-aware datasets such as `pouring`. |
| `--shuffle --seed N` | Shuffle samples deterministically before capping. |
| `--predictions-out PATH` | Write per-task JSONL records with prediction and ground truth. |
| `--metrics-out PATH` | Write aggregate metrics JSON. |
| `--data-root dataset=/path` | Override a dataset root. Can be repeated. |
| `--tol task=value` | Override numeric within-tolerance scoring. Can be repeated. |

For `pouring`, `--max-samples` is applied per task. For example,
`--max-samples 200` selects up to 200 liquid-grounding frame samples and up to
200 pouring-frame-ordering samples.

## Manifest Workflow

Use manifests when you want a stable, shareable evaluation set that is decoupled
from the live dataloader traversal order.

Export a manifest:

```bash
python scripts/export_tlg_manifest.py \
    --datasets labliquidvision dtld pouring ot2 \
    --out manifests/tlg_eval.json \
    --max-samples-per-dataset 500 \
    --shuffle \
    --seed 42 \
    --train-ratio 0.8
```

Evaluate the generated test split:

```bash
python run_vlm_tasks.py \
    --manifest manifests/tlg_eval.json \
    --manifest-split test \
    --model qwen \
    --predictions-out results/manifest_qwen_preds.jsonl \
    --metrics-out results/manifest_qwen_metrics.json
```

Filter a manifest evaluation to specific datasets:

```bash
python run_vlm_tasks.py \
    --manifest manifests/tlg_eval.json \
    --manifest-split all \
    --datasets dtld pouring \
    --dry-run
```

Manifest samples include only JSON-serializable fields needed by the evaluator:
image paths, task metadata, split labels, selected routing fields, and
standardized annotations. Large arrays such as depth maps are intentionally not
embedded.

## SD-VLM

[SD-VLM](https://github.com/cpystan/SD-VLM) uses depth information with a
LLaVA-style VLM. Install its `llava` package separately:

```bash
git clone https://github.com/cpystan/SD-VLM.git
cd SD-VLM
pip install -e .
```

Run SD-VLM:

```bash
# Estimate depth internally.
python run_vlm_tasks.py --model sdvlm --datasets labliquidvision

# Use LabLiquidVision ground-truth depth arrays.
python run_vlm_tasks.py --model sdvlm --datasets labliquidvision --sdvlm-gt-depth

# Use precomputed depth arrays from a directory.
python run_vlm_tasks.py \
    --model sdvlm \
    --datasets labliquidvision \
    --sdvlm-depth-dir results/precomputed_depth
```

SD-VLM-specific flags:

| Flag | Description |
| --- | --- |
| `--sdvlm-model-path` | Hugging Face ID or local path for the SD-VLM checkpoint. |
| `--sdvlm-vision-tower` | Vision encoder identifier. |
| `--sdvlm-gt-depth` | Load dataset-provided depth where available. |
| `--sdvlm-depth-dir` | Load precomputed `.npy` depth maps instead of running a depth model. |
| `--sdvlm-no-depth` | Disable depth encoding. |

## Debugging and Visualization

Attention overlays are available for Qwen and SD-VLM:

```bash
python run_vlm_tasks.py \
    --datasets labliquidvision \
    --model qwen \
    --max-samples 5 \
    --viz-attn \
    --viz-attn-mode first_answer_token \
    --viz-out results/qwen_attn
```

For DTLD target-box debugging:

```bash
python run_vlm_tasks.py \
    --dry-run \
    --datasets dtld \
    --max-samples 10 \
    --debug-dtld-boxes \
    --debug-image-out results/debug_dtld_boxes
```

Use `--dtld-no-occlusion-filter` to disable DTLD occlusion filtering for
target-object and container-count tasks during inspection.

## HPC Scripts

The `hpc_scripts/` directory contains LSF job templates used in development.
They include local DTU paths and should be edited before reuse.

Current entry points:

| Script | Purpose |
| --- | --- |
| `hpc_scripts/run_open_vlm.sh` | Run one open VLM/dataset job. Override `MODEL`, `DATASET`, and `MAX_SAMPLES` before submission. |
| `hpc_scripts/run_open_vlm_array.sh` | Array over `internvl35_8b`, `llavaov2_8b`, `kimi_vl_a3b` and the configured datasets. |
| `hpc_scripts/run_sd.sh` | Run SD-VLM. |
| `hpc_scripts/run_coqe_qwen.sh` | Run the experimental COQE/Qwen evaluator. |
| `hpc_scripts/run_array.sh` | Older array runner for benchmark jobs. |

Example:

```bash
MODEL=llavaov2_8b DATASET=dtld MAX_SAMPLES=50 bsub < hpc_scripts/run_open_vlm.sh
```

## COQE Experimental Workflow

COQE support is not part of the main TLG runner yet. It lives under
`src/experimental/coqe` and has its own Qwen evaluator, prompts, parser, metrics,
and README.

Run a small COQE dry run:

```bash
python src/experimental/coqe/run_qwen_coqe.py \
    --split test \
    --data-root data/COQE_dataset \
    --tasks volume content comparative pouring \
    --dry-run \
    --max-samples 5
```

Run COQE with Qwen:

```bash
python src/experimental/coqe/run_qwen_coqe.py \
    --split test \
    --data-root data/COQE_dataset \
    --tasks volume content comparative pouring \
    --model Qwen/Qwen3-VL-8B-Instruct \
    --predictions-out results/coqe_predictions.jsonl \
    --metrics-out results/coqe_metrics.json
```

See `src/experimental/coqe/README.md` for the COQE-specific task definitions and
evaluation details.

## Tasks and Prompts

Prompt configuration is data-driven:

| File | Role |
| --- | --- |
| `src/tlg/prompt_bank.yaml` | Reusable task prompts, choices, and output format instructions. |
| `src/tlg/dataset_prompts.yaml` | Dataset-to-task assignments and dataset-specific prompt prefixes. |
| `src/tlg/annotations.py` | Standard annotation object used by loaders and manifests. |
| `src/tlg/evaluation.py` | Parsers and metrics for model outputs. |

The main runner accepts alternate prompt files per run:

```bash
python3 run_vlm_tasks.py \
  --prompt-bank experiments/prompts/prompt_bank_v2.yaml \
  --dataset-prompts experiments/prompts/dataset_prompts_v2.yaml \
  --datasets labpics_chemistry \
  --model qwen
```

Current task families include:

| Task | Examples |
| --- | --- |
| Fill level | `fill_percentage`, `fill_level_ordinal`, `fill_height_mm` |
| Container understanding | `container_count`, `container_type`, `container_type_simple` |
| Appearance | `liquid_color`, `tip_presence` |
| Grounding | `liquid_grounding`, `highlighted_region_type` |
| Materials | `material_types_present`, `multiphase_material_presence` |
| Temporal ordering | `pouring_frame_ordering` |

## Dataset Coverage

TLG standardizes several datasets into image-plus-prompt evaluation samples.
Tasks are generated only when the source dataset has reliable supervision for
that question.

| Dataset | Current use |
| --- | --- |
| LabLiquidVision | Controlled lab images with vessel type, liquid color, liquid/vessel volume, optional masks, and depth maps. Used for fill level, counting, type recognition, color, and grounding. |
| Glasses in the Wild | Real-world glass images with geometry and fluid-level keypoints. Used for fill percentage and ordinal fill-level evaluation. |
| DTLD | Transparent-container RGB scenes with object boxes, visible masks, liquid labels, and liquid height where available. Used for counting, target-box container type, and fill height. |
| LabPics Chemistry | Laboratory chemistry scenes with vessel, material, part, and semantic-mask annotations. Used for highlighted-region, material-presence, multiphase, liquid-grounding, container-count, and container-type tasks. |
| Pouring | Video frames and annotations from pouring sequences. Used for single-frame liquid grounding and four-frame chronological ordering. |
| OT2 | YOLO-style lab video frames with Tip and Liquid boxes. Used for grounding, tip presence, counting, simple labware prompts, and qualitative sequence analysis. |

Additional datasets under discussion or experimental development:

| Dataset | Status |
| --- | --- |
| COQE | Experimental standalone evaluator under `src/experimental/coqe`; not wired into the main TLG runner. |
| Liquid Level Detection | Not included in the main benchmark runner. |

## Project Structure

```text
run_vlm_tasks.py                  Main VLM evaluator
scripts/export_tlg_manifest.py    Export reproducible manifest JSON
scripts/precompute_*_depth.py      Depth precomputation helpers
scripts/calculate_*               Analysis helpers
src/tlg/                           Prompts, evaluation, runners, manifests
src/datasets/                      Dataset loaders and sync helpers
src/experimental/coqe/             Standalone COQE/Qwen experiment
hpc_scripts/                       LSF job templates
notebooks/                         Dataset preview notebooks
tests/                             Unit and integration-style smoke tests
```

## Development Checks

Run the test suite:

```bash
python -m unittest discover tests
```

Run focused tests while changing runner behavior:

```bash
python -m unittest tests.test_labliquidvision_dry_run tests.test_manifest
```

The tests construct small synthetic datasets where possible, so they are useful
for checking loader, prompt, parser, metric, and manifest behavior without
requiring the full datasets.

## Benchmark Goal

Transparent liquid perception is hard for general-purpose VLMs because liquid
boundaries can be faint, refracted, reflected, occluded, or ambiguous. TLG
turns heterogeneous supervision from lab and real-world datasets into
standardized VQA-style tasks so models can be compared on:

- quantitative liquid-level reasoning;
- transparent-container recognition and counting;
- liquid-region localization;
- robustness to reflections and refractions;
- material and multiphase reasoning;
- temporal ordering in pouring sequences.

Metrics include parse rate, exact/category accuracy, multi-label scores,
grounding scores where applicable, mean absolute error for numeric tasks, and
within-tolerance accuracy for configured numeric targets.

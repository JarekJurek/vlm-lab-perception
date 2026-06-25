# TLG: Transparent Liquid Gauge

TLG is a codebase for evaluating vision-language models on transparent liquid
perception tasks: fill level estimation, container recognition, counting,
liquid grounding, material reasoning, and temporal pouring order.

The project is currently organized around three practical workflows:

1. run the current dataset loaders directly with `run_vlm_tasks.py`;
2. export a reproducible JSON manifest and evaluate from that manifest;
3. run the manifest through VLMEvalKit-compatible model wrappers;
4. run the standalone COQE/Qwen experiment when you need the original
   COQE-style task framing.

The root runner supports Qwen3-VL, SD-VLM, and several open Hugging Face VLMs.
COQE also has a TLG-compatible dataset loader in the main runner; the richer
COQE/Qwen workflow remains under `src/experimental/coqe`.

## Quick Start

Install the Python dependencies from the repository root:

```bash
pip install -r requirements.txt
```

For the VLMEvalKit runner, install the extra evaluation stack:

```bash
pip install -r requirements-vlmevalkit.txt
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
`KIMI_VL_MODEL_ID`. Generation lengths can be controlled globally with
`VLM_MAX_NEW_TOKENS` or per model with `QWEN_MAX_NEW_TOKENS`,
`INTERNVL35_MAX_NEW_TOKENS`, `LLAVAOV2_MAX_NEW_TOKENS`, and
`KIMI_VL_MAX_NEW_TOKENS`. Qwen image limits can also be adjusted with
`QWEN_MAX_PIXELS` and `QWEN_MIN_PIXELS`.

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
| `coqe` | `data/COQE_dataset` |

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
| `--coqe-split` | COQE split for the main `coqe` loader. Choices: `train`, `val`, `test`; default `test`. |
| `--labpics-chemistry-split` | LabPics Chemistry split. Choices: `train`, `test`, `all`; default `all`. |

For task-aware datasets, `--max-samples` is applied per primary task. For
example, `--max-samples 200` on `pouring` selects up to 200 samples for each of
its primary task pools.

## Manifest Workflow

Use manifests when you want a stable, shareable evaluation set that is decoupled
from the live dataloader traversal order.

Export a manifest:

```bash
python scripts/export_tlg_manifest.py \
    --datasets labliquidvision dtld pouring ot2 coqe \
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

Manifest export defaults to all datasets configured in
`src/tlg/dataset_prompts.yaml`. It uses the current dataloaders, skips samples
without standardized annotations, and supports `--labpics-chemistry-split` for
LabPics Chemistry source selection.

## VLMEvalKit Workflow

The manifest benchmark can also be run through a
[VLMEvalKit](https://github.com/open-compass/VLMEvalKit)-compatible adapter.
This keeps TLG's existing prompts, parsers, and metrics while using VLMEvalKit
model wrappers for inference.

Dry-run the flattened manifest rows and prompt construction:

```bash
python scripts/run_tlg_vlmeval.py \
    --manifest manifests/tlg_all_seed42.json \
    --manifest-split test \
    --datasets labliquidvision \
    --max-samples 5 \
    --dump-rows-out results/tlg_vlmeval/labliquidvision_rows.jsonl \
    --dry-run
```

Run a VLMEvalKit model key and write current-compatible outputs:

```bash
python scripts/run_tlg_vlmeval.py \
    --manifest manifests/tlg_all_seed42.json \
    --manifest-split test \
    --datasets labliquidvision \
    --model Qwen2-VL-7B-Instruct \
    --work-dir results/tlg_vlmeval
```

The runner writes raw VLMEvalKit-style rows to
`results/tlg_vlmeval/{model}/vlmeval_predictions.jsonl`, plus compatibility
files at `preds.jsonl` and `metrics.json`. Use `--resume` to append only missing
raw predictions. To rescore an existing VLMEvalKit table without running a
model, pass `--predictions-in path/to/predictions.jsonl|csv|tsv|xlsx`.

On the LSF cluster, first run the CPU-side implementation check:

```bash
bsub < hpc_scripts/check_tlg_vlmevalkit.sh
```

Then run a small GPU smoke job:

```bash
MODEL=Qwen2-VL-7B-Instruct DATASET=labliquidvision TASK=container_count MAX_SAMPLES=5 \
    bsub < hpc_scripts/run_tlg_vlmevalkit_smoke.sh
```

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

## COQE Workflows

COQE can be evaluated through the main TLG runner with the `coqe` dataset key:

```bash
python run_vlm_tasks.py \
    --dry-run \
    --datasets coqe \
    --coqe-split test \
    --max-samples 5
```

The main runner maps COQE into standardized TLG tasks:
`container_capacity_ml`, `container_capacity_bin`, `container_content_bin`, and
`liquid_transfer_feasibility`.

The standalone experimental workflow lives under `src/experimental/coqe` and
keeps its own Qwen evaluator, prompts, parser, metrics, and README. Use it when
you want the COQE-specific JSON prompts, visual query-box rendering, pouring
sequence evaluation, or paper-style rescoring utilities.

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
| COQE reasoning | `container_capacity_ml`, `container_capacity_bin`, `container_content_bin`, `liquid_transfer_feasibility` |

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
| COQE | Container capacity, content-bin, and transfer-feasibility reasoning from COQE annotations. Available in the main TLG runner and in the standalone experimental COQE/Qwen evaluator. |

Additional datasets under discussion or experimental development:

| Dataset | Status |
| --- | --- |
| Liquid Level Detection | Not included in the main benchmark runner. |

## Project Structure

```text
run_vlm_tasks.py                  Main VLM evaluator
scripts/export_tlg_manifest.py    Export reproducible manifest JSON
scripts/run_tlg_vlmeval.py        VLMEvalKit-compatible manifest runner
scripts/precompute_*_depth.py      Depth precomputation helpers
scripts/calculate_*               Analysis helpers
src/tlg/                           Prompts, evaluation, runners, manifests
src/datasets/                      Dataset loaders and sync helpers
src/experimental/coqe/             Standalone COQE/Qwen experiment
hpc_scripts/                       LSF job templates and VLMEvalKit smoke checks
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
python -m unittest tests.test_labliquidvision_dry_run tests.test_manifest tests.test_vlmevalkit_adapter
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

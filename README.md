# TLG (Transparent Liquid Gauge): A Benchmark for Vision–Language Models in Transparent Liquid Perception

TLG is a multi-task vision–language benchmark designed to evaluate Vision–Language Models (VLMs) on perception and reasoning tasks involving transparent liquid containers. The benchmark focuses on challenges specific to transparency, including ambiguous liquid boundaries, reflections, refractions, multi-phase materials, and fine-grained fill-level estimation.

## Dataset Sources and Alignment

TLG integrates and standardizes complementary datasets that provide reliable supervision for liquid perception and grounding tasks:

**LabLiquidVision**  
Controlled laboratory images of single vessels (e.g., beakers, flasks) with precise volume metadata.  
Provides:

- Vessel type labels  
- Liquid color labels  
- Exact liquid and vessel volumes (mL) – precise fill percentage  
- Optional vessel/content masks and depth maps

This dataset supports **quantitative fill estimation** and structured container recognition.  

**Glasses in the Wild**  
Real-world drinking glass images with annotated geometry and fluid-level keypoints.  
Provides:

- Glass geometry keypoints  
- Fluid-level keypoints  
- Derived fill percentage  
- Visibility flags

This dataset supports surface localization and robustness to viewpoint and lighting variation.

**DTLD**  
Reflection- and refraction-heavy RGB scenes of transparent containers.  
Provides:

- Object identities and bounding boxes  
- Liquid height in millimeters (when available)  
- Visible masks and optional depth  
- Liquid label points

This dataset supports grounding and robustness under strong transparency artifacts.

**Pouring (Video Dataset)**
Video-centric dataset of cup/container filling/pouring sequences (separate opaque vs transparent training; separate test videos).
Provides:

- RGB frames organized by video folders
- Test split: per-frame LabelMe-style JSON polygon annotations plus sidecar pixel labels (`label.png`, `label_viz.png`, `label_names.txt`)
- Training split: RGB sequences plus empty-cup reference frames and per-frame cup masks (`*_cup_mask.png`) in dedicated mask folders
- Natural “sequence” structure via `video_#` directories for frame-by-frame temporal sampling

This dataset supports liquid-region segmentation (polygon + pixel supervision), cup-context learning (cup masks), and per-video qualitative/temporal analysis across material domains (opaque vs transparent). It is useful for future temporal reasoning benchmarks, but not required for the initial static-image benchmark version.

**OT2 (Tips + Liquid)**
YOLO-style object detection dataset built from lab videos exported frame-by-frame.
Provides:
- RGB frames split into train/, valid/, test/
- YOLO.txt labels per frame with normalized bounding boxes
- Two classes: Tip and Liquid (multi-instance per frame)
- Heuristic “sequence” grouping via filename patterns (e.g., *_###)

This dataset supports detection, counting, and per-sequence qualitative analysis of tips and liquid regions in lab scenes.

### Datasets not included in V1 of the benchmark:

**COQE (Container and Quantity Estimation)** – not in the laboratory setting
Provides container bounding boxes, category labels, and continuous/discrete volume annotations.  
Useful for benchmarking quantitative reasoning and comparative volume estimation.

**Liquid Level Detection** – not in the laboratory setting
Provides segmentation masks of liquid regions (COCO-style RLE).  
Useful for liquid-region grounding tasks.

**LabPicsV1** – annotations not too relevant to our case
Provides instance and semantic segmentation masks of materials in laboratory scenes, including multi-phase annotations.  
Useful for material reasoning and multi-phase detection.

## Dataset Composition

TLG converts heterogeneous annotations (volumes, keypoints, masks, bounding boxes) into a unified Vision–Language Question Answering (VQA) format inspired by structured medical VLM benchmarks.

Example:  
```json
{
    "image_id": "...",
    "dataset": "...",
    "task_type": "...",
    "question": "...",
    "choices": [...],        # optional (for MCQ tasks)
    "answer": "...",         # categorical or numeric
    "metadata": {...}        # original GT references
}
```

For LabLiquidVision:  
```json
{  
    "image_id": "LLV_01234",
    "dataset": "LabLiquidVision",
    "task_type": "Fill-Level Estimation (continuous)",
    "question": "Estimate the fill level of the liquid in this container as a percentage (0–100).",
    "answer": 50,
    "metadata": {
        "vol_liquid": 150,
        "vol_vessel": 300,
        "derived_fill_percentage": 50
        }
}
```

Each image is associated with multiple structured prompts derived directly from available ground truth. Tasks are only defined when reliable supervision exists in the source dataset.

## Task Categories

### Container Type Recognition (CTR)

Identify the type of transparent container shown.  
Closed vocabulary derived from dataset labels (e.g., beaker, flask, wine glass, tumbler).  
Task type: **multi-class classification.**

### Liquid Presence Detection (LPD)

Determine whether the container contains liquid.  
Derived from fill volume or mask presence.  
Output: YES / NO.  
Task type: **binary classification.**

### Fill-Level Estimation (FLE)

Two evaluation modes depending on dataset:

- **Continuous Regression:**  
  Percentage derived from liquid and vessel volumes (LabLiquidVision) or liquid height (DTLD).  
- **Ordinal Classification:**  
  Discretized levels (EMPTY, LOW, MID, HIGH, FULL).

Task type: regression and/or multi-class classification.

### Visual Grounding (VG)

Given a bounding box or segmentation overlay, identify the highlighted region  
(e.g., liquid region, container body, reflection artifact, background).  
Derived from segmentation masks (Roboflow, LabPicsV1, DTLD).  
Task type: grounded multi-class classification.

### Multi-Phase and Material Reasoning (MMR)

Identify whether multiple liquid phases or materials are present.  
Supported by LabPicsV1 instance/material metadata.  
Task type: multi-class classification.

### Transparency Robustness Evaluation (TRE)

Evaluate performance degradation under strong reflections and refractions.  
Measured via DTLD subset performance compared to controlled lab images.  
Task type: cross-domain evaluation rather than a standalone question type.

## Evaluation Protocol

Tasks are formulated as:

- Multiple-choice questions (MCQs)  
- Structured categorical outputs  
- Numeric regression outputs (where continuous ground truth exists)

Reported metrics include:

- Overall accuracy  
- Per-task accuracy  
- Mean Absolute Error (for continuous fill estimation)  
- Grounding accuracy (IoU-based where segmentation is available)  
- Cross-domain generalization performance (lab vs real-world vs reflection-heavy scenes)

## Core Objective

TLG evaluates whether VLMs can:

- Perform quantitative reasoning on transparent liquid quantities  
- Localize liquid regions under transparency  
- Distinguish true liquid boundaries from reflection artifacts  
- Generalize across controlled laboratory and unconstrained real-world environments  
- Handle multi-phase materials and fine-grained spatial cues

TLG provides a structured and reproducible evaluation framework for multimodal reasoning in transparent liquid perception, enabling systematic comparison between general-purpose VLMs and domain-adapted models.

## Project structure

- `src/datasets/labliquidvision`: `LabLiquidVisionDataset`
- `src/datasets/glassesinthewild`: `GlassesInTheWildDataset`
- `src/datasets/dtld`: `DTLDDataset`
- `src/tlg/prompt_bank.yaml`: prompt definitions (prompt_id -> question/choices)
- `src/tlg/dataset_prompts.yaml`: dataset -> included tasks, with optional overrides
- `test_qwen_vl_tasks.py`: Qwen3-VL task runner across datasets
- `notebooks/`: dataset preview notebooks
- `hpc_scripts/`: helper scripts for HPC setup and runs
- `data/`: dataset roots

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

Main runtime packages include `numpy`, `matplotlib`, `torch`, `transformers`, and `Pillow`.

## Usage

### 1) Load datasets

From repo root:

```python
import sys
sys.path.append("src")

from datasets.labliquidvision import LabLiquidVisionDataset
from datasets.glassesinthewild import GlassesInTheWildDataset
from datasets.dtld import DTLDDataset

lab_ds = LabLiquidVisionDataset("data/LabLiquidVision/processed", rgb_only=True)
glasses_ds = GlassesInTheWildDataset("data/glasses_in_the_wild/glasses_in_the_wild_1000")
dtld_ds = DTLDDataset("data/DTLD_dataset", rgb_only=True)

sample = lab_ds[0]
print(sample["id"], sample.get("vol_liquid"), sample.get("rgb_image"))
```

### 2) Run Qwen3-VL tasks

Evaluate one or more datasets:

```bash
python test_qwen_vl_tasks.py --datasets labliquidvision
python test_qwen_vl_tasks.py --datasets labliquidvision glasses dtld
```

Default model in the script: `Qwen/Qwen3-VL-30B-A3B-Instruct`.

### 3) Run SD-VLM tasks

[SD-VLM](https://github.com/cpystan/SD-VLM) is a depth-encoded VLM built on LLaVA-1.5-7B.
It can leverage the depth maps stored in LabLiquidVision (`Input_DepthMap.npy`) to provide
spatially-aware reasoning.

#### Install SD-VLM

SD-VLM ships its own version of the `llava` package.  Install it from source:

```bash
git clone https://github.com/cpystan/SD-VLM.git
cd SD-VLM
pip install -e .
```

#### Run SD-VLM only

```bash
# Default: SD-VLM estimates depth internally from the RGB image
python test_qwen_vl_tasks.py --model sdvlm --datasets labliquidvision

# Use ground-truth depth maps from LabLiquidVision (Input_DepthMap.npy)
python test_qwen_vl_tasks.py --model sdvlm --datasets labliquidvision --sdvlm-gt-depth

# Use a local SD-VLM checkpoint and a local vision tower
python test_qwen_vl_tasks.py \
    --model sdvlm \
    --datasets labliquidvision \
    --sdvlm-model-path /path/to/SD-VLM-7B \
    --sdvlm-vision-tower /path/to/clip-vit-large-patch14-336 \
    --sdvlm-gt-depth
```

#### Run Qwen3-VL and SD-VLM side-by-side

```bash
python test_qwen_vl_tasks.py \
    --model both \
    --datasets labliquidvision \
    --sdvlm-gt-depth \
    --predictions-out results/preds.jsonl \
    --metrics-out results/metrics.json
```

The `--model both` option loads both models, evaluates each sample with each model, and writes
per-model metrics to the output files.  The `"model"` field in the JSONL predictions file
distinguishes Qwen (`"qwen"`) from SD-VLM (`"sdvlm"`) rows.

#### SD-VLM CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `qwen` | Which model(s) to run: `qwen`, `sdvlm`, or `both` |
| `--sdvlm-model-path` | `cpystan/SD-VLM-7B` | HuggingFace ID or local path for the SD-VLM checkpoint |
| `--sdvlm-vision-tower` | `openai/clip-vit-large-patch14-336` | Vision encoder identifier |
| `--sdvlm-gt-depth` | off | Use ground-truth depth from the dataset (`Input_DepthMap.npy`) |
| `--sdvlm-no-depth` | off | Disable depth encoding entirely (`use_depth=False`) |

#### Depth compatibility with LabLiquidVision

LabLiquidVision provides per-sample depth maps stored as `.npy` files
(`Input_DepthMap.npy`).  When `--sdvlm-gt-depth` is passed, the loader reads
these arrays, normalises them to 8-bit grayscale, and passes them to SD-VLM as
ground-truth depth images.  Without `--sdvlm-gt-depth`, SD-VLM estimates depth
internally using Depth Anything V2.

#### Attention visualization for SD-VLM

SD-VLM supports the same `--viz-attn` / `--viz-out` flags as Qwen3-VL.  The
visualization logic lives in `attention_viz.py` and works by:

1. Locating LLaVA's `IMAGE_TOKEN_INDEX` placeholder in the flat token sequence.
2. Expanding it to the patch grid (576 tokens for the default 336 × 336 CLIP
   ViT-L/14 vision tower; derived automatically from the model's config).
3. Averaging attention weights from generated tokens over the patch grid and
   upsampling the result to the original image resolution.

```bash
python test_qwen_vl_tasks.py \
    --model sdvlm \
    --datasets labliquidvision \
    --sdvlm-gt-depth \
    --viz-attn \
    --viz-attn-mode first_answer_token \
    --viz-out results/sdvlm_attn
```

The output PNGs follow the same directory structure as the Qwen visualizations:
`{viz_out}/{dataset}/{sample_id}/{task}.png`.

## Notebooks

Available preview notebooks:

- `notebooks/dataset_preview_labliquidvision.ipynb`
- `notebooks/dataset_preview_glasses.ipynb`
- `notebooks/dataset_preview_dtld.ipynb`
- `notebooks/dataset_preview_pouring.ipynb`
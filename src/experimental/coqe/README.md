# Standalone COQE Qwen Evaluation

This directory contains the experimental COQE evaluation path for Qwen VLMs. It is intentionally separate from the main benchmark runner. The reusable dataset loader lives in `src/datasets/coqe`, while this directory owns prompt construction, visual query-box rendering, Qwen inference, parsing, scoring, and result export.

The goal is to compare Qwen-style VLM inference against the tasks from Mottaghi et al. (2017), *See the Glass Half Full: Reasoning About Liquid Containers, Their Volume and Content*, using the released COQE annotations.

## Approach

The original paper used task-specific CNN/RNN models. For the image tasks, the model received RGB image features plus a query-container mask derived from the bounding box. For pouring prediction, the model received RGB, the query mask, and an angle channel. Ground-truth pouring labels were generated from CAD/simulation geometry, but the paper did not use CAD models at test inference.

For Qwen, we adapt that setup as follows:

- Use the full RGB image.
- Render a colored bounding box directly on the image to localize the query container.
- Use a yellow box for a single query/source container.
- Use a green box for the target container in comparative volume estimation.
- Provide the tilt angle as text for pouring prediction.
- Restrict outputs to the exact COQE label spaces.
- Require JSON-only answers in the prompt.
- Parse model responses into task labels and evaluate against COQE ground truth.

This is not byte-identical to the paper's tensor input, but it is the practical VLM equivalent: the model sees the same RGB scene, explicit localization of the query object, and the same task-specific answer classes.

## Files

- `prompts.py`: prompt templates and prompt version.
- `overlays.py`: rendering of yellow/green query boxes.
- `run_qwen_coqe.py`: standalone Qwen inference and evaluation script.
- `parsing.py`: response parser for task JSON.
- `metrics.py`: COQE metrics, including paper-style metrics.
- `rescore_predictions.py`: reparses existing JSONL predictions with the current parser and recomputes metrics.

## Dataset Inputs

The loader expects the COQE dataset under `data/COQE_dataset` by default.

For `train.json`, `val.json`, and `test.json`, each row contains:

- `image_id`
- `x1`, `y1`, `x2`, `y2` query bbox
- `volume_discrete`
- `content`
- `comparative_vol`
- metadata such as `category`, `cad_model`, and `volume_continuous`

For `*_pour_prediction.json`, each row contains:

- `image_id`
- `angle`
- `t0_content` through `t4_content`

The pouring sequence is converted with:

- `1 -> 0.0`
- `2 -> 0.1`
- ...
- `11 -> 1.0`
- `12 -> "opaque"`

## Running

Example full test run:

```bash
python3 -B src/experimental/coqe/run_qwen_coqe.py \
    --split test \
    --data-root data/COQE_dataset \
    --tasks volume content comparative pouring \
    --model Qwen/Qwen3-VL-8B-Instruct \
    --max-new-tokens 128 \
    --predictions-out bakery/coqe_qwen_8b/predictions.jsonl \
    --metrics-out bakery/coqe_qwen_8b/metrics.json
```

Dry-run without loading a model:

```bash
python3 -B src/experimental/coqe/run_qwen_coqe.py \
    --split test \
    --data-root data/COQE_dataset \
    --tasks volume content comparative pouring \
    --max-samples 2 \
    --input-images-out bakery/coqe_inputs_preview \
    --dry-run
```

Rescore an existing prediction file:

```bash
python3 -B src/experimental/coqe/rescore_predictions.py \
    --predictions-in bakery/10.06.2026/experiment_coqe_og_qwen_8B/predictions.jsonl \
    --rescored-out bakery/10.06.2026/experiment_coqe_og_qwen_8B/predictions_rescored.jsonl \
    --metrics-out bakery/10.06.2026/experiment_coqe_og_qwen_8B/metrics_rescored.json
```

## Prompt Version

Current prompt version: `coqe_qwen_v1`.

## Shared Instruction

```text
You are evaluating a visual reasoning task about liquid containers.
Use only the visual information in the image and the highlighted query container(s).
Do not use metadata, annotations, filenames, or ground-truth labels; answer only from the provided image and highlighted query container(s).
Think about object size, scene context, container shape, transparency, visible liquid level, and available free space.
Return only valid JSON in the specified schema.
Do not include explanations.
```

## Task Prompts

### Volume Estimation

```text
Task: Estimate the total capacity of the highlighted container.

The highlighted container is inside the yellow bounding box. Estimate the volume of the container when it is completely full, not the amount of liquid currently inside it.

Use visual scale cues from the whole image, including surrounding objects, container shape, and scene context. Choose the closest volume bin.

Allowed answers:
- "0_50_ml" for 0 mL <= volume < 50 mL
- "50_100_ml" for 50 mL <= volume < 100 mL
- "100_200_ml" for 100 mL <= volume < 200 mL
- "200_300_ml" for 200 mL <= volume < 300 mL
- "300_500_ml" for 300 mL <= volume < 500 mL
- "500_750_ml" for 500 mL <= volume < 750 mL
- "750_1000_ml" for 750 mL <= volume < 1000 mL
- "1000_2000_ml" for 1000 mL <= volume < 2000 mL
- "2000_3000_ml" for 2000 mL <= volume < 3000 mL
- "3000_inf_ml" for 3000 mL <= volume

Return only:
{"volume_bin": "<one_allowed_answer>"}
```

### Content Estimation

```text
Task: Estimate how full the highlighted container is.

The highlighted container is inside the yellow bounding box. Estimate the fraction of the container's total capacity that is currently filled with liquid.

Use visible cues such as the liquid surface line, color change, transparency, reflections, and container opacity. If the container is opaque and the amount of liquid inside cannot be visually estimated, choose "opaque".

Allowed answers:
- "0_percent" for empty
- "33_percent" for about one third full
- "50_percent" for about half full
- "66_percent" for about two thirds full
- "100_percent" for full or almost full
- "opaque" if the liquid amount cannot be visually determined because the container is opaque

Return only:
{"fill_level": "<one_allowed_answer>"}
```

### Comparative Volume Estimation

```text
Task: Determine whether the entire liquid content of the source container can be poured into the target container without overflowing.

The source container is inside the yellow bounding box.
The target container is inside the green bounding box.

Estimate:
1. how much liquid is currently in the source container,
2. how much free capacity remains in the target container,
3. whether the source liquid can fit into the target container.

Choose "yes" only if all liquid currently in the source container can fit into the remaining empty space of the target container.
Choose "no" if it would overflow.
Choose "cant_tell" if at least one relevant container is opaque or the liquid amount cannot be visually estimated.

Allowed answers:
- "yes"
- "no"
- "cant_tell"

Return only:
{"can_pour": "<one_allowed_answer>"}
```

### Pouring Prediction

```text
Task: Predict how much liquid will remain in the highlighted container over time if it is tilted.

The highlighted container is inside the yellow bounding box.
Tilt angle: [ANGLE_DEGREES] degrees.

Assume the container starts upright and is gradually tilted from 0 degrees to [ANGLE_DEGREES] degrees. Predict the fraction of liquid remaining inside the container at each of exactly five timesteps, corresponding to t0_content, t1_content, t2_content, t3_content, and t4_content.

Use the visible initial fill level, container geometry, opening shape, neck width, and tilt angle. A narrow opening may retain liquid longer. A wide opening may spill earlier.

Allowed sequence values:
- 0.0
- 0.1
- 0.2
- 0.3
- 0.4
- 0.5
- 0.6
- 0.7
- 0.8
- 0.9
- 1.0
- "opaque"

Return exactly five values.
If the container is initially empty, return [0.0, 0.0, 0.0, 0.0, 0.0].
If the container is opaque, return ["opaque", "opaque", "opaque", "opaque", "opaque"].

Return only:
{"remaining_sequence": [<five_allowed_values>]}
```

## Model Output Schemas

```json
{"volume_bin": "100_200_ml"}
```

```json
{"fill_level": "33_percent"}
```

```json
{"can_pour": "no"}
```

```json
{"remaining_sequence": [1.0, 0.9, 0.8, 0.7, 0.6]}
```

The parser accepts strict JSON and common Qwen wrappers around otherwise valid JSON, including `assistant\n{...}` and Markdown-fenced JSON. Parsed values are still validated against the exact allowed keys and labels.

## Prediction JSONL Fields

Each row written by `run_qwen_coqe.py` includes:

- `sample_id`
- `task`
- `split`
- `image_id`
- `model`
- `prompt_version`
- `prompt`
- `raw_response`
- `parsed_prediction`
- `ground_truth`
- `valid`
- `parse_error`
- `correct`
- `metadata`

Pouring rows also include `edit_distance` and `edit_distance_le_0` through `edit_distance_le_4`.

## Metrics

For volume, content, and comparative volume, the paper-style metric is average per-class accuracy.

For pouring, the paper-style metrics are exact sequence match and edit-distance threshold accuracies:

- `edit_distance <= 0`
- `edit_distance <= 1`
- `edit_distance <= 2`
- `edit_distance <= 3`
- `edit_distance <= 4`

## Inference Examples

Representative pouring input examples are exported under:

`bakery/10.06.2026/coqe_pouring_input_examples/`

Each case contains the exact rendered input image, prompt, raw annotation, ground truth, and Qwen predictions.

### Opaque Ground Truth Missed By Both Models

Input image:

`bakery/10.06.2026/coqe_pouring_input_examples/01_opaque_gt_missed_by_both/model_input.png`

```json
{
  "sample_id": "test/pouring/00010/08882.png/angle_47",
  "angle": 47,
  "ground_truth": ["opaque", "opaque", "opaque", "opaque", "opaque"],
  "qwen_8b_prediction": [1.0, 1.0, 1.0, 1.0, 1.0],
  "qwen_30b_prediction": [1.0, 1.0, 1.0, 1.0, 1.0]
}
```

### Empty Ground Truth Missed By Both Models

Input image:

`bakery/10.06.2026/coqe_pouring_input_examples/02_empty_gt_missed_by_both/model_input.png`

```json
{
  "sample_id": "test/pouring/00020/01792.png/angle_57",
  "angle": 57,
  "ground_truth": [0.0, 0.0, 0.0, 0.0, 0.0],
  "qwen_8b_prediction": [1.0, 0.9, 0.8, 0.7, 0.6],
  "qwen_30b_prediction": [0.9, 0.8, 0.7, 0.6, 0.5]
}
```

### Constant Partial Ground Truth Missed By Both Models

Input image:

`bakery/10.06.2026/coqe_pouring_input_examples/03_constant_partial_gt_missed_by_both/model_input.png`

```json
{
  "sample_id": "test/pouring/00001/03582.png/angle_90",
  "angle": 90,
  "ground_truth": [0.3, 0.3, 0.3, 0.3, 0.3],
  "qwen_8b_prediction": [0.8, 0.7, 0.6, 0.5, 0.4],
  "qwen_30b_prediction": [0.8, 0.6, 0.4, 0.2, 0.0]
}
```

## Notes On Pouring Performance

The low pouring score appears to be mostly a task/input limitation rather than a parser or label-mapping bug. The COQE pouring labels are generated from CAD/simulation geometry and annotated initial content, while Qwen receives only an RGB image, a visual query box, and a text angle.

Observed failure modes:

- Opaque containers can look visually like ordinary empty mugs or cups.
- Empty translucent containers often lack an obvious liquid surface cue.
- Interior geometry, neck width, and lip shape are weakly visible or not visible from a single RGB image.
- Qwen frequently predicts generic smooth decreasing sequences or constant full sequences.

Useful follow-up ablations:

- Oracle content label in the prompt.
- Oracle content plus volume labels in the prompt.
- Full image plus query crop.
- Cleaner query overlay with thinner border and no label covering the container opening.


# vlm-lab-perception

Minimal dataset tooling and VLM task evaluation for liquid-container perception.

## Project structure

- `src/labliquidvision`: `LabLiquidVisionDataset`
- `src/glassesinthewild`: `GlassesInTheWildDataset`
- `src/dtld`: `DTLDDataset`
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

from labliquidvision import LabLiquidVisionDataset
from glassesinthewild import GlassesInTheWildDataset
from dtld import DTLDDataset

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

## Notebooks

Available preview notebooks:

- `notebooks/dataset_preview_labliquidvision.ipynb`
- `notebooks/dataset_preview_glasses.ipynb`
- `notebooks/dataset_preview_dtld.ipynb`
- `notebooks/dataset_preview_pouring.ipynb`
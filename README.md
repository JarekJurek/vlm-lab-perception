# vlm-lab-perception

Minimal tooling for the LabLiquidVision dataset.

## Project structure

- src/labliquidvision: minimal dataloader
- notebooks: dataset preview notebook
- data: dataset root (already present)

## Requirements

Install:

- numpy
- matplotlib

## Usage

Example (from repo root):

```python
import sys
sys.path.append("src")

from labliquidvision import LabLiquidVisionDataset

ds = LabLiquidVisionDataset("data/LabLiquidVision/processed")
sample = ds[0]
print(sample["id"], sample.get("vol_liquid"))
```

## Notebook

Open the preview notebook:

- notebooks/labliquidvision_preview.ipynb
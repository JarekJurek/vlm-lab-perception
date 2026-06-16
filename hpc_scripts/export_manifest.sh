#!/bin/bash
### --------------- job name ------------------
#BSUB -J export_tlg_manifest

### --------------- queue name ----------------
#BSUB -q hpc

### --------------- number of cores -----------
#BSUB -n 4
#BSUB -R "span[hosts=1]"

### --------------- CPU memory requirements ---
#BSUB -R "rusage[mem=4GB]"

### --------------- wall-clock time ---------------
#BSUB -W 4:00

### --------------- output and error files ---------------
#BSUB -o /zhome/a2/c/213547/vlm-lab-perception/bakery/manifest_export_%J.out
#BSUB -e /zhome/a2/c/213547/vlm-lab-perception/bakery/manifest_export_%J.err

### --------------- send email notifications -------------
#BSUB -u s242911@dtu.dk
#BSUB -B
#BSUB -N

set -euo pipefail

cd /zhome/a2/c/213547/vlm-lab-perception

source /zhome/a2/c/213547/work3/venv/bin/activate

python3 scripts/export_tlg_manifest.py \
    --datasets labliquidvision glasses dtld labpics_chemistry pouring ot2 coqe \
    --out manifests/tlg_all_seed42.json \
    --shuffle \
    --seed 42 \
    --train-ratio 0.8 \
    --coqe-split test \
    --labpics-chemistry-split all

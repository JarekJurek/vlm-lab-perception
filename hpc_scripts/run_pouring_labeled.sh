#!/bin/bash
### --------------- job name ------------------
#BSUB -J qwen_pouring_labeled

### --------------- queue name ----------------
#BSUB -q gpua100

### --------------- GPU request ---------------
#BSUB -gpu "num=1:mode=exclusive_process"

### --------------- number of cores -----------
#BSUB -n 4
#BSUB -R "span[hosts=1]"

### --------------- CPU memory requirements ---
#BSUB -R "rusage[mem=2GB]"

### --------------- wall-clock time ---------------
#BSUB -W 20:00

### --------------- output and error files ---------------
#BSUB -o /zhome/a2/c/213547/vlm-lab-perception/bakery/qwen_pouring_labeled_%J.out
#BSUB -e /zhome/a2/c/213547/vlm-lab-perception/bakery/qwen_pouring_labeled_%J.err

### --------------- send email notifications -------------
#BSUB -u s242911@dtu.dk
#BSUB -B
#BSUB -N

### --------------- Load environment and run Python script ---------------
set -euo pipefail

cd /zhome/a2/c/213547/vlm-lab-perception

source /zhome/a2/c/213547/work3/venv/bin/activate

export HF_HOME="/zhome/a2/c/213547/work3/master_thesis/models/"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export QWEN_MAX_PIXELS=$((512 * 28 * 28))
export QWEN_MAX_NEW_TOKENS=512

OUT_DIR="bakery/qwen_pouring_labeled"
mkdir -p "${OUT_DIR}"

python3 test_qwen_vl_tasks.py \
    --datasets pouring \
    --model qwen \
    --predictions-out "${OUT_DIR}/preds.jsonl" \
    --metrics-out "${OUT_DIR}/metrics.json"

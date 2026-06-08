#!/bin/bash
### --------------- job name ------------------
#BSUB -J qwen_coqe_test_30B

### --------------- queue name ----------------
#BSUB -q gpua100

### --------------- GPU request ---------------
#BSUB -gpu "num=2:mode=exclusive_process"

### --------------- number of cores -----------
#BSUB -n 8
#BSUB -R "span[hosts=1]"

### --------------- CPU memory requirements ---
#BSUB -R "rusage[mem=32GB]"

### --------------- wall-clock time ---------------
#BSUB -W 24:00

### --------------- output and error files ---------------
#BSUB -o /zhome/a2/c/213547/vlm-lab-perception/bakery/experiment_coqe_og_qwen_30B/qwen_coqe_test_%J.out
#BSUB -e /zhome/a2/c/213547/vlm-lab-perception/bakery/experiment_coqe_og_qwen_30B/qwen_coqe_test_%J.err

### --------------- send email notifications -------------
#BSUB -u s242911@dtu.dk
#BSUB -B
#BSUB -N

### --------------- Load environment and run Python script ---------------
source /zhome/a2/c/213547/work3/venv/bin/activate

cd /zhome/a2/c/213547/vlm-lab-perception

export HF_HOME="/zhome/a2/c/213547/work3/master_thesis/models/"
export HF_HUB_CACHE="${HF_HOME}/hub"
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_ENABLE_PARALLEL_LOADING=true

# Qwen processor image cap. The runner reads this environment variable.
export QWEN_MAX_PIXELS=$((512 * 28 * 28))

mkdir -p bakery/experiment_coqe_og_qwen_30B

python3 -B src/experimental/coqe/run_qwen_coqe.py \
    --split test \
    --data-root data/COQE_dataset \
    --tasks volume content comparative pouring \
    --model Qwen/Qwen3-VL-30B-A3B-Instruct \
    --max-new-tokens 128 \
    --predictions-out bakery/experiment_coqe_og_qwen_30B/predictions.jsonl \
    --metrics-out bakery/experiment_coqe_og_qwen_30B/metrics.json

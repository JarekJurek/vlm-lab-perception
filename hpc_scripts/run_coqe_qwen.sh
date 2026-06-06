#!/bin/bash
### --------------- job name ------------------
#BSUB -J qwen_coqe_test

### --------------- queue name ----------------
#BSUB -q gpuv100

### --------------- GPU request ---------------
#BSUB -gpu "num=1:mode=exclusive_process"

### --------------- number of cores -----------
#BSUB -n 4
#BSUB -R "span[hosts=1]"

### --------------- CPU memory requirements ---
#BSUB -R "rusage[mem=3GB]"

### --------------- wall-clock time ---------------
#BSUB -W 24:00

### --------------- output and error files ---------------
#BSUB -o /zhome/a2/c/213547/vlm-lab-perception/bakery/qwen_coqe_test_%J.out
#BSUB -e /zhome/a2/c/213547/vlm-lab-perception/bakery/qwen_coqe_test_%J.err

### --------------- send email notifications -------------
#BSUB -u s242911@dtu.dk
#BSUB -B
#BSUB -N

### --------------- Load environment and run Python script ---------------
source /zhome/a2/c/213547/work3/venv/bin/activate

cd /zhome/a2/c/213547/vlm-lab-perception

export HF_HOME="/zhome/a2/c/213547/work3/master_thesis/models/"
export HF_HUB_CACHE="${HF_HOME}/hub"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Qwen visual-token and generation caps. Override at submission if needed.
export QWEN_MAX_PIXELS="${QWEN_MAX_PIXELS:-$((512 * 28 * 28))}"
export QWEN_MAX_NEW_TOKENS="${QWEN_MAX_NEW_TOKENS:-128}"

COQE_SPLIT="${COQE_SPLIT:-test}"
COQE_TASKS="${COQE_TASKS:-volume content comparative pouring}"
COQE_DATA_ROOT="${COQE_DATA_ROOT:-data/COQE_dataset}"
COQE_OUT_DIR="${COQE_OUT_DIR:-bakery/qwen_coqe_${COQE_SPLIT}}"
COQE_MODEL="${COQE_MODEL:-Qwen/Qwen3-VL-8B-Instruct}"

mkdir -p "${COQE_OUT_DIR}"

MAX_SAMPLES_ARG=()
if [ -n "${COQE_MAX_SAMPLES:-}" ]; then
    MAX_SAMPLES_ARG=(--max-samples "${COQE_MAX_SAMPLES}")
fi

python3 -B src/experimental/coqe/run_qwen_coqe.py \
    --split "${COQE_SPLIT}" \
    --data-root "${COQE_DATA_ROOT}" \
    --tasks ${COQE_TASKS} \
    --model "${COQE_MODEL}" \
    "${MAX_SAMPLES_ARG[@]}" \
    --predictions-out "${COQE_OUT_DIR}/predictions.jsonl" \
    --metrics-out "${COQE_OUT_DIR}/metrics.json"

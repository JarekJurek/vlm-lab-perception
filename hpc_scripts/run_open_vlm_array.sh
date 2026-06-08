#!/bin/bash
### --------------- job name ------------------
#BSUB -J open_vlm_dataset[1-18]

### --------------- queue name ----------------
#BSUB -q gpuv100

### --------------- GPU request ---------------
#BSUB -gpu "num=1:mode=exclusive_process"

### --------------- number of cores -----------
#BSUB -n 4
#BSUB -R "span[hosts=1]"

### --------------- CPU memory requirements ---
#BSUB -R "rusage[mem=2GB]"

### --------------- wall-clock time ---------------
#BSUB -W 10:00

### --------------- output and error files ---------------
#BSUB -o /zhome/a2/c/213547/vlm-lab-perception/bakery/open_vlm_dataset_%J_%I.out
#BSUB -e /zhome/a2/c/213547/vlm-lab-perception/bakery/open_vlm_dataset_%J_%I.err

### --------------- send email notifications -------------
#BSUB -u s242911@dtu.dk
#BSUB -B
#BSUB -N

set -euo pipefail

cd /zhome/a2/c/213547/vlm-lab-perception

source /zhome/a2/c/213547/work3/venv/bin/activate

export HF_HOME="/zhome/a2/c/213547/work3/master_thesis/models/"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export HF_XET_CACHE="${HF_HOME}/xet"
export TORCH_HOME="${HF_HOME}/torch"
export XDG_CACHE_HOME="${HF_HOME}/xdg"
mkdir -p "${HF_HOME}" "${HF_HUB_CACHE}" "${HF_XET_CACHE}" "${TORCH_HOME}" "${XDG_CACHE_HOME}"

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export QWEN_MAX_PIXELS=$((512 * 28 * 28))
export QWEN_MAX_NEW_TOKENS="${QWEN_MAX_NEW_TOKENS:-256}"
export INTERNVL35_MAX_NEW_TOKENS="${INTERNVL35_MAX_NEW_TOKENS:-256}"
export LLAVAOV2_MAX_NEW_TOKENS="${LLAVAOV2_MAX_NEW_TOKENS:-256}"
export KIMI_VL_MAX_NEW_TOKENS="${KIMI_VL_MAX_NEW_TOKENS:-256}"

MODELS=(
    internvl35_8b
    llavaov2_8b
    kimi_vl_a3b
)

DATASETS=(
    labliquidvision
    glasses
    dtld
    labpicsv1
    pouring
    ot2
)

MODEL_INDEX=$(( (LSB_JOBINDEX - 1) / ${#DATASETS[@]} ))
DATASET_INDEX=$(( (LSB_JOBINDEX - 1) % ${#DATASETS[@]} ))
MODEL="${MODELS[${MODEL_INDEX}]}"
DATASET="${DATASETS[${DATASET_INDEX}]}"
TIMESTAMP=$(date +%Y%m%d_%H)
OUT_DIR="bakery/open_vlms_${TIMESTAMP}/${MODEL}/${DATASET}"
mkdir -p "${OUT_DIR}"

echo "Array job index: ${LSB_JOBINDEX}"
echo "Model: ${MODEL}"
echo "Dataset: ${DATASET}"
echo "HF_HOME: ${HF_HOME}"
echo "Output directory: ${OUT_DIR}"

python3 run_vlm_tasks.py \
    --datasets "${DATASET}" \
    --model "${MODEL}" \
    --predictions-out "${OUT_DIR}/preds.jsonl" \
    --metrics-out "${OUT_DIR}/metrics.json" \
    --shuffle \
    --seed 42

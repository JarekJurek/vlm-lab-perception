#!/bin/bash
### --------------- job name ------------------
#BSUB -J tlg_vlmeval_smoke

### --------------- queue name ----------------
#BSUB -q gpuv100

### --------------- GPU request ---------------
#BSUB -gpu "num=1:mode=exclusive_process"

### --------------- number of cores -----------
#BSUB -n 4
#BSUB -R "span[hosts=1]"

### --------------- CPU memory requirements ---
#BSUB -R "rusage[mem=4GB]"

### --------------- wall-clock time ---------------
#BSUB -W 2:00

### --------------- output and error files ---------------
#BSUB -o /zhome/a2/c/213547/vlm-lab-perception/bakery/vlmevalkit_smoke_%J.out
#BSUB -e /zhome/a2/c/213547/vlm-lab-perception/bakery/vlmevalkit_smoke_%J.err

### --------------- send email notifications -------------
#BSUB -u s242911@dtu.dk
#BSUB -B
#BSUB -N

set -euo pipefail

cd /zhome/a2/c/213547/vlm-lab-perception
source /zhome/a2/c/213547/work3/venv/bin/activate

export HF_HOME="${HF_HOME:-/zhome/a2/c/213547/work3/master_thesis/models/}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
export HF_XET_CACHE="${HF_XET_CACHE:-${HF_HOME}/xet}"
export TORCH_HOME="${TORCH_HOME:-${HF_HOME}/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${HF_HOME}/xdg}"
mkdir -p "${HF_HOME}" "${HF_HUB_CACHE}" "${HF_XET_CACHE}" "${TORCH_HOME}" "${XDG_CACHE_HOME}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

MANIFEST="${MANIFEST:-manifests/tlg_all_seed42.json}"
MODEL="${MODEL:-Qwen2-VL-7B-Instruct}"
DATASET="${DATASET:-labliquidvision}"
TASK="${TASK:-container_count}"
MAX_SAMPLES="${MAX_SAMPLES:-5}"
OUT_DIR="${OUT_DIR:-bakery/vlmevalkit_smoke/${MODEL}/${DATASET}_${TASK}}"

echo "Model: ${MODEL}"
echo "Manifest: ${MANIFEST}"
echo "Dataset: ${DATASET}"
echo "Task: ${TASK}"
echo "Max samples: ${MAX_SAMPLES}"
echo "Output directory: ${OUT_DIR}"
echo "HF_HOME: ${HF_HOME}"

python3 - <<'PY'
import importlib
config = importlib.import_module("vlmeval.config")
keys = []
for value in vars(config).values():
    if isinstance(value, dict):
        keys.extend(str(k) for k, v in value.items() if callable(v))
print("VLMEvalKit model registry sample:", ", ".join(sorted(set(keys))[:20]))
PY

python3 scripts/run_tlg_vlmeval.py \
    --manifest "${MANIFEST}" \
    --manifest-split test \
    --datasets "${DATASET}" \
    --tasks "${TASK}" \
    --model "${MODEL}" \
    --max-samples "${MAX_SAMPLES}" \
    --work-dir "${OUT_DIR}" \
    --resume

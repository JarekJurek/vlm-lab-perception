#!/bin/bash
### --------------- job name ------------------
#BSUB -J qwen_run_1

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
#BSUB -W 1:00

### --------------- output and error files ---------------
#BSUB -o /zhome/a2/c/213547/vlm-lab-perception/bakery/qwen_run_1_%J.out
#BSUB -e /zhome/a2/c/213547/vlm-lab-perception/bakery/qwen_run_1_%J.err
    
### --------------- send email notifications -------------
#BSUB -u s242911@dtu.dk
#BSUB -B
#BSUB -N

### --------------- Load environment and run Python script ---------------
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

# Qwen-VL memory is dominated by visual tokens. LabPicsV1 contains larger
# source images than the other datasets, so cap processor pixels for 16 GB GPUs.
export QWEN_MAX_PIXELS=$((512 * 28 * 28))
export QWEN_MAX_NEW_TOKENS=128
export INTERNVL35_MAX_NEW_TOKENS="${INTERNVL35_MAX_NEW_TOKENS:-128}"
export LLAVAOV2_MAX_NEW_TOKENS="${LLAVAOV2_MAX_NEW_TOKENS:-128}"
export KIMI_VL_MAX_NEW_TOKENS="${KIMI_VL_MAX_NEW_TOKENS:-128}"

# Single-model runner. Supported --model values:
#   qwen           -> Qwen/Qwen3-VL-8B-Instruct
#   sdvlm          -> cpystan/SD-VLM-7B; add --sdvlm-gt-depth or --sdvlm-depth-dir if needed
#   internvl35_8b  -> OpenGVLab/InternVL3_5-8B
#   llavaov2_8b    -> lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct
#   kimi_vl_a3b    -> moonshotai/Kimi-VL-A3B-Thinking-2506
mkdir -p bakery/qwen_run_1

python3 run_vlm_tasks.py \
    --datasets labliquidvision glasses dtld labpicsv1 pouring ot2 \
    --max-samples 500 \
    --shuffle \
    --seed 42 \
    --model qwen \
    --predictions-out bakery/qwen_run_1/preds.jsonl \
    --metrics-out bakery/qwen_run_1/metrics.json

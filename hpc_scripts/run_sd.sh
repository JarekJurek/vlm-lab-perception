#!/bin/bash
### --------------- job name ------------------
#BSUB -J sdvlm_cached_depth_all

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
#BSUB -W 8:00

### --------------- output and error files ---------------
#BSUB -o /zhome/a2/c/213547/vlm-lab-perception/bakery/sdvlm_cached_depth_all_%J.out
#BSUB -e /zhome/a2/c/213547/vlm-lab-perception/bakery/sdvlm_cached_depth_all_%J.err
    
### --------------- send email notifications -------------
#BSUB -u s242911@dtu.dk
#BSUB -B
#BSUB -N

### --------------- Load environment and run Python script ---------------
source /zhome/a2/c/213547/work3/venv_sd/bin/activate

export HF_HOME="/zhome/a2/c/213547/work3/master_thesis/models/"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export HF_XET_CACHE="${HF_HOME}/xet"
export TORCH_HOME="${HF_HOME}/torch"
export XDG_CACHE_HOME="${HF_HOME}/xdg"
mkdir -p "${HF_HOME}" "${HF_HUB_CACHE}" "${HF_XET_CACHE}" "${TORCH_HOME}" "${XDG_CACHE_HOME}"

export SDVLM_DEPTH_DIR="${SDVLM_DEPTH_DIR:-bakery/sdvlm_da3_all/da3_depth_cache}"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python3 run_vlm_tasks.py \
    --model sdvlm \
    --sdvlm-depth-dir "${SDVLM_DEPTH_DIR}" \
    --datasets labliquidvision glasses dtld labpics_chemistry pouring ot2 \
    --max-samples 10 \
    --shuffle \
    --seed 42 \
    --predictions-out bakery/sdvlm_cached_depth_all/preds.jsonl \
    --metrics-out bakery/sdvlm_cached_depth_all/metrics.json

#!/bin/bash
### --------------- job name ------------------
#BSUB -J da3_depth_precompute

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
#BSUB -o /zhome/a2/c/213547/vlm-lab-perception/bakery/da3_depth_precompute_%J.out
#BSUB -e /zhome/a2/c/213547/vlm-lab-perception/bakery/da3_depth_precompute_%J.err

### --------------- send email notifications -------------
#BSUB -u s242911@dtu.dk
#BSUB -B
#BSUB -N

### --------------- Load environment and run DA3 precompute ---------------
source /zhome/a2/c/213547/work3/venv_sd/bin/activate

export HF_HOME="/zhome/a2/c/213547/work3/master_thesis/models/"
export HF_HUB_CACHE="${HF_HOME}/hub"
export DA3_MODEL_ID="depth-anything/DA3METRIC-LARGE"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python3 scripts/precompute_da3_depth.py \
    --da3-model-path "${DA3_MODEL_ID}" \
    --da3-cache-dir bakery/sdvlm_da3_all/da3_depth_cache \
    --datasets labliquidvision dtld \
    --max-samples 10 \
    --shuffle \
    --seed 42 \
    --manifest-out bakery/sdvlm_da3_all/da3_depth_manifest.jsonl

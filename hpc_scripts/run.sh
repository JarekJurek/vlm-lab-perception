#!/bin/bash
### --------------- job name ------------------
#BSUB -J qwen_labpicsv1

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
#BSUB -o /zhome/a2/c/213547/vlm-lab-perception/bakery/qwen_labpicsv1_%J.out
#BSUB -e /zhome/a2/c/213547/vlm-lab-perception/bakery/qwen_labpicsv1_%J.err
    
### --------------- send email notifications -------------
#BSUB -u s242911@dtu.dk
#BSUB -B
#BSUB -N

### --------------- Load environment and run Python script ---------------
source /zhome/a2/c/213547/work3/venv/bin/activate

export HF_HOME="/zhome/a2/c/213547/work3/master_thesis/models/"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Qwen-VL memory is dominated by visual tokens. LabPicsV1 contains larger
# source images than the other datasets, so cap processor pixels for 16 GB GPUs.
export QWEN_MAX_PIXELS=$((512 * 28 * 28))
export QWEN_MAX_NEW_TOKENS=128

python3 test_qwen_vl_tasks.py \
    --datasets labpicsv1 \
    --max-samples 500 \
    --shuffle \
    --seed 42 \
    --model qwen \
    --predictions-out bakery/qwen_labpicsv1/preds.jsonl \
    --metrics-out bakery/qwen_labpicsv1/metrics.json

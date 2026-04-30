#!/bin/bash
### --------------- job name ------------------
#BSUB -J sd_vlm_1

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
#BSUB -W 0:45

### --------------- output and error files ---------------
#BSUB -o /zhome/a2/c/213547/vlm-lab-perception/bakery/sd_vlm_1_%J.out
#BSUB -e /zhome/a2/c/213547/vlm-lab-perception/bakery/sd_vlm_1_%J.err
    
### --------------- send email notifications -------------
#BSUB -u s242911@dtu.dk
#BSUB -B
#BSUB -N

### --------------- Load environment and run Python script ---------------
source /zhome/a2/c/213547/work3/venv_sd/bin/activate

export HF_HOME="/zhome/a2/c/213547/work3/master_thesis/models/"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

python3 test_qwen_vl_tasks.py \
    --model sdvlm \
    --sdvlm-gt-depth \
    --datasets labliquidvision \
    --max-samples 10 \
    --viz-attn \
    --viz-attn-mode first_answer_token \
    --viz-out bakery/sd_vlm_1/attn \
    --predictions-out bakery/sd_vlm_1/preds.jsonl \
    --metrics-out bakery/sd_vlm_1/metrics.json

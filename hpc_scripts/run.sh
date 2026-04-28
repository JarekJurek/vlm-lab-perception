#!/bin/bash
### --------------- job name ------------------
#BSUB -J attn_2_a

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
#BSUB -W 0:20

### --------------- output and error files ---------------
#BSUB -o /zhome/a2/c/213547/vlm-lab-perception/bakery/attention_viz2V_%J.out
#BSUB -e /zhome/a2/c/213547/vlm-lab-perception/bakery/attention_viz2V_%J.err
    
### --------------- send email notifications -------------
#BSUB -u s242911@dtu.dk
#BSUB -B
#BSUB -N

### --------------- Load environment and run Python script ---------------
source /zhome/a2/c/213547/hypersight/venv/bin/activate

export HF_HOME="/zhome/a2/c/213547/work3/master_thesis/models/"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

python3 test_qwen_vl_tasks.py \
    --datasets labliquidvision glasses dtld pouring ot2 \
    --max-samples 3 \
    --viz-attn \
    --viz-out bakery/run2V/attn \
    --predictions-out bakery/run2V/preds.jsonl \
    --metrics-out bakery/run2V/metrics.json

#!/bin/bash
### --------------- job name ------------------
#BSUB -J attn_5

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
#BSUB -W 0:45

### --------------- output and error files ---------------
#BSUB -o /zhome/a2/c/213547/vlm-lab-perception/bakery/attention_viz5_%J.out
#BSUB -e /zhome/a2/c/213547/vlm-lab-perception/bakery/attention_viz5_%J.err
    
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

python3 test_qwen_vl_tasks.py \
    --datasets labliquidvision glasses dtld pouring ot2 \
    --max-samples 4 \
    --model qwen \
    --viz-attn \
    --viz-attn-mode first_answer_token \
    --viz-out /zhome/a2/c/213547/work3/master_thesis/run5/attn \
    --predictions-out /zhome/a2/c/213547/work3/master_thesis/run5/preds.jsonl \
    --metrics-out /zhome/a2/c/213547/work3/master_thesis/run5/metrics.json

# Example: run SD-VLM only on LabLiquidVision with ground-truth depth
# python3 test_qwen_vl_tasks.py \
#     --model sdvlm \
#     --datasets labliquidvision \
#     --sdvlm-gt-depth \
#     --max-samples 4 \
#     --predictions-out bakery/run_sdvlm/preds.jsonl \
#     --metrics-out bakery/run_sdvlm/metrics.json

# Example: run both Qwen and SD-VLM side-by-side on LabLiquidVision
# python3 test_qwen_vl_tasks.py \
#     --model both \
#     --datasets labliquidvision \
#     --sdvlm-gt-depth \
#     --max-samples 4 \
#     --predictions-out bakery/run_both/preds.jsonl \
#     --metrics-out bakery/run_both/metrics.json

#!/bin/bash
### --------------- job name ------------------
#BSUB -J test_run2

### --------------- queue name ----------------
#BSUB -q gpuv100

### --------------- GPU request ---------------
#BSUB -gpu "num=1:mode=exclusive_process"

### --------------- number of cores -----------
#BSUB -n 6
#BSUB -R "span[hosts=1]"

### --------------- CPU memory requirements ---
#BSUB -R "rusage[mem=8GB]"

### --------------- wall-clock time ---------------
#BSUB -W 23:59

### --------------- output and error files ---------------
#BSUB -o /zhome/a2/c/213547/vlm-lab-perception/bakery/test_run2_%J.out
#BSUB -e /zhome/a2/c/213547/vlm-lab-perception/bakery/test_run2_%J.err
    
### --------------- send email notifications -------------
#BSUB -u s242911@dtu.dk
#BSUB -B
#BSUB -N

### --------------- Load environment and run Python script ---------------
source /zhome/a2/c/213547/vlm-lab-perception/venv/bin/activate
export HF_HOME="/zhome/a2/c/213547/work3/master_thesis/models/"
python3 /zhome/a2/c/213547/vlm-lab-perception/test_qwen_vl_tasks.py --datasets labliquidvision glasses

#!/bin/bash
### --------------- job name ------------------
#BSUB -J check_tlg_vlmeval

### --------------- queue name ----------------
#BSUB -q hpc

### --------------- number of cores -----------
#BSUB -n 4
#BSUB -R "span[hosts=1]"

### --------------- CPU memory requirements ---
#BSUB -R "rusage[mem=4GB]"

### --------------- wall-clock time ---------------
#BSUB -W 1:00

### --------------- output and error files ---------------
#BSUB -o /zhome/a2/c/213547/vlm-lab-perception/bakery/vlmevalkit_check_%J.out
#BSUB -e /zhome/a2/c/213547/vlm-lab-perception/bakery/vlmevalkit_check_%J.err

### --------------- send email notifications -------------
#BSUB -u s242911@dtu.dk
#BSUB -B
#BSUB -N

set -euo pipefail

cd /zhome/a2/c/213547/vlm-lab-perception
source /zhome/a2/c/213547/work3/venv/bin/activate

MANIFEST="${MANIFEST:-manifests/tlg_all_seed42.json}"
DATASET="${DATASET:-labliquidvision}"
MAX_SAMPLES="${MAX_SAMPLES:-10}"
OUT_DIR="${OUT_DIR:-bakery/vlmevalkit_check}"
mkdir -p "${OUT_DIR}"

echo "Python: $(python3 --version)"
echo "Manifest: ${MANIFEST}"
echo "Dataset: ${DATASET}"
echo "Max samples: ${MAX_SAMPLES}"

python3 - <<'PY'
try:
    import vlmeval
except Exception as exc:
    print(f"VLMEvalKit import: NOT INSTALLED ({type(exc).__name__}: {exc})")
else:
    print(f"VLMEvalKit import: OK ({vlmeval.__file__})")
PY

python3 -m unittest tests.test_vlmevalkit_adapter tests.test_manifest

python3 scripts/run_tlg_vlmeval.py \
    --manifest "${MANIFEST}" \
    --manifest-split test \
    --datasets "${DATASET}" \
    --max-samples "${MAX_SAMPLES}" \
    --dump-rows-out "${OUT_DIR}/${DATASET}_rows.jsonl" \
    --dry-run

echo "VLMEvalKit adapter check complete."

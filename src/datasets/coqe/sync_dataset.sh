#!/bin/bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <source_root> <destination_root>" >&2
  echo "Example: $0 /path/to/COQE_dataset /zhome/user/datasets/COQE_dataset" >&2
  exit 1
fi

src_root="$1"
dst_root="$2"

rsync -av --prune-empty-dirs \
  --include="*/" \
  --include="README" \
  --include="train.json" \
  --include="val.json" \
  --include="test.json" \
  --include="train_pour_prediction.json" \
  --include="val_pour_prediction.json" \
  --include="test_pour_prediction.json" \
  --include="images/*.png" \
  --exclude="*" \
  "$src_root/" "$dst_root/"

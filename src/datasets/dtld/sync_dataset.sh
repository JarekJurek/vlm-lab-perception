#!/bin/bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <source_root> <destination_root>" >&2
  echo "Example: $0 /path/to/DTLD_dataset /zhome/user/datasets/DTLD_dataset" >&2
  exit 1
fi

src_root="$1"
dst_root="$2"

rsync -av --prune-empty-dirs \
  --exclude="*/mask/" \
  --exclude="*/mask_visib/" \
  --exclude="*/liquid_label/" \
  --exclude="*/mask_visib_liquid/" \
  "$src_root/" "$dst_root/"
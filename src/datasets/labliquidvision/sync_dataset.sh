#!/bin/bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <source_root> <destination_root>" >&2
  echo "Example: $0 /path/to/labliquidvision /zhome/user/datasets/labliquidvision" >&2
  exit 1
fi

src_root="$1"
dst_root="$2"

src_processed="$src_root/processed/"
dst_processed="$dst_root/processed/"

rsync -av --prune-empty-dirs \
  --include="*/" \
  --include="Input_color_liquid.txt" \
  --include="Input_RGBImage.png" \
  --include="Input_vol_vessel.txt" \
  --include="Input_DepthMap.npy" \
  --include="Input_vessel.txt" \
  --include="Input_DepthMap.png" \
  --include="Input_vol_liquid.txt" \
  --exclude="*" \
  "$src_processed" "$dst_processed"

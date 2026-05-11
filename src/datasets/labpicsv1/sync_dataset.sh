#!/bin/bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <source_root> <destination_root>" >&2
  echo "Example: $0 /path/to/LabPicsV1 /zhome/user/datasets/LabPicsV1" >&2
  exit 1
fi

src_root="$1"
dst_root="$2"

rsync -av --prune-empty-dirs \
  --include="/Simple/" \
  --include="/Simple/Train/" \
  --include="/Simple/Test/" \
  --include="/Simple/Train/InstCategory.json" \
  --include="/Simple/Test/InstCategory.json" \
  --include="/Simple/Train/Image/" \
  --include="/Simple/Test/Image/" \
  --include="/Simple/Train/Image/***" \
  --include="/Simple/Test/Image/***" \
  --include="/Simple/Train/Instance/" \
  --include="/Simple/Test/Instance/" \
  --include="/Simple/Train/Instance/*.png" \
  --include="/Simple/Test/Instance/*.png" \
  --include="/Simple/Train/Semantic/" \
  --include="/Simple/Test/Semantic/" \
  --include="/Simple/Train/Semantic/1_Vessel/" \
  --include="/Simple/Test/Semantic/1_Vessel/" \
  --include="/Simple/Train/Semantic/1_Vessel/*.png" \
  --include="/Simple/Test/Semantic/1_Vessel/*.png" \
  --include="/Simple/Train/Semantic/6_Liquid GENERAL/" \
  --include="/Simple/Test/Semantic/6_Liquid GENERAL/" \
  --include="/Simple/Train/Semantic/6_Liquid GENERAL/*.png" \
  --include="/Simple/Test/Semantic/6_Liquid GENERAL/*.png" \
  --include="/Simple/Train/Semantic/7_Liquid Suspension/" \
  --include="/Simple/Test/Semantic/7_Liquid Suspension/" \
  --include="/Simple/Train/Semantic/7_Liquid Suspension/*.png" \
  --include="/Simple/Test/Semantic/7_Liquid Suspension/*.png" \
  --include="/Simple/Train/Semantic/8_Foam/" \
  --include="/Simple/Test/Semantic/8_Foam/" \
  --include="/Simple/Train/Semantic/8_Foam/*.png" \
  --include="/Simple/Test/Semantic/8_Foam/*.png" \
  --include="/Simple/Train/Semantic/9_Gel/" \
  --include="/Simple/Test/Semantic/9_Gel/" \
  --include="/Simple/Train/Semantic/9_Gel/*.png" \
  --include="/Simple/Test/Semantic/9_Gel/*.png" \
  --include="/Simple/Train/Semantic/10_Solid GENERAL/" \
  --include="/Simple/Test/Semantic/10_Solid GENERAL/" \
  --include="/Simple/Train/Semantic/10_Solid GENERAL/*.png" \
  --include="/Simple/Test/Semantic/10_Solid GENERAL/*.png" \
  --include="/Simple/Train/Semantic/11_Granular/" \
  --include="/Simple/Test/Semantic/11_Granular/" \
  --include="/Simple/Train/Semantic/11_Granular/*.png" \
  --include="/Simple/Test/Semantic/11_Granular/*.png" \
  --include="/Simple/Train/Semantic/12_Powder/" \
  --include="/Simple/Test/Semantic/12_Powder/" \
  --include="/Simple/Train/Semantic/12_Powder/*.png" \
  --include="/Simple/Test/Semantic/12_Powder/*.png" \
  --include="/Simple/Train/Semantic/13_Solid Bulk/" \
  --include="/Simple/Test/Semantic/13_Solid Bulk/" \
  --include="/Simple/Train/Semantic/13_Solid Bulk/*.png" \
  --include="/Simple/Test/Semantic/13_Solid Bulk/*.png" \
  --include="/Simple/Train/Semantic/14_Vapor/" \
  --include="/Simple/Test/Semantic/14_Vapor/" \
  --include="/Simple/Train/Semantic/14_Vapor/*.png" \
  --include="/Simple/Test/Semantic/14_Vapor/*.png" \
  --exclude="*" \
  "$src_root/" "$dst_root/"

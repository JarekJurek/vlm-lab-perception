#!/bin/bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <source_root> <destination_root>" >&2
  echo "Example: $0 '/path/to/LabPics_Chemistry' '/zhome/user/datasets/LabPics_Chemistry'" >&2
  exit 1
fi

src_root="$1"
dst_root="$2"

rsync -av --prune-empty-dirs \
  --include="/Train/" \
  --include="/Test/" \
  --include="/Train/*/" \
  --include="/Test/*/" \
  --include="/Train/*/Image.jpg" \
  --include="/Test/*/Image.jpg" \
  --include="/Train/*/Data.json" \
  --include="/Test/*/Data.json" \
  --include="/Train/*/Source.txt" \
  --include="/Test/*/Source.txt" \
  --include="/Train/*/Ignore.png" \
  --include="/Test/*/Ignore.png" \
  --include="/Train/*/Vessels/" \
  --include="/Test/*/Vessels/" \
  --include="/Train/*/Vessels/*.png" \
  --include="/Test/*/Vessels/*.png" \
  --include="/Train/*/MaterialsAndParts/" \
  --include="/Test/*/MaterialsAndParts/" \
  --include="/Train/*/MaterialsAndParts/*.png" \
  --include="/Test/*/MaterialsAndParts/*.png" \
  --include="/Train/*/SemanticMaps/" \
  --include="/Test/*/SemanticMaps/" \
  --include="/Train/*/SemanticMaps/FullImage/" \
  --include="/Test/*/SemanticMaps/FullImage/" \
  --include="/Train/*/SemanticMaps/FullImage/Vessel.png" \
  --include="/Test/*/SemanticMaps/FullImage/Vessel.png" \
  --include="/Train/*/SemanticMaps/FullImage/Liquid.png" \
  --include="/Test/*/SemanticMaps/FullImage/Liquid.png" \
  --include="/Train/*/SemanticMaps/FullImage/Suspension.png" \
  --include="/Test/*/SemanticMaps/FullImage/Suspension.png" \
  --include="/Train/*/SemanticMaps/FullImage/Foam.png" \
  --include="/Test/*/SemanticMaps/FullImage/Foam.png" \
  --include="/Train/*/SemanticMaps/FullImage/Gel.png" \
  --include="/Test/*/SemanticMaps/FullImage/Gel.png" \
  --include="/Train/*/SemanticMaps/FullImage/Solid.png" \
  --include="/Test/*/SemanticMaps/FullImage/Solid.png" \
  --include="/Train/*/SemanticMaps/FullImage/SolidLargChunk.png" \
  --include="/Test/*/SemanticMaps/FullImage/SolidLargChunk.png" \
  --include="/Train/*/SemanticMaps/FullImage/Powder.png" \
  --include="/Test/*/SemanticMaps/FullImage/Powder.png" \
  --include="/Train/*/SemanticMaps/FullImage/Granular.png" \
  --include="/Test/*/SemanticMaps/FullImage/Granular.png" \
  --include="/Train/*/SemanticMaps/FullImage/Vapor.png" \
  --include="/Test/*/SemanticMaps/FullImage/Vapor.png" \
  --exclude="*" \
  "$src_root/" "$dst_root/"

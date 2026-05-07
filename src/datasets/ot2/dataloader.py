from __future__ import annotations

import inspect
import re
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from tlg import PromptBank, TLGAnnotation, make_tlg_annotation


_IMG_EXTS = {".jpg", ".jpeg", ".png"}
_DEFAULT_SPLITS: Tuple[str, ...] = ("train", "valid", "test")
_SEQ_RE = re.compile(r"^(?P<base>.*)_(?P<idx>\d+)$")


_DATACLASS_KWARGS = {"slots": True} if "slots" in inspect.signature(dataclass).parameters else {}


@dataclass(frozen=True, **_DATACLASS_KWARGS)
class YoloBox:
    cls: int
    xc: float
    yc: float
    w: float
    h: float

    def to_dict(self, class_names: Optional[Dict[int, str]] = None) -> Dict[str, object]:
        d: Dict[str, object] = {
            "cls": int(self.cls),
            "xc": float(self.xc),
            "yc": float(self.yc),
            "w": float(self.w),
            "h": float(self.h),
        }
        if class_names is not None:
            d["class_name"] = str(class_names.get(int(self.cls), str(self.cls)))
        return d


class OT2Dataset:
    """Minimal folder-based dataset loader for OT2 (Tips + Liquid).

    Expected dataset layout (YOLO-style):
        root/
          data.yaml                # optional; provides class names
          train/images/*.jpg
          train/labels/*.txt
          valid/images/*.jpg
          valid/labels/*.txt
          test/images/*.jpg
          test/labels/*.txt

    Notes:
    - Each label file line must be: `class xc yc w h` in normalized coords.
    - Samples are frames; `sequence_id` and `frame_index` are derived from filenames
      like `tipsWater_4.jpg` (heuristic).
    """

    def __init__(
        self,
        root: Union[str, Path],
        ids: Optional[Iterable[str]] = None,
        splits: Sequence[str] = _DEFAULT_SPLITS,
        prompt_bank: Optional[PromptBank] = None,
        shuffle: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.root}")

        self._splits = tuple(str(s) for s in splits)
        self._prompt_bank = prompt_bank or PromptBank.default()
        self._class_names = _load_yolo_class_names(self.root)

        # Support both:
        # 1) Standard YOLO split layout: root/{train,valid,test}/{images,labels}/...
        # 2) Flat layout: root/*.jpg + root/*.txt (labels alongside images)
        self._flat_dir: Optional[Path] = None
        if not self._has_yolo_split_layout():
            self._flat_dir = self._discover_flat_dir()

        if ids is not None:
            self._entries = self._resolve_ids(list(ids))
        else:
            self._entries = self._discover_entries()

        if shuffle:
            rng = np.random.default_rng(seed)
            self._entries = [
                self._entries[i]
                for i in rng.permutation(len(self._entries))
            ]

    def _has_yolo_split_layout(self) -> bool:
        for split in self._splits:
            images_dir = self.root / split / "images"
            labels_dir = self.root / split / "labels"
            if images_dir.is_dir() and labels_dir.is_dir():
                return True
        return False

    def _discover_flat_dir(self) -> Optional[Path]:
        """Heuristic: find a directory containing YOLO-style image/label pairs."""

        candidates = [self.root]
        # Common layout for the OT2-Computer-Vision dataset drop.
        candidates.extend(
            [
                self.root / "Training Data" / "Images_with_Labels_tips_n_Liquid",
                self.root / "Training Data" / "Images with labels_tips only",
                self.root / "Training Data" / "Images",
            ]
        )

        for d in candidates:
            if not d.is_dir():
                continue
            imgs = [p for p in d.iterdir() if p.is_file() and p.suffix.lower() in _IMG_EXTS]
            if not imgs:
                continue
            # Require at least one matching .txt label file.
            for p in imgs[:50]:
                if p.with_suffix(".txt").exists():
                    return d
        return None

    def _discover_entries(self) -> List[Tuple[str, Path]]:
        entries: List[Tuple[str, Path]] = []

        # Flat mode.
        if self._flat_dir is not None:
            for p in sorted(self._flat_dir.iterdir()):
                if p.is_file() and p.suffix.lower() in _IMG_EXTS:
                    entries.append(("all", p))
            return entries

        # YOLO split mode.
        for split in self._splits:
            images_dir = self.root / split / "images"
            if not images_dir.is_dir():
                continue
            for p in sorted(images_dir.iterdir()):
                if p.is_file() and p.suffix.lower() in _IMG_EXTS:
                    entries.append((split, p))
        return entries

    def _resolve_ids(self, ids: List[str]) -> List[Tuple[str, Path]]:
        """Resolve ids to entries.

        Supported forms (YOLO split mode):
        - "train/tipsWater_4" (split/stem)
        - "train/images/tipsWater_4.jpg" (relative path under split)

        Supported forms (flat mode):
        - "tipsWater_4" (stem)
        - "tipsWater_4.jpg" (filename)

        Returns entries with existing images only.
        """

        entries: List[Tuple[str, Path]] = []
        for raw in ids:
            s = str(raw).strip()
            if not s:
                continue

            # Flat mode ids.
            if self._flat_dir is not None and "/" not in s:
                p = self._flat_dir / s
                if p.suffix.lower() not in _IMG_EXTS:
                    # Try all extensions.
                    candidate: Optional[Path] = None
                    for ext in sorted(_IMG_EXTS):
                        q = self._flat_dir / f"{s}{ext}"
                        if q.exists():
                            candidate = q
                            break
                    if candidate is None:
                        raise FileNotFoundError(f"Could not resolve OT2 id={raw!r} in flat_dir={self._flat_dir}")
                    p = candidate
                if not p.exists():
                    raise FileNotFoundError(f"Could not resolve OT2 id={raw!r} in flat_dir={self._flat_dir}")
                entries.append(("all", p))
                continue

            parts = s.split("/", 1)
            if len(parts) != 2:
                raise ValueError(
                    "OT2Dataset ids must be of form 'split/stem' or 'split/relative_path'. "
                    f"Got: {raw!r}"
                )
            split, rest = parts[0], parts[1]

            # If a file extension is present, treat as relative path.
            candidate: Optional[Path] = None
            if Path(rest).suffix.lower() in _IMG_EXTS:
                p = self.root / split / rest
                if p.exists():
                    candidate = p
            else:
                images_dir = self.root / split / "images"
                for ext in sorted(_IMG_EXTS):
                    p = images_dir / f"{rest}{ext}"
                    if p.exists():
                        candidate = p
                        break

            if candidate is None:
                raise FileNotFoundError(f"Could not resolve OT2 id={raw!r} under split={split!r}")

            entries.append((split, candidate))

        return entries

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]

    def __getitem__(self, index: int) -> Dict[str, object]:
        split, img_path = self._entries[index]

        if split == "all" and self._flat_dir is not None:
            label_path = img_path.with_suffix(".txt")
        else:
            label_path = self.root / split / "labels" / f"{img_path.stem}.txt"
        boxes = _parse_yolo_label_file(label_path)

        sequence_id, frame_index = _derive_sequence_fields(img_path.stem)

        image_id = f"{split}/{img_path.stem}" if split != "all" else str(img_path.stem)
        sample: Dict[str, object] = {
            "id": image_id,
            "split": split,
            "path": str(img_path),
            "rgb_image": img_path,
            "label_path": str(label_path),
            "class_names": dict(self._class_names),
            "yolo_boxes": [b.to_dict(self._class_names) for b in boxes],
        }
        if sequence_id is not None:
            sample["sequence_id"] = sequence_id
        if frame_index is not None:
            sample["frame_index"] = frame_index

        sample["annotations"] = self._build_annotations(image_id=image_id, boxes=boxes)
        return sample

    def _build_annotations(self, image_id: str, boxes: List[YoloBox]) -> List[TLGAnnotation]:
        dataset = "ot2"
        add = partial(
            make_tlg_annotation,
            self._prompt_bank,
            dataset=dataset,
            image_id=image_id,
        )

        tip_boxes = [b for b in boxes if int(b.cls) == 0]
        liquid_boxes = [b for b in boxes if int(b.cls) == 1]

        container_count = len(tip_boxes)
        liquid_box_count = len(liquid_boxes)
        fill_percentage = _compute_fill_percentage(tip_boxes, liquid_boxes)

        annotations = [
            add(
                task="tip_presence",
                answer="YES" if container_count > 0 else "NO",
                metadata={"container_count": container_count},
            ),
            add(
                task="liquid_presence",
                answer="YES" if liquid_box_count > 0 else "NO",
                metadata={"liquid_box_count": liquid_box_count},
            ),
            add(
                task="container_count",
                answer=int(container_count),
                metadata={"container_count": container_count, "source": "tip_boxes"},
            ),
            add(
                task="fill_percentage",
                answer=float(fill_percentage),
                metadata={"unit": "percent_of_tip_height"},
            ),
        ]

        if container_count > 0 and self._prompt_bank.supports(dataset, "container_type"):
            annotations.append(
                add(
                    task="container_type",
                    answer="Tip",
                    metadata={"container_count": container_count},
                )
            )

        return annotations


def _derive_sequence_fields(stem: str) -> Tuple[Optional[str], Optional[int]]:
    m = _SEQ_RE.match(stem)
    if not m:
        return None, None
    base = m.group("base")
    idx = int(m.group("idx"))
    return base, idx


def _parse_yolo_label_file(path: Path) -> List[YoloBox]:
    boxes: List[YoloBox] = []
    if not path.exists():
        return boxes

    for ln, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"Unexpected YOLO format in {path} line {ln}: {line!r}")
        cls, xc, yc, w, h = parts
        boxes.append(YoloBox(int(cls), float(xc), float(yc), float(w), float(h)))

    return boxes


def _compute_fill_percentage(tip_boxes: List[YoloBox], liquid_boxes: List[YoloBox]) -> float:
    """Estimate fill percentage as 100 * liquid_box_height / tip_box_height.

    Matching rule:
    - For each liquid box, find the smallest tip box that contains the liquid box center.
    - Return the highest fill percentage across matched pairs.
    - If nothing can be matched, return 0.0.
    """

    best_fill = 0.0

    for liquid in liquid_boxes:
        matched_tip: Optional[YoloBox] = None
        matched_tip_area: Optional[float] = None

        for tip in tip_boxes:
            x1 = tip.xc - tip.w / 2.0
            x2 = tip.xc + tip.w / 2.0
            y1 = tip.yc - tip.h / 2.0
            y2 = tip.yc + tip.h / 2.0

            inside = x1 <= liquid.xc <= x2 and y1 <= liquid.yc <= y2
            if not inside:
                continue

            area = tip.w * tip.h
            if matched_tip is None or area < matched_tip_area:
                matched_tip = tip
                matched_tip_area = area

        if matched_tip is None or matched_tip.h <= 0:
            continue

        fill = 100.0 * (liquid.h / matched_tip.h)
        fill = max(0.0, min(100.0, fill))
        best_fill = max(best_fill, fill)

    return float(best_fill)


def _load_yolo_class_names(root: Path) -> Dict[int, str]:
    """Load class names from YOLO `data.yaml` if present; else default."""

    default = {0: "Tip", 1: "Liquid"}
    cfg_path = root / "data.yaml"
    if not cfg_path.exists():
        return default

    try:
        import yaml  # type: ignore
    except Exception:  # pragma: no cover
        return default

    try:
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return default

    names = cfg.get("names")
    if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
        return default

    return {int(i): str(n) for i, n in enumerate(names)}

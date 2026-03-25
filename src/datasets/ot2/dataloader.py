from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

try:
    from tlg import PromptBank, TLGAnnotation
except ModuleNotFoundError:  # pragma: no cover
    from src.tlg import PromptBank, TLGAnnotation


_IMG_EXTS = {".jpg", ".jpeg", ".png"}
_DEFAULT_SPLITS: Tuple[str, ...] = ("train", "valid", "test")
_SEQ_RE = re.compile(r"^(?P<base>.*)_(?P<idx>\d+)$")


@dataclass(frozen=True, slots=True)
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
    ) -> None:
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.root}")

        self._splits = tuple(str(s) for s in splits)
        self._prompt_bank = prompt_bank or PromptBank.default()
        self._class_names = _load_yolo_class_names(self.root)

        if ids is not None:
            self._entries = self._resolve_ids(list(ids))
        else:
            self._entries = self._discover_entries()

    def _discover_entries(self) -> List[Tuple[str, Path]]:
        entries: List[Tuple[str, Path]] = []
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

        Supported forms:
        - "train/tipsWater_4" (split/stem)
        - "train/images/tipsWater_4.jpg" (relative path under split)

        Returns entries with existing images only.
        """

        entries: List[Tuple[str, Path]] = []
        for raw in ids:
            s = str(raw).strip()
            if not s:
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

        label_path = self.root / split / "labels" / f"{img_path.stem}.txt"
        boxes = _parse_yolo_label_file(label_path)

        sequence_id, frame_index = _derive_sequence_fields(img_path.stem)

        image_id = f"{split}/{img_path.stem}"
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
        annotations: List[TLGAnnotation] = []

        tip_count = sum(1 for b in boxes if int(b.cls) == 0)
        liquid_count = sum(1 for b in boxes if int(b.cls) == 1)

        def add(task: str, answer: object, metadata: Dict[str, object]) -> None:
            spec = self._prompt_bank.get(dataset, task)
            annotations.append(
                TLGAnnotation(
                    image_id=image_id,
                    dataset=dataset,
                    task_type=str(spec.get("task_type", task)),
                    question=str(spec["question"]),
                    choices=spec.get("choices"),
                    answer=answer,
                    metadata={"task": task, **metadata},
                )
            )

        add("tip_presence", "YES" if tip_count > 0 else "NO", {"tip_count": tip_count})
        add("liquid_presence", "YES" if liquid_count > 0 else "NO", {"liquid_count": liquid_count})
        add("tip_count", int(tip_count), {"tip_count": tip_count})
        add("liquid_count", int(liquid_count), {"liquid_count": liquid_count})

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

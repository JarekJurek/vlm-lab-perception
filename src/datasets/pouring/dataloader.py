from __future__ import annotations

import inspect
import json
import re
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union, cast

import numpy as np

from tlg import PromptBank, TLGAnnotation, make_tlg_annotation


_VIDEO_RE = re.compile(r"^video_\d+$")
_RGB_JPG_RE = re.compile(r"^rgb_(?P<idx>\d+)$")
_EMPTY_CUP_RE = re.compile(r"^empty_cup_(?P<idx>\d+)(?:_glass)?$")


_DATACLASS_KWARGS = {"slots": True} if "slots" in inspect.signature(dataclass).parameters else {}


@dataclass(frozen=True, **_DATACLASS_KWARGS)
class _Entry:
    split: str
    video_id: str
    frame_type: str  # "rgb" or "empty_cup"
    frame_index: int
    image_path: Path


class PouringDataset:
    """Minimal folder-based dataset loader for Pouring.

    Expected layout (as in `data/pouring/`):

        root/
          pouring_testset_videos/
            video_0/
              rgb_30.jpg
              rgb_30.json              # LabelMe-style polygons (liquid_mask)
              rgb_30_json/             # sidecar labels
                img.png
                label.png
                label_viz.png
                label_names.txt
              empty_cup_0_glass.png    # optional reference
          pouring_training_videos/
            opaque/
              video_0/
                rgb_0.jpg
                empty_cup_0.jpg
              video_0_cup_masks/
                rgb_0_cup_mask.png
                empty_cup_0_cup_mask.png
            transparent/
              ...

    Notes:
    - Test split supports liquid polygon annotations (LabelMe JSON) and rendered label images.
    - Training split supports cup-mask supervision; liquid masks are generally absent.
    """

    def __init__(
        self,
        root: Union[str, Path],
        ids: Optional[Iterable[str]] = None,
        splits: Sequence[str] = ("test",),
        include_empty_cup: bool = False,
        prompt_bank: Optional[PromptBank] = None,
        shuffle: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.root}")

        self._splits = tuple(str(s) for s in splits)
        self._include_empty_cup = bool(include_empty_cup)
        self._prompt_bank = prompt_bank or PromptBank.default()

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

    def _discover_entries(self) -> List[_Entry]:
        entries: List[_Entry] = []
        for split in self._splits:
            for video_dir in self._iter_video_dirs(split):
                video_id = video_dir.name
                # Always include rgb frames.
                entries.extend(self._discover_frames_for_video(split, video_id, video_dir, frame_type="rgb"))

                if self._include_empty_cup:
                    entries.extend(
                        self._discover_frames_for_video(split, video_id, video_dir, frame_type="empty_cup")
                    )

        return entries

    def _iter_video_dirs(self, split: str) -> List[Path]:
        base = self._split_base(split)
        if not base.is_dir():
            return []
        return sorted([p for p in base.iterdir() if p.is_dir() and _VIDEO_RE.fullmatch(p.name)])

    def _discover_frames_for_video(
        self, split: str, video_id: str, video_dir: Path, frame_type: str
    ) -> List[_Entry]:
        if frame_type == "rgb":
            paths = [p for p in video_dir.glob("rgb_*.jpg") if p.is_file()]
            key_fn = _rgb_frame_index
        elif frame_type == "empty_cup":
            # Training: empty_cup_*.jpg; Test: empty_cup_*_glass.png (optional)
            paths = [p for p in video_dir.glob("empty_cup_*.*") if p.is_file()]
            key_fn = _empty_cup_index
        else:
            raise ValueError(f"Unknown frame_type: {frame_type!r}")

        out: List[_Entry] = []
        for p in sorted(paths, key=key_fn):
            idx = key_fn(p)
            if idx < 0:
                continue
            out.append(_Entry(split=split, video_id=video_id, frame_type=frame_type, frame_index=idx, image_path=p))
        return out

    def _resolve_ids(self, ids: List[str]) -> List[_Entry]:
        """Resolve ids to entries.

        Supported forms:
        - "test/video_0/rgb_30"
        - "train/opaque/video_0/rgb_0"
        - "train/opaque/video_0/empty_cup_0"  (requires include_empty_cup=True)
        """

        entries: List[_Entry] = []
        for raw in ids:
            s = str(raw).strip()
            if not s:
                continue

            parts = s.split("/")
            if len(parts) == 3:
                # test/video_0/rgb_30
                split = parts[0]
                video_id = parts[1]
                frame_stem = parts[2]
            elif len(parts) == 4:
                # train/opaque/video_0/rgb_0
                split = f"{parts[0]}/{parts[1]}"
                video_id = parts[2]
                frame_stem = parts[3]
            else:
                raise ValueError(
                    "PouringDataset ids must be of form 'test/video_id/rgb_<idx>' or "
                    "'train/<material>/video_id/rgb_<idx>' (or empty_cup_<idx>). "
                    f"Got: {raw!r}"
                )

            video_dir = self._split_base(split) / video_id
            if not video_dir.is_dir():
                raise FileNotFoundError(f"Video directory not found for id={raw!r}: {video_dir}")

            if frame_stem.startswith("rgb_"):
                idx = _stem_index(frame_stem, _RGB_JPG_RE)
                if idx is None:
                    raise ValueError(f"Could not parse frame index from id={raw!r}")
                img_path = video_dir / f"rgb_{idx}.jpg"
                frame_type = "rgb"
            elif frame_stem.startswith("empty_cup_"):
                if not self._include_empty_cup:
                    raise ValueError(
                        "Requested empty_cup frame ids but include_empty_cup=False. "
                        "Construct the dataset with include_empty_cup=True."
                    )
                idx = _stem_index(frame_stem, _EMPTY_CUP_RE)
                if idx is None:
                    raise ValueError(f"Could not parse empty_cup index from id={raw!r}")
                # Training uses jpg; test uses png.
                img_path = video_dir / f"empty_cup_{idx}.jpg"
                if not img_path.exists():
                    candidates = sorted(video_dir.glob(f"empty_cup_{idx}*.*"))
                    if candidates:
                        img_path = candidates[0]
                frame_type = "empty_cup"
            else:
                raise ValueError(f"Unknown frame stem in id={raw!r}: {frame_stem!r}")

            if not img_path.exists():
                raise FileNotFoundError(f"Image not found for id={raw!r}: {img_path}")

            entries.append(
                _Entry(
                    split=split,
                    video_id=video_id,
                    frame_type=frame_type,
                    frame_index=int(idx),
                    image_path=img_path,
                )
            )

        return entries

    def _split_base(self, split: str) -> Path:
        if split == "test":
            return self.root / "pouring_testset_videos"

        if split.startswith("train/"):
            material = split.split("/", 1)[1]
            return self.root / "pouring_training_videos" / material

        raise ValueError(
            "Unknown split for PouringDataset. Use 'test' or 'train/<material>' "
            "(material is typically 'opaque' or 'transparent'). "
            f"Got: {split!r}"
        )

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]

    def __getitem__(self, index: int) -> Dict[str, object]:
        e = self._entries[index]
        split = e.split
        video_id = e.video_id
        frame_index = int(e.frame_index)
        frame_type = e.frame_type
        img_path = e.image_path

        # id strings mirror other loaders: split/video/frame
        id_parts = [split, video_id, f"{frame_type}_{frame_index}"]
        sample_id = "/".join(id_parts)

        sample: Dict[str, object] = {
            "id": sample_id,
            "split": split,
            "video_id": video_id,
            "frame_type": frame_type,
            "frame_index": frame_index,
            "path": str(img_path),
            "rgb_image": img_path,
        }

        if split == "test" and frame_type == "rgb":
            video_dir = self._split_base(split) / video_id
            json_path = video_dir / f"rgb_{frame_index}.json"
            sidecar_dir = video_dir / f"rgb_{frame_index}_json"

            if json_path.exists():
                labelme = _load_labelme_json(json_path)
                sample["labelme_json_path"] = json_path
                sample["labelme_shapes"] = labelme.get("shapes", [])
                if "imageWidth" in labelme:
                    sample["image_width"] = int(cast(int, labelme["imageWidth"]))
                if "imageHeight" in labelme:
                    sample["image_height"] = int(cast(int, labelme["imageHeight"]))

            if sidecar_dir.is_dir():
                img_png = sidecar_dir / "img.png"
                label_png = sidecar_dir / "label.png"
                label_viz_png = sidecar_dir / "label_viz.png"
                label_names_txt = sidecar_dir / "label_names.txt"

                if img_png.exists():
                    sample["sidecar_img"] = img_png
                if label_png.exists():
                    sample["label_image"] = label_png
                if label_viz_png.exists():
                    sample["label_viz_image"] = label_viz_png
                if label_names_txt.exists():
                    sample["label_names"] = _read_label_names(label_names_txt)

            sample["annotations"] = self._build_annotations_test(sample_id, sample)
            return sample

        # Training (and optional empty-cup entries): add cup mask when available.
        if split.startswith("train/"):
            base = self._split_base(split)
            mask_dir = base / f"{video_id}_cup_masks"
            if mask_dir.is_dir():
                mask_path = mask_dir / f"{frame_type}_{frame_index}_cup_mask.png"
                if mask_path.exists():
                    sample["cup_mask_image"] = mask_path

            # Expose available empty-cup references for this video (paths only).
            video_dir = base / video_id
            empty_refs = sorted(video_dir.glob("empty_cup_*.*"), key=_empty_cup_index)
            if empty_refs:
                sample["empty_cup_images"] = [p for p in empty_refs if _empty_cup_index(p) >= 0]

        sample["annotations"] = []
        return sample

    def _build_annotations_test(self, image_id: str, sample: Dict[str, object]) -> List[TLGAnnotation]:
        dataset = "pouring"
        annotations: List[TLGAnnotation] = []

        add = partial(
            make_tlg_annotation,
            self._prompt_bank,
            dataset=dataset,
            image_id=image_id,
        )

        shapes = sample.get("labelme_shapes")
        liquid_poly_count = 0
        if isinstance(shapes, list):
            for s in shapes:
                if not isinstance(s, dict):
                    continue
                if str(s.get("label", "")).strip() == "liquid_mask":
                    liquid_poly_count += 1

        annotations.append(
            add(
                task="liquid_presence",
                answer="YES" if liquid_poly_count > 0 else "NO",
                metadata={"liquid_polygon_count": int(liquid_poly_count)},
            )
        )

        return annotations


def _rgb_frame_index(path: Path) -> int:
    m = _RGB_JPG_RE.fullmatch(path.stem)
    return int(m.group("idx")) if m else -1


def _empty_cup_index(path: Path) -> int:
    m = _EMPTY_CUP_RE.fullmatch(path.stem)
    return int(m.group("idx")) if m else -1


def _stem_index(stem: str, rx: re.Pattern[str]) -> Optional[int]:
    m = rx.fullmatch(stem)
    if not m:
        return None
    try:
        return int(m.group("idx"))
    except Exception:
        return None


def _load_labelme_json(path: Path) -> Dict[str, Any]:
    """Load a LabelMe JSON file but drop the embedded `imageData` to keep memory small."""

    with path.open("r", encoding="utf-8") as f:
        data = cast(Dict[str, Any], json.load(f))
    if isinstance(data, dict):
        data.pop("imageData", None)
    return data


def _read_label_names(path: Path) -> List[str]:
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]

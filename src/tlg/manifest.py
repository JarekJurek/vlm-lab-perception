from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Union, cast

from tlg.annotations import TLGAnnotation


MANIFEST_VERSION = 1
_SAMPLE_FIELD_ALLOWLIST = {
    "id",
    "primary_task",
    "rgb_image",
    "comparison_rgb_image",
    "ordered_rgb_images",
    "ordered_image_labels",
    "bbox_visib",
    "bbox_obj",
    "target_occluded",
    "container_count_ambiguous",
    "scene_id",
    "frame_id",
    "depth_map_path",
}


class TLGManifestDataset:
    """Lightweight dataset backed by an exported TLG manifest JSON."""

    def __init__(
        self,
        manifest_path: Union[str, Path],
        *,
        split: str = "test",
        datasets: Optional[Iterable[str]] = None,
    ) -> None:
        self.path = Path(manifest_path)
        payload = load_manifest_payload(self.path)
        samples = cast(List[Dict[str, Any]], payload.get("samples", []))

        wanted_datasets = None if datasets is None else {str(d) for d in datasets}
        split = str(split)
        if split not in {"train", "test", "all"}:
            raise ValueError(f"manifest split must be 'train', 'test', or 'all'. Got: {split!r}")

        self.samples: List[Dict[str, Any]] = []
        for raw_sample in samples:
            dataset_name = str(raw_sample.get("dataset", ""))
            if wanted_datasets is not None and dataset_name not in wanted_datasets:
                continue
            if split != "all" and str(raw_sample.get("split", "")) != split:
                continue
            self.samples.append(deserialize_manifest_sample(raw_sample))

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for idx in range(len(self)):
            yield self[idx]

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.samples[index]


def load_manifest_payload(path: Union[str, Path]) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"TLG manifest must be a JSON object: {path}")
    samples = payload.get("samples")
    if not isinstance(samples, list):
        raise ValueError(f"TLG manifest missing list field 'samples': {path}")
    return payload


def serialize_manifest_sample(
    sample: Mapping[str, Any],
    *,
    dataset_name: str,
    split: str,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    source_split = sample.get("split")
    for key, value in sample.items():
        if key == "annotations":
            out[key] = [_annotation_to_dict(ann) for ann in _annotation_list(value)]
        elif key == "split":
            continue
        elif key not in _SAMPLE_FIELD_ALLOWLIST:
            continue
        else:
            converted = to_jsonable(value)
            if converted is not _SKIP:
                out[key] = converted

    out["dataset"] = str(dataset_name)
    out["split"] = str(split)
    if source_split is not None:
        out["source_split"] = to_jsonable(source_split)
    return out


def deserialize_manifest_sample(raw_sample: Mapping[str, Any]) -> Dict[str, Any]:
    sample = dict(raw_sample)
    annotations = sample.get("annotations", [])
    if not isinstance(annotations, list):
        raise ValueError(f"Manifest sample annotations must be a list: id={sample.get('id')!r}")
    sample["annotations"] = [annotation_from_dict(ann) for ann in annotations]
    return sample


def annotation_from_dict(data: Mapping[str, Any]) -> TLGAnnotation:
    if not isinstance(data, Mapping):
        raise ValueError(f"Manifest annotation must be an object. Got: {type(data).__name__}")
    choices = data.get("choices")
    if choices is not None:
        if not isinstance(choices, list) or not all(isinstance(c, str) for c in choices):
            raise ValueError(f"Manifest annotation choices must be list[str]: {data!r}")
    metadata = data.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise ValueError(f"Manifest annotation metadata must be an object: {data!r}")
    return TLGAnnotation(
        image_id=str(data.get("image_id", "")),
        dataset=str(data.get("dataset", "")),
        task_type=str(data.get("task_type", "")),
        question=str(data.get("question", "")),
        choices=cast(Optional[Sequence[str]], choices),
        answer=data.get("answer"),
        metadata=dict(metadata),
    )


def to_jsonable(value: Any) -> Any:
    if isinstance(value, TLGAnnotation):
        return value.to_dict()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Mapping):
        out: Dict[str, Any] = {}
        for key, item in value.items():
            converted = to_jsonable(item)
            if converted is not _SKIP:
                out[str(key)] = converted
        return out
    if isinstance(value, (list, tuple)):
        out_list = []
        for item in value:
            converted = to_jsonable(item)
            if converted is not _SKIP:
                out_list.append(converted)
        return out_list

    try:
        import numpy as np

        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return _SKIP
    except Exception:
        pass

    return _SKIP


def _annotation_list(value: Any) -> List[TLGAnnotation]:
    if not isinstance(value, list):
        return []
    return [ann for ann in value if isinstance(ann, TLGAnnotation)]


def _annotation_to_dict(ann: TLGAnnotation) -> Dict[str, Any]:
    return cast(Dict[str, Any], to_jsonable(ann.to_dict()))


class _SkipValue:
    pass


_SKIP = _SkipValue()

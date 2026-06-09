from __future__ import annotations

import inspect
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple, Union, cast

from tlg import PromptBank, TLGAnnotation


COQE_VOLUME_LABELS: Mapping[int, str] = {
    1: "0_50_ml",
    2: "50_100_ml",
    3: "100_200_ml",
    4: "200_300_ml",
    5: "300_500_ml",
    6: "500_750_ml",
    7: "750_1000_ml",
    8: "1000_2000_ml",
    9: "2000_3000_ml",
    10: "3000_inf_ml",
}

COQE_CONTENT_LABELS: Mapping[int, str] = {
    1: "0_percent",
    2: "33_percent",
    3: "50_percent",
    4: "66_percent",
    5: "100_percent",
    6: "opaque",
}

COQE_COMPARATIVE_LABELS: Mapping[int, str] = {
    1: "yes",
    2: "no",
    3: "cant_tell",
}

COQE_POUR_LABELS: Mapping[int, Union[float, str]] = {
    1: 0.0,
    2: 0.1,
    3: 0.2,
    4: 0.3,
    5: 0.4,
    6: 0.5,
    7: 0.6,
    8: 0.7,
    9: 0.8,
    10: 0.9,
    11: 1.0,
    12: "opaque",
}

COQE_TASKS = ("volume", "content", "comparative", "pouring")
COQE_TLG_TASKS = (
    "container_capacity_ml",
    "container_capacity_bin",
    "container_content_bin",
    "liquid_transfer_feasibility",
)
_POUR_TIMESTEP_KEYS = ("t0_content", "t1_content", "t2_content", "t3_content", "t4_content")
_DATACLASS_KWARGS = {"slots": True} if "slots" in inspect.signature(dataclass).parameters else {}


@dataclass(frozen=True, **_DATACLASS_KWARGS)
class COQERecord:
    """One COQE container annotation from train/val/test JSON."""

    split: str
    index: int
    image_id: str
    image_path: Path
    cad_model: int
    category: str
    comparative_class: int
    content_class: int
    volume_continuous: float
    volume_class: int
    bbox_xyxy: Tuple[int, int, int, int]

    @property
    def id(self) -> str:
        return f"{self.split}/{self.index:05d}/{self.image_id}"

    @property
    def bbox_xywh(self) -> Tuple[int, int, int, int]:
        x1, y1, x2, y2 = self.bbox_xyxy
        return x1, y1, x2 - x1, y2 - y1

    @property
    def volume_label(self) -> str:
        return _lookup_label(COQE_VOLUME_LABELS, self.volume_class, field="volume_discrete")

    @property
    def content_label(self) -> str:
        return _lookup_label(COQE_CONTENT_LABELS, self.content_class, field="content")

    @property
    def comparative_label(self) -> str:
        return _lookup_label(COQE_COMPARATIVE_LABELS, self.comparative_class, field="comparative_vol")


@dataclass(frozen=True, **_DATACLASS_KWARGS)
class COQETaskExample:
    """Task-level example consumed by the standalone COQE evaluator."""

    id: str
    task: str
    split: str
    image_id: str
    image_path: Path
    answer: Any
    source: COQERecord
    target: Optional[COQERecord] = None
    angle: Optional[int] = None
    raw: Mapping[str, Any] = None  # type: ignore[assignment]

    @property
    def bbox_metadata(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "source_bbox_xyxy": list(self.source.bbox_xyxy),
            "source_bbox_xywh": list(self.source.bbox_xywh),
        }
        if self.target is not None:
            out["target_bbox_xyxy"] = list(self.target.bbox_xyxy)
            out["target_bbox_xywh"] = list(self.target.bbox_xywh)
        return out


class COQEExpDataset:
    """Reusable loader for COQE annotations and task examples.

    The base split JSON contains one record per annotated container. Adjacent
    records are paired containers in the same scene; comparative labels are
    directional from the current record's container to the adjacent paired
    container.
    """

    def __init__(
        self,
        root: Union[str, Path],
        *,
        split: str = "test",
        shuffle: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"COQE root not found: {self.root}")

        self.split = str(split)
        if self.split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported COQE split: {split!r}")

        self.images_dir = self.root / "images"
        if not self.images_dir.is_dir():
            raise FileNotFoundError(f"COQE images directory not found: {self.images_dir}")

        self.annotations_path = self.root / f"{self.split}.json"
        self.pour_annotations_path = self.root / f"{self.split}_pour_prediction.json"
        self.records = self._load_records(self.annotations_path)
        self.records_by_image_id = {r.image_id: r for r in self.records}
        self.pairs_by_image_id = self._build_directional_pairs(self.records)
        self.pour_records = self._load_pour_records(self.pour_annotations_path)

        if shuffle:
            rng = random.Random(seed)
            rng.shuffle(self.records)
            rng.shuffle(self.pour_records)

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self) -> Iterator[COQERecord]:
        return iter(self.records)

    def __getitem__(self, index: int) -> COQERecord:
        return self.records[index]

    def iter_examples(self, tasks: Sequence[str] = COQE_TASKS) -> Iterator[COQETaskExample]:
        task_set = _normalize_tasks(tasks)
        if "volume" in task_set:
            for record in self.records:
                yield COQETaskExample(
                    id=f"{record.id}/volume",
                    task="volume",
                    split=self.split,
                    image_id=record.image_id,
                    image_path=record.image_path,
                    answer=record.volume_label,
                    source=record,
                    raw={},
                )
        if "content" in task_set:
            for record in self.records:
                yield COQETaskExample(
                    id=f"{record.id}/content",
                    task="content",
                    split=self.split,
                    image_id=record.image_id,
                    image_path=record.image_path,
                    answer=record.content_label,
                    source=record,
                    raw={},
                )
        if "comparative" in task_set:
            for record in self.records:
                target = self.pairs_by_image_id.get(record.image_id)
                if target is None:
                    continue
                yield COQETaskExample(
                    id=f"{record.id}/comparative",
                    task="comparative",
                    split=self.split,
                    image_id=record.image_id,
                    image_path=record.image_path,
                    answer=record.comparative_label,
                    source=record,
                    target=target,
                    raw={},
                )
        if "pouring" in task_set:
            for i, pour in enumerate(self.pour_records):
                image_id = str(pour["image_id"])
                record = self.records_by_image_id.get(image_id)
                if record is None:
                    continue
                angle = int(pour["angle"])
                sequence = pour_sequence_from_raw(pour)
                yield COQETaskExample(
                    id=f"{self.split}/pouring/{i:05d}/{image_id}/angle_{angle}",
                    task="pouring",
                    split=self.split,
                    image_id=image_id,
                    image_path=record.image_path,
                    answer=sequence,
                    source=record,
                    angle=angle,
                    raw=pour,
                )

    def task_counts(self) -> Dict[str, int]:
        return {
            "volume": len(self.records),
            "content": len(self.records),
            "comparative": sum(1 for r in self.records if r.image_id in self.pairs_by_image_id),
            "pouring": sum(1 for p in self.pour_records if str(p["image_id"]) in self.records_by_image_id),
        }

    def pouring_records_per_image_distribution(self) -> Dict[int, int]:
        counts: Dict[str, int] = {}
        for row in self.pour_records:
            image_id = str(row["image_id"])
            counts[image_id] = counts.get(image_id, 0) + 1
        out: Dict[int, int] = {}
        for count in counts.values():
            out[count] = out.get(count, 0) + 1
        return dict(sorted(out.items()))

    def _load_records(self, path: Path) -> List[COQERecord]:
        rows = _load_json_list(path)
        records: List[COQERecord] = []
        for index, row in enumerate(rows):
            image_id = str(row["image_id"])
            image_path = self.images_dir / image_id
            x1 = int(row["x1"])
            y1 = int(row["y1"])
            x2 = int(row["x2"])
            y2 = int(row["y2"])
            if x2 <= x1 or y2 <= y1:
                raise ValueError(f"Invalid bbox for {image_id}: {(x1, y1, x2, y2)}")
            records.append(
                COQERecord(
                    split=self.split,
                    index=index,
                    image_id=image_id,
                    image_path=image_path,
                    cad_model=int(row["cad_model"]),
                    category=str(row["category"]),
                    comparative_class=int(row["comparative_vol"]),
                    content_class=int(row["content"]),
                    volume_continuous=float(row["volume_continuous"]),
                    volume_class=int(row["volume_discrete"]),
                    bbox_xyxy=(x1, y1, x2, y2),
                )
            )
        return records

    def _load_pour_records(self, path: Path) -> List[Dict[str, Any]]:
        return [dict(row) for row in _load_json_list(path)]

    def _build_directional_pairs(self, records: Sequence[COQERecord]) -> Dict[str, COQERecord]:
        if len(records) % 2 != 0:
            raise ValueError(f"Expected an even number of COQE records, got {len(records)}")

        out: Dict[str, COQERecord] = {}
        for i in range(0, len(records), 2):
            first = records[i]
            second = records[i + 1]
            if first.image_path != second.image_path:
                # In COQE, paired records point at identical scene image files whose
                # names are consecutive IDs. The files are distinct but pixel-identical.
                pass
            out[first.image_id] = second
            out[second.image_id] = first
        return out


class COQEDataset:
    """Benchmark-facing COQE dataset with native TLG annotations."""

    def __init__(
        self,
        root: Union[str, Path],
        *,
        split: str = "test",
        prompt_bank: Optional[PromptBank] = None,
        tasks: Sequence[str] = COQE_TLG_TASKS,
        shuffle: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        self._dataset = COQEExpDataset(root, split=split, shuffle=False)
        self.root = self._dataset.root
        self.split = self._dataset.split
        self._prompt_bank = prompt_bank or PromptBank.default()
        task_set = _normalize_tlg_tasks(tasks)

        entries: List[Tuple[str, COQERecord]] = []
        for task in task_set:
            if task == "liquid_transfer_feasibility":
                entries.extend(
                    (task, record)
                    for record in self._dataset.records
                    if record.image_id in self._dataset.pairs_by_image_id
                )
            else:
                entries.extend((task, record) for record in self._dataset.records)

        if shuffle:
            rng = random.Random(seed)
            rng.shuffle(entries)
        self._entries = entries

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[Dict[str, object]]:
        for idx in range(len(self)):
            yield self[idx]

    def __getitem__(self, index: int) -> Dict[str, object]:
        task, record = self._entries[index]
        sample_id = f"{record.id}/{task}"
        ann = self._build_annotation(task, record, sample_id)
        return {
            "id": sample_id,
            "image_id": record.image_id,
            "rgb_image": record.image_path,
            "split": self.split,
            "primary_task": task,
            "annotations": [ann],
        }

    def task_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for task, _record in self._entries:
            counts[task] = counts.get(task, 0) + 1
        return counts

    def _build_annotation(self, task: str, record: COQERecord, sample_id: str) -> TLGAnnotation:
        spec = self._prompt_bank.get("coqe", task)
        answer: object
        metadata: Dict[str, object] = {
            "task": task,
            "source_image_id": record.image_id,
            "source_bbox_xyxy": list(record.bbox_xyxy),
            "source_bbox_xywh": list(record.bbox_xywh),
            "category": record.category,
            "cad_model": record.cad_model,
        }

        if task == "container_capacity_ml":
            answer = float(record.volume_continuous)
            metadata["bbox_xyxy"] = list(record.bbox_xyxy)
            metadata["volume_discrete"] = record.volume_class
        elif task == "container_capacity_bin":
            answer = record.volume_label
            metadata["bbox_xyxy"] = list(record.bbox_xyxy)
            metadata["volume_continuous"] = float(record.volume_continuous)
            metadata["volume_discrete"] = record.volume_class
        elif task == "container_content_bin":
            answer = record.content_label
            metadata["bbox_xyxy"] = list(record.bbox_xyxy)
            metadata["content_class"] = record.content_class
        elif task == "liquid_transfer_feasibility":
            target = self._dataset.pairs_by_image_id.get(record.image_id)
            if target is None:
                raise ValueError(f"Missing COQE comparative target for {record.image_id}")
            answer = record.comparative_label
            metadata.update(
                {
                    "target_image_id": target.image_id,
                    "target_bbox_xyxy": list(target.bbox_xyxy),
                    "target_bbox_xywh": list(target.bbox_xywh),
                    "comparative_class": record.comparative_class,
                    "bbox_boxes": [
                        {"bbox_xyxy": list(record.bbox_xyxy), "label": "SOURCE", "color": "red"},
                        {"bbox_xyxy": list(target.bbox_xyxy), "label": "TARGET", "color": "green"},
                    ],
                }
            )
        else:
            raise ValueError(f"Unknown COQE TLG task: {task!r}")

        return TLGAnnotation(
            image_id=sample_id,
            dataset="coqe",
            task_type=str(spec.get("task_type", task)),
            question=str(spec["question"]),
            choices=spec.get("choices"),
            answer=answer,
            metadata=metadata,
        )


def pour_sequence_from_raw(row: Mapping[str, Any]) -> List[Union[float, str]]:
    sequence: List[Union[float, str]] = []
    for key in _POUR_TIMESTEP_KEYS:
        klass = int(row[key])
        sequence.append(_lookup_label(COQE_POUR_LABELS, klass, field=key))
    return sequence


def _normalize_tasks(tasks: Sequence[str]) -> Tuple[str, ...]:
    out = tuple(str(t).strip() for t in tasks if str(t).strip())
    unknown = sorted(set(out) - set(COQE_TASKS))
    if unknown:
        raise ValueError(f"Unknown COQE task(s): {', '.join(unknown)}")
    return out


def _normalize_tlg_tasks(tasks: Sequence[str]) -> Tuple[str, ...]:
    out = tuple(str(t).strip() for t in tasks if str(t).strip())
    unknown = sorted(set(out) - set(COQE_TLG_TASKS))
    if unknown:
        raise ValueError(f"Unknown COQE TLG task(s): {', '.join(unknown)}")
    return out


def _load_json_list(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"COQE annotation file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list JSON in {path}, got {type(data).__name__}")
    return cast(List[Dict[str, Any]], data)


def _lookup_label(mapping: Mapping[int, Any], key: int, *, field: str) -> Any:
    try:
        return mapping[int(key)]
    except KeyError as exc:
        raise ValueError(f"Unknown COQE {field} class: {key!r}") from exc

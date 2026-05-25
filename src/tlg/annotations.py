from __future__ import annotations

import inspect
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

from .prompts import PromptBank

JsonPrimitive = Union[str, int, float, bool, None]
JsonValue = Union[JsonPrimitive, List["JsonValue"], Dict[str, "JsonValue"]]


_CONTAINER_TYPE_SIMPLE_MAP = {
    "Tube_50mL": "Tube",
    "Tube_15mL": "Tube",
    "Pyrex_100mL": "Pyrex",
    "StorageBottle_250mL": "StorageBottle",
    "StorageBottle_500mL": "StorageBottle",
    "Duran_500mL": "Duran",
    "Duran_250mL": "Duran",
    "Duran_100mL": "Duran",
    "Cell_Flask_160mL": "Flask",
    "Cell_Flask_400mL": "Flask",
    "Gibco_125mL": "Gibco",
    "Gibco_500mL": "Gibco",
    "T25 Flask": "Flask",
    "T75 Flask": "Flask",
    "T175 Flask": "Flask",
    "G-rex": "G-rex",
    "Tip": "Tip",
}


_DATACLASS_KWARGS = {"slots": True} if "slots" in inspect.signature(dataclass).parameters else {}


@dataclass(frozen=True, **_DATACLASS_KWARGS)
class TLGAnnotation:
    """Standardized VQA-style annotation.

    Matches the schema described in README:
        {
            "image_id": "...",
            "dataset": "...",
            "task_type": "...",
            "question": "...",
            "choices": [...],  # optional
            "answer": "...",   # categorical or numeric
            "metadata": {...}
        }

    Notes:
    - `answer` is intentionally typed as `Any` to support numeric regression.
    - `metadata` should remain JSON-serializable.
    """

    image_id: str
    dataset: str
    task_type: str
    question: str
    choices: Optional[Sequence[str]] = None
    answer: Any = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Keep schema stable: omit optional keys if absent
        if self.choices is None:
            d.pop("choices", None)
        return d


def make_tlg_annotation(
    prompt_bank: PromptBank,
    *,
    dataset: str,
    task: str,
    image_id: str,
    answer: Any,
    metadata: Optional[Mapping[str, Any]] = None,
) -> TLGAnnotation:
    """Create a `TLGAnnotation` using the PromptBank as a source of truth.

    This keeps dataset loaders concise and consistent.
    """

    spec = prompt_bank.get(str(dataset), str(task))
    md: Dict[str, Any] = dict(metadata or {})
    md["task"] = str(task)
    question = str(spec["question"])
    prefix = prompt_bank.task_prompt_prefix(str(dataset), str(task))
    if prefix and "target_region_bbox_prompt" not in prompt_bank.task_prompt_prefix_ids(str(dataset), str(task)):
        question = f"{prefix.rstrip()}\n\n{question}"

    return TLGAnnotation(
        image_id=str(image_id),
        dataset=str(dataset),
        task_type=str(spec.get("task_type", task)),
        question=question,
        choices=spec.get("choices"),
        answer=answer,
        metadata=md,
    )


def simplify_container_type(container_type: object) -> str:
    """Return the no-volume container type label for the easier type task."""

    label = str(container_type).strip()
    return _CONTAINER_TYPE_SIMPLE_MAP.get(label, label)


def discretize_fill_level_ordinal(fill_percentage: float) -> str:
    """Discretize a 0-100 fill percentage into EMPTY/LOW/MID/HIGH/FULL."""
    p = max(0.0, min(100.0, float(fill_percentage)))
    if p <= 1.0:
        return "EMPTY"
    if p < 25.0:
        return "LOW"
    if p < 60.0:
        return "MID"
    if p < 90.0:
        return "HIGH"
    return "FULL"

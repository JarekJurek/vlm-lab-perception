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

CONTAINER_CAPACITY_BIN_LABELS = (
    "0_50_ml",
    "50_100_ml",
    "100_200_ml",
    "200_300_ml",
    "300_500_ml",
    "500_750_ml",
    "750_1000_ml",
    "1000_2000_ml",
    "2000_3000_ml",
    "3000_inf_ml",
)

_CONTAINER_CAPACITY_BIN_EDGES = (
    0.0,
    50.0,
    100.0,
    200.0,
    300.0,
    500.0,
    750.0,
    1000.0,
    2000.0,
    3000.0,
)


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


def discretize_fill_level_ordinal(fill_percentage: float, *, empty_threshold: float = 1.0) -> str:
    """Discretize a 0-100 fill percentage into ordinal percent bins."""
    p = max(0.0, min(100.0, float(fill_percentage)))
    threshold = max(0.0, min(100.0, float(empty_threshold)))
    if p <= threshold:
        return "0_percent"
    if p < 41.5:
        return "33_percent"
    if p < 58.0:
        return "50_percent"
    if p < 83.0:
        return "66_percent"
    return "100_percent"


def discretize_container_capacity_ml(capacity_ml: float) -> str:
    """Discretize a vessel capacity in milliliters into configured capacity bins."""
    capacity = max(0.0, float(capacity_ml))
    for index, lower_edge in enumerate(_CONTAINER_CAPACITY_BIN_EDGES[1:], start=1):
        if capacity < lower_edge:
            return CONTAINER_CAPACITY_BIN_LABELS[index - 1]
    return CONTAINER_CAPACITY_BIN_LABELS[-1]

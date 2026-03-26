from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

from .prompts import PromptBank

JsonPrimitive = Union[str, int, float, bool, None]
JsonValue = Union[JsonPrimitive, List["JsonValue"], Dict[str, "JsonValue"]]


@dataclass(frozen=True, slots=True)
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

    return TLGAnnotation(
        image_id=str(image_id),
        dataset=str(dataset),
        task_type=str(spec.get("task_type", task)),
        question=str(spec["question"]),
        choices=spec.get("choices"),
        answer=answer,
        metadata=md,
    )


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

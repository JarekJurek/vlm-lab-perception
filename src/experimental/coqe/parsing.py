from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple, Union


VOLUME_CHOICES = {
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
}
CONTENT_CHOICES = {"0_percent", "33_percent", "50_percent", "66_percent", "100_percent", "opaque"}
COMPARATIVE_CHOICES = {"yes", "no", "cant_tell"}
POURING_VALUES: Tuple[Union[float, str], ...] = (
    0.0,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    1.0,
    "opaque",
)


def parse_prediction(task: str, response: str) -> Tuple[Optional[Any], bool, Optional[str]]:
    try:
        payload = json.loads(_extract_json_text(response))
    except json.JSONDecodeError as exc:
        return None, False, f"invalid_json: {exc.msg}"
    if not isinstance(payload, dict):
        return None, False, "json_root_not_object"

    if task == "volume":
        return _parse_scalar(payload, key="volume_bin", choices=VOLUME_CHOICES)
    if task == "content":
        return _parse_scalar(payload, key="fill_level", choices=CONTENT_CHOICES)
    if task == "comparative":
        return _parse_scalar(payload, key="can_pour", choices=COMPARATIVE_CHOICES)
    if task == "pouring":
        return _parse_pouring(payload)
    return None, False, f"unknown_task: {task}"


def _parse_scalar(payload: Dict[str, Any], *, key: str, choices: set[str]) -> Tuple[Optional[str], bool, Optional[str]]:
    if set(payload.keys()) != {key}:
        return None, False, f"expected_only_key: {key}"
    value = payload.get(key)
    if not isinstance(value, str):
        return None, False, f"{key}_not_string"
    if value not in choices:
        return None, False, f"{key}_unknown_label"
    return value, True, None


def _parse_pouring(payload: Dict[str, Any]) -> Tuple[Optional[List[Union[float, str]]], bool, Optional[str]]:
    key = "remaining_sequence"
    if set(payload.keys()) != {key}:
        return None, False, f"expected_only_key: {key}"
    value = payload.get(key)
    if not isinstance(value, list):
        return None, False, "remaining_sequence_not_list"
    if len(value) != 5:
        return None, False, "remaining_sequence_not_length_5"

    parsed: List[Union[float, str]] = []
    for item in value:
        canonical = _canonical_pour_value(item)
        if canonical is None:
            return None, False, "remaining_sequence_unknown_value"
        parsed.append(canonical)
    return parsed, True, None


def _canonical_pour_value(value: Any) -> Optional[Union[float, str]]:
    if value == "opaque":
        return "opaque"
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        value = float(value)
    if isinstance(value, float):
        rounded = round(value, 1)
        if abs(value - rounded) > 1e-8:
            return None
        for allowed in POURING_VALUES:
            if isinstance(allowed, float) and abs(allowed - rounded) < 1e-8:
                return allowed
    return None


def _extract_json_text(response: object) -> str:
    text = str(response).strip()
    if text.startswith("assistant\n"):
        text = text.split("\n", 1)[1].strip()
    return text

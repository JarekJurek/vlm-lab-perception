from __future__ import annotations

import inspect
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, TypedDict, Union, cast

try:
    from typing import NotRequired
except ImportError:  # pragma: no cover (py<3.11)
    from typing_extensions import NotRequired  # type: ignore


class PromptSpec(TypedDict):
    question: str
    task_type: NotRequired[str]
    choices: NotRequired[List[str]]


class DatasetPromptAssignment(TypedDict):
    include: List[str]
    overrides: NotRequired[Dict[str, str]]


PromptDefsDict = Mapping[str, PromptSpec]
DatasetAssignmentsDict = Mapping[str, DatasetPromptAssignment]


_DATACLASS_KWARGS = {"slots": True} if "slots" in inspect.signature(dataclass).parameters else {}


@dataclass(frozen=True, **_DATACLASS_KWARGS)
class PromptBank:
    """YAML-backed prompt bank.

    Intended usage:
        bank = PromptBank.default()
        spec = bank.get("labliquidvision", "fill_percentage")

    The underlying file is meant to be edited frequently.
    """

    prompt_defs: PromptDefsDict
    dataset_assignments: DatasetAssignmentsDict

    def get(self, dataset: str, task: str) -> PromptSpec:
        dataset = str(dataset)
        task = str(task)

        try:
            assignment = self.dataset_assignments[dataset]
        except KeyError as e:
            raise KeyError(f"Unknown dataset in prompt assignments: {dataset!r}") from e

        include = assignment.get("include", [])
        overrides = cast(Dict[str, str], assignment.get("overrides", {}))

        if task not in include and task not in overrides:
            raise KeyError(
                f"Task not enabled for dataset={dataset!r}: {task!r}. "
                f"Enabled tasks: {sorted(set(include) | set(overrides.keys()))}"
            )

        prompt_id = str(overrides.get(task, task))
        try:
            return cast(PromptSpec, self.prompt_defs[prompt_id])
        except KeyError as e:
            raise KeyError(
                f"Unknown prompt_id referenced by dataset assignments: {prompt_id!r} "
                f"(dataset={dataset!r}, task={task!r})"
            ) from e

    def supports(self, dataset: str, task: str) -> bool:
        dataset = str(dataset)
        task = str(task)
        assignment = self.dataset_assignments.get(dataset)
        if not assignment:
            return False
        include = assignment.get("include", [])
        overrides = cast(Dict[str, str], assignment.get("overrides", {}))
        return task in include or task in overrides

    def datasets_for_prompt(self, prompt_id: str) -> List[str]:
        """List datasets that reference the given prompt id.

        Note: when no override is present, tasks implicitly map to prompt_id == task.
        """

        prompt_id = str(prompt_id)
        out: List[str] = []
        for ds, assignment in self.dataset_assignments.items():
            include = assignment.get("include", [])
            overrides = cast(Dict[str, str], assignment.get("overrides", {}))
            if prompt_id in overrides.values() or prompt_id in include:
                out.append(str(ds))
        return sorted(set(out))

    @staticmethod
    def default_path() -> Path:
        # Prompt definitions live here (prompt_id -> spec).
        return Path(__file__).with_name("prompt_bank.yaml")

    @staticmethod
    def default_assignments_path() -> Path:
        # Dataset mapping lives here (dataset -> include list and optional overrides).
        return Path(__file__).with_name("dataset_prompts.yaml")

    @classmethod
    def default(cls) -> "PromptBank":
        return cls.from_files(cls.default_path(), cls.default_assignments_path())

    @classmethod
    def from_files(
        cls,
        prompt_defs_path: Union[str, Path],
        dataset_assignments_path: Union[str, Path],
    ) -> "PromptBank":
        prompt_defs_path = Path(prompt_defs_path)
        dataset_assignments_path = Path(dataset_assignments_path)
        prompt_defs = _load_prompt_defs_yaml(prompt_defs_path)
        assignments = _load_dataset_assignments_yaml(dataset_assignments_path)
        return cls(prompt_defs=prompt_defs, dataset_assignments=assignments)


@lru_cache(maxsize=8)
def _require_yaml() -> Any:
    try:
        import yaml  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Missing dependency for YAML prompt bank. "
            "Install `PyYAML` (e.g., `pip install pyyaml`)."
        ) from e
    return yaml


@lru_cache(maxsize=8)
def _load_prompt_defs_yaml(path: Path) -> Dict[str, PromptSpec]:
    """Load and lightly validate prompt definitions (prompt_id -> spec)."""

    yaml = _require_yaml()

    if not path.exists():
        raise FileNotFoundError(f"Prompt definitions YAML not found: {path}")

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Prompt definitions must be a mapping at top-level: {path}")

    prompt_defs: Dict[str, PromptSpec] = {}
    for prompt_id, spec in raw.items():
        if not isinstance(prompt_id, str) or not isinstance(spec, dict):
            continue

        question = spec.get("question")
        task_type = spec.get("task_type")
        choices = spec.get("choices")

        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Missing/empty question for prompt_id={prompt_id!r} in {path}")
        if task_type is not None and not isinstance(task_type, str):
            raise ValueError(f"task_type must be a string for prompt_id={prompt_id!r} in {path}")
        if choices is not None:
            if not isinstance(choices, list) or not all(isinstance(c, str) for c in choices):
                raise ValueError(f"choices must be a list[str] for prompt_id={prompt_id!r} in {path}")

        out: PromptSpec = {"question": question}
        if isinstance(task_type, str):
            out["task_type"] = task_type
        if isinstance(choices, list):
            out["choices"] = cast(List[str], choices)

        prompt_defs[prompt_id] = out

    return prompt_defs


@lru_cache(maxsize=8)
def _load_dataset_assignments_yaml(path: Path) -> Dict[str, DatasetPromptAssignment]:
    """Load dataset prompt assignments.

    Supported YAML shapes per dataset:
    - list[str]: interpreted as {include: [...]}
    - mapping: {include: [...], overrides: {task: prompt_id}}
    """

    yaml = _require_yaml()

    if not path.exists():
        raise FileNotFoundError(f"Dataset prompt assignments YAML not found: {path}")

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Dataset prompt assignments must be a mapping at top-level: {path}")

    out: Dict[str, DatasetPromptAssignment] = {}
    for dataset, cfg in raw.items():
        if not isinstance(dataset, str):
            continue

        include: List[str] = []
        overrides: Dict[str, str] = {}

        if isinstance(cfg, list):
            if not all(isinstance(x, str) for x in cfg):
                raise ValueError(f"Dataset {dataset!r} assignments list must be list[str] in {path}")
            include = [str(x) for x in cfg]
        elif isinstance(cfg, dict):
            inc = cfg.get("include", [])
            if not isinstance(inc, list) or not all(isinstance(x, str) for x in inc):
                raise ValueError(f"Dataset {dataset!r} include must be list[str] in {path}")
            include = [str(x) for x in inc]

            ov = cfg.get("overrides", {})
            if ov is None:
                ov = {}
            if not isinstance(ov, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in ov.items()):
                raise ValueError(f"Dataset {dataset!r} overrides must be dict[str,str] in {path}")
            overrides = {str(k): str(v) for k, v in ov.items()}
        else:
            raise ValueError(
                f"Dataset {dataset!r} assignments must be a list[str] or mapping in {path}; got {type(cfg).__name__}"
            )

        # Normalize/validate.
        include_norm = sorted({s.strip() for s in include if str(s).strip()})
        overrides_norm = {k.strip(): v.strip() for k, v in overrides.items() if k.strip() and v.strip()}
        out[dataset] = {"include": include_norm}
        if overrides_norm:
            out[dataset]["overrides"] = overrides_norm

    return out



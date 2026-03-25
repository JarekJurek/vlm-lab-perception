from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Mapping, TypedDict, Union, cast

try:
    from typing import NotRequired
except ImportError:  # pragma: no cover (py<3.11)
    from typing_extensions import NotRequired  # type: ignore


class PromptSpec(TypedDict):
    question: str
    task_type: NotRequired[str]
    choices: NotRequired[List[str]]


PromptBankDict = Mapping[str, Mapping[str, PromptSpec]]


@dataclass(frozen=True, slots=True)
class PromptBank:
    """YAML-backed prompt bank.

    Intended usage:
        bank = PromptBank.default()
        spec = bank.get("labliquidvision", "fill_percentage")

    The underlying file is meant to be edited frequently.
    """

    prompts: PromptBankDict

    def get(self, dataset: str, task: str) -> PromptSpec:
        try:
            ds = self.prompts[dataset]
        except KeyError as e:
            raise KeyError(f"Unknown dataset in prompt bank: {dataset!r}") from e

        try:
            return cast(PromptSpec, ds[task])
        except KeyError as e:
            raise KeyError(
                f"Unknown task in prompt bank for dataset={dataset!r}: {task!r}"
            ) from e

    @staticmethod
    def default_path() -> Path:
        return Path(__file__).with_name("prompt_bank.yaml")

    @classmethod
    def default(cls) -> "PromptBank":
        return cls.from_yaml(cls.default_path())

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "PromptBank":
        path = Path(path)
        return cls(prompts=_load_prompt_bank_yaml(path))


@lru_cache(maxsize=8)
def _load_prompt_bank_yaml(path: Path) -> Dict[str, Dict[str, PromptSpec]]:
    """Load and lightly validate the prompt bank.

    Cached by path to keep dataset iteration cheap.
    """

    try:
        import yaml  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Missing dependency for YAML prompt bank. "
            "Install `PyYAML` (e.g., `pip install pyyaml`)."
        ) from e

    if not path.exists():
        raise FileNotFoundError(f"Prompt bank YAML not found: {path}")

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Prompt bank must be a mapping at top-level: {path}")

    prompts: Dict[str, Dict[str, PromptSpec]] = {}
    for dataset, tasks in raw.items():
        if not isinstance(dataset, str) or not isinstance(tasks, dict):
            continue

        prompts[dataset] = {}
        for task, spec in tasks.items():
            if not isinstance(task, str) or not isinstance(spec, dict):
                continue

            question = spec.get("question")
            task_type = spec.get("task_type")
            choices = spec.get("choices")

            if not isinstance(question, str) or not question.strip():
                raise ValueError(f"Missing/empty question for {dataset}.{task} in {path}")
            if task_type is not None and not isinstance(task_type, str):
                raise ValueError(f"task_type must be a string for {dataset}.{task} in {path}")
            if choices is not None:
                if not isinstance(choices, list) or not all(isinstance(c, str) for c in choices):
                    raise ValueError(f"choices must be a list[str] for {dataset}.{task} in {path}")

            out: PromptSpec = {"question": question}
            if isinstance(task_type, str):
                out["task_type"] = task_type
            if isinstance(choices, list):
                out["choices"] = cast(List[str], choices)

            prompts[dataset][task] = out

    return prompts

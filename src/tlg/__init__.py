"""TLG (Transparent Liquid Gauge) shared primitives.

This package provides:
- A shared annotation type (VQA-style tasks)
- A YAML-backed prompt bank used by dataset loaders

The repo adds `src/` to `PYTHONPATH` in examples, so this package can be
imported as `import tlg`.
"""

from .annotations import TLGAnnotation, make_tlg_annotation, simplify_container_type
from .prompts import PromptBank

__all__ = ["TLGAnnotation", "make_tlg_annotation", "simplify_container_type", "PromptBank"]

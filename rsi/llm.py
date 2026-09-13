"""Tiny backend seam for optional language-model integrations.

No provider SDK is imported here.  Students can wrap a local model, an API, or a
record/replay fixture by implementing ``complete(prompt) -> str``.
"""

from __future__ import annotations

import importlib
from typing import Protocol


class TextBackend(Protocol):
    def complete(self, prompt: str) -> str:
        ...


class UnconfiguredBackend:
    def complete(self, prompt: str) -> str:
        raise RuntimeError(
            "LLM backend is not configured. Pass --llm-backend module:object; "
            "the object must expose complete(prompt) -> str."
        )


def load_backend(import_path: str | None) -> TextBackend:
    """Load ``module:object``; instantiate it when it is a class or factory."""

    if not import_path:
        return UnconfiguredBackend()
    if ":" not in import_path:
        raise ValueError("backend path must have the form module:object")
    module_name, object_name = import_path.split(":", 1)
    value = getattr(importlib.import_module(module_name), object_name)
    backend = value() if isinstance(value, type) else value
    if not callable(getattr(backend, "complete", None)):
        raise TypeError(f"{import_path} does not expose complete(prompt)")
    return backend

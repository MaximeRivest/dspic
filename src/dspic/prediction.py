"""Prediction result container."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any


class Prediction(dict[str, Any]):
    """Dictionary result with attribute access, matching DSPy ergonomics."""

    def __init__(self, values: Mapping[str, Any] | None = None, **kwargs: Any) -> None:
        super().__init__(values or {}, **kwargs)

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        try:
            del self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def copy(self, **updates: Any) -> Prediction:
        return Prediction({**self, **updates})

    def __iter__(self) -> Iterator[str]:
        return super().__iter__()

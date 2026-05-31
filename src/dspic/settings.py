"""Small global settings for DSPIC programs."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace

from dspic.base_vm import BaseVM


@dataclass(frozen=True)
class DSPICSettings:
    vm: BaseVM | list[BaseVM] | None = None
    adapter: object | None = None


_DEFAULT_SETTINGS = DSPICSettings()
_SETTINGS: ContextVar[DSPICSettings | None] = ContextVar(
    "dspic_settings",
    default=None,
)


def settings() -> DSPICSettings:
    return _SETTINGS.get() or _DEFAULT_SETTINGS


def configure(
    *,
    vm: BaseVM | list[BaseVM] | None = None,
    adapter: object | None = None,
) -> None:
    """Set process-local defaults for `Predict` calls."""

    current = settings()
    _SETTINGS.set(
        replace(
            current,
            vm=current.vm if vm is None else vm,
            adapter=current.adapter if adapter is None else adapter,
        )
    )


@contextmanager
def context(
    *,
    vm: BaseVM | list[BaseVM] | None = None,
    adapter: object | None = None,
) -> Iterator[None]:
    """Temporarily override DSPIC defaults."""

    current = settings()
    token = _SETTINGS.set(
        replace(
            current,
            vm=current.vm if vm is None else vm,
            adapter=current.adapter if adapter is None else adapter,
        )
    )
    try:
        yield
    finally:
        _SETTINGS.reset(token)

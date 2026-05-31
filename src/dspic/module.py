"""Base class for DSPIC trainable programs."""

from __future__ import annotations

import copy
from typing import Any


class Module:
    """Minimal DSPy-style module base.

    DSPIC optimizers can mutate/copy modules and especially their `demos`, which
    act as trainable visual prompting examples for specialized VMs.
    """

    demos: list[dict[str, Any]]

    def named_parameters(self) -> dict[str, Any]:
        return {"demos": self.demos} if hasattr(self, "demos") else {}

    def deepcopy(self):
        return copy.deepcopy(self)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

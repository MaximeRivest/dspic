"""Default vision-model adapter placeholder."""

from __future__ import annotations

from dspic.base_vm import EchoVM


class VM(EchoVM):
    """Default typed vision model placeholder.

    Real providers should subclass `BaseVM` directly or replace this class with
    an adapter that implements `forward(request) -> VisionResponse`.
    """

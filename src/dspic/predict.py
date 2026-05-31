"""DSPy-style Predict module for DSPIC vision models."""

from __future__ import annotations

from typing import Any

from dspic.adapters import ImageAdapter
from dspic.base_vm import BaseVM
from dspic.module import Module
from dspic.prediction import Prediction
from dspic.settings import settings
from dspic.signatures import Signature, ensure_signature

_DEFAULT_CUSTOM_TYPES: dict[str, type] = {}


def _default_custom_types() -> dict[str, type]:
    if _DEFAULT_CUSTOM_TYPES:
        return _DEFAULT_CUSTOM_TYPES
    import dspic
    from dspic.foundation_vision_types import (
        BoxPrompt,
        MaskPrompt,
        PointPrompt,
        TextPrompt,
    )

    _DEFAULT_CUSTOM_TYPES.update(
        {
            "Image": dspic.Image,
            "Video": dspic.Video,
            "Text": dspic.Text,
            "Box": dspic.Box,
            "Boxes": dspic.Boxes,
            "Point": dspic.Point,
            "Points": dspic.Points,
            "Mask": dspic.Mask,
            "Masks": dspic.Masks,
            "Tracks": dspic.Tracks,
            "Keypoints": dspic.Keypoints,
            "RawText": dspic.RawText,
            "PointPrompt": PointPrompt,
            "BoxPrompt": BoxPrompt,
            "MaskPrompt": MaskPrompt,
            "TextPrompt": TextPrompt,
        }
    )
    return _DEFAULT_CUSTOM_TYPES


class Predict(Module):
    """Run a typed vision signature against one or more VMs.

    Examples:

    ```python
    segment = dspic.Predict("image: Image, point: PointPrompt -> mask: Masks")
    output = segment(image="frame.png", point={...}, vm=sam_vm)
    ```

    `demos` are the DSPIC analogue of DSPy few-shot demos. For vision models,
    they are normalized example frames/prompts prepended to the VM request so an
    optimizer can search over visual prompting examples.
    """

    def __init__(
        self,
        signature: str | type[Signature],
        *,
        vm: BaseVM | list[BaseVM] | None = None,
        adapter: ImageAdapter | None = None,
        demos: list[dict[str, Any]] | None = None,
        instructions: str | None = None,
        custom_types: dict[str, type] | None = None,
        **config: Any,
    ) -> None:
        self.signature = self._coerce_signature(
            signature,
            instructions=instructions,
            custom_types=custom_types,
        )
        self.vm = vm
        self.adapter = adapter
        self.demos = list(demos or [])
        self.config = dict(config)

    def forward(
        self,
        *args: Any,
        vm: BaseVM | list[BaseVM] | None = None,
        adapter: ImageAdapter | None = None,
        demos: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Prediction:
        if args:
            kwargs = self._positional_args_to_kwargs(args, kwargs)

        active_settings = settings()
        selected_vm = vm or self.vm or active_settings.vm
        if selected_vm is None:
            raise ValueError(
                "No VM configured. Pass vm=..., set it on Predict(...), or call "
                "dspic.configure(vm=...)."
            )

        selected_adapter = (
            adapter or self.adapter or active_settings.adapter or ImageAdapter()
        )
        if not isinstance(selected_adapter, ImageAdapter):
            # Keep the check broad later if we add more adapter classes; for now
            # all public DSPIC adapters share this implementation contract.
            if not callable(selected_adapter):
                raise TypeError("adapter must be callable.")

        call_demos = self.demos if demos is None else list(demos)
        outputs = selected_adapter(
            selected_vm,
            self.signature,
            kwargs,
            demos=call_demos,
            **self.config,
        )
        return Prediction(outputs)

    async def acall(
        self,
        *args: Any,
        vm: BaseVM | list[BaseVM] | None = None,
        adapter: ImageAdapter | None = None,
        demos: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Prediction:
        if args:
            kwargs = self._positional_args_to_kwargs(args, kwargs)

        active_settings = settings()
        selected_vm = vm or self.vm or active_settings.vm
        if selected_vm is None:
            raise ValueError("No VM configured.")

        selected_adapter = (
            adapter or self.adapter or active_settings.adapter or ImageAdapter()
        )
        call_demos = self.demos if demos is None else list(demos)
        outputs = await selected_adapter.acall(
            selected_vm,
            self.signature,
            kwargs,
            demos=call_demos,
            **self.config,
        )
        return Prediction(outputs)

    def _coerce_signature(
        self,
        signature: str | type[Signature],
        *,
        instructions: str | None,
        custom_types: dict[str, type] | None,
    ) -> type[Signature]:
        if isinstance(signature, str):
            merged_custom_types = {**_default_custom_types(), **(custom_types or {})}
            return Signature(
                signature,
                instructions,
                custom_types=merged_custom_types,
            )
        return ensure_signature(signature, instructions)

    def _positional_args_to_kwargs(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        input_names = list(self.signature.input_fields)
        if len(args) > len(input_names):
            raise TypeError(
                f"Expected at most {len(input_names)} positional inputs, "
                f"got {len(args)}."
            )
        updates = dict(kwargs)
        for name, value in zip(input_names, args, strict=False):
            if name in updates:
                raise TypeError(f"Input {name!r} was passed twice.")
            updates[name] = value
        return updates

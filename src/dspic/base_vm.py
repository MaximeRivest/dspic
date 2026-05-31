"""Typed base class for foundation vision model adapters.

`BaseVM` is the vision-model analogue of DSPy's typed `BaseLM` path: adapters
receive a normalized `VisionRequest` and must return a normalized
`VisionResponse`. There is intentionally no legacy provider-response contract in
this class.
"""

from __future__ import annotations

import copy as copy_module
import datetime as datetime_module
import importlib
import inspect
import pprint
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar, TextIO

from dspic.foundation_vision_types import (
    ModelUsage,
    VisionConfig,
    VisionInput,
    VisionOutput,
    VisionRequest,
    VisionResponse,
    VisionTask,
    coerce_vision_input,
)

MAX_HISTORY_SIZE = 10_000
GLOBAL_HISTORY: list[dict[str, Any]] = []
VM_CLASS_STATE_KEY = "_dspic_vm_class"
_BUILTIN_VM_CLASS_PATH = "dspic.vm.VM"


@dataclass(frozen=True)
class VMCapabilities:
    """Declared capabilities for a vision-model adapter.

    Keep capability metadata centralized instead of adding many capability
    properties to `BaseVM`. Adapter authors can set
    `default_capabilities` on the class or pass `capabilities=` at
    construction time.
    """

    tasks: frozenset[VisionTask] = field(default_factory=frozenset)
    input_types: frozenset[str] = field(default_factory=frozenset)
    output_types: frozenset[str] = field(default_factory=frozenset)
    config_params: frozenset[str] = field(default_factory=frozenset)
    batching: bool = False
    streaming: bool = False
    async_calls: bool = False
    max_batch_size: int | None = None
    provider_data: Mapping[str, Any] = field(default_factory=dict)

    def has_task(self, task: VisionTask) -> bool:
        return task in self.tasks

    def has_input(self, input_type: str) -> bool:
        return input_type in self.input_types

    def has_output(self, output_type: str) -> bool:
        return output_type in self.output_types

    def to_dict(self) -> dict[str, Any]:
        return {
            "tasks": sorted(self.tasks),
            "input_types": sorted(self.input_types),
            "output_types": sorted(self.output_types),
            "config_params": sorted(self.config_params),
            "batching": self.batching,
            "streaming": self.streaming,
            "async_calls": self.async_calls,
            "max_batch_size": self.max_batch_size,
            "provider_data": dict(self.provider_data),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> VMCapabilities:
        data = dict(value)
        for key in ("tasks", "input_types", "output_types", "config_params"):
            data[key] = frozenset(data.get(key, ()) or ())
        return cls(**data)


def _coerce_capabilities(
    value: VMCapabilities | Mapping[str, Any] | None,
) -> VMCapabilities:
    if value is None:
        return VMCapabilities()
    if isinstance(value, VMCapabilities):
        return value
    if isinstance(value, Mapping):
        return VMCapabilities.from_mapping(value)
    raise TypeError(f"Cannot convert {type(value)!r} to VMCapabilities.")


def _import_class(class_path: str) -> type:
    parts = class_path.split(".")
    last_error: Exception | None = None

    for split_index in range(len(parts) - 1, 0, -1):
        module_name = ".".join(parts[:split_index])
        try:
            obj = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            if exc.name == module_name or module_name.startswith(f"{exc.name}."):
                last_error = exc
                continue
            raise

        try:
            for attr in parts[split_index:]:
                obj = getattr(obj, attr)
        except AttributeError as exc:
            last_error = exc
            continue

        if not isinstance(obj, type):
            raise TypeError(
                f"Serialized VM class `{class_path}` did not resolve to a class."
            )
        return obj

    raise ImportError(
        f"Could not import serialized VM class `{class_path}`."
    ) from last_error


class BaseVM:
    """Base class for typed foundation vision model calls.

    Subclasses implement only one contract:

    ```python
    def forward(self, request: VisionRequest) -> VisionResponse: ...
    ```

    `BaseVM.__call__` is a small normalization boundary. It accepts either an
    explicit `VisionRequest` or direct `VisionInput` objects/dicts, merges
    instance defaults with call kwargs into `VisionConfig`, validates the typed
    response, and records typed history.
    """

    default_capabilities: ClassVar[VMCapabilities] = VMCapabilities()

    def __init__(
        self,
        model: str,
        *,
        task: VisionTask = "unknown",
        cache: bool = True,
        num_retries: int = 3,
        capabilities: VMCapabilities | Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self.model = model
        self.task = task
        self.cache = cache
        self.num_retries = num_retries
        self.capabilities = (
            _coerce_capabilities(capabilities)
            if capabilities is not None
            else self.default_capabilities
        )
        self.kwargs = self._get_initial_kwargs(**kwargs)
        self.history: list[dict[str, Any]] = []

    def _get_initial_kwargs(self, **kwargs: Any) -> dict[str, Any]:
        return dict(kwargs)

    def __call__(
        self,
        *inputs: VisionInput | dict[str, Any],
        request: VisionRequest | None = None,
        task: VisionTask | None = None,
        **kwargs: Any,
    ) -> VisionResponse:
        normalized_request = self._normalize_call(
            *inputs,
            request=request,
            task=task,
            **kwargs,
        )
        return self._call_with_request(normalized_request)

    async def acall(
        self,
        *inputs: VisionInput | dict[str, Any],
        request: VisionRequest | None = None,
        task: VisionTask | None = None,
        **kwargs: Any,
    ) -> VisionResponse:
        normalized_request = self._normalize_call(
            *inputs,
            request=request,
            task=task,
            **kwargs,
        )
        return await self._acall_with_request(normalized_request)

    def _normalize_call(
        self,
        *inputs: VisionInput | dict[str, Any],
        request: VisionRequest | None = None,
        task: VisionTask | None = None,
        **kwargs: Any,
    ) -> VisionRequest:
        if request is None and inputs and isinstance(inputs[0], VisionRequest):
            request = inputs[0]
            inputs = inputs[1:]

        if request is not None:
            if inputs:
                raise ValueError(
                    "Pass either a VisionRequest or direct inputs, not both."
                )
            request = self._with_request_defaults(request)
            if task is not None:
                request = request.model_copy(update={"task": task}, deep=True)
            return self._with_config_overrides(request, **kwargs) if kwargs else request

        merged_kwargs = {**self.kwargs, **kwargs}
        return VisionRequest.from_inputs(
            model=self.model,
            task=task or self.task,
            inputs=[coerce_vision_input(input_) for input_ in inputs],
            **merged_kwargs,
        )

    def _with_request_defaults(self, request: VisionRequest) -> VisionRequest:
        updates: dict[str, Any] = {}
        if not request.model:
            updates["model"] = self.model
        if request.task == "unknown" and self.task != "unknown":
            updates["task"] = self.task
        return request.model_copy(update=updates, deep=True) if updates else request

    def _with_config_overrides(
        self,
        request: VisionRequest,
        **kwargs: Any,
    ) -> VisionRequest:
        current = request.config.model_dump()
        extensions = dict(current.pop("extensions", {}) or {})
        override = VisionConfig.from_kwargs(**kwargs).model_dump(exclude_none=True)
        extensions.update(override.pop("extensions", {}) or {})
        current.update(override)
        current["extensions"] = extensions
        return request.model_copy(update={"config": VisionConfig(**current)}, deep=True)

    def _call_with_request(self, request: VisionRequest) -> VisionResponse:
        response = self.forward(request)
        return self._finalize_response(request, self._validate_response(response))

    async def _acall_with_request(self, request: VisionRequest) -> VisionResponse:
        response = await self.aforward(request)
        return self._finalize_response(request, self._validate_response(response))

    def _validate_response(self, response: Any) -> VisionResponse:
        if isinstance(response, VisionResponse):
            return response
        raise TypeError(
            f"{type(self).__name__}.forward() must return VisionResponse, "
            f"but got {type(response).__name__}."
        )

    def _finalize_response(
        self,
        request: VisionRequest,
        response: VisionResponse,
    ) -> VisionResponse:
        if response.model is None or response.task is None:
            response = response.model_copy(
                update={
                    "model": response.model or request.model,
                    "task": response.task or request.task,
                },
                deep=True,
            )

        self.update_history(
            {
                "request": request,
                "response": response,
                "timestamp": datetime_module.datetime.now().isoformat(),
                "uuid": str(uuid.uuid4()),
                "model": request.model,
                "task": request.task,
            }
        )
        return response

    def forward(self, request: VisionRequest) -> VisionResponse:
        """Run the vision model on a normalized request.

        Subclasses must translate `VisionRequest` into provider/model-specific
        tensors or arguments and translate provider outputs back into
        `VisionResponse`.
        """

        raise NotImplementedError("Subclasses must implement forward(request).")

    async def aforward(self, request: VisionRequest) -> VisionResponse:
        """Async vision-model forward pass.

        Async-capable subclasses should override this. The base implementation
        intentionally does not call sync `forward()` to avoid surprising event
        loop blocking for large local vision models.
        """

        raise NotImplementedError("Subclasses must implement aforward(request).")

    def dump_state(self) -> dict[str, Any]:
        filtered_kwargs = {
            key: value
            for key, value in self.kwargs.items()
            if key not in {"api_key", "token", VM_CLASS_STATE_KEY}
        }
        return {
            VM_CLASS_STATE_KEY: f"{type(self).__module__}.{type(self).__qualname__}",
            "model": self.model,
            "task": self.task,
            "cache": self.cache,
            "num_retries": self.num_retries,
            "capabilities": self.capabilities.to_dict(),
            **filtered_kwargs,
        }

    @classmethod
    def load_state(
        cls,
        state: dict[str, Any],
        *,
        allow_custom_vm_class: bool = False,
    ) -> BaseVM:
        state = dict(state)
        class_path = state.pop(VM_CLASS_STATE_KEY, None)

        if cls is BaseVM:
            if class_path is None:
                from dspic.vm import VM

                return VM(**state)

            if class_path != _BUILTIN_VM_CLASS_PATH and not allow_custom_vm_class:
                raise ValueError(
                    f"Refusing to import custom serialized VM class `{class_path}`. "
                    "Pass allow_custom_vm_class=True when loading trusted state."
                )

            vm_cls = _import_class(class_path)
            if not issubclass(vm_cls, BaseVM):
                raise TypeError(
                    f"Serialized VM class `{class_path}` must be a BaseVM subclass."
                )
            if (
                "allow_custom_vm_class"
                in inspect.signature(vm_cls.load_state).parameters
            ):
                return vm_cls.load_state(
                    state,
                    allow_custom_vm_class=allow_custom_vm_class,
                )
            return vm_cls.load_state(state)

        return cls(**state)

    def copy(self, **kwargs: Any) -> BaseVM:
        new_instance = copy_module.copy(self)
        new_instance.history = []
        new_instance.kwargs = dict(getattr(self, "kwargs", {}) or {})

        for key, value in kwargs.items():
            if hasattr(new_instance, key):
                setattr(new_instance, key, value)
            if key in new_instance.kwargs or not hasattr(self, key):
                if value is None:
                    new_instance.kwargs.pop(key, None)
                else:
                    new_instance.kwargs[key] = value
        return new_instance

    def update_history(self, entry: dict[str, Any]) -> None:
        if len(GLOBAL_HISTORY) >= MAX_HISTORY_SIZE:
            GLOBAL_HISTORY.pop(0)
        GLOBAL_HISTORY.append(entry)
        self.history.append(entry)

    def inspect_history(self, n: int = 1, file: TextIO | None = None) -> None:
        pprint.pprint(self.history[-n:], stream=file)


class EchoVM(BaseVM):
    """Tiny typed adapter useful for tests and examples."""

    def forward(self, request: VisionRequest) -> VisionResponse:
        return VisionResponse.from_outputs(
            *self._outputs_from_request(request),
            model=request.model,
            task=request.task,
            usage=ModelUsage(output_objects=0),
        )

    def _outputs_from_request(self, request: VisionRequest) -> list[VisionOutput]:
        return []


def inspect_history(n: int = 1, file: TextIO | None = None) -> None:
    pprint.pprint(GLOBAL_HISTORY[-n:], stream=file)

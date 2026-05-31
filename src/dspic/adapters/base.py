"""Base adapter for mapping DSPIC signatures to VM requests."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Any

from pydantic.fields import FieldInfo

from dspic.adapters.strategies import (
    DEFAULT_STRATEGIES,
    AdapterTrace,
    TypeStrategy,
    parse_with_annotation,
)
from dspic.base_vm import BaseVM
from dspic.foundation_vision_types import VisionInput, VisionRequest, VisionResponse
from dspic.signatures.signature import Signature


@dataclass(frozen=True)
class PlannedOutput:
    field_name: str
    field: FieldInfo
    strategy: TypeStrategy
    task: str
    output_type: str | None


@dataclass
class AdapterPlan:
    signature: type[Signature]
    inputs: dict[str, Any]
    vision_inputs: list[VisionInput] = dataclass_field(default_factory=list)
    outputs: list[PlannedOutput] = dataclass_field(default_factory=list)
    trace: list[AdapterTrace] = dataclass_field(default_factory=list)


class Adapter:
    """Map a typed vision signature onto one or more specialized VMs.

    DSPIC VMs are often narrower than LMs. An adapter therefore owns routing:
    it renders shared inputs once, selects VM(s) for requested outputs, sends
    normalized `VisionRequest`s, and reconstructs signature outputs from
    normalized `VisionResponse`s.
    """

    def __init__(self, strategies: list[TypeStrategy] | None = None) -> None:
        self.strategies = tuple(strategies or DEFAULT_STRATEGIES)

    def __call__(
        self,
        vm: BaseVM | list[BaseVM],
        signature: type[Signature],
        inputs: dict[str, Any],
        *,
        demos: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        plan = self.plan(signature, inputs, demos=demos)
        responses = self.call_vms(vm, plan, **kwargs)
        return self.parse(plan, responses)

    async def acall(
        self,
        vm: BaseVM | list[BaseVM],
        signature: type[Signature],
        inputs: dict[str, Any],
        *,
        demos: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        plan = self.plan(signature, inputs, demos=demos)
        responses = await self.acall_vms(vm, plan, **kwargs)
        return self.parse(plan, responses)

    def plan(
        self,
        signature: type[Signature],
        inputs: dict[str, Any],
        *,
        demos: list[dict[str, Any]] | None = None,
    ) -> AdapterPlan:
        plan = AdapterPlan(signature=signature, inputs=dict(inputs))
        for demo_index, demo in enumerate(demos or []):
            self._render_inputs_into_plan(
                plan,
                signature,
                demo,
                role="demo",
                demo_index=demo_index,
            )
        self._render_inputs_into_plan(plan, signature, inputs, role="input")

        for name, field in signature.output_fields.items():
            strategy = self._output_strategy(field.annotation)
            if strategy is None:
                continue
            task = strategy.task(name, field) or "unknown"
            plan.outputs.append(
                PlannedOutput(
                    field_name=name,
                    field=field,
                    strategy=strategy,
                    task=task,
                    output_type=strategy.output_type(name, field),
                )
            )
            plan.trace.append(
                AdapterTrace(name, type(strategy).__name__, "parse", "output field")
            )
        return plan

    def _render_inputs_into_plan(
        self,
        plan: AdapterPlan,
        signature: type[Signature],
        inputs: dict[str, Any],
        *,
        role: str,
        demo_index: int | None = None,
    ) -> None:
        for name, field in signature.input_fields.items():
            if name not in inputs:
                continue
            strategy = self._input_strategy(field.annotation)
            if strategy is None:
                continue
            rendered = strategy.render_input(name, field, inputs[name])
            if rendered is None:
                continue
            for vision_input in rendered.inputs:
                provider_data = {
                    **vision_input.provider_data,
                    "adapter_role": role,
                }
                if demo_index is not None:
                    provider_data["demo_index"] = demo_index
                plan.vision_inputs.append(
                    vision_input.model_copy(
                        update={"provider_data": provider_data},
                        deep=True,
                    )
                )
            plan.trace.append(rendered.trace)

    def call_vms(
        self,
        vm: BaseVM | list[BaseVM],
        plan: AdapterPlan,
        **kwargs: Any,
    ) -> dict[str, VisionResponse]:
        vms = vm if isinstance(vm, list) else [vm]
        responses = {}
        for output in plan.outputs:
            selected = self.select_vm(vms, output)
            request = self.render_request(selected, plan, output, **kwargs)
            responses[output.field_name] = selected(request=request)
        return responses

    async def acall_vms(
        self,
        vm: BaseVM | list[BaseVM],
        plan: AdapterPlan,
        **kwargs: Any,
    ) -> dict[str, VisionResponse]:
        vms = vm if isinstance(vm, list) else [vm]
        responses = {}
        for output in plan.outputs:
            selected = self.select_vm(vms, output)
            request = self.render_request(selected, plan, output, **kwargs)
            responses[output.field_name] = await selected.acall(request=request)
        return responses

    def render_request(
        self,
        vm: BaseVM,
        plan: AdapterPlan,
        output: PlannedOutput,
        **kwargs: Any,
    ) -> VisionRequest:
        return VisionRequest.from_inputs(
            model=vm.model,
            task=output.task,  # type: ignore[arg-type]
            inputs=plan.vision_inputs,
            **{**vm.kwargs, **kwargs},
        )

    def select_vm(self, vms: list[BaseVM], output: PlannedOutput) -> BaseVM:
        for vm in vms:
            if output.output_type and vm.capabilities.has_output(output.output_type):
                return vm
            if output.task != "unknown" and vm.capabilities.has_task(output.task):
                return vm
        if len(vms) == 1:
            return vms[0]
        raise ValueError(
            f"No VM can satisfy output field {output.field_name!r} "
            f"(task={output.task!r}, output_type={output.output_type!r})."
        )

    def parse(
        self,
        plan: AdapterPlan,
        responses: dict[str, VisionResponse],
    ) -> dict[str, Any]:
        values = {}
        for output in plan.outputs:
            response = responses[output.field_name]
            value = output.strategy.parse_output(
                output.field_name, output.field, response
            )
            values[output.field_name] = parse_with_annotation(
                value,
                output.field.annotation,
            )
        return values

    def _input_strategy(self, annotation: Any) -> TypeStrategy | None:
        return next(
            (
                strategy
                for strategy in self.strategies
                if strategy.matches_input(annotation)
            ),
            None,
        )

    def _output_strategy(self, annotation: Any) -> TypeStrategy | None:
        return next(
            (
                strategy
                for strategy in self.strategies
                if strategy.matches_output(annotation)
            ),
            None,
        )

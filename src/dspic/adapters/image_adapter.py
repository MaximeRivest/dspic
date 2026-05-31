"""Default adapter for image/vision model programs."""

from __future__ import annotations

from typing import Any

from dspic.adapters.base import Adapter, AdapterPlan, PlannedOutput
from dspic.base_vm import BaseVM
from dspic.foundation_vision_types import VisionRequest


class ImageAdapter(Adapter):
    """Simple VM adapter for one or many specialized vision models.

    If several output fields can be served by the same VM/task, this adapter
    still keeps requests field-oriented for now. That makes routing and parsing
    obvious, and leaves room for later batching/fusion strategies.
    """

    def render_request(
        self,
        vm: BaseVM,
        plan: AdapterPlan,
        output: PlannedOutput,
        **kwargs: Any,
    ) -> VisionRequest:
        request = super().render_request(vm, plan, output, **kwargs)
        return request.model_copy(
            update={
                "metadata": {
                    **request.metadata,
                    "signature": plan.signature.signature,
                    "output_field": output.field_name,
                    "adapter": type(self).__name__,
                }
            },
            deep=True,
        )

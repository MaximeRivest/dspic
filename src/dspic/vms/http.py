"""HTTP transport for remotely served normalized vision models."""

from __future__ import annotations

from typing import Any

import httpx

from dspic.base_vm import BaseVM
from dspic.foundation_vision_types import VisionRequest, VisionResponse


class HTTPVisionModelVM(BaseVM):
    """Base VM for servers that accept `VisionRequest` and return `VisionResponse`.

    The server contract is intentionally simple:

    - request body: `VisionRequest.model_dump(mode="json", exclude_none=True)`
    - response body: JSON matching `VisionResponse`
    """

    def __init__(
        self,
        model: str,
        *,
        endpoint: str,
        timeout: float | httpx.Timeout = 120.0,
        headers: dict[str, str] | None = None,
        transport: httpx.BaseTransport | None = None,
        async_transport: httpx.AsyncBaseTransport | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, **kwargs)
        self.endpoint = endpoint
        self.timeout = timeout
        self.headers = dict(headers or {})
        self.transport = transport
        self.async_transport = async_transport

    def forward(self, request: VisionRequest) -> VisionResponse:
        with httpx.Client(
            timeout=self.timeout,
            headers=self.headers,
            transport=self.transport,
        ) as client:
            response = client.post(self.endpoint, json=self._request_json(request))
            response.raise_for_status()
            return VisionResponse.model_validate(response.json())

    async def aforward(self, request: VisionRequest) -> VisionResponse:
        async with httpx.AsyncClient(
            timeout=self.timeout,
            headers=self.headers,
            transport=self.async_transport,
        ) as client:
            response = await client.post(
                self.endpoint,
                json=self._request_json(request),
            )
            response.raise_for_status()
            return VisionResponse.model_validate(response.json())

    def _request_json(self, request: VisionRequest) -> dict[str, Any]:
        return request.model_dump(mode="json", exclude_none=True)

    def dump_state(self) -> dict[str, Any]:
        return {
            **super().dump_state(),
            "endpoint": self.endpoint,
            "timeout": self.timeout
            if isinstance(self.timeout, float | int)
            else None,
            "headers": self.headers,
        }

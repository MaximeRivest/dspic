"""Optional idiomatic signature + adapter + VM end-to-end test for SAM 2.1.

Run with:

    export DSPIC_SAM21_ENDPOINT=http://192.168.2.24:8078/v1/vision
    uv run pytest tests/test_sam21_signature_adapter_integration.py
"""

from __future__ import annotations

import base64
import os
from io import BytesIO

import httpx
import pytest

import dspic
from dspic.foundation_vision_types import PointPrompt
from dspic.vms import SAM21VM


def _inline_png(width: int = 96, height: int = 96) -> str:
    try:
        from PIL import Image as PILImage
    except ImportError as exc:
        pytest.skip(f"Pillow is required for this integration test: {exc}")

    image = PILImage.new("RGB", (width, height), "black")
    for x in range(24, 72):
        for y in range(24, 72):
            image.putpixel((x, y), (255, 255, 255))

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


SegmentFromPoint = dspic.make_signature(
    "image: Image, point: PointPrompt -> masks: Masks",
    instructions="Segment the object indicated by a positive point prompt.",
    custom_types={
        "Image": dspic.Image,
        "Masks": dspic.Masks,
        "PointPrompt": PointPrompt,
    },
)


@pytest.mark.integration
def test_sam21_signature_adapter_segments_inline_png_from_point_prompt() -> None:
    endpoint = os.environ.get("DSPIC_SAM21_ENDPOINT")
    if endpoint is None:
        pytest.skip("Set DSPIC_SAM21_ENDPOINT to run the SAM 2.1 integration test.")

    health_url = endpoint.rsplit("/v1/vision", 1)[0].rstrip("/") + "/health"
    try:
        health = httpx.get(health_url, timeout=5.0)
        health.raise_for_status()
    except httpx.HTTPError as exc:
        pytest.skip(f"SAM 2.1 server is not reachable at {health_url}: {exc}")

    adapter = dspic.ImageAdapter()
    vm = SAM21VM(endpoint=endpoint, timeout=60.0)

    outputs = adapter(
        vm,
        SegmentFromPoint,
        {
            "image": {
                "data": _inline_png(),
                "media_type": "image/png",
                "width": 96,
                "height": 96,
            },
            "point": {
                "type": "point_prompt",
                "point": {"x": 48, "y": 48},
                "label": "positive",
            },
        },
        mask_format="bitmap",
    )

    masks = outputs["masks"].masks
    assert masks

    mask = masks[0].mask
    assert mask.encoding == "bitmap"
    assert mask.size == (96, 96)
    assert mask.data is not None

    if isinstance(mask.data, list):
        true_pixels = sum(bool(pixel) for row in mask.data for pixel in row)
        assert true_pixels > 0

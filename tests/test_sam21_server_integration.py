"""Optional end-to-end test for a running SAM 2.1 server.

Run with:

    export DSPIC_SAM21_ENDPOINT=http://192.168.2.24:8078/v1/vision
    uv run pytest tests/test_sam21_server_integration.py
"""

from __future__ import annotations

import base64
import os
import struct
import zlib

import httpx
import pytest

from dspic.vms import SAM21VM


def _inline_png(width: int = 96, height: int = 96) -> str:
    rows = []
    for y in range(height):
        row = bytearray([0])  # PNG filter type: none.
        for x in range(width):
            pixel = (
                b"\xff\xff\xff"
                if 24 <= x < 72 and 24 <= y < 72
                else b"\x00\x00\x00"
            )
            row.extend(pixel)
        rows.append(bytes(row))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(
        b"IHDR",
        struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0),
    )
    png += chunk(b"IDAT", zlib.compress(b"".join(rows)))
    png += chunk(b"IEND", b"")
    return base64.b64encode(png).decode("ascii")


@pytest.mark.integration
def test_sam21_server_segments_inline_png_from_point_prompt() -> None:
    endpoint = os.environ.get("DSPIC_SAM21_ENDPOINT")
    if endpoint is None:
        pytest.skip("Set DSPIC_SAM21_ENDPOINT to run the SAM 2.1 integration test.")

    health_url = endpoint.rsplit("/v1/vision", 1)[0].rstrip("/") + "/health"
    try:
        health = httpx.get(health_url, timeout=5.0)
        health.raise_for_status()
    except httpx.HTTPError as exc:
        pytest.skip(f"SAM 2.1 server is not reachable at {health_url}: {exc}")

    vm = SAM21VM(endpoint=endpoint, timeout=60.0)
    response = vm(
        {
            "type": "image",
            "image": {
                "data": _inline_png(),
                "media_type": "image/png",
                "width": 96,
                "height": 96,
            },
        },
        {
            "type": "point_prompt",
            "point": {"x": 48, "y": 48},
            "label": "positive",
        },
        mask_format="bitmap",
    )

    assert response.model == "facebook/sam2.1-hiera-large"
    assert response.task == "segment"
    assert response.parts

    masks = response.parts[0].masks
    assert masks

    mask = masks[0].mask
    assert mask.encoding == "bitmap"
    assert mask.size == (96, 96)
    assert mask.data is not None

    if isinstance(mask.data, list):
        true_pixels = sum(bool(pixel) for row in mask.data for pixel in row)
        assert true_pixels > 0

"""Optional full DSPIC Predict integration test for a running SAM 2.1 server.

This exercises the idiomatic path:

    dspic.configure(vm=sam)
    segment = dspic.Predict("image: Image, point: PointPrompt -> mask: Masks")
    pred = segment(...)

Run with:

    export DSPIC_SAM21_ENDPOINT=http://192.168.2.24:8078/v1/vision
    uv run pytest tests/test_sam21_predict_integration.py
"""

from __future__ import annotations

import base64
import os
import struct
import zlib

import httpx
import pytest

import dspic


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


def _require_sam21_endpoint() -> str:
    endpoint = os.environ.get(
        "DSPIC_SAM21_ENDPOINT",
        "http://192.168.2.24:8078/v1/vision",
    )
    health_url = endpoint.rsplit("/v1/vision", 1)[0].rstrip("/") + "/health"
    try:
        health = httpx.get(health_url, timeout=5.0)
        health.raise_for_status()
    except httpx.HTTPError as exc:
        pytest.skip(f"SAM 2.1 server is not reachable at {health_url}: {exc}")
    return endpoint


@pytest.mark.integration
def test_predict_configure_sam21_segments_inline_png_from_point_prompt() -> None:
    endpoint = _require_sam21_endpoint()
    previous_vm = dspic.settings().vm
    sam = dspic.SAM21VM(endpoint=endpoint, timeout=60.0)

    try:
        dspic.configure(vm=sam)
        segment = dspic.Predict(
            "image: Image, point: PointPrompt -> mask: Masks",
            mask_format="bitmap",
        )

        pred = segment(
            image={
                "data": _inline_png(),
                "media_type": "image/png",
                "width": 96,
                "height": 96,
            },
            point={
                "type": "point_prompt",
                "point": {"x": 48, "y": 48},
                "label": "positive",
            },
        )
    finally:
        dspic.configure(vm=previous_vm)

    masks = pred.mask.masks
    assert masks

    mask = masks[0].mask
    assert mask.encoding == "bitmap"
    assert mask.size == (96, 96)
    assert mask.data is not None

    if isinstance(mask.data, list):
        true_pixels = sum(bool(pixel) for row in mask.data for pixel in row)
        assert true_pixels > 0

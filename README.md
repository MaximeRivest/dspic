# dspic

> **Status: vibe-coded experiment.** This project is exploratory and not
> production-quality. If it reaches `1.0`, it means the code and API have been
> fully reviewed by qualified humans.

DSPIC is a DSPy-inspired library for foundation vision models. It uses typed
signatures, adapters, and normalized request/response objects for vision models
instead of language models.

Created by Maxime Rivest and Aurelie Guisnet.

## Quick start

```python
import dspic

sam = dspic.SAM21VM(endpoint="http://192.168.2.24:8078/v1/vision")
dspic.configure(vm=sam)

segment = dspic.Predict("image: Image, point: Point -> mask: Masks")

pred = segment(
    image="image.png",
    point=(48, 48),
)

mask = pred.mask
```

For remote servers, use image data, URLs, or paths the server can access.

## Core ideas

- `VM`: a vision model client, analogous to a DSPy `LM`.
- `Predict`: a DSPy-style module for typed vision signatures.
- `ImageAdapter`: routes signatures to one or more specialized VMs.
- `demos`: visual prompting examples, analogous to DSPy demos.

## Signatures

String signatures:

```python
segment = dspic.Predict("image: Image, point: Point -> mask: Masks")
```

Class signatures:

```python
class Segment(dspic.Signature):
    image: dspic.Image = dspic.InputField()
    point: dspic.Point = dspic.InputField()
    mask: dspic.Masks = dspic.OutputField()

segment = dspic.Predict(Segment)
```

## Simple input types

```python
image = "image.png"
point = (48, 48)
box = (10, 20, 100, 120)
```

These become normalized VM inputs:

- `Image` -> `ImageInput`
- `Point` -> `PointPrompt`
- `Box` -> `BoxPrompt`
- `Mask` -> `MaskPrompt`

Common outputs:

- `Masks`
- `Boxes`
- `Points`
- `Tracks`
- `Keypoints`
- `RawText`

## Multiple VMs

Vision models are specialized, so one program can use several VMs:

```python
program = dspic.Predict("image: Image, query: Text -> boxes: Boxes, mask: Masks")

pred = program(
    image="image.png",
    query="cat",
    vm=[detector_vm, segmenter_vm],
)
```

## Visual prompting demos

```python
segment = dspic.Predict(
    "image: Image, point: Point -> mask: Masks",
    vm=sam,
    demos=[{"image": "first-frame.png", "point": (100, 120)}],
)

pred = segment(image="current-frame.png", point=(130, 140))
```

## SAM 2.1 integration test

If a SAM 2.1 server is running at the default endpoint, this test runs end to
end:

```bash
uv run pytest tests/test_sam21_predict_integration.py
```

Default endpoint:

```text
http://192.168.2.24:8078/v1/vision
```

Override it with:

```bash
export DSPIC_SAM21_ENDPOINT=http://your-server:8078/v1/vision
uv run pytest tests/test_sam21_predict_integration.py
```

## Development

```bash
uv sync --dev
uv run pytest
uv run ruff check .
```

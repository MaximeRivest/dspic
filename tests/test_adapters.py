import dspic
from dspic.adapters import Boxes, Image, ImageAdapter, Masks
from dspic.base_vm import BaseVM, VMCapabilities
from dspic.foundation_vision_types import (
    BBox,
    BoxOutput,
    Detection,
    MaskOutput,
    VisionResponse,
)
from dspic.signatures import InputField, OutputField, Signature


class DetectVM(BaseVM):
    default_capabilities = VMCapabilities(
        tasks=frozenset({"detect"}),
        input_types=frozenset({"image", "text_prompt"}),
        output_types=frozenset({"boxes"}),
    )

    def forward(self, request):
        return VisionResponse.from_outputs(
            BoxOutput(
                boxes=[
                    Detection(
                        label="cat",
                        score=0.9,
                        box=BBox(x1=1, y1=2, x2=3, y2=4),
                    )
                ]
            ),
            model=request.model,
            task=request.task,
        )


class SegmentVM(BaseVM):
    default_capabilities = VMCapabilities(
        tasks=frozenset({"segment"}),
        input_types=frozenset({"image"}),
        output_types=frozenset({"masks"}),
    )

    def forward(self, request):
        return VisionResponse.from_outputs(
            MaskOutput(mode="promptable"),
            model=request.model,
            task=request.task,
        )


class FindAndSegment(Signature):
    """Find objects and segment the image."""

    image: Image = InputField()
    query: str = InputField()
    boxes: Boxes = OutputField()
    masks: Masks = OutputField()


def test_image_adapter_routes_outputs_to_specialized_vms() -> None:
    adapter = ImageAdapter()
    outputs = adapter(
        [DetectVM("detector"), SegmentVM("segmenter")],
        FindAndSegment,
        {"image": "image.png", "query": "cat"},
    )

    assert outputs["boxes"].boxes[0].label == "cat"
    assert outputs["masks"].masks == []


def test_signature_string_supports_custom_types() -> None:
    signature = dspic.make_signature(
        "image: Image -> boxes: Boxes",
        custom_types={"Image": Image, "Boxes": Boxes},
    )

    assert list(signature.input_fields) == ["image"]
    assert list(signature.output_fields) == ["boxes"]

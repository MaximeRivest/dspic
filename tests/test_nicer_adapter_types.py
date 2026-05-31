import dspic
from dspic.base_vm import BaseVM, VMCapabilities
from dspic.foundation_vision_types import MaskOutput, VisionResponse


class SegmentVM(BaseVM):
    default_capabilities = VMCapabilities(
        tasks=frozenset({"segment"}),
        input_types=frozenset({"image", "point_prompt", "box_prompt"}),
        output_types=frozenset({"masks"}),
    )

    def forward(self, request):
        self.last_request = request
        return VisionResponse.from_outputs(
            MaskOutput(mode="promptable"),
            model=request.model,
            task=request.task,
        )


def test_predict_accepts_simpler_point_prompt_type() -> None:
    vm = SegmentVM("sam")
    segment = dspic.Predict("image: Image, point: Point -> mask: Masks", vm=vm)

    prediction = segment(image="image.png", point=(48, 48))

    assert prediction.mask.masks == []
    assert vm.last_request.inputs[1].type == "point_prompt"
    assert vm.last_request.inputs[1].point.x == 48
    assert vm.last_request.inputs[1].point.y == 48
    assert vm.last_request.inputs[1].label == "positive"


def test_predict_accepts_simpler_box_prompt_type() -> None:
    vm = SegmentVM("sam")
    segment = dspic.Predict("image: Image, box: Box -> mask: Masks", vm=vm)

    segment(image="image.png", box=(1, 2, 30, 40))

    assert vm.last_request.inputs[1].type == "box_prompt"
    assert vm.last_request.inputs[1].box.x1 == 1
    assert vm.last_request.inputs[1].box.y2 == 40

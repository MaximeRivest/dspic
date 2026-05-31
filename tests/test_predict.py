import dspic
from dspic.base_vm import BaseVM, VMCapabilities
from dspic.foundation_vision_types import MaskOutput, VisionResponse


class PromptAwareSegmentVM(BaseVM):
    default_capabilities = VMCapabilities(
        tasks=frozenset({"segment"}),
        input_types=frozenset({"image", "point_prompt"}),
        output_types=frozenset({"masks"}),
    )

    def forward(self, request):
        self.last_request = request
        return VisionResponse.from_outputs(
            MaskOutput(mode="promptable"),
            model=request.model,
            task=request.task,
        )


def test_predict_accepts_string_arrow_signature_and_returns_prediction() -> None:
    vm = PromptAwareSegmentVM("sam")
    segment = dspic.Predict("image: Image -> mask: Masks", vm=vm)

    prediction = segment(image="image.png")

    assert isinstance(prediction, dspic.Prediction)
    assert prediction.mask.masks == []
    assert vm.last_request.task == "segment"
    assert vm.last_request.inputs[0].type == "image"


def test_predict_uses_configured_vm() -> None:
    vm = PromptAwareSegmentVM("sam")
    segment = dspic.Predict("image: Image -> mask: Masks")

    with dspic.context(vm=vm):
        prediction = segment(image="image.png")

    assert prediction.mask.masks == []


def test_predict_uses_global_configure_vm() -> None:
    vm = PromptAwareSegmentVM("sam")
    previous_vm = dspic.settings().vm
    segment = dspic.Predict("image: Image -> mask: Masks")

    try:
        dspic.configure(vm=vm)
        prediction = segment(image="image.png")
    finally:
        dspic.configure(vm=previous_vm)

    assert prediction.mask.masks == []


def test_predict_demos_are_prepended_as_visual_prompting_examples() -> None:
    vm = PromptAwareSegmentVM("sam")
    segment = dspic.Predict(
        "image: Image, point: PointPrompt -> mask: Masks",
        vm=vm,
        demos=[
            {
                "image": "first-frame.png",
                "point": {
                    "type": "point_prompt",
                    "point": {"x": 12, "y": 34},
                    "label": "positive",
                },
            }
        ],
    )

    segment(
        image="current-frame.png",
        point={
            "type": "point_prompt",
            "point": {"x": 56, "y": 78},
            "label": "positive",
        },
    )

    roles = [item.provider_data["adapter_role"] for item in vm.last_request.inputs]
    assert roles == ["demo", "demo", "input", "input"]
    assert vm.last_request.inputs[0].image.path == "first-frame.png"
    assert vm.last_request.inputs[2].image.path == "current-frame.png"

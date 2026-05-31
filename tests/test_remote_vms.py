import httpx

from dspic.foundation_vision_types import MaskOutput, VisionResponse
from dspic.vms import SAM21VM, LocateAnythingVM


def _echo_response(request: httpx.Request) -> httpx.Response:
    payload = request.read()
    assert payload
    return httpx.Response(
        200,
        json=VisionResponse.from_outputs(
            MaskOutput(mode="promptable"),
            model="remote-model",
            task="segment",
        ).model_dump(mode="json", exclude_none=True),
    )


def test_sam21_vm_posts_normalized_request() -> None:
    transport = httpx.MockTransport(_echo_response)
    vm = SAM21VM(endpoint="https://vision.example/sam2", transport=transport)

    response = vm(
        {"type": "image", "image": {"path": "image.png"}},
        {"type": "point_prompt", "point": {"x": 10, "y": 20}, "label": "positive"},
    )

    assert response.model == "remote-model"
    assert response.parts[0].type == "masks"
    assert vm.capabilities.has_task("segment")


def test_locate_anything_request_helpers_and_parser() -> None:
    vm = LocateAnythingVM(endpoint="https://vision.example/locate")

    request = vm.ground_request(
        image={
            "type": "image",
            "image": {"path": "image.png", "width": 200, "height": 100},
        },
        phrase="red car",
    )
    outputs = LocateAnythingVM.parse_outputs_from_text(
        "car <box><100><200><300><400></box> center <box><500><500></box>",
        image_width=200,
        image_height=100,
    )

    assert request.task == "ground"
    assert request.inputs[1].type == "text_prompt"
    assert outputs[1].boxes[0].box.x1 == 20
    assert outputs[2].points[0].point.x == 100

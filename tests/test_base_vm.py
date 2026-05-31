from dspic.base_vm import BaseVM, VMCapabilities
from dspic.foundation_vision_types import (
    ImageRef,
    LabelOutput,
    LabelScore,
    VisionResponse,
)


class ToyVM(BaseVM):
    default_capabilities = VMCapabilities(
        tasks=frozenset({"classify"}),
        input_types=frozenset({"image"}),
        output_types=frozenset({"labels"}),
    )

    def forward(self, request):
        return VisionResponse.from_outputs(
            LabelOutput(labels=[LabelScore(label="ok", score=1.0)]),
            model=request.model,
            task=request.task,
        )


def test_base_vm_normalizes_direct_inputs_and_records_history() -> None:
    vm = ToyVM(model="toy", task="classify", threshold=0.5)

    response = vm({"type": "image", "image": {"path": "image.png"}})

    assert response.model == "toy"
    assert response.task == "classify"
    assert response.parts[0].labels[0].label == "ok"
    assert len(vm.history) == 1
    assert vm.history[0]["request"].config.threshold == 0.5
    assert vm.capabilities.has_task("classify")
    assert vm.capabilities.has_input("image")
    assert vm.capabilities.has_output("labels")


def test_base_vm_accepts_explicit_request_overrides() -> None:
    vm = ToyVM(model="toy", task="classify")
    request = vm._normalize_call(
        {"type": "image", "image": ImageRef(path="image.png")},
        threshold=0.1,
    )

    response = vm(request=request, threshold=0.9)

    assert response.model == "toy"
    assert vm.history[0]["request"].config.threshold == 0.9


def test_base_vm_serializes_capabilities() -> None:
    vm = ToyVM(model="toy")

    state = vm.dump_state()

    assert state["capabilities"]["tasks"] == ["classify"]
    assert state["capabilities"]["input_types"] == ["image"]
    assert state["capabilities"]["output_types"] == ["labels"]

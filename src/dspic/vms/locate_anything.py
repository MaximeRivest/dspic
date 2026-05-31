"""Remote VM adapter for NVIDIA LocateAnything-style grounding servers."""

from __future__ import annotations

import re
from typing import Any

from dspic.base_vm import VMCapabilities
from dspic.foundation_vision_types import (
    BBox,
    BoxOutput,
    Detection,
    ImageInput,
    Point2D,
    PointDetection,
    PointOutput,
    RawTextOutput,
    TextPrompt,
    VisionRequest,
)
from dspic.vms.http import HTTPVisionModelVM


class LocateAnythingVM(HTTPVisionModelVM):
    """Client for a remote normalized LocateAnything service.

    The remote service is expected to accept `VisionRequest` JSON and return
    `VisionResponse` JSON. This adapter does not load model weights locally.
    """

    default_capabilities = VMCapabilities(
        tasks=frozenset({"detect", "ground", "locate", "ocr"}),
        input_types=frozenset({"image", "text_prompt"}),
        output_types=frozenset({"boxes", "points", "raw_text"}),
        config_params=frozenset({"threshold", "max_outputs", "generation_mode"}),
        batching=True,
        async_calls=True,
        provider_data={"family": "locateanything"},
    )

    def __init__(
        self,
        *,
        endpoint: str,
        model: str = "nvidia/LocateAnything-3B",
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, endpoint=endpoint, task="locate", **kwargs)

    def request_for_prompt(
        self,
        *,
        image: ImageInput | dict[str, Any],
        prompt: str,
        task: str = "locate",
        role: str = "referring_expression",
        **kwargs: Any,
    ) -> VisionRequest:
        """Build a normalized request with one image and one text prompt."""

        return VisionRequest.from_inputs(
            model=self.model,
            task=task,  # type: ignore[arg-type]
            inputs=[image, TextPrompt(text=prompt, role=role)],  # type: ignore[list-item]
            **{**self.kwargs, **kwargs},
        )

    def detect_request(
        self,
        *,
        image: ImageInput | dict[str, Any],
        categories: list[str],
        **kwargs: Any,
    ) -> VisionRequest:
        categories_text = ", ".join(categories)
        return self.request_for_prompt(
            image=image,
            prompt=(
                "Locate all the instances that matches the following "
                f"description: {categories_text}."
            ),
            task="detect",
            role="category",
            **kwargs,
        )

    def ground_request(
        self,
        *,
        image: ImageInput | dict[str, Any],
        phrase: str,
        multiple: bool = True,
        **kwargs: Any,
    ) -> VisionRequest:
        quantifier = "all the instances" if multiple else "a single instance"
        verb = "match" if multiple else "matches"
        return self.request_for_prompt(
            image=image,
            prompt=(
                f"Locate {quantifier} that {verb} the following "
                f"description: {phrase}."
            ),
            task="ground",
            **kwargs,
        )

    def point_request(
        self,
        *,
        image: ImageInput | dict[str, Any],
        phrase: str,
        **kwargs: Any,
    ) -> VisionRequest:
        return self.request_for_prompt(
            image=image,
            prompt=f"Point to: {phrase}.",
            task="locate",
            **kwargs,
        )

    @staticmethod
    def parse_outputs_from_text(
        text: str,
        *,
        image_width: int,
        image_height: int,
    ) -> list[BoxOutput | PointOutput | RawTextOutput]:
        """Parse LocateAnything coordinate tokens from raw generated text.

        Coordinates are normalized integer coordinates in `[0, 1000]` in the
        public LocateAnything examples. This helper is useful for servers that
        choose to return only `RawTextOutput`.
        """

        boxes = []
        for match in re.finditer(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>", text):
            x1, y1, x2, y2 = [int(group) for group in match.groups()]
            boxes.append(
                Detection(
                    box=BBox(
                        x1=x1 / 1000 * image_width,
                        y1=y1 / 1000 * image_height,
                        x2=x2 / 1000 * image_width,
                        y2=y2 / 1000 * image_height,
                    )
                )
            )

        points = []
        for match in re.finditer(r"<box><(\d+)><(\d+)></box>", text):
            x, y = [int(group) for group in match.groups()]
            points.append(
                PointDetection(
                    point=Point2D(
                        x=x / 1000 * image_width,
                        y=y / 1000 * image_height,
                    )
                )
            )

        outputs: list[BoxOutput | PointOutput | RawTextOutput] = [
            RawTextOutput(text=text)
        ]
        if boxes:
            outputs.append(BoxOutput(boxes=boxes))
        if points:
            outputs.append(PointOutput(points=points))
        return outputs

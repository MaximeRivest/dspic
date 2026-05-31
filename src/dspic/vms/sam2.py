"""Remote VM adapter for SAM 2.1 segmentation servers."""

from __future__ import annotations

from typing import Any

from dspic.base_vm import VMCapabilities
from dspic.foundation_vision_types import (
    BoxPrompt,
    ImageInput,
    MaskPrompt,
    PointPrompt,
    VideoInput,
    VisionInput,
    VisionRequest,
)
from dspic.vms.http import HTTPVisionModelVM


class SAM21VM(HTTPVisionModelVM):
    """Client for a remote normalized `facebook/sam2.1-hiera-large` service.

    The service should handle SAM2-specific tensorization and postprocessing on
    the server. This client only sends normalized images/videos and visual
    prompts and receives normalized masks/tracks.
    """

    default_capabilities = VMCapabilities(
        tasks=frozenset({"segment", "track"}),
        input_types=frozenset(
            {"image", "video", "point_prompt", "box_prompt", "mask_prompt"}
        ),
        output_types=frozenset({"masks", "tracks"}),
        config_params=frozenset(
            {
                "threshold",
                "max_outputs",
                "mask_format",
                "multimask_output",
                "points_per_batch",
            }
        ),
        batching=True,
        async_calls=True,
        provider_data={"family": "sam2", "variant": "2.1-hiera-large"},
    )

    def __init__(
        self,
        *,
        endpoint: str,
        model: str = "facebook/sam2.1-hiera-large",
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, endpoint=endpoint, task="segment", **kwargs)

    def segment_request(
        self,
        *,
        image: ImageInput | dict[str, Any],
        prompts: list[
            PointPrompt | BoxPrompt | MaskPrompt | dict[str, Any]
        ] | None = None,
        **kwargs: Any,
    ) -> VisionRequest:
        """Build an image segmentation request."""

        inputs: list[VisionInput | dict[str, Any]] = [image]
        inputs.extend(prompts or [])
        return VisionRequest.from_inputs(
            model=self.model,
            task="segment",
            inputs=inputs,
            **{**self.kwargs, **kwargs},
        )

    def track_request(
        self,
        *,
        video: VideoInput | dict[str, Any],
        prompts: list[
            PointPrompt | BoxPrompt | MaskPrompt | dict[str, Any]
        ] | None = None,
        **kwargs: Any,
    ) -> VisionRequest:
        """Build a video tracking request."""

        inputs: list[VisionInput | dict[str, Any]] = [video]
        inputs.extend(prompts or [])
        return VisionRequest.from_inputs(
            model=self.model,
            task="track",
            inputs=inputs,
            **{**self.kwargs, **kwargs},
        )

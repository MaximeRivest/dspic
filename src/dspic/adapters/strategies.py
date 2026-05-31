"""Adapter type strategies for rendering vision signatures to VM requests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, get_args, get_origin

from pydantic import TypeAdapter
from pydantic.fields import FieldInfo

from dspic.adapters.types import (
    Box,
    Boxes,
    Image,
    Keypoints,
    Mask,
    Masks,
    Point,
    Points,
    RawText,
    Text,
    Tracks,
    Video,
)
from dspic.foundation_vision_types import (
    BoxOutput,
    BoxPrompt,
    EmbeddingOutput,
    ImageInput,
    ImageRef,
    KeypointOutput,
    LabelOutput,
    MaskOutput,
    MaskPrompt,
    PointOutput,
    PointPrompt,
    RawTextOutput,
    TextPrompt,
    TrackOutput,
    VideoInput,
    VideoRef,
    VisionInput,
    VisionOutput,
    VisionResponse,
    coerce_vision_input,
)


@dataclass(frozen=True)
class AdapterTrace:
    field_name: str | None
    strategy: str
    action: str
    reason: str = ""


@dataclass(frozen=True)
class InputRender:
    field_name: str
    inputs: tuple[VisionInput, ...]
    trace: AdapterTrace


class TypeStrategy:
    """Base class for signature-field rendering/parsing strategies."""

    def matches_input(self, annotation: Any) -> bool:
        return False

    def matches_output(self, annotation: Any) -> bool:
        return False

    def render_input(
        self, field_name: str, field: FieldInfo, value: Any
    ) -> InputRender | None:
        return None

    def output_type(self, field_name: str, field: FieldInfo) -> str | None:
        return None

    def task(self, field_name: str, field: FieldInfo) -> str | None:
        return None

    def parse_output(
        self, field_name: str, field: FieldInfo, response: VisionResponse
    ) -> Any:
        return None


class ImageInputStrategy(TypeStrategy):
    def matches_input(self, annotation: Any) -> bool:
        return _contains(annotation, Image) or annotation in {ImageInput, ImageRef}

    def render_input(
        self, field_name: str, field: FieldInfo, value: Any
    ) -> InputRender | None:
        if isinstance(value, ImageInput):
            image_input = value
        elif isinstance(value, ImageRef):
            image_input = ImageInput(id=field_name, image=value)
        else:
            image_input = ImageInput(
                id=field_name, image=Image.model_validate(value).ref
            )
        return InputRender(
            field_name=field_name,
            inputs=(image_input,),
            trace=AdapterTrace(
                field_name, type(self).__name__, "render", "image input"
            ),
        )


class VideoInputStrategy(TypeStrategy):
    def matches_input(self, annotation: Any) -> bool:
        return _contains(annotation, Video) or annotation in {VideoInput, VideoRef}

    def render_input(
        self, field_name: str, field: FieldInfo, value: Any
    ) -> InputRender | None:
        if isinstance(value, VideoInput):
            video_input = value
        elif isinstance(value, VideoRef):
            video_input = VideoInput(id=field_name, video=value)
        else:
            video_input = VideoInput(
                id=field_name, video=Video.model_validate(value).ref
            )
        return InputRender(
            field_name=field_name,
            inputs=(video_input,),
            trace=AdapterTrace(
                field_name, type(self).__name__, "render", "video input"
            ),
        )


class TextInputStrategy(TypeStrategy):
    def matches_input(self, annotation: Any) -> bool:
        return _contains(annotation, Text) or annotation in {str, TextPrompt}

    def render_input(
        self, field_name: str, field: FieldInfo, value: Any
    ) -> InputRender | None:
        if isinstance(value, TextPrompt):
            prompt = value
        else:
            text = value.text if isinstance(value, Text) else str(value)
            prompt = TextPrompt(id=field_name, text=text)
        return InputRender(
            field_name=field_name,
            inputs=(prompt,),
            trace=AdapterTrace(
                field_name, type(self).__name__, "render", "text prompt"
            ),
        )


class PointPromptStrategy(TypeStrategy):
    def matches_input(self, annotation: Any) -> bool:
        return _contains(annotation, Point)

    def render_input(
        self, field_name: str, field: FieldInfo, value: Any
    ) -> InputRender | None:
        point = Point.model_validate(value)
        return InputRender(
            field_name=field_name,
            inputs=(
                PointPrompt(
                    id=field_name,
                    point=point.point,
                    label=point.label,  # type: ignore[arg-type]
                    frame_index=point.frame_index,
                    object_id=point.object_id,
                ),
            ),
            trace=AdapterTrace(
                field_name, type(self).__name__, "render", "point prompt"
            ),
        )


class BoxPromptStrategy(TypeStrategy):
    def matches_input(self, annotation: Any) -> bool:
        return _contains(annotation, Box)

    def render_input(
        self, field_name: str, field: FieldInfo, value: Any
    ) -> InputRender | None:
        box = Box.model_validate(value)
        return InputRender(
            field_name=field_name,
            inputs=(
                BoxPrompt(
                    id=field_name,
                    box=box.box,
                    label=box.label,
                    frame_index=box.frame_index,
                    object_id=box.object_id,
                ),
            ),
            trace=AdapterTrace(field_name, type(self).__name__, "render", "box prompt"),
        )


class MaskPromptStrategy(TypeStrategy):
    def matches_input(self, annotation: Any) -> bool:
        return _contains(annotation, Mask)

    def render_input(
        self, field_name: str, field: FieldInfo, value: Any
    ) -> InputRender | None:
        mask = Mask.model_validate(value)
        return InputRender(
            field_name=field_name,
            inputs=(
                MaskPrompt(
                    id=field_name,
                    mask=mask.mask,
                    label=mask.label,
                    frame_index=mask.frame_index,
                    object_id=mask.object_id,
                ),
            ),
            trace=AdapterTrace(
                field_name,
                type(self).__name__,
                "render",
                "mask prompt",
            ),
        )


class VisionInputPassthroughStrategy(TypeStrategy):
    """Render already-normalized prompt/input types as VisionInput values."""

    passthrough_types = {PointPrompt, BoxPrompt, MaskPrompt, TextPrompt}

    def matches_input(self, annotation: Any) -> bool:
        return annotation in self.passthrough_types

    def render_input(
        self, field_name: str, field: FieldInfo, value: Any
    ) -> InputRender | None:
        vision_input = coerce_vision_input(value)
        if vision_input.id is None:
            vision_input = vision_input.model_copy(update={"id": field_name})
        return InputRender(
            field_name=field_name,
            inputs=(vision_input,),
            trace=AdapterTrace(
                field_name,
                type(self).__name__,
                "render",
                "normalized vision input",
            ),
        )


class BoxesOutputStrategy(TypeStrategy):
    def matches_output(self, annotation: Any) -> bool:
        return _contains(annotation, Boxes) or annotation is BoxOutput

    def output_type(self, field_name: str, field: FieldInfo) -> str | None:
        return "boxes"

    def task(self, field_name: str, field: FieldInfo) -> str | None:
        return "detect"

    def parse_output(
        self, field_name: str, field: FieldInfo, response: VisionResponse
    ) -> Any:
        output = _first(response, BoxOutput)
        if output is None:
            return None
        return output if field.annotation is BoxOutput else Boxes(boxes=output.boxes)


class PointsOutputStrategy(TypeStrategy):
    def matches_output(self, annotation: Any) -> bool:
        return _contains(annotation, Points) or annotation is PointOutput

    def output_type(self, field_name: str, field: FieldInfo) -> str | None:
        return "points"

    def task(self, field_name: str, field: FieldInfo) -> str | None:
        return "locate"

    def parse_output(
        self, field_name: str, field: FieldInfo, response: VisionResponse
    ) -> Any:
        output = _first(response, PointOutput)
        if output is None:
            return None
        return (
            output if field.annotation is PointOutput else Points(points=output.points)
        )


class MasksOutputStrategy(TypeStrategy):
    def matches_output(self, annotation: Any) -> bool:
        return _contains(annotation, Masks) or annotation is MaskOutput

    def output_type(self, field_name: str, field: FieldInfo) -> str | None:
        return "masks"

    def task(self, field_name: str, field: FieldInfo) -> str | None:
        return "segment"

    def parse_output(
        self, field_name: str, field: FieldInfo, response: VisionResponse
    ) -> Any:
        output = _first(response, MaskOutput)
        if output is None:
            return None
        return output if field.annotation is MaskOutput else Masks(masks=output.masks)


class TracksOutputStrategy(TypeStrategy):
    def matches_output(self, annotation: Any) -> bool:
        return _contains(annotation, Tracks) or annotation is TrackOutput

    def output_type(self, field_name: str, field: FieldInfo) -> str | None:
        return "tracks"

    def task(self, field_name: str, field: FieldInfo) -> str | None:
        return "track"

    def parse_output(
        self, field_name: str, field: FieldInfo, response: VisionResponse
    ) -> Any:
        output = _first(response, TrackOutput)
        if output is None:
            return None
        return (
            output if field.annotation is TrackOutput else Tracks(tracks=output.tracks)
        )


class KeypointsOutputStrategy(TypeStrategy):
    def matches_output(self, annotation: Any) -> bool:
        return _contains(annotation, Keypoints) or annotation is KeypointOutput

    def output_type(self, field_name: str, field: FieldInfo) -> str | None:
        return "keypoints"

    def task(self, field_name: str, field: FieldInfo) -> str | None:
        return "pose"

    def parse_output(
        self, field_name: str, field: FieldInfo, response: VisionResponse
    ) -> Any:
        output = _first(response, KeypointOutput)
        if output is None:
            return None
        return (
            output
            if field.annotation is KeypointOutput
            else Keypoints(instances=output.instances)
        )


class RawTextOutputStrategy(TypeStrategy):
    def matches_output(self, annotation: Any) -> bool:
        return _contains(annotation, RawText) or annotation in {RawTextOutput, str}

    def output_type(self, field_name: str, field: FieldInfo) -> str | None:
        return "raw_text"

    def parse_output(
        self, field_name: str, field: FieldInfo, response: VisionResponse
    ) -> Any:
        output = _first(response, RawTextOutput)
        if output is None:
            return None
        return output.text if field.annotation is str else output


class PassthroughOutputStrategy(TypeStrategy):
    def matches_output(self, annotation: Any) -> bool:
        return annotation in {LabelOutput, EmbeddingOutput}

    def output_type(self, field_name: str, field: FieldInfo) -> str | None:
        if field.annotation is LabelOutput:
            return "labels"
        if field.annotation is EmbeddingOutput:
            return "embeddings"
        return None

    def task(self, field_name: str, field: FieldInfo) -> str | None:
        if field.annotation is LabelOutput:
            return "classify"
        if field.annotation is EmbeddingOutput:
            return "embed"
        return None

    def parse_output(
        self, field_name: str, field: FieldInfo, response: VisionResponse
    ) -> Any:
        return _first(response, field.annotation)


DEFAULT_STRATEGIES: tuple[TypeStrategy, ...] = (
    ImageInputStrategy(),
    VideoInputStrategy(),
    TextInputStrategy(),
    PointPromptStrategy(),
    BoxPromptStrategy(),
    MaskPromptStrategy(),
    VisionInputPassthroughStrategy(),
    BoxesOutputStrategy(),
    PointsOutputStrategy(),
    MasksOutputStrategy(),
    TracksOutputStrategy(),
    KeypointsOutputStrategy(),
    RawTextOutputStrategy(),
    PassthroughOutputStrategy(),
)


def parse_with_annotation(value: Any, annotation: Any) -> Any:
    if value is None:
        return None
    return TypeAdapter(annotation).validate_python(value)


def _contains(annotation: Any, expected: type) -> bool:
    try:
        if isinstance(annotation, type) and issubclass(annotation, expected):
            return True
    except TypeError:
        pass
    origin = get_origin(annotation)
    if origin is None:
        return False
    return any(_contains(arg, expected) for arg in get_args(annotation))


def _first(response: VisionResponse, output_type: type[VisionOutput]) -> Any | None:
    for part in response.outputs:
        found = part.first(output_type)
        if found is not None:
            return found
    return None

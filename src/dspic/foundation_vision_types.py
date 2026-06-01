"""Small normalized I/O types for foundation vision model adapters.

These models do not share a native API. SAM returns masks, DINO returns feature
vectors, YOLO returns result objects, and LocateAnything returns text that encodes
boxes or points. This module keeps only the common data shape we need at the DSPIC
boundary. Model-specific details should live in adapter code or `provider_data`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Any, Literal, TypeAlias

import pydantic
from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "ArrayRef",
    "AssetRef",
    "BBox",
    "BinaryRef",
    "BoxOutput",
    "BoxPrompt",
    "CoordinateSpace",
    "Detection",
    "EmbeddingOutput",
    "EmbeddingRef",
    "Geometry3DOutput",
    "ImageInput",
    "ImageRef",
    "Keypoint",
    "KeypointInstance",
    "KeypointOutput",
    "LabelOutput",
    "LabelScore",
    "MaskInstance",
    "MaskOutput",
    "MaskPrompt",
    "MaskRef",
    "MeshRef",
    "ModelUsage",
    "Point2D",
    "Point3D",
    "PointCloudRef",
    "PointOutput",
    "PointPrompt",
    "Polygon",
    "RawTextOutput",
    "SourceRef",
    "TextPrompt",
    "Track",
    "TrackFrame",
    "TrackOutput",
    "VideoInput",
    "VideoRef",
    "VisionConfig",
    "VisionInput",
    "VisionOutput",
    "VisionOutputPart",
    "VisionRequest",
    "VisionResponse",
    "VisionTask",
    "coerce_vision_input",
    "coerce_vision_output",
]


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")


class MetadataModel(StrictBaseModel):
    metadata: dict[str, Any] = Field(default_factory=dict)
    provider_data: dict[str, Any] = Field(default_factory=dict)


class SourceRef(MetadataModel):
    """Reference to inline data, a local file, a URL, or provider-managed asset."""

    media_type: str = "application/octet-stream"
    data: Any | None = None
    path: str | None = None
    url: str | None = None
    file_id: str | None = None
    asset_id: str | None = None
    name: str | None = None

    @model_validator(mode="after")
    def validate_one_source(self) -> SourceRef:
        sources = [self.data, self.path, self.url, self.file_id, self.asset_id]
        if sum(source is not None for source in sources) != 1:
            raise ValueError(
                "provide exactly one of data, path, url, file_id, or asset_id"
            )
        return self


class ImageRef(SourceRef):
    media_type: str = "image/png"
    width: int | None = Field(default=None, ge=1)
    height: int | None = Field(default=None, ge=1)


class VideoRef(SourceRef):
    media_type: str = "video/mp4"
    width: int | None = Field(default=None, ge=1)
    height: int | None = Field(default=None, ge=1)
    fps: float | None = Field(default=None, gt=0)
    frame_count: int | None = Field(default=None, ge=0)
    duration_seconds: float | None = Field(default=None, ge=0)


class BinaryRef(SourceRef):
    filename: str | None = None


class AssetRef(SourceRef):
    role: str | None = None


class ArrayRef(SourceRef):
    media_type: str = "application/x-numpy"
    shape: tuple[int, ...] | None = None
    dtype: str | None = None
    layout: str | None = None


class MaskRef(ArrayRef):
    media_type: str = "application/json"
    encoding: Literal["bitmap", "rle", "coco_rle", "polygon", "alpha", "array"] = "rle"
    size: tuple[int, int] | None = None  # height, width


class EmbeddingRef(ArrayRef):
    dim: int | None = Field(default=None, ge=1)
    pooling: str | None = None


class MeshRef(SourceRef):
    media_type: str = "model/gltf-binary"
    vertex_count: int | None = Field(default=None, ge=0)
    face_count: int | None = Field(default=None, ge=0)


class PointCloudRef(SourceRef):
    point_count: int | None = Field(default=None, ge=0)
    coordinate_space: str | None = None


CoordinateSpace: TypeAlias = Literal[
    "pixel",
    "normalized",
    "camera",
    "world",
    "object",
    "unknown",
]


class Point2D(StrictBaseModel):
    x: float
    y: float
    coordinate_space: CoordinateSpace = "pixel"


class Point3D(StrictBaseModel):
    x: float
    y: float
    z: float
    coordinate_space: CoordinateSpace = "world"
    unit: str | None = None


class BBox(StrictBaseModel):
    """Axis-aligned XYXY box."""

    x1: float
    y1: float
    x2: float
    y2: float
    coordinate_space: CoordinateSpace = "pixel"

    @model_validator(mode="after")
    def validate_order(self) -> BBox:
        if self.x2 < self.x1 or self.y2 < self.y1:
            raise ValueError("BBox expects x1 <= x2 and y1 <= y2")
        return self

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1


class Polygon(StrictBaseModel):
    points: list[Point2D] = Field(min_length=3)


class LabelScore(StrictBaseModel):
    label: str
    score: float | None = Field(default=None, ge=0, le=1)
    id: str | None = None


class Keypoint(StrictBaseModel):
    point: Point2D | Point3D
    name: str | None = None
    score: float | None = Field(default=None, ge=0, le=1)
    visible: bool | None = None


class ModelUsage(StrictBaseModel):
    input_pixels: int | None = Field(default=None, ge=0)
    input_frames: int | None = Field(default=None, ge=0)
    output_objects: int | None = Field(default=None, ge=0)
    latency_ms: float | None = Field(default=None, ge=0)
    details: dict[str, Any] = Field(default_factory=dict)


class VisionInputBase(MetadataModel):
    type: str
    id: str | None = None


class ImageInput(VisionInputBase):
    type: Literal["image"] = "image"
    image: ImageRef


class VideoInput(VisionInputBase):
    type: Literal["video"] = "video"
    video: VideoRef
    frame_indices: list[int] | None = None


class TextPrompt(VisionInputBase):
    type: Literal["text_prompt"] = "text_prompt"
    text: str
    role: Literal[
        "query",
        "category",
        "referring_expression",
        "instruction",
        "negative",
    ] = "query"


class PointPrompt(VisionInputBase):
    type: Literal["point_prompt"] = "point_prompt"
    point: Point2D
    label: Literal["positive", "negative", "neutral"] | None = None
    frame_index: int | None = Field(default=None, ge=0)
    object_id: str | None = None


class BoxPrompt(VisionInputBase):
    type: Literal["box_prompt"] = "box_prompt"
    box: BBox
    label: Literal["positive", "negative", "neutral"] | str | None = None
    frame_index: int | None = Field(default=None, ge=0)
    object_id: str | None = None


class MaskPrompt(VisionInputBase):
    type: Literal["mask_prompt"] = "mask_prompt"
    mask: MaskRef
    label: Literal["positive", "negative", "neutral"] | str | None = None
    frame_index: int | None = Field(default=None, ge=0)
    object_id: str | None = None


VisionInput = Annotated[
    ImageInput | VideoInput | TextPrompt | PointPrompt | BoxPrompt | MaskPrompt,
    Field(discriminator="type"),
]


class VisionOutputBase(MetadataModel):
    type: str
    id: str | None = None
    source_input_ids: list[str] = Field(default_factory=list)
    model: str | None = None


class LabelOutput(VisionOutputBase):
    type: Literal["labels"] = "labels"
    labels: list[LabelScore] = Field(default_factory=list)


class EmbeddingOutput(VisionOutputBase):
    type: Literal["embeddings"] = "embeddings"
    embeddings: list[EmbeddingRef] = Field(default_factory=list)
    model_family: str | None = None


class Detection(StrictBaseModel):
    box: BBox
    id: str | None = None
    label: str | None = None
    score: float | None = Field(default=None, ge=0, le=1)
    phrase: str | None = None
    frame_index: int | None = Field(default=None, ge=0)
    attributes: dict[str, Any] = Field(default_factory=dict)


class BoxOutput(VisionOutputBase):
    type: Literal["boxes"] = "boxes"
    boxes: list[Detection] = Field(default_factory=list)


class PointDetection(StrictBaseModel):
    point: Point2D
    id: str | None = None
    label: str | None = None
    score: float | None = Field(default=None, ge=0, le=1)
    frame_index: int | None = Field(default=None, ge=0)


class PointOutput(VisionOutputBase):
    type: Literal["points"] = "points"
    points: list[PointDetection] = Field(default_factory=list)


class MaskInstance(StrictBaseModel):
    mask: MaskRef
    id: str | None = None
    label: str | None = None
    score: float | None = Field(default=None, ge=0, le=1)
    predicted_iou: float | None = Field(default=None, ge=0, le=1)
    box: BBox | None = None
    polygon: Polygon | None = None
    frame_index: int | None = Field(default=None, ge=0)


class MaskOutput(VisionOutputBase):
    type: Literal["masks"] = "masks"
    masks: list[MaskInstance] = Field(default_factory=list)
    mode: Literal[
        "semantic",
        "instance",
        "panoptic",
        "promptable",
        "unknown",
    ] = "unknown"


class TrackFrame(StrictBaseModel):
    frame_index: int = Field(ge=0)
    box: BBox | None = None
    mask: MaskRef | None = None
    points: list[Point2D] = Field(default_factory=list)
    score: float | None = Field(default=None, ge=0, le=1)
    visible: bool | None = None


class Track(StrictBaseModel):
    id: str
    label: str | None = None
    frames: list[TrackFrame] = Field(default_factory=list)
    score: float | None = Field(default=None, ge=0, le=1)


class TrackOutput(VisionOutputBase):
    type: Literal["tracks"] = "tracks"
    tracks: list[Track] = Field(default_factory=list)


class KeypointInstance(StrictBaseModel):
    keypoints: list[Keypoint] = Field(default_factory=list)
    id: str | None = None
    label: str | None = None
    score: float | None = Field(default=None, ge=0, le=1)
    box: BBox | None = None
    frame_index: int | None = Field(default=None, ge=0)
    skeleton_edges: list[tuple[int, int]] = Field(default_factory=list)


class KeypointOutput(VisionOutputBase):
    type: Literal["keypoints"] = "keypoints"
    instances: list[KeypointInstance] = Field(default_factory=list)


class Geometry3DOutput(VisionOutputBase):
    type: Literal["geometry_3d"] = "geometry_3d"
    mesh: MeshRef | None = None
    point_cloud: PointCloudRef | None = None
    gaussian_splat: BinaryRef | None = None
    vertices: ArrayRef | None = None
    faces: ArrayRef | None = None
    joints_3d: list[Keypoint] = Field(default_factory=list)
    assets: list[AssetRef] = Field(default_factory=list)
    coordinate_space: CoordinateSpace = "world"


class RawTextOutput(VisionOutputBase):
    """Unparsed model text, useful for VLM-style grounding models."""

    type: Literal["raw_text"] = "raw_text"
    text: str


VisionOutput = Annotated[
    LabelOutput
    | EmbeddingOutput
    | BoxOutput
    | PointOutput
    | MaskOutput
    | TrackOutput
    | KeypointOutput
    | Geometry3DOutput
    | RawTextOutput,
    Field(discriminator="type"),
]


VisionTask: TypeAlias = Literal[
    "classify",
    "embed",
    "detect",
    "ground",
    "locate",
    "segment",
    "track",
    "pose",
    "reconstruct_3d",
    "ocr",
    "unknown",
]


class VisionConfig(StrictBaseModel):
    threshold: float | None = Field(default=None, ge=0, le=1)
    max_outputs: int | None = Field(default=None, ge=1)
    device: str | None = None
    dtype: str | None = None
    batch_size: int | None = Field(default=None, ge=1)
    coordinate_space: CoordinateSpace = "pixel"
    mask_format: Literal["rle", "coco_rle", "bitmap", "polygon", "alpha", "any"] = "rle"
    return_intermediates: bool = False
    extensions: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_kwargs(cls, **kwargs: Any) -> VisionConfig:
        fields = set(cls.model_fields)
        data: dict[str, Any] = {}
        extensions = dict(kwargs.get("extensions", {}) or {})
        for key, value in kwargs.items():
            if key == "extensions":
                continue
            if key in fields:
                data[key] = value
            else:
                extensions[key] = value
        data["extensions"] = extensions
        return cls(**data)


class VisionRequest(MetadataModel):
    model: str
    task: VisionTask = "unknown"
    inputs: list[VisionInput] = Field(default_factory=list)
    config: VisionConfig = Field(default_factory=VisionConfig)

    @classmethod
    def from_inputs(
        cls,
        *,
        model: str,
        task: VisionTask = "unknown",
        inputs: list[VisionInput | Mapping[str, Any]] | None = None,
        **kwargs: Any,
    ) -> VisionRequest:
        return cls(
            model=model,
            task=task,
            inputs=[coerce_vision_input(item) for item in inputs or []],
            config=VisionConfig.from_kwargs(**kwargs),
        )

    def as_tool_args(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=True)


class VisionOutputPart(MetadataModel):
    outputs: list[VisionOutput] = Field(default_factory=list)
    finish_reason: str | None = None
    truncated: bool = False

    def first(self, output_type: type[Any]) -> Any | None:
        for output in self.outputs:
            if isinstance(output, output_type):
                return output
        return None


class VisionResponse(MetadataModel):
    outputs: list[VisionOutputPart] = Field(min_length=1)
    model: str | None = None
    task: VisionTask | None = None
    usage: ModelUsage | dict[str, Any] | None = None
    response_id: str | None = None

    @property
    def output(self) -> VisionOutputPart:
        return self.outputs[0]

    @property
    def parts(self) -> list[VisionOutput]:
        return self.output.outputs

    @classmethod
    def from_outputs(
        cls,
        *outputs: VisionOutput | Mapping[str, Any],
        model: str | None = None,
        task: VisionTask | None = None,
        usage: ModelUsage | dict[str, Any] | None = None,
    ) -> VisionResponse:
        return cls(
            model=model,
            task=task,
            usage=usage,
            outputs=[
                VisionOutputPart(
                    outputs=[coerce_vision_output(output) for output in outputs]
                )
            ],
        )


_VISION_INPUT_ADAPTER = pydantic.TypeAdapter(VisionInput)
_VISION_OUTPUT_ADAPTER = pydantic.TypeAdapter(VisionOutput)


def coerce_vision_input(value: VisionInput | Mapping[str, Any]) -> VisionInput:
    if isinstance(
        value,
        (ImageInput, VideoInput, TextPrompt, PointPrompt, BoxPrompt, MaskPrompt),
    ):
        return value
    if isinstance(value, Mapping):
        return _VISION_INPUT_ADAPTER.validate_python(dict(value))
    raise TypeError(f"Cannot convert {type(value)!r} to VisionInput.")


def coerce_vision_output(value: VisionOutput | Mapping[str, Any]) -> VisionOutput:
    if isinstance(
        value,
        (
            LabelOutput,
            EmbeddingOutput,
            BoxOutput,
            PointOutput,
            MaskOutput,
            TrackOutput,
            KeypointOutput,
            Geometry3DOutput,
            RawTextOutput,
        ),
    ):
        return value
    if isinstance(value, Mapping):
        return _VISION_OUTPUT_ADAPTER.validate_python(dict(value))
    raise TypeError(f"Cannot convert {type(value)!r} to VisionOutput.")

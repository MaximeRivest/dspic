"""Semantic adapter marker types for DSPIC vision signatures."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dspic.foundation_vision_types import (
    BBox,
    Detection,
    ImageRef,
    KeypointInstance,
    MaskInstance,
    MaskRef,
    Point2D,
    PointDetection,
    Track,
    VideoRef,
)


class Type(BaseModel):
    """Base class for DSPIC adapter marker types."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class Image(Type):
    ref: ImageRef

    @model_validator(mode="before")
    @classmethod
    def validate_input(cls, value: Any) -> Any:
        if isinstance(value, cls):
            return value
        if isinstance(value, ImageRef):
            return {"ref": value}
        if isinstance(value, str):
            if value.startswith(("http://", "https://")):
                return {"ref": ImageRef(url=value)}
            return {"ref": ImageRef(path=value)}
        if isinstance(value, dict):
            if "ref" in value:
                return value
            return {"ref": ImageRef.model_validate(value)}
        return value


class Video(Type):
    ref: VideoRef

    @model_validator(mode="before")
    @classmethod
    def validate_input(cls, value: Any) -> Any:
        if isinstance(value, cls):
            return value
        if isinstance(value, VideoRef):
            return {"ref": value}
        if isinstance(value, str):
            if value.startswith(("http://", "https://")):
                return {"ref": VideoRef(url=value)}
            return {"ref": VideoRef(path=value)}
        if isinstance(value, dict):
            if "ref" in value:
                return value
            return {"ref": VideoRef.model_validate(value)}
        return value


class Text(Type):
    text: str

    @model_validator(mode="before")
    @classmethod
    def validate_input(cls, value: Any) -> Any:
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return {"text": value}
        return value


class Point(Type):
    """A prompt point. Accepts `(x, y)`, `{x, y}`, or a `Point2D`."""

    point: Point2D
    label: str = "positive"
    frame_index: int | None = None
    object_id: str | None = None

    @model_validator(mode="before")
    @classmethod
    def validate_input(cls, value: Any) -> Any:
        if isinstance(value, cls):
            return value
        if isinstance(value, Point2D):
            return {"point": value}
        if isinstance(value, (tuple, list)) and len(value) == 2:
            return {"point": {"x": value[0], "y": value[1]}}
        if isinstance(value, dict):
            if "point" in value:
                return value
            if {"x", "y"} <= set(value):
                data = dict(value)
                return {"point": {"x": data.pop("x"), "y": data.pop("y")}, **data}
        return value


class Box(Type):
    """A prompt box. Accepts `(x1, y1, x2, y2)` or a `BBox`."""

    box: BBox
    label: str = "positive"
    frame_index: int | None = None
    object_id: str | None = None

    @model_validator(mode="before")
    @classmethod
    def validate_input(cls, value: Any) -> Any:
        if isinstance(value, cls):
            return value
        if isinstance(value, BBox):
            return {"box": value}
        if isinstance(value, (tuple, list)) and len(value) == 4:
            return {
                "box": {
                    "x1": value[0],
                    "y1": value[1],
                    "x2": value[2],
                    "y2": value[3],
                }
            }
        if isinstance(value, dict):
            if "box" in value:
                return value
            if {"x1", "y1", "x2", "y2"} <= set(value):
                data = dict(value)
                return {
                    "box": {
                        "x1": data.pop("x1"),
                        "y1": data.pop("y1"),
                        "x2": data.pop("x2"),
                        "y2": data.pop("y2"),
                    },
                    **data,
                }
        return value


class Mask(Type):
    mask: MaskRef
    label: str = "positive"
    frame_index: int | None = None
    object_id: str | None = None

    @model_validator(mode="before")
    @classmethod
    def validate_input(cls, value: Any) -> Any:
        if isinstance(value, cls):
            return value
        if isinstance(value, MaskRef):
            return {"mask": value}
        if isinstance(value, dict):
            if "mask" in value:
                return value
            return {"mask": MaskRef.model_validate(value)}
        return value


class Boxes(Type):
    boxes: list[Detection] = Field(default_factory=list)


class Points(Type):
    points: list[PointDetection] = Field(default_factory=list)


class Masks(Type):
    masks: list[MaskInstance] = Field(default_factory=list)


class Tracks(Type):
    tracks: list[Track] = Field(default_factory=list)


class Keypoints(Type):
    instances: list[KeypointInstance] = Field(default_factory=list)


class RawText(Type):
    text: str

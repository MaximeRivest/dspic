"""Adapters and adapter marker types for DSPIC."""

from dspic.adapters.base import Adapter, AdapterPlan, PlannedOutput
from dspic.adapters.image_adapter import ImageAdapter
from dspic.adapters.strategies import AdapterTrace, TypeStrategy
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
    Type,
    Video,
)

__all__ = [
    "Adapter",
    "AdapterPlan",
    "AdapterTrace",
    "Box",
    "Boxes",
    "Image",
    "ImageAdapter",
    "Keypoints",
    "Mask",
    "Masks",
    "PlannedOutput",
    "Point",
    "Points",
    "RawText",
    "Text",
    "Tracks",
    "Type",
    "TypeStrategy",
    "Video",
]

"""dspic Python library."""

from dspic.adapters import (
    Adapter,
    Box,
    Boxes,
    Image,
    ImageAdapter,
    Keypoints,
    Mask,
    Masks,
    Point,
    Points,
    RawText,
    Text,
    Tracks,
    Type,
    TypeStrategy,
    Video,
)
from dspic.base_vm import BaseVM, VMCapabilities
from dspic.predict import Predict
from dspic.prediction import Prediction
from dspic.settings import configure, context, settings
from dspic.signatures import InputField, OutputField, Signature, make_signature
from dspic.vm import VM
from dspic.vms import SAM21VM, LocateAnythingVM

__version__ = "0.1.0"

__all__ = [
    "Adapter",
    "BaseVM",
    "Box",
    "Boxes",
    "Image",
    "ImageAdapter",
    "InputField",
    "Keypoints",
    "LocateAnythingVM",
    "Mask",
    "Masks",
    "OutputField",
    "Point",
    "Points",
    "Predict",
    "Prediction",
    "RawText",
    "SAM21VM",
    "Signature",
    "Text",
    "Tracks",
    "Type",
    "TypeStrategy",
    "VM",
    "VMCapabilities",
    "Video",
    "__version__",
    "configure",
    "context",
    "make_signature",
    "settings",
]

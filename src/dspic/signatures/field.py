"""Signature field helpers for DSPIC vision programs."""

from __future__ import annotations

from typing import Literal

import pydantic
from pydantic.fields import FieldInfo

DSPIC_FIELD_TYPE = "__dspic_field_type"


def _move_kwargs(field_type: Literal["input", "output"], **kwargs):
    pydantic_kwargs = {}
    json_schema_extra = dict(kwargs.pop("json_schema_extra", {}) or {})
    for key in ("desc", "prefix"):
        if key in kwargs:
            json_schema_extra[key] = kwargs.pop(key)
    if "description" in kwargs and "desc" not in json_schema_extra:
        json_schema_extra["desc"] = kwargs["description"]
    json_schema_extra[DSPIC_FIELD_TYPE] = field_type
    pydantic_kwargs.update(kwargs)
    pydantic_kwargs["json_schema_extra"] = json_schema_extra
    return pydantic_kwargs


def InputField(**kwargs) -> FieldInfo:  # noqa: N802
    """Declare a vision-program input field."""

    return pydantic.Field(**_move_kwargs("input", **kwargs))


def OutputField(**kwargs) -> FieldInfo:  # noqa: N802
    """Declare a vision-program output field."""

    return pydantic.Field(**_move_kwargs("output", **kwargs))

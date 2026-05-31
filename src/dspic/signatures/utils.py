"""Utilities for DSPIC signatures."""

from __future__ import annotations

from typing import Literal

from pydantic.fields import FieldInfo

from dspic.signatures.field import DSPIC_FIELD_TYPE


def get_dspic_field_type(field: FieldInfo) -> Literal["input", "output"]:
    field_type = (field.json_schema_extra or {}).get(DSPIC_FIELD_TYPE)
    if field_type not in {"input", "output"}:
        raise ValueError(f"Field {field} does not have a valid {DSPIC_FIELD_TYPE}.")
    return field_type

"""Typed signatures for DSPIC vision programs."""

from __future__ import annotations

import ast
import inspect
import re
import types
import typing
from collections.abc import Iterator
from copy import deepcopy
from typing import Any

from pydantic import BaseModel, create_model
from pydantic.fields import FieldInfo

from dspic.signatures.field import DSPIC_FIELD_TYPE, InputField, OutputField


def _default_instructions(cls) -> str:
    inputs = ", ".join(f"`{name}`" for name in cls.input_fields)
    outputs = ", ".join(f"`{name}`" for name in cls.output_fields)
    return f"Given {inputs}, produce {outputs}."


class SignatureMeta(type(BaseModel)):
    def __call__(cls, *args, **kwargs):
        if cls is Signature:
            return make_signature(*args, **kwargs)
        return super().__call__(*args, **kwargs)

    def __new__(mcs, signature_name, bases, namespace, **kwargs):
        field_order = [
            name for name, value in namespace.items() if isinstance(value, FieldInfo)
        ]
        annotations = dict(namespace.get("__annotations__", {}))
        for name in field_order:
            annotations.setdefault(name, str)
        namespace["__annotations__"] = annotations

        cls = super().__new__(mcs, signature_name, bases, namespace, **kwargs)
        if cls.__doc__ is None:
            cls.__doc__ = _default_instructions(cls)
        cls._validate_fields()
        for name, field in cls.model_fields.items():
            extra = field.json_schema_extra or {}
            extra.setdefault("prefix", infer_prefix(name) + ":")
            extra.setdefault("desc", f"${{{name}}}")
            field.json_schema_extra = extra
        return cls

    def _validate_fields(cls) -> None:
        if cls.__name__ == "Signature" and not cls.model_fields:
            return
        for name, field in cls.model_fields.items():
            field_type = (field.json_schema_extra or {}).get(DSPIC_FIELD_TYPE)
            if field_type not in {"input", "output"}:
                raise TypeError(
                    f"Field `{name}` in `{cls.__name__}` must use "
                    "InputField() or OutputField()."
                )

    @property
    def instructions(cls) -> str:
        return inspect.cleandoc(getattr(cls, "__doc__", ""))

    @instructions.setter
    def instructions(cls, value: str) -> None:
        cls.__doc__ = value

    @property
    def input_fields(cls) -> dict[str, FieldInfo]:
        return cls._fields_with_type("input")

    @property
    def output_fields(cls) -> dict[str, FieldInfo]:
        return cls._fields_with_type("output")

    @property
    def fields(cls) -> dict[str, FieldInfo]:
        return {**cls.input_fields, **cls.output_fields}

    @property
    def signature(cls) -> str:
        return f"{', '.join(cls.input_fields)} -> {', '.join(cls.output_fields)}"

    def _fields_with_type(cls, field_type: str) -> dict[str, FieldInfo]:
        return {
            name: field
            for name, field in cls.model_fields.items()
            if (field.json_schema_extra or {}).get(DSPIC_FIELD_TYPE) == field_type
        }

    def __repr__(cls) -> str:
        return f"{cls.__name__}({cls.signature})"


class Signature(BaseModel, metaclass=SignatureMeta):
    """Base class for DSPIC typed vision signatures."""

    @classmethod
    def with_instructions(cls, instructions: str) -> type[Signature]:
        return Signature(deepcopy(cls.fields), instructions)

    @classmethod
    def delete(cls, name: str) -> type[Signature]:
        fields = dict(cls.fields)
        fields.pop(name, None)
        return Signature(fields, cls.instructions)

    @classmethod
    def append(
        cls,
        name: str,
        field: FieldInfo,
        type_: type | None = None,
    ) -> type[Signature]:
        fields = dict(cls.fields)
        fields[name] = (type_ or field.annotation or str, field)
        return Signature(fields, cls.instructions)


def ensure_signature(
    signature: str | type[Signature] | None,
    instructions: str | None = None,
) -> type[Signature] | None:
    if signature is None:
        return None
    if isinstance(signature, str):
        return Signature(signature, instructions)
    if instructions is not None:
        raise ValueError("Do not pass instructions with an existing Signature class.")
    return signature


def make_signature(
    signature: str | dict[str, tuple[type, FieldInfo] | FieldInfo],
    instructions: str | None = None,
    signature_name: str = "StringSignature",
    custom_types: dict[str, type] | None = None,
) -> type[Signature]:
    fields = (
        _parse_signature(signature, custom_types)
        if isinstance(signature, str)
        else signature
    )
    fixed_fields: dict[str, tuple[type, FieldInfo]] = {}
    for name, value in fields.items():
        if isinstance(value, FieldInfo):
            fixed_fields[name] = (value.annotation or str, value)
        else:
            type_, field = value
            fixed_fields[name] = (type_ or str, field)
    if instructions is None:
        temp = create_model("TempSignature", __base__=Signature, **fixed_fields)
        instructions = _default_instructions(temp)
    return create_model(
        signature_name,
        __base__=Signature,
        __doc__=instructions,
        **fixed_fields,
    )


def _parse_signature(
    signature: str,
    custom_types: dict[str, type] | None = None,
) -> dict[str, tuple[type, FieldInfo]]:
    if signature.count("->") != 1:
        raise ValueError(f"Invalid signature format: {signature!r}.")
    inputs_str, outputs_str = signature.split("->")
    names = dict(vars(typing))
    names.update(vars(types))
    names.update(
        {
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "list": list,
            "dict": dict,
            "tuple": tuple,
            "set": set,
            "frozenset": frozenset,
        }
    )
    names.update(custom_types or {})
    fields = {}
    for name, type_ in _parse_field_string(inputs_str, names):
        fields[name] = (type_, InputField())
    for name, type_ in _parse_field_string(outputs_str, names):
        if name in fields:
            raise ValueError(f"Duplicate input/output field name: {name!r}.")
        fields[name] = (type_, OutputField())
    return fields


def _parse_field_string(
    field_string: str, names: dict[str, Any]
) -> Iterator[tuple[str, type]]:
    field_string = field_string.strip()
    if not field_string:
        return iter(())
    args = ast.parse(f"def f({field_string}): pass").body[0].args.args
    return ((arg.arg, _parse_annotation(arg.annotation, names)) for arg in args)


def _parse_annotation(node: ast.expr | None, names: dict[str, Any]) -> type:
    if node is None:
        return str
    expr = ast.unparse(node)
    try:
        return eval(expr, names)  # noqa: S307 - controlled type expression parsing.
    except Exception as exc:
        raise ValueError(f"Could not resolve annotation {expr!r}.") from exc


def infer_prefix(attribute_name: str) -> str:
    snake = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", attribute_name)
    snake = re.sub("([a-z0-9])([A-Z])", r"\1_\2", snake)
    snake = re.sub(r"([a-zA-Z])(\d)", r"\1_\2", snake)
    snake = re.sub(r"(\d)([a-zA-Z])", r"\1_\2", snake)
    return " ".join(
        part.upper() if part.isupper() else part.capitalize()
        for part in snake.split("_")
    )

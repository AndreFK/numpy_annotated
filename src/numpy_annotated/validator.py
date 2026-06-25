from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
from pydantic_core import core_schema

from numpy_annotated.shape import Shape_Specification
from numpy_annotated.json_format import (
    DICT_TO_ARRAY_SCHEMA,
    JSON_DICT_TO_ARRAY_SCHEMA,
    decode_json_dict,
)

_PYTHON_SCALAR_TYPES = (int, float, bool, complex)


@dataclass(frozen=True)
class ResolvedDtype:
    """What kind of dtype constraint the user asked for.

    Exactly one mode is active:

    - **any**      — no dtype constraint (``concrete`` and ``abstract`` are None)
    - **concrete** — exact match, e.g. ``np.float64``
    - **abstract** — family match via ``np.issubdtype``, e.g. ``np.floating``
    """

    concrete: np.dtype | None = None
    abstract: type | None = None

    def __post_init__(self) -> None:
        if self.concrete is not None and self.abstract is not None:
            raise ValueError("concrete and abstract dtype cannot both be set")

    @property
    def is_concrete(self) -> bool:
        return self.concrete is not None

    @property
    def is_abstract(self) -> bool:
        return self.abstract is not None

    @property
    def label(self) -> str:
        if self.is_concrete:
            return str(self.concrete)
        if self.is_abstract:
            return self.abstract.__name__
        return "any"


def _as_concrete_dtype(dtype: Any) -> np.dtype:
    """Convert a user dtype hint to a concrete ``np.dtype``, with clear errors."""
    if isinstance(dtype, np.dtype):
        return dtype

    if isinstance(dtype, np.generic):
        dtype = type(dtype)

    if isinstance(dtype, (str, bytes)) or dtype in _PYTHON_SCALAR_TYPES:
        pass
    elif isinstance(dtype, type) and dtype in _PYTHON_SCALAR_TYPES:
        pass
    else:
        raise TypeError(
            f"Invalid dtype {dtype!r}: expected np.float64, np.floating, "
            f"'float64', float, etc.; got {type(dtype).__name__}"
        )

    try:
        return np.dtype(dtype)
    except TypeError as exc:
        if isinstance(dtype, (str, bytes)):
            raise ValueError(f"Invalid dtype {dtype!r}: {exc}") from exc
        raise TypeError(
            f"Invalid dtype {dtype!r}: expected np.float64, np.floating, "
            f"'float64', float, etc.; got {type(dtype).__name__}"
        ) from exc
    except ValueError as exc:
        raise ValueError(f"Invalid dtype {dtype!r}: {exc}") from exc


def _resolve_numpy_generic(dtype: type) -> ResolvedDtype:
    """Tell concrete types (np.float64) apart from abstract ones (np.floating)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            as_dtype = np.dtype(dtype)
        except TypeError:
            return ResolvedDtype(abstract=dtype)

    if as_dtype.type is dtype:
        return ResolvedDtype(concrete=as_dtype)

    return ResolvedDtype(abstract=dtype)


def resolve_dtype(dtype: Any) -> ResolvedDtype:
    """Classify a user dtype hint as concrete, abstract, or any."""
    if dtype is None:
        return ResolvedDtype()

    if isinstance(dtype, type) and issubclass(dtype, np.generic):
        return _resolve_numpy_generic(dtype)

    return ResolvedDtype(concrete=_as_concrete_dtype(dtype))


def validate_shape(arr: np.ndarray, shape: Shape_Specification) -> np.ndarray:
    arr = shape.normalize_scalar(arr)
    shape.validate(arr.shape)
    return arr


def validate_dtype(arr: np.ndarray, resolved: ResolvedDtype) -> None:
    if resolved.is_concrete and arr.dtype != resolved.concrete:
        raise ValueError(
            f"Array dtype must be {resolved.concrete}, but got {arr.dtype}"
        )

    if resolved.is_abstract and not np.issubdtype(arr.dtype, resolved.abstract):
        raise ValueError(
            f"Array dtype must be a subtype of {resolved.label}, but got {arr.dtype}"
        )


def coerce(value: Any, resolved: ResolvedDtype) -> np.ndarray:
    """Coerce input to ndarray. Lists are built with target dtype; ndarrays are cast too."""
    try:
        if resolved.is_concrete:
            return np.asarray(value, dtype=resolved.concrete)
        return np.asarray(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"Could not create array with dtype {resolved.label} from "
            f"{type(value).__name__}: {exc}"
        ) from exc


def _coerce_input(
    value: Any, resolved: ResolvedDtype, *, strict_type: bool
) -> np.ndarray:
    if isinstance(value, dict):
        return decode_json_dict(value)

    if isinstance(value, np.ndarray):
        if strict_type:
            return value
        return coerce(value, resolved)

    if strict_type:
        return np.asarray(value)

    return coerce(value, resolved)


def _apply_constraints(
    arr: np.ndarray,
    shape: Shape_Specification,
    resolved: ResolvedDtype,
) -> np.ndarray:
    arr = validate_shape(arr, shape)
    validate_dtype(arr, resolved)
    return arr


def build_numpy_array_core_schemas(
    *,
    shape: Shape_Specification,
    dtype: Any,
    strict_type: bool = False,
) -> tuple[core_schema.CoreSchema, core_schema.CoreSchema]:
    """Return ``(python_schema, json_schema)`` for ndarray coercion + constraints."""
    resolved = resolve_dtype(dtype)

    def apply_constraints(arr: np.ndarray) -> np.ndarray:
        return _apply_constraints(arr, shape, resolved)

    constraint_step = core_schema.no_info_plain_validator_function(apply_constraints)

    python_input = core_schema.union_schema(
        [
            DICT_TO_ARRAY_SCHEMA,
            core_schema.chain_schema(
                [
                    core_schema.is_instance_schema(
                        (np.ndarray, list, tuple, int, float, complex, bool)
                    ),
                    core_schema.no_info_plain_validator_function(
                        lambda value: _coerce_input(
                            value, resolved, strict_type=strict_type
                        )
                    ),
                ]
            ),
        ]
    )
    python_schema = core_schema.chain_schema([python_input, constraint_step])
    json_schema = core_schema.chain_schema([JSON_DICT_TO_ARRAY_SCHEMA, constraint_step])

    return python_schema, json_schema


def validate_numpy_array(
    value: Any,
    shape: Shape_Specification,
    dtype: Any,
    strict_type: bool = False,
) -> np.ndarray:
    resolved = resolve_dtype(dtype)
    arr = _coerce_input(value, resolved, strict_type=strict_type)
    return _apply_constraints(arr, shape, resolved)

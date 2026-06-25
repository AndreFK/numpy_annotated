from __future__ import annotations

import base64
from typing import Any

import numpy as np
from pydantic_core import core_schema


def _decode_complex_list(data: list[Any], shape: tuple[int, ...]) -> np.ndarray:
    """Rebuild a complex ndarray from nested ``[re, im]`` leaf lists."""
    if len(shape) == 0:
        if not isinstance(data, list) or len(data) != 2:
            raise ValueError("0-d complex data must be a [re, im] pair")
        return np.array(data[0] + 1j * data[1])

    if len(shape) == 1:
        values = [row[0] + 1j * row[1] for row in data]
        return np.array(values)

    return np.array([_decode_complex_list(row, shape[1:]) for row in data])


def _decode_list_data(data: list[Any], dtype: np.dtype, shape: tuple[int, ...]) -> np.ndarray:
    if np.issubdtype(dtype, np.complexfloating):
        arr = _decode_complex_list(data, shape)
        return arr.astype(dtype, copy=False)
    return np.array(data, dtype=dtype).reshape(shape)


def decode_json_dict(value: dict[str, Any]) -> np.ndarray:
    """Decode a JSON dict into an ndarray.

    ``data`` may be a base64 string or a nested list; format is detected automatically.
    """
    dtype = np.dtype(value["dtype"])
    shape = tuple(value["shape"])
    data = value["data"]

    if isinstance(data, str):
        raw = base64.b64decode(data)
        return np.frombuffer(raw, dtype=dtype).reshape(shape).copy()

    if isinstance(data, list):
        return _decode_list_data(data, dtype, shape)

    raise ValueError(
        f"data must be a base64 string or nested list, got {type(data).__name__}"
    )


JSON_DICT_SCHEMA = core_schema.typed_dict_schema(
    {
        "dtype": core_schema.typed_dict_field(core_schema.str_schema(), required=True),
        "shape": core_schema.typed_dict_field(
            core_schema.list_schema(items_schema=core_schema.int_schema(ge=0)),
            required=True,
        ),
        "data": core_schema.typed_dict_field(
            core_schema.union_schema(
                [
                    core_schema.str_schema(),
                    core_schema.list_schema(items_schema=core_schema.any_schema()),
                ]
            ),
            required=True,
        ),
    },
)

JSON_DICT_TO_ARRAY_SCHEMA = core_schema.chain_schema(
    [
        JSON_DICT_SCHEMA,
        core_schema.no_info_plain_validator_function(decode_json_dict),
    ]
)

DICT_TO_ARRAY_SCHEMA = core_schema.chain_schema(
    [
        core_schema.is_instance_schema(dict),
        JSON_DICT_TO_ARRAY_SCHEMA,
    ]
)

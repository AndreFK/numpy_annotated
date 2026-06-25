from typing import Annotated, Any
from pydantic_core import CoreSchema, core_schema
from pydantic import GetCoreSchemaHandler, GetJsonSchemaHandler, BaseModel, create_model
from pydantic.json_schema import JsonSchemaValue
import numpy as np

from numpy_annotated.shape import Shape_Specification
from numpy_annotated.validator import (
    ResolvedDtype,
    build_numpy_array_core_schemas,
    resolve_dtype,
    validate_numpy_array,
)
from numpy_annotated.serializer import serialize_numpy_array


def _json_schema_item_type(resolved: ResolvedDtype) -> dict[str, Any]:
    """Map a resolved dtype constraint to a JSON Schema type for leaf values."""
    if resolved.is_concrete:
        kind = np.dtype(resolved.concrete).type
    elif resolved.is_abstract:
        kind = resolved.abstract
    else:
        return {}

    if issubclass(kind, (np.floating, np.integer)):
        return {"type": "integer" if issubclass(kind, np.integer) else "number"}
    if issubclass(kind, np.bool_):
        return {"type": "boolean"}
    if issubclass(kind, (np.str_, np.bytes_)):
        return {"type": "string"}
    if issubclass(kind, np.complexfloating):
        return {
            "type": "array",
            "prefixItems": [{"type": "number"}, {"type": "number"}],
            "minItems": 2,
            "maxItems": 2,
            "description": "complex number as [re, im]",
        }
    return {}


def _nested_array_schema(item_schema: dict[str, Any], depth: int) -> dict[str, Any]:
    """Build a nested JSON Schema ``array`` for ``depth`` dimensions."""
    if depth <= 0:
        return item_schema
    result: dict[str, Any] = item_schema
    for _ in range(depth):
        result = {"type": "array", "items": result}
    return result


class NumpyAnnotated:
    def __init__(
        self,
        shape: Shape_Specification,
        dtype: Any,
        strict_type: bool = False,
        to_base64: bool = False,
    ):
        self.shape = shape
        self.dtype = dtype
        self.strict_type = strict_type
        self.to_base64 = to_base64
        self.resolved_dtype = resolve_dtype(dtype)

    def _serialize(self, value: np.ndarray) -> dict:
        return serialize_numpy_array(value, self.to_base64)

    def _validate(self, value: Any) -> np.ndarray:
        return validate_numpy_array(value, self.shape, self.dtype, self.strict_type)

    
    def __get_pydantic_core_schema__(
        self, source_type: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:

        python_schema, json_schema = build_numpy_array_core_schemas(
            shape=self.shape,
            dtype=self.dtype,
            strict_type=self.strict_type,
        )

        return core_schema.json_or_python_schema(
            python_schema=python_schema,
            json_schema=json_schema,
            serialization=core_schema.plain_serializer_function_ser_schema(
                self._serialize,
                when_used="json",
            ),
        )

    def __get_pydantic_json_schema__(
        self, schema: CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        label = self.resolved_dtype.label
        shape_desc = str(self.shape.dims)

        dtype_schema: dict[str, Any] = {
            "type": "string",
            "title": "dtype",
            "description": "NumPy dtype string (e.g. '<f8' for float64)",
        }
        if self.resolved_dtype.is_concrete:
            dtype_schema["examples"] = [self.resolved_dtype.concrete.str]

        shape_schema: dict[str, Any] = {
            "type": "array",
            "items": {"type": "integer", "minimum": 0},
            "title": "shape",
            "description": f"Expected rank {len(self.shape.dims)}; spec {shape_desc}",
        }

        list_data_schema = _nested_array_schema(
            _json_schema_item_type(self.resolved_dtype),
            len(self.shape.dims),
        )
        list_data_schema["description"] = "Nested list (tolist()) wire format"

        base64_data_schema: dict[str, Any] = {
            "type": "string",
            "format": "byte",
            "contentEncoding": "base64",
            "description": "Base64-encoded raw array buffer (lossless)",
        }

        if self.to_base64:
            data_schema: dict[str, Any] = {
                "anyOf": [base64_data_schema, list_data_schema],
                "description": "Serialized as base64; nested list also accepted on input",
            }
            encoding_note = "base64 (to_base64=True)"
        else:
            data_schema = {
                "anyOf": [list_data_schema, base64_data_schema],
                "description": "Serialized as nested list; base64 also accepted on input",
            }
            encoding_note = "nested list (to_base64=False)"

        return {
            "type": "object",
            "title": "NDArray",
            "description": (
                f"NumPy ndarray as {{dtype, shape, data}}; "
                f"dtype={label}, shape={shape_desc}, JSON output={encoding_note}"
            ),
            "required": ["dtype", "shape", "data"],
            "properties": {
                "dtype": dtype_schema,
                "shape": shape_schema,
                "data": data_schema,
            },
        }
        
    @staticmethod
    def make_ndarray_type(
        shape: Shape_Specification | tuple[Any, ...],
        dtype: Any,
        *,
        strict_type: bool = False,
        to_base64: bool = False,
    ) -> Any:
        """Build ``Annotated[np.ndarray, NumpyAnnotated(...)]``."""
        if not isinstance(shape, Shape_Specification):
            shape = Shape_Specification(shape)
        return Annotated[
            np.ndarray,
            NumpyAnnotated(
                shape,
                dtype,
                strict_type=strict_type,
                to_base64=to_base64,
            ),
        ]

    @staticmethod
    def make_model(
        model_name: str,
        field_name: str,
        shape: Shape_Specification | tuple[Any, ...],
        dtype: Any,
        *,
        strict_type: bool = False,
        to_base64: bool = False,
    ) -> type[BaseModel]:
        """Create a one-field Pydantic model with a constrained ndarray."""
        return create_model(
            model_name,
            **{
                field_name: (
                    NumpyAnnotated.make_ndarray_type(
                        shape,
                        dtype,
                        strict_type=strict_type,
                        to_base64=to_base64,
                    ),
                    ...,
                )
            },
        )


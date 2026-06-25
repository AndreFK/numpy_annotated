"""Smoke tests for numpy_annotated — run with ``uv run python main.py``.

Each block prints OK lines for passing cases and expected rejections.
Use pytest (``tests/``) for the full photonic/quantum suite.
"""
from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ValidationError

from numpy_annotated import NDArray, make_model


def expect_ok(label: str, fn) -> None:
    """Run fn; print result if no exception."""
    result = fn()
    print(f"OK: {label} -> {result}")


def expect_validation_error(label: str, fn) -> None:
    try:
        fn()
    except ValidationError as exc:
        print(f"OK: {label} rejected -> {exc.errors()[0]['msg']}")
        return
    raise AssertionError(f"expected ValidationError for: {label}")


def expect_type_error(label: str, fn) -> None:
    try:
        fn()
    except TypeError as exc:
        print(f"OK: {label} rejected -> {exc}")
        return
    raise AssertionError(f"expected TypeError for: {label}")


def expect_value_error(label: str, fn) -> None:
    try:
        fn()
    except ValueError as exc:
        print(f"OK: {label} rejected -> {exc}")
        return
    raise AssertionError(f"expected ValueError for: {label}")


def main() -> None:
    # --- Shape + dtype via make_model (image/tensor examples) ---
    RGBImage = make_model("RGBImage", "pixels", (None, None, 3), np.uint8)
    
    RGBImageStrict = make_model(
        "RGBImageStrict", "pixels", (None, None, 3), np.uint8, strict_type=True
    )
    
    RGBImageNamed = make_model("RGBImageNamed", "pixels", ("N", "N", 3), np.uint8)

    # Fixed shape, coercion, strict_type, and named "N" dimensions
    expect_ok(
        "4x4 RGB image",
        lambda: RGBImage(pixels=np.zeros((4, 4, 3), dtype=np.uint8)).pixels.shape,
    )
    expect_validation_error(
        "wrong channel count",
        lambda: RGBImage(pixels=np.zeros((4, 4, 4), dtype=np.uint8)),
    )
    expect_ok(
        "float32 coerced to uint8",
        lambda: RGBImage(pixels=np.zeros((4, 4, 3), dtype=np.float32)).pixels.dtype,
    )
    expect_validation_error(
        "strict_type rejects dtype mismatch before coercion",
        lambda: RGBImageStrict(pixels=np.zeros((4, 4, 3), dtype=np.float32)),
    )
    expect_ok(
        "named dimensions: square RGB image (N, N, 3)",
        lambda: RGBImageNamed(pixels=np.zeros((4, 4, 3), dtype=np.uint8)).pixels.shape,
    )
    expect_validation_error(
        "named dimensions reject non-square image",
        lambda: RGBImageNamed(pixels=np.zeros((4, 5, 3), dtype=np.uint8)),
    )

    # Named dims repeated across axes: ("N", "M", "N") binds N at indices 0 and 2
    TensorNamed = make_model("TensorNamed", "data", ("N", "M", "N"), np.float64)
    expect_ok(
        "multiple named dims (N, M, N): first and last match",
        lambda: TensorNamed(data=np.zeros((2, 7, 2))).data.shape,
    )
    expect_validation_error(
        "multiple named dims (N, M, N): trailing N mismatch",
        lambda: TensorNamed(data=np.zeros((2, 7, 3))),
    )

    # 0-d scalars: () vs (None,) vs ("N",) vs (0,) vs fixed (3,)
    Scalar = make_model("Scalar", "value", (), np.float64)
    Vector = make_model("Vector", "value", (None,), np.float64)
    NamedVector = make_model("NamedVector", "value", ("N",), np.float64)
    EmptyAxis = make_model("EmptyAxis", "value", (0,), np.float64)
    FixedVector = make_model("FixedVector", "value", (3,), np.float64)
    expect_ok(
        "0-d array with empty shape spec ()",
        lambda: (Scalar(value=np.array(2.0)).value.shape, Scalar(value=np.array(2.0)).value.item()),
    )
    expect_ok(
        "0-d scalar promoted for (None,)",
        lambda: Vector(value=np.array(2.0)).value.shape,
    )
    expect_ok(
        "0-d scalar promoted for (N,)",
        lambda: NamedVector(value=np.array(2.0)).value.shape,
    )
    expect_ok(
        "0-d scalar accepted for (0,)",
        lambda: EmptyAxis(value=np.array(2.0)).value.shape,
    )
    expect_validation_error(
        "0-d scalar not accepted for fixed (3,)",
        lambda: FixedVector(value=np.array(2.0)),
    )

    # Invalid dtype hints fail at model construction time
    expect_type_error(
        "invalid dtype hint: list type",
        lambda: make_model("Bad", "x", (2, 2), list),
    )
    expect_type_error(
        "invalid dtype hint: arbitrary object",
        lambda: make_model("Bad", "x", (2, 2), object()),
    )
    expect_value_error(
        "invalid dtype hint: unknown string",
        lambda: make_model("Bad", "x", (2, 2), "not_a_dtype"),
    )
    expect_ok(
        "valid string dtype hint",
        lambda: make_model("Ok", "x", (2, 2), "float64")(x=[[1.0, 2.0], [3.0, 4.0]]).x.dtype,
    )

    # Complex dtypes — concrete complex128 vs abstract np.complexfloating
    Operator = make_model("Operator", "matrix", ("N", "N"), np.complex128)
    Unitary = make_model("Unitary", "matrix", ("N", "N"), np.complexfloating)

    expect_ok(
        "complex128 square operator from list",
        lambda: Operator(matrix=[[1 + 2j, 3j], [0j, 1 - 1j]]).matrix.dtype,
    )
    expect_ok(
        "complexfloating accepts complex128",
        lambda: Unitary(
            matrix=np.array([[1 + 1j, 0j], [0j, 1 - 1j]], dtype=np.complex128)
        ).matrix.dtype,
    )
    expect_ok(
        "complexfloating accepts complex64",
        lambda: Unitary(matrix=np.eye(2, dtype=np.complex64)).matrix.dtype,
    )
    expect_validation_error(
        "complexfloating rejects real float matrix",
        lambda: Unitary(matrix=np.eye(2, dtype=np.float64)),
    )

    # NDArray[...] on a BaseModel field (vs make_model factory)
    class Vector(BaseModel):
        value: NDArray[(3,), np.int64]

    expect_ok(
        "NDArray[shape, dtype] from list",
        lambda: Vector(value=[1, 2, 3]).value.shape,
    )
    expect_ok(
        "NDArray JSON round-trip",
        lambda: np.array_equal(
            Vector.model_validate_json(
                Vector(value=np.array([1, 2, 3])).model_dump_json()
            ).value,
            np.array([1, 2, 3]),
        ),
    )

    class RGBImageModel(BaseModel):
        pixels: NDArray[(None, None, 3), np.uint8]

    expect_ok(
        "NDArray named/wildcard shape",
        lambda: RGBImageModel(
            pixels=np.zeros((2, 2, 3), dtype=np.uint8)
        ).pixels.shape,
    )

    # OpenAPI: NDArray fields describe the {dtype, shape, data} wire object
    schema = Vector.model_json_schema()["properties"]["value"]

    expect_ok(
        "OpenAPI JSON schema for NDArray field",
        lambda: (
            schema["type"],
            schema["required"],
            "dtype" in schema["properties"],
        ),
    )


if __name__ == "__main__":
    main()
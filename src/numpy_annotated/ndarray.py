from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from numpy_annotated.annotation import NumpyAnnotated
from numpy_annotated.shape import Shape_Specification


@dataclass(frozen=True)
class NDArrayConfig:
    """Flags for ``NDArray[shape, dtype, config]``."""

    strict_type: bool = False
    to_base64: bool = False


def _is_dtype_hint(item: Any) -> bool:
    if item is None:
        return True
    if isinstance(item, (np.dtype, str, bytes)):
        return True
    if isinstance(item, type) and issubclass(item, (np.generic, int, float, bool, complex)):
        return True
    return False


class NDArray:
    """Parametrizable annotation: ``NDArray[shape, dtype]`` or with flags via config.

    Examples::

        NDArray[(3, 4), np.float64]
        NDArray[("N", "N"), np.complex128]
        NDArray[((None,), np.complex128, NDArrayConfig(to_base64=True))]
        NDArray[(("N", "N"), np.complex128, NDArrayConfig(strict_type=True))]
    """

    def __class_getitem__(cls, item: Any) -> Any:
        shape_part: Any = None
        dtype_part: Any = None
        config = NDArrayConfig()

        if isinstance(item, tuple):
            if len(item) == 3:
                shape_part, dtype_part, third = item
                if not isinstance(third, NDArrayConfig):
                    raise TypeError(
                        "NDArray[shape, dtype, config] requires NDArrayConfig "
                        "as the third element"
                    )
                config = third
            elif len(item) == 2:
                shape_part, dtype_part = item
            elif len(item) == 1:
                shape_part = item[0]
            else:
                raise TypeError(
                    "NDArray expects NDArray[shape, dtype], NDArray[shape, dtype, config], "
                    f"or NDArray[(shape, dtype)]; got {len(item)} subscript arguments"
                )
        elif isinstance(item, Shape_Specification):
            shape_part = item
        elif _is_dtype_hint(item):
            raise TypeError(
                "NDArray requires an explicit shape tuple; use NDArray[shape, dtype]"
            )
        else:
            shape_part = item

        if shape_part is None:
            raise TypeError("NDArray requires a shape tuple; use NDArray[shape, dtype]")
        if dtype_part is None:
            raise TypeError("NDArray requires a dtype; use NDArray[shape, dtype]")

        return NumpyAnnotated.make_ndarray_type(
            shape_part,
            dtype_part,
            strict_type=config.strict_type,
            to_base64=config.to_base64,
        )

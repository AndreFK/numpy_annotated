from typing import Any

import numpy as np

class Shape_Specification:
    """
    specification for numpy arrays
    
    Args:
        dims: tuple of dimensions
        
    Returns:
        None
        
    Raises:
        ValueError: if the shape is invalid
        
    Accepted dimensions:
        - int: expected size
        - str: dimension name
        - None: wildcard
        
    Examples:
        >>> shape_spec = Shape_Specification((1, 2, 3))
        >>> shape_spec.validate((1, 2, 3))
        >>> shape_spec = Shape_Specification(("N", "N", 3))
        >>> shape_spec.validate((1, 1, 3))
        >>> shape_spec = Shape_Specification((1, None, 3))
        >>> shape_spec.validate((1, 10, 3))
    """
    
    def __init__(self, dims: tuple[Any, ...]):
        self.dims = dims

    def normalize_scalar(self, arr: np.ndarray) -> np.ndarray:
        """Promote 0-d arrays when the spec is a single flexible or zero-size axis."""
        if arr.ndim != 0 or len(self.dims) != 1:
            return arr
        dim = self.dims[0]
        if dim is None or isinstance(dim, str):
            return np.atleast_1d(arr)
        return arr

    def validate(self, shape: tuple[Any, ...]) -> None:
        """Validate the shape of the array"""
        if len(shape) == 0 and len(self.dims) == 1 and self.dims[0] == 0:
            return

        if len(shape) != len(self.dims):
            raise ValueError(f"Shape must have {len(self.dims)} dimensions, but got {len(shape)}")
        
        bindings: dict[str, int] = {}
        for i, (dim, actual_size) in enumerate(zip(self.dims, shape)):
            if isinstance(dim, int):
                if dim != actual_size:
                    raise ValueError(
                        f"Dimension index {i}: expected size {dim}, got {actual_size}"
                    )

            elif dim is None:
                continue

            elif isinstance(dim, str):
                if dim in bindings and bindings[dim] != actual_size:
                    raise ValueError(
                        f"Dimension index {i}: dimension '{dim}' was {bindings[dim]} "
                        f"but also {actual_size}. Once dimension is bound, it cannot be bound again"
                    )
                bindings[dim] = actual_size
            else:
                raise TypeError(
                    f"axis {i}: invalid spec entry {dim!r} ({type(dim).__name__})"
                )
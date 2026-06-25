import base64
from typing import Any

import numpy as np


def _tolist_json_safe(data: Any) -> Any:
    """Convert ndarray.tolist() output to JSON-serializable values."""
    if isinstance(data, complex):
        return [data.real, data.imag]
    if isinstance(data, list):
        return [_tolist_json_safe(item) for item in data]
    return data


def serialize_numpy_array(arr: np.ndarray, to_base64: bool = False) -> dict:
    payload = {
        "dtype": arr.dtype.str,
        "shape": list(arr.shape),
    }
    if to_base64:
        buffer = np.ascontiguousarray(arr).tobytes()
        payload["data"] = base64.b64encode(buffer).decode("ascii")
    else:
        payload["data"] = _tolist_json_safe(arr.tolist())
    return payload

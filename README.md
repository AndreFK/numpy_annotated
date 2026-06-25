# numpy_annotated

Pydantic v2 type annotations for `numpy.ndarray` with **dtype validation**, **shape validation**, and **JSON round-trip serialization**.

Use it when API models need to accept, validate, and serialize numerical arrays — including **complex** state vectors and unitaries common in quantum and photonic simulation.

## Install

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/) (or pip).

```bash
uv sync
# optional extras
uv sync --extra dev        # pytest
uv sync --extra notebook   # jupyter, pennylane
```

## Quick example

```python
import numpy as np
from pydantic import BaseModel

from numpy_annotated import NDArray, NDArrayConfig

class StateVector(BaseModel):
    amplitudes: NDArray[(4,), np.complex128]

class Unitary(BaseModel):
    matrix: NDArray[
        (("N", "N"), np.complex128, NDArrayConfig(to_base64=True))
    ]

state = StateVector(amplitudes=np.array([1, 0, 0, 1], dtype=np.complex128) / np.sqrt(2))
json_str = state.model_dump_json()
restored = StateVector.model_validate_json(json_str)
```

## Features

| Concern | Behavior |
| --- | --- |
| **Definition** | `NDArray[shape, dtype]` expands to `Annotated[np.ndarray, ...]` with a Pydantic core schema |
| **dtype** | Concrete dtypes (`np.float64`, `np.complex128`) or abstract families (`np.floating`, `np.complexfloating`) via `np.issubdtype` |
| **shape** | Fixed sizes, `None` wildcards, and named dimensions (`"N"`, `"N"`) that must match across axes |
| **coercion** | Non-strict mode coerces list/tuple input with `np.asarray(..., dtype=...)`; `strict_type=True` rejects dtype mismatches |
| **JSON out** | `{dtype, shape, data}` — nested lists by default, base64 raw buffer when `to_base64=True` |
| **JSON in** | Accepts both list and base64 `data` regardless of the serialization flag |
| **OpenAPI** | `__get_pydantic_json_schema__` describes the wire object |

### Shape specification

Shapes are tuple-based:

```python
NDArray[(3, 4), np.float64]           # exactly 3×4
NDArray[(None,), np.float64]          # any 1-D length
NDArray[("N", "N"), np.complex128]    # square N×N
NDArray[(None, None, 3), np.uint8]    # H×W×3 (e.g. RGB)
```

### Serialization flags

```python
NDArray[shape, dtype]                                          # default: nested list
NDArray[(shape, dtype, NDArrayConfig(strict_type=True))]       # no dtype coercion
NDArray[(shape, dtype, NDArrayConfig(to_base64=True))]         # base64 wire format
```

**List format** — human-readable; complex entries encode as `[re, im]` pairs (JSON has no native complex type).

**Base64 format** — compact and bit-exact; suited to large `complex128` matrices (e.g. interferometer unitaries).

### Factory helper

For one-field demo models:

```python
from numpy_annotated import make_model

RGBImage = make_model("RGBImage", "pixels", (None, None, 3), np.uint8)
```

## JSON wire format

```json
{
  "dtype": "<c16",
  "shape": [2, 2],
  "data": [[[0.0, 0.0], [0.0, -1.0]], [[0.0, 1.0], [0.0, 0.0]]]
}
```

With `to_base64=True`, `data` is a base64 string of the contiguous array buffer.

## Run

```bash
uv run python main.py          # interactive validation smoke tests
uv run pytest -q               # test suite (photonic / quantum scenarios)
uv run jupyter notebook notebooks/pennylane_integration.ipynb
```

Open `notebooks/pennylane_integration.ipynb` in Cursor/VS Code and select the project `.venv` kernel (see notebook setup cell).

## Project layout

```
src/numpy_annotated/
  annotation.py    # Pydantic hooks + OpenAPI schema
  ndarray.py       # NDArray[...] + NDArrayConfig
  shape.py         # Shape_Specification DSL
  validator.py     # dtype/shape validation + coercion
  serializer.py    # list / base64 encoding
  json_format.py   # decode {dtype, shape, data}
main.py            # runnable demos
tests/             # pytest (Fock states, beam splitters, JSON round-trip)
notebooks/         # PennyLane integration walkthrough
```

## Related libraries

- **[numpydantic](https://github.com/p2p-ld/numpydantic)** — multi-backend arrays (NumPy, zarr, HDF5, …), rich shape syntax, mypy integration.
- **[pydantic-numpy](https://github.com/caniko/pydantic-numpy)** — ndarray typing and file I/O helpers.

This package is intentionally small and NumPy-only, with first-class support for **complex JSON round-trip** (list and base64), which those libraries do not handle out of the box.

## Future work

Static type checker integration (e.g. **mypy** and **Pyright**) for `NDArray[shape, dtype]` may be added later — similar to what [numpydantic](https://github.com/p2p-ld/numpydantic) provides today. Today, fields are typed as `np.ndarray` via `Annotated[...]`; runtime validation is fully supported.

## License

See project metadata in `pyproject.toml`.

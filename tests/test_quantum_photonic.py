"""Tests for numpy_annotated in photonic quantum computing scenarios.

Photon-based platforms often represent:

- **Fock states** — complex amplitudes in the occupation-number basis
- **Unitaries** — interferometers, beam splitters, and linear optical networks
- **Batch circuits** — many input states transformed by the same unitary

Run with::

    uv run pytest -q
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from pydantic import BaseModel, ValidationError

from numpy_annotated import NDArray, NDArrayConfig


# --------------------------------------------------------------------------- #
# Fock-basis state vectors
# --------------------------------------------------------------------------- #


class FockState(BaseModel):
    """Single-mode or multi-mode state vector (cutoff-dependent length)."""

    amplitudes: NDArray[(None,), np.complex128]


def _normalized_single_photon_superposition() -> np.ndarray:
    """|psi> = (|0> + |1>) / sqrt(2) in a single-mode Fock truncation."""
    return np.array([1 / np.sqrt(2), 1 / np.sqrt(2)], dtype=np.complex128)


def test_fock_state_accepts_complex128_superposition():
    state = FockState(amplitudes=_normalized_single_photon_superposition())
    assert state.amplitudes.dtype == np.complex128
    assert state.amplitudes.shape == (2,)
    assert np.allclose(np.linalg.norm(state.amplitudes), 1.0)


def test_fock_state_accepts_dual_rail_encoding():
    """Dual-rail: one photon in two modes -> |1,0> or |0,1> basis element."""
    one_photon_mode_a = np.array([0, 1, 0, 0], dtype=np.complex128)
    parsed = FockState(amplitudes=one_photon_mode_a)
    assert parsed.amplitudes[1] == 1


def test_fock_state_coerces_real_to_complex128_when_non_strict():
    """Non-strict mode builds complex128 from a real vector (common for list input)."""
    parsed = FockState(amplitudes=[1.0, 0.0])
    assert parsed.amplitudes.dtype == np.complex128


def test_fock_state_strict_rejects_float64_array():


    class StrictFockState(BaseModel):
        amplitudes: NDArray[
            ((None,), np.complex128, NDArrayConfig(strict_type=True))
        ]

    with pytest.raises(ValidationError):
        StrictFockState(amplitudes=np.array([1.0, 0.0], dtype=np.float64))


def test_fock_state_json_round_trip_list_format():
    original = FockState(amplitudes=_normalized_single_photon_superposition())
    restored = FockState.model_validate_json(original.model_dump_json())
    assert np.allclose(restored.amplitudes, original.amplitudes)


def test_fock_state_json_round_trip_base64():
    class FockStateB64(BaseModel):
        amplitudes: NDArray[
            ((None,), np.complex128, NDArrayConfig(to_base64=True))
        ]

    arr = _normalized_single_photon_superposition()
    original = FockStateB64(amplitudes=arr)
    payload = json.loads(original.model_dump_json())
    assert isinstance(payload["amplitudes"]["data"], str)

    restored = FockStateB64.model_validate_json(original.model_dump_json())
    assert np.array_equal(restored.amplitudes, arr)


# --------------------------------------------------------------------------- #
# Linear optical unitaries
# --------------------------------------------------------------------------- #


class BeamSplitter(BaseModel):
    """2x2 unitary describing a 50:50 beam splitter with phase."""

    unitary: NDArray[("N", "N"), np.complex128]


def _fifty_fifty_beam_splitter() -> np.ndarray:
    """Standard 50:50 beam splitter with i phase (common in photonic circuits)."""
    return (1 / np.sqrt(2)) * np.array(
        [[1, 1j], [1j, 1]],
        dtype=np.complex128,
    )


def test_beam_splitter_accepts_square_unitary():
    u = BeamSplitter(unitary=_fifty_fifty_beam_splitter())
    assert u.unitary.shape == (2, 2)
    assert np.allclose(u.unitary @ u.unitary.conj().T, np.eye(2), atol=1e-10)


def test_beam_splitter_rejects_non_square_matrix():
    with pytest.raises(ValidationError):
        BeamSplitter(
            unitary=np.array([[1, 0, 0], [0, 1, 0]], dtype=np.complex128)
        )


class LinearOpticalNetwork(BaseModel):
    """N-mode unitary; accepts complex64 or complex128 (abstract complexfloating)."""

    matrix: NDArray[("N", "N"), np.complexfloating]


def test_linear_network_accepts_complex64_and_complex128():
    eye64 = np.eye(3, dtype=np.complex64)
    eye128 = np.eye(3, dtype=np.complex128)
    assert LinearOpticalNetwork(matrix=eye64).matrix.dtype == np.complex64
    assert LinearOpticalNetwork(matrix=eye128).matrix.dtype == np.complex128


def test_linear_network_rejects_real_matrix():
    with pytest.raises(ValidationError):
        LinearOpticalNetwork(matrix=np.eye(2, dtype=np.float64))


def test_beam_splitter_json_round_trip_list_format():
    """2×2 complex unitaries round-trip through nested [re, im] JSON lists."""
    u = _fifty_fifty_beam_splitter()
    original = BeamSplitter(unitary=u)
    payload = json.loads(original.model_dump_json())["unitary"]

    assert payload["shape"] == [2, 2]
    assert isinstance(payload["data"], list)
    assert payload["data"][0][0] == [1 / np.sqrt(2), 0.0]
    assert payload["data"][0][1] == [0.0, 1 / np.sqrt(2)]  # 1j

    restored = BeamSplitter.model_validate_json(original.model_dump_json())
    assert np.allclose(restored.unitary, u)
    assert np.allclose(restored.unitary.imag, u.imag)

    json_again = restored.model_dump_json()
    assert json_again == original.model_dump_json()
    assert np.allclose(
        BeamSplitter.model_validate_json(json_again).unitary, u
    )


def test_unitary_base64_json_preserves_complex_exactly():

    class U(BaseModel):
        matrix: NDArray[
            (("N", "N"), np.complex128, NDArrayConfig(to_base64=True))
        ]

    u = _fifty_fifty_beam_splitter()
    original = U(matrix=u)
    restored = U.model_validate_json(original.model_dump_json())
    assert np.array_equal(restored.matrix, u)


def test_large_interferometer_unitary_base64_round_trip():
    """Large N×N unitaries stay compact and bit-exact through base64 JSON."""
    n = 512
    rng = np.random.default_rng(0)
    # Random complex128 matrix — proxy for a large linear optical network
    u = (
        rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    ).astype(np.complex128)

    class LargeUnitary(BaseModel):
        matrix: NDArray[
            (("N", "N"), np.complex128, NDArrayConfig(to_base64=True))
        ]

    original = LargeUnitary(matrix=u)
    json_str = original.model_dump_json()
    payload = json.loads(json_str)["matrix"]

    assert payload["shape"] == [n, n]
    assert isinstance(payload["data"], str)
    assert u.nbytes == n * n * 16  # 512×512 complex128 ≈ 4 MiB

    # base64 expands raw bytes by ~4/3; nested-list JSON is impractical for complex
    b64_len = len(payload["data"])
    assert (4 * u.nbytes) // 3 - 10 <= b64_len <= (4 * u.nbytes) // 3 + 10

    restored = LargeUnitary.model_validate_json(json_str)
    assert restored.matrix.shape == (n, n)
    assert np.array_equal(restored.matrix, u)


# --------------------------------------------------------------------------- #
# Batched Fock states (e.g. many shots / parallel inputs)
# --------------------------------------------------------------------------- #


class FockBatch(BaseModel):
    """Batch of state vectors: (num_states, cutoff_dim)."""

    states: NDArray[(None, None), np.complex128]


def test_fock_batch_accepts_multiple_input_states():
    # Three different single-photon inputs in a 4-dim Fock truncation
    batch = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ],
        dtype=np.complex128,
    )
    parsed = FockBatch(states=batch)
    assert parsed.states.shape == (3, 4)


def test_fock_batch_rejects_wrong_rank():
    with pytest.raises(ValidationError):
        FockBatch(states=_normalized_single_photon_superposition())


# --------------------------------------------------------------------------- #
# OpenAPI schema smoke test
# --------------------------------------------------------------------------- #


def test_fock_state_openapi_schema_describes_wire_format():
    schema = FockState.model_json_schema()["properties"]["amplitudes"]
    assert schema["type"] == "object"
    assert set(schema["required"]) == {"dtype", "shape", "data"}
    assert "dtype" in schema["properties"]

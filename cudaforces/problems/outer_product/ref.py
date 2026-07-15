"""Reference for outer product: out[i, j] = a[i] * b[j]."""

from typing import Any

import numpy as np
import numpy.typing as npt

from ..types import RefCase

FloatArray = npt.NDArray[np.float32]

RTOL = 1e-5
ATOL = 1e-6


def tests() -> list[RefCase]:
    cases = [
        RefCase(
            {
                "M": 2,
                "N": 3,
                "a": np.array([2.0, 3.0], dtype=np.float32),
                "b": np.array([4.0, 5.0, 6.0], dtype=np.float32),
            }
        ),
        RefCase(
            {
                "M": 1,
                "N": 1,
                "a": np.array([0.0], dtype=np.float32),
                "b": np.array([-8.0], dtype=np.float32),
            }
        ),
        RefCase(
            {
                "M": 3,
                "N": 4,
                "a": np.array([1.0, -2.0, 0.25], dtype=np.float32),
                "b": np.array([0.0, 1.0, -1.0, 2.0], dtype=np.float32),
            }
        ),
    ]
    rng = np.random.default_rng(131)
    cases.append(
        RefCase(
            {
                "M": 257,
                "N": 509,
                "a": rng.standard_normal(257).astype(np.float32),
                "b": rng.standard_normal(509).astype(np.float32),
            }
        )
    )
    return cases


def solve(inputs: dict[str, Any]) -> list[FloatArray]:
    a: FloatArray = inputs["a"].astype(np.float32)
    b: FloatArray = inputs["b"].astype(np.float32)
    out: FloatArray = np.multiply.outer(a, b).astype(np.float32)
    return [out]

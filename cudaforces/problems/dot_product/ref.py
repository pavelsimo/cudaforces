"""Reference for dot-product: out[0] = sum_i a[i] * b[i]."""

from typing import Any

import numpy as np
import numpy.typing as npt

from ..types import RefCase

FloatArray = npt.NDArray[np.float32]

RTOL = 1e-3
ATOL = 1e-2


def tests() -> list[RefCase]:
    rng = np.random.default_rng(202)
    signs = np.where(np.arange(4096) % 2 == 0, 1.0, -1.0).astype(np.float32)
    return [
        RefCase(
            {
                "N": 3,
                "a": np.array([1.0, 2.0, 3.0], dtype=np.float32),
                "b": np.array([4.0, 5.0, 6.0], dtype=np.float32),
            }
        ),
        RefCase(
            {
                "N": 1,
                "a": np.array([-3.25], dtype=np.float32),
                "b": np.array([2.0], dtype=np.float32),
            }
        ),
        RefCase(
            {
                "N": signs.size,
                "a": signs,
                "b": np.linspace(0.5, 1.5, signs.size, dtype=np.float32),
            }
        ),
        RefCase(
            {
                "N": 100_003,
                "a": rng.standard_normal(100_003).astype(np.float32),
                "b": rng.standard_normal(100_003).astype(np.float32),
            }
        ),
    ]


def solve(inputs: dict[str, Any]) -> list[FloatArray]:
    a: FloatArray = inputs["a"]
    b: FloatArray = inputs["b"]
    result = np.sum(a * b, dtype=np.float32)
    return [np.array([result], dtype=np.float32)]

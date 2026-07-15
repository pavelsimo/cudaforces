"""Reference for scatter-add into a zero-initialized output."""

from typing import Any

import numpy as np
import numpy.typing as npt

from ..types import RefCase

FloatArray = npt.NDArray[np.float32]
IntArray = npt.NDArray[np.int32]

RTOL = 1e-3
ATOL = 1e-2


def tests() -> list[RefCase]:
    rng = np.random.default_rng(909)
    skewed_indices = np.zeros(4096, dtype=np.int32)
    skewed_indices[::13] = np.arange(skewed_indices[::13].size, dtype=np.int32) % 33
    return [
        RefCase(
            {
                "N": 4,
                "M": 3,
                "values": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
                "indices": np.array([0, 2, 0, 2], dtype=np.int32),
            }
        ),
        RefCase(
            {
                "N": 1,
                "M": 4,
                "values": np.array([-2.25], dtype=np.float32),
                "indices": np.array([2], dtype=np.int32),
            }
        ),
        RefCase(
            {
                "N": skewed_indices.size,
                "M": 33,
                "values": rng.standard_normal(skewed_indices.size).astype(np.float32),
                "indices": skewed_indices,
            }
        ),
        RefCase(
            {
                "N": 100_003,
                "M": 2048,
                "values": rng.standard_normal(100_003).astype(np.float32),
                "indices": rng.integers(0, 2048, size=100_003, dtype=np.int32),
            }
        ),
    ]


def solve(inputs: dict[str, Any]) -> list[FloatArray]:
    output_size = inputs["M"]
    values: FloatArray = inputs["values"]
    indices: IntArray = inputs["indices"]
    out = np.zeros(output_size, dtype=np.float32)
    np.add.at(out, indices, values)
    return [out]

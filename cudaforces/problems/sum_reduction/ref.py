"""Reference for sum-reduction: out[0] = sum_i inp[i]."""

from typing import Any

import numpy as np
import numpy.typing as npt

from ..types import RefCase

FloatArray = npt.NDArray[np.float32]

# Parallel reductions may add values in a different order from NumPy's fp32
# reduction, particularly for cancellation-heavy inputs.
RTOL = 1e-3
ATOL = 1e-2


def tests() -> list[RefCase]:
    rng = np.random.default_rng(101)
    cancellation = np.tile(np.array([1.0, -1.0, 0.125, -0.125], dtype=np.float32), 1024)
    return [
        RefCase(
            {
                "N": 5,
                "inp": np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32),
            }
        ),
        RefCase({"N": 1, "inp": np.array([-7.25], dtype=np.float32)}),
        RefCase({"N": cancellation.size, "inp": cancellation}),
        RefCase({"N": 100_003, "inp": rng.standard_normal(100_003).astype(np.float32)}),
    ]


def solve(inputs: dict[str, Any]) -> list[FloatArray]:
    inp: FloatArray = inputs["inp"]
    total = np.sum(inp, dtype=np.float32)
    return [np.array([total], dtype=np.float32)]

"""Reference for stable stream compaction retaining strictly positive values."""

from typing import Any

import numpy as np
import numpy.typing as npt

from ..types import RefCase

FloatArray = npt.NDArray[np.float32]
IntArray = npt.NDArray[np.int32]

RTOL = 0.0
ATOL = 0.0


def tests() -> list[RefCase]:
    rng = np.random.default_rng(707)
    random_values = rng.standard_normal(100_003).astype(np.float32)
    random_values[::19] = np.float32(0.0)
    return [
        RefCase(
            {
                "N": 6,
                "inp": np.array([-2.0, 3.0, 0.0, 4.0, -1.0, 5.0], dtype=np.float32),
            }
        ),
        RefCase(
            {
                "N": 5,
                "inp": np.array([0.0, -1.0, -2.0, 0.0, -0.5], dtype=np.float32),
            }
        ),
        RefCase({"N": 1, "inp": np.array([6.25], dtype=np.float32)}),
        RefCase({"N": random_values.size, "inp": random_values}),
    ]


def solve(inputs: dict[str, Any]) -> list[FloatArray | IntArray]:
    inp: FloatArray = inputs["inp"]
    out = inp[inp > np.float32(0.0)].copy()
    count = np.array([out.size], dtype=np.int32)
    return [count, out]

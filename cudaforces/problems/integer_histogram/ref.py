"""Reference for a histogram over integer inputs in [0, B)."""

from typing import Any

import numpy as np
import numpy.typing as npt

from ..types import RefCase

IntArray = npt.NDArray[np.int32]

RTOL = 0.0
ATOL = 0.0


def tests() -> list[RefCase]:
    rng = np.random.default_rng(808)
    skewed = np.zeros(4096, dtype=np.int32)
    skewed[::17] = np.arange(skewed[::17].size, dtype=np.int32) % 17
    return [
        RefCase(
            {
                "N": 6,
                "B": 4,
                "inp": np.array([1, 0, 1, 3, 1, 0], dtype=np.int32),
            }
        ),
        RefCase({"N": 1, "B": 1, "inp": np.array([0], dtype=np.int32)}),
        RefCase({"N": skewed.size, "B": 17, "inp": skewed}),
        RefCase(
            {
                "N": 100_003,
                "B": 257,
                "inp": rng.integers(0, 257, size=100_003, dtype=np.int32),
            }
        ),
    ]


def solve(inputs: dict[str, Any]) -> list[IntArray]:
    inp: IntArray = inputs["inp"]
    bin_count = inputs["B"]
    bins = np.bincount(inp, minlength=bin_count).astype(np.int32)
    return [bins]

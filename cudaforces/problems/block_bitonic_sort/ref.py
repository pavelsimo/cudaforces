"""Reference for block bitonic sort: ascending sort of one power-of-two block."""

from typing import Any

import numpy as np
import numpy.typing as npt

from ..types import RefCase

IntArray = npt.NDArray[np.int32]

RTOL = 0.0
ATOL = 0.0


def tests() -> list[RefCase]:
    cases = [
        RefCase(
            {
                "N": 8,
                "inp": np.array([7, 2, 9, 1, 5, 3, 8, 4], dtype=np.int32),
            }
        ),
        RefCase({"N": 1, "inp": np.array([42], dtype=np.int32)}),
        RefCase(
            {
                "N": 16,
                "inp": np.array([3, 3, -1, 8, 0, -1, 8, 2, 7, 6, 5, 4, 3, 2, 1, 0], dtype=np.int32),
            }
        ),
    ]
    rng = np.random.default_rng(103)
    cases.append(
        RefCase(
            {
                "N": 1_024,
                "inp": rng.integers(-1_000_000, 1_000_001, size=1_024, dtype=np.int32),
            }
        )
    )
    return cases


def solve(inputs: dict[str, Any]) -> list[IntArray]:
    out: IntArray = np.sort(inputs["inp"].astype(np.int32))
    return [out]

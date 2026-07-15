"""Reference for row-wise-sum: reduce each row of an R by C matrix."""

from typing import Any

import numpy as np
import numpy.typing as npt

from ..types import RefCase

FloatArray = npt.NDArray[np.float32]

RTOL = 1e-3
ATOL = 1e-3


def tests() -> list[RefCase]:
    rng = np.random.default_rng(303)
    return [
        RefCase(
            {
                "R": 2,
                "C": 3,
                "inp": np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32),
            }
        ),
        RefCase({"R": 1, "C": 1, "inp": np.array([-9.5], dtype=np.float32)}),
        RefCase(
            {
                "R": 17,
                "C": 31,
                "inp": rng.standard_normal(17 * 31).astype(np.float32),
            }
        ),
        RefCase(
            {
                "R": 128,
                "C": 1024,
                "inp": rng.standard_normal(128 * 1024).astype(np.float32),
            }
        ),
    ]


def solve(inputs: dict[str, Any]) -> list[FloatArray]:
    rows, cols = inputs["R"], inputs["C"]
    inp: FloatArray = inputs["inp"]
    out = np.sum(inp.reshape(rows, cols), axis=1, dtype=np.float32)
    return [out.astype(np.float32)]

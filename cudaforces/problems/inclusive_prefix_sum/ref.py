"""Reference for inclusive-prefix-sum."""

from typing import Any

import numpy as np
import numpy.typing as npt

from ..types import RefCase

FloatArray = npt.NDArray[np.float32]

# A parallel scan has a different association tree from sequential cumsum.
RTOL = 2e-3
ATOL = 1e-2


def tests() -> list[RefCase]:
    rng = np.random.default_rng(505)
    cancellation = np.tile(np.array([1.0, -1.0, 0.25, -0.25], dtype=np.float32), 1024)
    return [
        RefCase(
            {
                "N": 5,
                "inp": np.array([3.0, 1.0, 4.0, 1.0, 5.0], dtype=np.float32),
            }
        ),
        RefCase({"N": 1, "inp": np.array([-2.5], dtype=np.float32)}),
        RefCase({"N": cancellation.size, "inp": cancellation}),
        RefCase(
            {
                "N": 100_003,
                "inp": (0.25 * rng.standard_normal(100_003)).astype(np.float32),
            }
        ),
    ]


def solve(inputs: dict[str, Any]) -> list[FloatArray]:
    inp: FloatArray = inputs["inp"]
    return [np.cumsum(inp, dtype=np.float32)]

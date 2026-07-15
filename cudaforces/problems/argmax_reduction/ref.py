"""Reference for argmax-reduction with the lowest index winning ties."""

from typing import Any

import numpy as np
import numpy.typing as npt

from ..types import RefCase

FloatArray = npt.NDArray[np.float32]
IntArray = npt.NDArray[np.int32]

RTOL = 0.0
ATOL = 0.0


def tests() -> list[RefCase]:
    rng = np.random.default_rng(404)
    tied_random = rng.standard_normal(100_003).astype(np.float32)
    tied_random[123] = np.float32(50.0)
    tied_random[99_999] = np.float32(50.0)
    return [
        RefCase(
            {
                "N": 5,
                "inp": np.array([3.0, 9.0, 2.0, 9.0, 1.0], dtype=np.float32),
            }
        ),
        RefCase({"N": 1, "inp": np.array([-12.75], dtype=np.float32)}),
        RefCase({"N": 33, "inp": np.full(33, 4.5, dtype=np.float32)}),
        RefCase({"N": tied_random.size, "inp": tied_random}),
    ]


def solve(inputs: dict[str, Any]) -> list[FloatArray | IntArray]:
    inp: FloatArray = inputs["inp"]
    index = int(np.argmax(inp))
    return [
        np.array([inp[index]], dtype=np.float32),
        np.array([index], dtype=np.int32),
    ]

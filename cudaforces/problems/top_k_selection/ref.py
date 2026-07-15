"""Reference for top-k selection with deterministic index tie-breaking."""

from typing import Any

import numpy as np
import numpy.typing as npt

from ..types import RefCase

FloatArray = npt.NDArray[np.float32]
IntArray = npt.NDArray[np.int32]

RTOL = 0.0
ATOL = 0.0


def tests() -> list[RefCase]:
    cases = [
        RefCase(
            {
                "N": 5,
                "K": 3,
                "inp": np.array([4.0, 9.0, 2.0, 9.0, 7.0], dtype=np.float32),
            }
        ),
        RefCase({"N": 1, "K": 1, "inp": np.array([-7.25], dtype=np.float32)}),
        RefCase(
            {
                "N": 8,
                "K": 8,
                "inp": np.array([2.0, 2.0, -1.0, 4.0, 4.0, 4.0, 0.0, -1.0], dtype=np.float32),
            }
        ),
    ]
    rng = np.random.default_rng(109)
    n = 100_003
    inp = rng.integers(-100, 101, size=n).astype(np.float32)
    cases.append(RefCase({"N": n, "K": 257, "inp": inp}))
    return cases


def solve(inputs: dict[str, Any]) -> list[FloatArray | IntArray]:
    inp: FloatArray = inputs["inp"].astype(np.float32)
    k = int(inputs["K"])
    indices = np.arange(inp.size, dtype=np.int32)
    order = np.lexsort((indices, -inp))[:k]
    selected_indices = order.astype(np.int32)
    selected_values: FloatArray = inp[selected_indices]
    return [selected_values, selected_indices]

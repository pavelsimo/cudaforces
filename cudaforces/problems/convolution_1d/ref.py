"""Reference for same-size zero-padded 1D cross-correlation."""

from typing import Any

import numpy as np
import numpy.typing as npt

from ..types import RefCase

FloatArray = npt.NDArray[np.float32]

RTOL = 1e-4
ATOL = 1e-5

NINE_KERNEL = np.array([0.25, -0.5, 1.0, 0.0, 2.0, 0.0, -1.0, 0.5, -0.25], dtype=np.float32)


def tests() -> list[RefCase]:
    cases = [
        RefCase(
            {
                "N": 4,
                "K": 3,
                "inp": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
                "kernel": np.array([1.0, 0.0, -1.0], dtype=np.float32),
            }
        ),
        RefCase(
            {
                "N": 1,
                "K": 1,
                "inp": np.array([-2.0], dtype=np.float32),
                "kernel": np.array([3.0], dtype=np.float32),
            }
        ),
        RefCase(
            {
                "N": 4,
                "K": 5,
                "inp": np.array([1.0, -1.0, 2.0, -2.0], dtype=np.float32),
                "kernel": np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32),
            }
        ),
    ]
    rng = np.random.default_rng(139)
    cases.append(
        RefCase(
            {
                "N": 100_003,
                "K": NINE_KERNEL.size,
                "inp": rng.standard_normal(100_003).astype(np.float32),
                "kernel": NINE_KERNEL,
            }
        )
    )
    return cases


def solve(inputs: dict[str, Any]) -> list[FloatArray]:
    n, k = int(inputs["N"]), int(inputs["K"])
    inp: FloatArray = inputs["inp"].astype(np.float32)
    kernel: FloatArray = inputs["kernel"].astype(np.float32)
    radius = k // 2
    padded = np.pad(inp, (radius, radius))
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        out[i] = np.sum(padded[i : i + k] * kernel, dtype=np.float32)
    return [out]

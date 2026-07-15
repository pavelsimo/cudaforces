"""Reference for same-size zero-padded 2D cross-correlation."""

from typing import Any

import numpy as np
import numpy.typing as npt

from ..types import RefCase

FloatArray = npt.NDArray[np.float32]

RTOL = 1e-4
ATOL = 1e-5


def tests() -> list[RefCase]:
    cases = [
        RefCase(
            {
                "H": 2,
                "W": 2,
                "K": 1,
                "inp": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
                "kernel": np.array([1.0], dtype=np.float32),
            }
        ),
        RefCase(
            {
                "H": 1,
                "W": 1,
                "K": 1,
                "inp": np.array([-3.0], dtype=np.float32),
                "kernel": np.array([2.0], dtype=np.float32),
            }
        ),
        RefCase(
            {
                "H": 2,
                "W": 4,
                "K": 3,
                "inp": np.array([1.0, 2.0, 3.0, 4.0, -1.0, -2.0, -3.0, -4.0], dtype=np.float32),
                "kernel": np.ones(9, dtype=np.float32),
            }
        ),
    ]
    rng = np.random.default_rng(151)
    cases.append(
        RefCase(
            {
                "H": 193,
                "W": 257,
                "K": 7,
                "inp": rng.standard_normal(193 * 257).astype(np.float32),
                "kernel": rng.standard_normal(49).astype(np.float32),
            }
        )
    )
    return cases


def solve(inputs: dict[str, Any]) -> list[FloatArray]:
    h, w, k = int(inputs["H"]), int(inputs["W"]), int(inputs["K"])
    inp = inputs["inp"].reshape(h, w).astype(np.float32)
    kernel = inputs["kernel"].reshape(k, k).astype(np.float32)
    radius = k // 2
    padded = np.pad(inp, ((radius, radius), (radius, radius)))
    out = np.empty((h, w), dtype=np.float32)
    for row in range(h):
        for col in range(w):
            window = padded[row : row + k, col : col + k]
            out[row, col] = np.sum(window * kernel, dtype=np.float32)
    return [out]

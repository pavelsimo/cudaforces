"""Reference for RGB-to-grayscale conversion of an HWC image."""

from typing import Any

import numpy as np
import numpy.typing as npt

from ..types import RefCase

FloatArray = npt.NDArray[np.float32]

RTOL = 1e-5
ATOL = 1e-5


def tests() -> list[RefCase]:
    cases = [
        RefCase(
            {
                "H": 1,
                "W": 2,
                "rgb": np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32),
            }
        ),
        RefCase(
            {
                "H": 1,
                "W": 1,
                "rgb": np.array([0.0, 0.0, 0.0], dtype=np.float32),
            }
        ),
        RefCase(
            {
                "H": 2,
                "W": 2,
                "rgb": np.array(
                    [1.0, 1.0, 1.0, 10.0, 20.0, 30.0, -1.0, 2.0, -3.0, 4.5, 0.0, 8.0],
                    dtype=np.float32,
                ),
            }
        ),
    ]
    rng = np.random.default_rng(137)
    cases.append(
        RefCase(
            {
                "H": 257,
                "W": 257,
                "rgb": rng.uniform(0.0, 255.0, size=257 * 257 * 3).astype(np.float32),
            }
        )
    )
    return cases


def solve(inputs: dict[str, Any]) -> list[FloatArray]:
    h, w = int(inputs["H"]), int(inputs["W"])
    rgb = inputs["rgb"].reshape(h, w, 3).astype(np.float32)
    weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    out: FloatArray = np.sum(rgb * weights, axis=2, dtype=np.float32)
    return [out]

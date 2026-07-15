"""Reference for row-major matrix transpose."""

from typing import Any

import numpy as np
import numpy.typing as npt

from ..types import RefCase

FloatArray = npt.NDArray[np.float32]


def tests() -> list[RefCase]:
    cases = [
        RefCase(
            {
                "H": 2,
                "W": 3,
                "inp": np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32),
            }
        ),
        RefCase(
            {
                "H": 1,
                "W": 1,
                "inp": np.array([-3.5], dtype=np.float32),
            }
        ),
    ]
    for seed, h, w in [(61, 1, 257), (62, 65, 129)]:
        rng = np.random.default_rng(seed)
        cases.append(
            RefCase(
                {
                    "H": h,
                    "W": w,
                    "inp": rng.standard_normal(h * w).astype(np.float32),
                }
            )
        )
    return cases


def solve(inputs: dict[str, Any]) -> list[FloatArray]:
    h, w = int(inputs["H"]), int(inputs["W"])
    out: FloatArray = np.ascontiguousarray(inputs["inp"].reshape(h, w).T)
    return [out]

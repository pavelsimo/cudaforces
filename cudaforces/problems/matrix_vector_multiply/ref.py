"""Reference for matrix-vector multiplication: out = matrix @ x."""

from typing import Any

import numpy as np
import numpy.typing as npt

from ..types import RefCase

FloatArray = npt.NDArray[np.float32]

RTOL = 1e-4
ATOL = 1e-4


def tests() -> list[RefCase]:
    cases = [
        RefCase(
            {
                "R": 2,
                "C": 3,
                "matrix": np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32),
                "x": np.array([1.0, 0.0, 2.0], dtype=np.float32),
            }
        ),
        RefCase(
            {
                "R": 1,
                "C": 1,
                "matrix": np.array([-3.0], dtype=np.float32),
                "x": np.array([2.5], dtype=np.float32),
            }
        ),
    ]
    for seed, (r, c) in [(113, (127, 31)), (127, (257, 513))]:
        rng = np.random.default_rng(seed)
        cases.append(
            RefCase(
                {
                    "R": r,
                    "C": c,
                    "matrix": rng.standard_normal(r * c).astype(np.float32),
                    "x": rng.standard_normal(c).astype(np.float32),
                }
            )
        )
    return cases


def solve(inputs: dict[str, Any]) -> list[FloatArray]:
    r, c = int(inputs["R"]), int(inputs["C"])
    matrix = inputs["matrix"].reshape(r, c).astype(np.float32)
    x: FloatArray = inputs["x"].astype(np.float32)
    out: FloatArray = (matrix @ x).astype(np.float32)
    return [out]

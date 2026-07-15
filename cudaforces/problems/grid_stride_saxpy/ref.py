"""Reference for grid-stride SAXPY: y[i] = a * x[i] + y[i]."""

from typing import Any

import numpy as np
import numpy.typing as npt

from ..types import RefCase

FloatArray = npt.NDArray[np.float32]


def tests() -> list[RefCase]:
    cases = [
        RefCase(
            {
                "N": 3,
                "a": 2.0,
                "x": np.array([1.0, 2.0, 3.0], dtype=np.float32),
                "y": np.array([4.0, 5.0, 6.0], dtype=np.float32),
            }
        ),
        RefCase(
            {
                "N": 1,
                "a": -3.0,
                "x": np.array([2.0], dtype=np.float32),
                "y": np.array([7.0], dtype=np.float32),
            }
        ),
    ]
    for seed, n, a in [(11, 257, 0.0), (12, 4_097, -0.375)]:
        rng = np.random.default_rng(seed)
        cases.append(
            RefCase(
                {
                    "N": n,
                    "a": a,
                    "x": rng.standard_normal(n).astype(np.float32),
                    "y": rng.standard_normal(n).astype(np.float32),
                }
            )
        )
    return cases


def solve(inputs: dict[str, Any]) -> list[FloatArray]:
    a = np.float32(inputs["a"])
    out: FloatArray = a * inputs["x"] + inputs["y"]
    return [out.astype(np.float32)]

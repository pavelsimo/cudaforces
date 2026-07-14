"""Reference for gelu-forward: GELU tanh approximation, element-wise."""

from typing import Any

import numpy as np
import numpy.typing as npt

from ..types import RefCase

FloatArray = npt.NDArray[np.float32]


def tests() -> list[RefCase]:
    cases = [
        RefCase(
            {
                "N": 4,
                "inp": np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32),
            }
        )
    ]
    for seed, n in [(1, 1_000), (2, 65_536), (3, 100_001)]:
        rng = np.random.default_rng(seed)
        cases.append(
            RefCase(
                {
                    "N": n,
                    "inp": (rng.standard_normal(n) * 3.0).astype(np.float32),
                }
            )
        )
    return cases


def solve(inputs: dict[str, Any]) -> list[FloatArray]:
    x: FloatArray = inputs["inp"].astype(np.float32)
    k = np.float32(np.sqrt(2.0 / np.pi))
    cube = np.float32(0.044715) * x * x * x
    out = np.float32(0.5) * x * (np.float32(1.0) + np.tanh(k * (x + cube)))
    return [np.asarray(out, dtype=np.float32)]

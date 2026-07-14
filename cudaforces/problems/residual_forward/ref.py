"""Reference for residual-forward: out[i] = inp1[i] + inp2[i]."""

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
                "inp1": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
                "inp2": np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32),
            }
        )
    ]
    for seed, n in [(1, 1_000), (2, 65_536), (3, 100_001)]:
        rng = np.random.default_rng(seed)
        cases.append(
            RefCase(
                {
                    "N": n,
                    "inp1": rng.standard_normal(n).astype(np.float32),
                    "inp2": rng.standard_normal(n).astype(np.float32),
                }
            )
        )
    return cases


def solve(inputs: dict[str, Any]) -> list[FloatArray]:
    out: FloatArray = inputs["inp1"] + inputs["inp2"]
    return [out]

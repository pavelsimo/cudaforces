"""Reference for collision-free scatter into a pre-zeroed output array."""

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
                "M": 5,
                "values": np.array([5.0, 7.0, 9.0], dtype=np.float32),
                "indices": np.array([2, 0, 4], dtype=np.int32),
            }
        ),
        RefCase(
            {
                "N": 1,
                "M": 1,
                "values": np.array([9.0], dtype=np.float32),
                "indices": np.array([0], dtype=np.int32),
            }
        ),
    ]
    for seed, n, m in [(51, 129, 257), (52, 2_049, 4_099)]:
        rng = np.random.default_rng(seed)
        cases.append(
            RefCase(
                {
                    "N": n,
                    "M": m,
                    "values": rng.standard_normal(n).astype(np.float32),
                    "indices": rng.choice(m, size=n, replace=False).astype(np.int32),
                }
            )
        )
    return cases


def solve(inputs: dict[str, Any]) -> list[FloatArray]:
    out = np.zeros(int(inputs["M"]), dtype=np.float32)
    out[inputs["indices"]] = inputs["values"]
    return [out]

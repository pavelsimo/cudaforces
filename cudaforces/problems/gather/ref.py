"""Reference for indexed gather: out[i] = src[indices[i]]."""

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
                "M": 3,
                "src": np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32),
                "indices": np.array([3, 0, 3], dtype=np.int32),
            }
        ),
        RefCase(
            {
                "N": 1,
                "M": 1,
                "src": np.array([-7.5], dtype=np.float32),
                "indices": np.array([0], dtype=np.int32),
            }
        ),
    ]
    for seed, n, m in [(41, 257, 513), (42, 4_099, 3_001)]:
        rng = np.random.default_rng(seed)
        cases.append(
            RefCase(
                {
                    "N": n,
                    "M": m,
                    "src": rng.standard_normal(n).astype(np.float32),
                    "indices": rng.integers(0, n, size=m, dtype=np.int32),
                }
            )
        )
    return cases


def solve(inputs: dict[str, Any]) -> list[FloatArray]:
    out: FloatArray = inputs["src"][inputs["indices"]]
    return [np.ascontiguousarray(out, dtype=np.float32)]

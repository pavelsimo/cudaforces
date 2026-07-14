"""Reference for global-norm: out[0] = sum of squares of the whole tensor."""

from typing import Any

import numpy as np
import numpy.typing as npt

from ..types import RefCase

FloatArray = npt.NDArray[np.float32]

# fp32 sums over up to 2e5 values depend on accumulation order (atomics,
# tree reductions); 1e-3 relative absorbs that while still rejecting fp64-only
# or wrong-formula submissions.
RTOL = 1e-3


def tests() -> list[RefCase]:
    cases = [
        RefCase(
            {
                "count": 3,
                "data": np.array([1.0, 2.0, 3.0], dtype=np.float32),
            }
        )
    ]
    for seed, count in [(1, 1_000), (2, 100_000), (3, 200_000)]:
        rng = np.random.default_rng(seed)
        cases.append(
            RefCase(
                {
                    "count": count,
                    "data": rng.standard_normal(count).astype(np.float32),
                }
            )
        )
    return cases


def solve(inputs: dict[str, Any]) -> list[FloatArray]:
    data: FloatArray = inputs["data"]
    total = np.sum(data * data, dtype=np.float32)
    return [np.array([total], dtype=np.float32)]

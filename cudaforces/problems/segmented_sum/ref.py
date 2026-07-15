"""Reference for sums over ranges described by a nondecreasing offsets array."""

from typing import Any

import numpy as np
import numpy.typing as npt

from ..types import RefCase

FloatArray = npt.NDArray[np.float32]

RTOL = 1e-3
ATOL = 1e-2


def tests() -> list[RefCase]:
    rng = np.random.default_rng(606)
    large_n = 100_003
    segment_count = 257
    cuts = np.sort(rng.integers(0, large_n + 1, size=segment_count - 1))
    large_offsets = np.concatenate(([0], cuts, [large_n])).astype(np.int32)
    large_offsets[100] = large_offsets[99]
    return [
        RefCase(
            {
                "N": 4,
                "S": 3,
                "values": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
                "offsets": np.array([0, 2, 2, 4], dtype=np.int32),
            }
        ),
        RefCase(
            {
                "N": 1,
                "S": 3,
                "values": np.array([-4.5], dtype=np.float32),
                "offsets": np.array([0, 0, 1, 1], dtype=np.int32),
            }
        ),
        RefCase(
            {
                "N": 17,
                "S": 5,
                "values": np.linspace(-2.0, 2.0, 17, dtype=np.float32),
                "offsets": np.array([0, 3, 3, 4, 9, 17], dtype=np.int32),
            }
        ),
        RefCase(
            {
                "N": large_n,
                "S": segment_count,
                "values": rng.standard_normal(large_n).astype(np.float32),
                "offsets": large_offsets,
            }
        ),
    ]


def solve(inputs: dict[str, Any]) -> list[FloatArray]:
    values: FloatArray = inputs["values"]
    offsets = inputs["offsets"]
    segment_count = inputs["S"]
    out = np.empty(segment_count, dtype=np.float32)
    for segment in range(segment_count):
        start, stop = int(offsets[segment]), int(offsets[segment + 1])
        out[segment] = np.sum(values[start:stop], dtype=np.float32)
    return [out]

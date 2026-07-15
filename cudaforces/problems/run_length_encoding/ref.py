"""Reference for run-length encoding consecutive equal int32 values."""

from typing import Any

import numpy as np
import numpy.typing as npt

from ..types import RefCase

IntArray = npt.NDArray[np.int32]

RTOL = 0.0
ATOL = 0.0


def tests() -> list[RefCase]:
    cases = [
        RefCase(
            {
                "N": 6,
                "inp": np.array([2, 2, 2, 5, 5, 1], dtype=np.int32),
            }
        ),
        RefCase({"N": 1, "inp": np.array([42], dtype=np.int32)}),
        RefCase({"N": 257, "inp": np.full(257, -3, dtype=np.int32)}),
    ]
    rng = np.random.default_rng(71)
    lengths = rng.integers(1, 12, size=700, dtype=np.int32)
    values = rng.integers(-50, 51, size=700, dtype=np.int32)
    values[1:][values[1:] == values[:-1]] += 101
    inp = np.repeat(values, lengths).astype(np.int32)
    cases.append(RefCase({"N": inp.size, "inp": inp}))
    return cases


def solve(inputs: dict[str, Any]) -> list[IntArray]:
    inp = inputs["inp"]
    starts = np.empty(inp.size, dtype=np.bool_)
    starts[0] = True
    starts[1:] = inp[1:] != inp[:-1]
    offsets = np.flatnonzero(starts)
    unique = inp[offsets].astype(np.int32, copy=True)
    counts = np.diff(np.append(offsets, inp.size)).astype(np.int32)
    runs = np.array([offsets.size], dtype=np.int32)
    return [runs, unique, counts]

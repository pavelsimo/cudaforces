"""Reference for stable ascending sort of nonnegative int32 key-value pairs."""

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
                "N": 4,
                "keys": np.array([9, 3, 9, 1], dtype=np.int32),
                "values": np.array([0, 1, 2, 3], dtype=np.int32),
            }
        ),
        RefCase(
            {
                "N": 1,
                "keys": np.array([2_147_483_647], dtype=np.int32),
                "values": np.array([-7], dtype=np.int32),
            }
        ),
        RefCase(
            {
                "N": 257,
                "keys": np.zeros(257, dtype=np.int32),
                "values": np.arange(257, dtype=np.int32),
            }
        ),
    ]
    rng = np.random.default_rng(81)
    n = 8_193
    cases.append(
        RefCase(
            {
                "N": n,
                "keys": rng.integers(0, 1 << 30, size=n, dtype=np.int32),
                "values": rng.integers(-(1 << 30), 1 << 30, size=n, dtype=np.int32),
            }
        )
    )
    return cases


def solve(inputs: dict[str, Any]) -> list[IntArray]:
    order = np.argsort(inputs["keys"], kind="stable")
    keys_out = np.ascontiguousarray(inputs["keys"][order], dtype=np.int32)
    values_out = np.ascontiguousarray(inputs["values"][order], dtype=np.int32)
    return [keys_out, values_out]

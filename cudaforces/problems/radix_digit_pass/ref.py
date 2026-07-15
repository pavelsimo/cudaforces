"""Reference for radix digit pass: stable sort by one unsigned key byte."""

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
                "shift": 0,
                "keys": np.array([258, 1, 257, 2], dtype=np.int32),
                "values": np.array([0, 1, 2, 3], dtype=np.int32),
            }
        ),
        RefCase(
            {
                "N": 1,
                "shift": 24,
                "keys": np.array([-1], dtype=np.int32),
                "values": np.array([99], dtype=np.int32),
            }
        ),
        RefCase(
            {
                "N": 9,
                "shift": 8,
                "keys": np.array([0x100, 0x2FF, 0x101, 0, -1, 0x200, 0x1FE, 0x201, -256], dtype=np.int32),
                "values": np.arange(9, dtype=np.int32),
            }
        ),
    ]
    rng = np.random.default_rng(107)
    n = 65_537
    cases.append(
        RefCase(
            {
                "N": n,
                "shift": 16,
                "keys": rng.integers(-(2**31), 2**31, size=n, dtype=np.int32),
                "values": np.arange(n, dtype=np.int32),
            }
        )
    )
    return cases


def solve(inputs: dict[str, Any]) -> list[IntArray]:
    keys: IntArray = inputs["keys"].astype(np.int32)
    values: IntArray = inputs["values"].astype(np.int32)
    shift = int(inputs["shift"])
    digits = (keys.view(np.uint32) >> np.uint32(shift)) & np.uint32(0xFF)
    order = np.argsort(digits, kind="stable")
    return [keys[order], values[order]]

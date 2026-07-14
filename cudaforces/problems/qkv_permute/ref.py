"""Reference for qkv-permute: scatter inp (B,T,3,NH,HS) into q, k, v (B,NH,T,HS)."""

from typing import Any

import numpy as np
import numpy.typing as npt

from ..types import RefCase

FloatArray = npt.NDArray[np.float32]


def tests() -> list[RefCase]:
    cases = [
        RefCase(
            {
                "B": 1,
                "T": 1,
                "NH": 1,
                "HS": 2,
                "inp": np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32),
            }
        )
    ]
    for seed, (b, t, nh, hs) in [(1, (2, 16, 4, 8)), (2, (4, 64, 8, 16))]:
        rng = np.random.default_rng(seed)
        cases.append(
            RefCase(
                {
                    "B": b,
                    "T": t,
                    "NH": nh,
                    "HS": hs,
                    "inp": rng.standard_normal(b * t * 3 * nh * hs).astype(np.float32),
                }
            )
        )
    return cases


def solve(inputs: dict[str, Any]) -> list[FloatArray]:
    b, t, nh, hs = inputs["B"], inputs["T"], inputs["NH"], inputs["HS"]
    inp = inputs["inp"].reshape(b, t, 3, nh, hs).astype(np.float32)
    # (B, T, NH, HS) -> (B, NH, T, HS)
    q = np.ascontiguousarray(inp[:, :, 0].transpose(0, 2, 1, 3))
    k = np.ascontiguousarray(inp[:, :, 1].transpose(0, 2, 1, 3))
    v = np.ascontiguousarray(inp[:, :, 2].transpose(0, 2, 1, 3))
    return [q, k, v]

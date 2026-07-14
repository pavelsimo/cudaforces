"""Reference for trimat-forward: lower-triangular preatt = Q @ K^T / sqrt(HS)."""

import math
from typing import Any

import numpy as np
import numpy.typing as npt

from ..types import RefCase

FloatArray = npt.NDArray[np.float32]

# fp32 dot-product accumulation order differs between GPU and reference.
RTOL = 1e-3


def tests() -> list[RefCase]:
    cases = [
        RefCase(
            {
                "B": 1,
                "NH": 1,
                "T": 2,
                "HS": 1,
                "q": np.array([1.0, 2.0], dtype=np.float32),
                "k": np.array([3.0, 4.0], dtype=np.float32),
            }
        )
    ]
    for seed, (b, nh, t, hs) in [(1, (2, 2, 32, 8)), (2, (2, 4, 64, 16))]:
        rng = np.random.default_rng(seed)
        n = b * nh * t * hs
        cases.append(
            RefCase(
                {
                    "B": b,
                    "NH": nh,
                    "T": t,
                    "HS": hs,
                    "q": rng.standard_normal(n).astype(np.float32),
                    "k": rng.standard_normal(n).astype(np.float32),
                }
            )
        )
    return cases


def solve(inputs: dict[str, Any]) -> list[FloatArray]:
    b, nh, t, hs = inputs["B"], inputs["NH"], inputs["T"], inputs["HS"]
    q = inputs["q"].reshape(b, nh, t, hs).astype(np.float32)
    k = inputs["k"].reshape(b, nh, t, hs).astype(np.float32)
    scale = np.float32(1.0 / math.sqrt(hs))
    preatt = (q @ np.swapaxes(k, 2, 3)).astype(np.float32) * scale
    # Emit only the lower triangle: for b, h, t1, then t2 in [0, t1].
    rows, cols = np.tril_indices(t)
    tri: FloatArray = np.ascontiguousarray(preatt[:, :, rows, cols], dtype=np.float32)
    return [tri]

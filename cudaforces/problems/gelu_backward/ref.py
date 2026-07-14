"""Reference for gelu-backward: dinp[i] = dout[i] * GELU'(inp[i]) (tanh approx)."""

from typing import Any

import numpy as np
import numpy.typing as npt

from ..types import RefCase

FloatArray = npt.NDArray[np.float32]


def tests() -> list[RefCase]:
    cases = [
        RefCase(
            {
                "N": 1,
                "inp": np.array([1.0], dtype=np.float32),
                "dout": np.array([1.0], dtype=np.float32),
            }
        )
    ]
    for seed, n in [(1, 1_000), (2, 65_536), (3, 100_001)]:
        rng = np.random.default_rng(seed)
        cases.append(
            RefCase(
                {
                    "N": n,
                    "inp": (rng.standard_normal(n) * 3.0).astype(np.float32),
                    "dout": rng.standard_normal(n).astype(np.float32),
                }
            )
        )
    return cases


def solve(inputs: dict[str, Any]) -> list[FloatArray]:
    x: FloatArray = inputs["inp"].astype(np.float32)
    dout: FloatArray = inputs["dout"].astype(np.float32)
    k = np.float32(np.sqrt(2.0 / np.pi))
    coef = np.float32(0.044715)
    u = k * (x + coef * x * x * x)
    t = np.tanh(u)
    sech2 = np.float32(1.0) - t * t
    local = np.float32(0.5) * (np.float32(1.0) + t) + np.float32(0.5) * x * sech2 * k * (
        np.float32(1.0) + np.float32(3.0) * coef * x * x
    )
    dinp: FloatArray = dout * local
    return [dinp.astype(np.float32)]

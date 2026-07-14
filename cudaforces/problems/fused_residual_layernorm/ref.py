"""Reference for fused-residual-layernorm.

residual = inp1 + inp2, then LayerNorm over each row (eps = 1e-5, population
variance): normed = (residual - mean) * rstd * weight + bias.
"""

from typing import Any

import numpy as np
import numpy.typing as npt

from ..types import RefCase

FloatArray = npt.NDArray[np.float32]

EPS = np.float32(1e-5)


def tests() -> list[RefCase]:
    cases = [
        RefCase(
            {
                "N": 1,
                "C": 3,
                "inp1": np.array([1.0, 2.0, 3.0], dtype=np.float32),
                "inp2": np.array([0.0, 0.0, 0.0], dtype=np.float32),
                "weight": np.array([1.0, 1.0, 1.0], dtype=np.float32),
                "bias": np.array([0.0, 0.0, 0.0], dtype=np.float32),
            }
        )
    ]
    for seed, (n, c) in [(1, (64, 128)), (2, (128, 768))]:
        rng = np.random.default_rng(seed)
        cases.append(
            RefCase(
                {
                    "N": n,
                    "C": c,
                    "inp1": rng.standard_normal(n * c).astype(np.float32),
                    "inp2": rng.standard_normal(n * c).astype(np.float32),
                    "weight": rng.standard_normal(c).astype(np.float32),
                    "bias": rng.standard_normal(c).astype(np.float32),
                }
            )
        )
    return cases


def solve(inputs: dict[str, Any]) -> list[FloatArray]:
    n, c = inputs["N"], inputs["C"]
    residual = (inputs["inp1"] + inputs["inp2"]).reshape(n, c)
    weight = inputs["weight"]
    bias = inputs["bias"]
    mean = residual.mean(axis=-1, dtype=np.float32)
    var = np.square(residual - mean[:, None]).mean(axis=-1, dtype=np.float32)
    rstd = np.float32(1.0) / np.sqrt(var + EPS, dtype=np.float32)
    normed = (residual - mean[:, None]) * rstd[:, None] * weight[None, :] + bias[None, :]
    return [
        residual.astype(np.float32),
        normed.astype(np.float32),
        mean.astype(np.float32),
        rstd.astype(np.float32),
    ]

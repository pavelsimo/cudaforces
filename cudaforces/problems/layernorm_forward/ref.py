"""Reference for layernorm-forward: row-wise layernorm over C with eps=1e-5."""

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
                "inp": np.array([1.0, 2.0, 3.0], dtype=np.float32),
                "weight": np.array([1.0, 1.0, 1.0], dtype=np.float32),
                "bias": np.array([0.0, 0.0, 0.0], dtype=np.float32),
            }
        )
    ]
    for seed, (n, c) in [(1, (64, 128)), (2, (256, 768))]:
        rng = np.random.default_rng(seed)
        cases.append(
            RefCase(
                {
                    "N": n,
                    "C": c,
                    "inp": rng.standard_normal(n * c).astype(np.float32),
                    "weight": rng.uniform(0.5, 1.5, c).astype(np.float32),
                    "bias": rng.uniform(-0.5, 0.5, c).astype(np.float32),
                }
            )
        )
    return cases


def solve(inputs: dict[str, Any]) -> list[FloatArray]:
    n, c = int(inputs["N"]), int(inputs["C"])
    x: FloatArray = inputs["inp"].reshape(n, c)
    weight: FloatArray = inputs["weight"]
    bias: FloatArray = inputs["bias"]
    mean: FloatArray = x.mean(axis=1, dtype=np.float32)
    centered: FloatArray = x - mean[:, None]
    var: FloatArray = np.mean(centered * centered, axis=1, dtype=np.float32)
    rstd: FloatArray = (1.0 / np.sqrt(var + EPS)).astype(np.float32)
    out: FloatArray = centered * rstd[:, None] * weight[None, :] + bias[None, :]
    return [out.astype(np.float32), mean.astype(np.float32), rstd]

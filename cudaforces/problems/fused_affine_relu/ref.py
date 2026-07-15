"""Reference for fused affine ReLU: out = max(scale * x + bias, 0)."""

from typing import Any

import numpy as np
import numpy.typing as npt

from ..types import RefCase

FloatArray = npt.NDArray[np.float32]


def tests() -> list[RefCase]:
    cases = [
        RefCase(
            {
                "N": 4,
                "scale": 2.0,
                "bias": 1.0,
                "x": np.array([-2.0, -0.5, 1.0, 3.0], dtype=np.float32),
            }
        ),
        RefCase(
            {
                "N": 1,
                "scale": -4.0,
                "bias": 2.0,
                "x": np.array([0.5], dtype=np.float32),
            }
        ),
    ]
    for seed, n, scale, bias in [
        (21, 513, 0.0, 1.25),
        (22, 4_096, -0.75, -0.125),
    ]:
        rng = np.random.default_rng(seed)
        cases.append(
            RefCase(
                {
                    "N": n,
                    "scale": scale,
                    "bias": bias,
                    "x": rng.standard_normal(n).astype(np.float32),
                }
            )
        )
    return cases


def solve(inputs: dict[str, Any]) -> list[FloatArray]:
    scale = np.float32(inputs["scale"])
    bias = np.float32(inputs["bias"])
    out: FloatArray = np.maximum(scale * inputs["x"] + bias, np.float32(0.0))
    return [out.astype(np.float32)]

"""Reference for matmul-forward: out[i, o] = bias[o] + sum_c inp[i, c] * weight[o, c]."""

from typing import Any

import numpy as np
import numpy.typing as npt

from ..types import RefCase

FloatArray = npt.NDArray[np.float32]

# fp32 dot-product accumulation order differs between GPU and reference.
RTOL = 1e-3
ATOL = 1e-3


def tests() -> list[RefCase]:
    cases = [
        RefCase(
            {
                "BT": 1,
                "C": 2,
                "OC": 2,
                "inp": np.array([1.0, 2.0], dtype=np.float32),
                "weight": np.array([3.0, 4.0, 5.0, 6.0], dtype=np.float32),
                "bias": np.array([10.0, 20.0], dtype=np.float32),
            }
        )
    ]
    for seed, (bt, c, oc) in [(1, (32, 64, 64)), (2, (128, 128, 96)), (3, (64, 256, 192))]:
        rng = np.random.default_rng(seed)
        cases.append(
            RefCase(
                {
                    "BT": bt,
                    "C": c,
                    "OC": oc,
                    "inp": rng.standard_normal(bt * c).astype(np.float32),
                    "weight": rng.standard_normal(oc * c).astype(np.float32),
                    "bias": rng.standard_normal(oc).astype(np.float32),
                }
            )
        )
    return cases


def solve(inputs: dict[str, Any]) -> list[FloatArray]:
    bt, c, oc = inputs["BT"], inputs["C"], inputs["OC"]
    inp = inputs["inp"].reshape(bt, c).astype(np.float32)
    weight = inputs["weight"].reshape(oc, c).astype(np.float32)
    bias = inputs["bias"].astype(np.float32)
    out: FloatArray = (inp @ weight.T + bias).astype(np.float32)
    return [out]

"""Reference for softmax-forward: numerically-stable row-wise softmax of (N, C)."""

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
                "C": 3,
                "inp": np.array([1.0, 2.0, 3.0], dtype=np.float32),
            }
        )
    ]
    for seed, (n, c), scale in [(1, (32, 64), 1.0), (2, (128, 1024), 10.0), (3, (8, 2048), 100.0)]:
        rng = np.random.default_rng(seed)
        inp = (rng.standard_normal(n * c) * scale).astype(np.float32)
        cases.append(RefCase({"N": n, "C": c, "inp": inp}))
    return cases


def solve(inputs: dict[str, Any]) -> list[FloatArray]:
    n, c = int(inputs["N"]), int(inputs["C"])
    inp: FloatArray = inputs["inp"].reshape(n, c)
    shifted: FloatArray = inp - inp.max(axis=1, keepdims=True)
    e: FloatArray = np.exp(shifted)
    out = e / e.sum(axis=1, keepdims=True, dtype=np.float32)
    return [out.astype(np.float32)]

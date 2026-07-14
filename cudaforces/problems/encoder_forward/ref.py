"""Reference for encoder-forward: out[b,t,c] = wte[inp[b,t],c] + wpe[t,c]."""

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
                "T": 2,
                "C": 2,
                "V": 3,
                "inp": np.array([2, 0], dtype=np.int32),
                "wte": np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=np.float32),
                "wpe": np.array([[1.0, 1.0], [2.0, 2.0]], dtype=np.float32),
            }
        )
    ]
    for seed, (b, t, c, v) in [
        (1, (2, 16, 8, 32)),
        (2, (4, 64, 64, 128)),
        (3, (2, 128, 96, 256)),
    ]:
        rng = np.random.default_rng(seed)
        cases.append(
            RefCase(
                {
                    "B": b,
                    "T": t,
                    "C": c,
                    "V": v,
                    "inp": rng.integers(0, v, size=b * t, dtype=np.int32),
                    "wte": rng.standard_normal((v, c)).astype(np.float32),
                    "wpe": rng.standard_normal((t, c)).astype(np.float32),
                }
            )
        )
    return cases


def solve(inputs: dict[str, Any]) -> list[FloatArray]:
    b, t, c = int(inputs["B"]), int(inputs["T"]), int(inputs["C"])
    v = int(inputs["V"])
    inp = np.asarray(inputs["inp"]).reshape(b, t)
    wte = np.asarray(inputs["wte"], dtype=np.float32).reshape(v, c)
    wpe = np.asarray(inputs["wpe"], dtype=np.float32).reshape(t, c)
    out: FloatArray = (wte[inp] + wpe[np.newaxis, :, :]).astype(np.float32)
    return [out]

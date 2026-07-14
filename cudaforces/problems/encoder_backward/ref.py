"""Reference for encoder-backward: scatter-add dout into dwte and dwpe."""

from typing import Any

import numpy as np
import numpy.typing as npt

from ..types import RefCase

FloatArray = npt.NDArray[np.float32]


def _skewed_tokens(rng: np.random.Generator, n: int, v: int) -> npt.NDArray[np.int32]:
    """Token ids in [0, V) biased toward small ids, guaranteeing repeats."""
    tokens = np.minimum((rng.random(n) ** 3 * v).astype(np.int32), v - 1)
    return tokens.astype(np.int32)


def tests() -> list[RefCase]:
    cases = [
        RefCase(
            {
                "B": 1,
                "T": 2,
                "C": 1,
                "V": 8,
                "dout": np.array([1.0, 2.0], dtype=np.float32),
                "inp": np.array([5, 5], dtype=np.int32),
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
                    "dout": rng.standard_normal(b * t * c).astype(np.float32),
                    "inp": _skewed_tokens(rng, b * t, v),
                }
            )
        )
    return cases


def solve(inputs: dict[str, Any]) -> list[FloatArray]:
    b, t, c = int(inputs["B"]), int(inputs["T"]), int(inputs["C"])
    v = int(inputs["V"])
    dout = np.asarray(inputs["dout"], dtype=np.float32).reshape(b, t, c)
    inp = np.asarray(inputs["inp"]).reshape(b, t)
    dwte: FloatArray = np.zeros((v, c), dtype=np.float32)
    dwpe: FloatArray = np.zeros((t, c), dtype=np.float32)
    np.add.at(dwte, inp.ravel(), dout.reshape(b * t, c))
    dwpe += dout.sum(axis=0, dtype=np.float32)
    return [dwte, dwpe]

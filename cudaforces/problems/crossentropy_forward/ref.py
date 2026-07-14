"""Reference for crossentropy-forward: losses[b,t] = -log(probs[b,t,target])."""

from typing import Any

import numpy as np
import numpy.typing as npt

from ..types import RefCase

FloatArray = npt.NDArray[np.float32]


def _softmax_rows(logits: FloatArray) -> FloatArray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(shifted, dtype=np.float32)
    out: FloatArray = (e / e.sum(axis=-1, keepdims=True, dtype=np.float32)).astype(np.float32)
    return out


def tests() -> list[RefCase]:
    cases = [
        RefCase(
            {
                "B": 1,
                "T": 2,
                "V": 3,
                "probs": np.array([[0.2, 0.3, 0.5], [0.9, 0.05, 0.05]], dtype=np.float32),
                "targets": np.array([2, 0], dtype=np.int32),
            }
        )
    ]
    for seed, (b, t, v) in [(1, (2, 64, 256)), (2, (2, 128, 512))]:
        rng = np.random.default_rng(seed)
        logits = rng.standard_normal((b * t, v)).astype(np.float32)
        cases.append(
            RefCase(
                {
                    "B": b,
                    "T": t,
                    "V": v,
                    "probs": _softmax_rows(logits),
                    "targets": rng.integers(0, v, size=b * t).astype(np.int32),
                }
            )
        )
    return cases


def solve(inputs: dict[str, Any]) -> list[FloatArray]:
    b, t, v = inputs["B"], inputs["T"], inputs["V"]
    probs = inputs["probs"].reshape(b * t, v)
    targets = inputs["targets"].reshape(b * t)
    losses: FloatArray = -np.log(probs[np.arange(b * t), targets], dtype=np.float32)
    return [losses]

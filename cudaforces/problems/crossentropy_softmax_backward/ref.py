"""Reference for crossentropy-softmax-backward.

dlogits[b,t,v] = (probs[b,t,v] - 1{v == targets[b,t]}) * dlosses[b,t]
"""

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
                "T": 1,
                "V": 3,
                "dlosses": np.array([1.0], dtype=np.float32),
                "probs": np.array([[0.2, 0.3, 0.5]], dtype=np.float32),
                "targets": np.array([2], dtype=np.int32),
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
                    "dlosses": rng.standard_normal(b * t).astype(np.float32),
                    "probs": _softmax_rows(logits),
                    "targets": rng.integers(0, v, size=b * t).astype(np.int32),
                }
            )
        )
    return cases


def solve(inputs: dict[str, Any]) -> list[FloatArray]:
    b, t, v = inputs["B"], inputs["T"], inputs["V"]
    probs = inputs["probs"].reshape(b * t, v)
    dlosses = inputs["dlosses"].reshape(b * t)
    targets = inputs["targets"].reshape(b * t)
    dlogits = probs.copy()
    dlogits[np.arange(b * t), targets] -= np.float32(1.0)
    dlogits *= dlosses[:, None]
    return [dlogits.astype(np.float32)]

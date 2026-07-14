"""Reference for fused-classifier: softmax + cross-entropy loss + gradient.

Per (b,t) row of logits: losses = -(x[target] - max - log(sumexp)) and the
row is overwritten in place with dlogits = (softmax(x) - onehot(target)) * dloss.
"""

from typing import Any

import numpy as np
import numpy.typing as npt

from ..types import RefCase

FloatArray = npt.NDArray[np.float32]

RTOL = 1e-3


def tests() -> list[RefCase]:
    cases = [
        RefCase(
            {
                "B": 1,
                "T": 1,
                "V": 3,
                "dloss": 1.0,
                "logits": np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
                "targets": np.array([1], dtype=np.int32),
            }
        )
    ]
    for seed, (b, t, v) in [(1, (2, 32, 512)), (2, (2, 64, 1024))]:
        rng = np.random.default_rng(seed)
        cases.append(
            RefCase(
                {
                    "B": b,
                    "T": t,
                    "V": v,
                    "dloss": 1.0 / (b * t),
                    "logits": rng.standard_normal((b * t, v)).astype(np.float32),
                    "targets": rng.integers(0, v, size=b * t).astype(np.int32),
                }
            )
        )
    return cases


def solve(inputs: dict[str, Any]) -> list[FloatArray]:
    b, t, v = inputs["B"], inputs["T"], inputs["V"]
    dloss = np.float32(inputs["dloss"])
    logits = inputs["logits"].reshape(b * t, v)
    targets = inputs["targets"].reshape(b * t)
    rows = np.arange(b * t)
    m = logits.max(axis=-1, keepdims=True)
    e = np.exp(logits - m, dtype=np.float32)
    sumexp = e.sum(axis=-1, keepdims=True, dtype=np.float32)
    losses: FloatArray = -(logits[rows, targets] - m[:, 0] - np.log(sumexp[:, 0], dtype=np.float32))
    dlogits = e / sumexp
    dlogits[rows, targets] -= np.float32(1.0)
    dlogits *= dloss
    return [losses.astype(np.float32), dlogits.astype(np.float32)]

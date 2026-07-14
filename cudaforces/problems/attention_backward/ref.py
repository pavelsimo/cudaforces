"""Reference for attention-backward: dq, dk, dv through values, softmax, scores."""

import math
from typing import Any

import numpy as np
import numpy.typing as npt

from ..attention_forward.ref import attention_forward
from ..types import RefCase

FloatArray = npt.NDArray[np.float32]

# Double chain of fp32 matmuls: accumulation order differences compound.
RTOL = 1e-2


def tests() -> list[RefCase]:
    cases = [
        RefCase(
            {
                "B": 1,
                "NH": 1,
                "T": 1,
                "HS": 1,
                "dout": np.array([1.0], dtype=np.float32),
                "q": np.array([1.0], dtype=np.float32),
                "k": np.array([2.0], dtype=np.float32),
                "v": np.array([3.0], dtype=np.float32),
                "att": np.array([1.0], dtype=np.float32),
            }
        )
    ]
    for seed, (b, nh, t, hs) in [(1, (1, 2, 16, 8)), (2, (2, 2, 48, 16))]:
        rng = np.random.default_rng(seed)
        n = b * nh * t * hs
        q = rng.standard_normal(n).astype(np.float32)
        k = rng.standard_normal(n).astype(np.float32)
        v = rng.standard_normal(n).astype(np.float32)
        att, _ = attention_forward(q.reshape(b, nh, t, hs), k.reshape(b, nh, t, hs), v.reshape(b, nh, t, hs))
        cases.append(
            RefCase(
                {
                    "B": b,
                    "NH": nh,
                    "T": t,
                    "HS": hs,
                    "dout": rng.standard_normal(n).astype(np.float32),
                    "q": q,
                    "k": k,
                    "v": v,
                    "att": att,
                }
            )
        )
    return cases


def solve(inputs: dict[str, Any]) -> list[FloatArray]:
    b, nh, t, hs = inputs["B"], inputs["NH"], inputs["T"], inputs["HS"]
    dout = inputs["dout"].reshape(b, nh, t, hs).astype(np.float32)
    q = inputs["q"].reshape(b, nh, t, hs).astype(np.float32)
    k = inputs["k"].reshape(b, nh, t, hs).astype(np.float32)
    att = inputs["att"].reshape(b, nh, t, t).astype(np.float32)
    v = inputs["v"].reshape(b, nh, t, hs).astype(np.float32)
    scale = np.float32(1.0 / math.sqrt(hs))
    dq = np.zeros_like(q)
    dk = np.zeros_like(k)
    dv = np.zeros_like(v)
    for bi in range(b):
        for hi in range(nh):
            # Backward through the weighted sum. att is zero above the diagonal,
            # so multiplying by att enforces causality without explicit masks.
            datt = (dout[bi, hi] @ v[bi, hi].T).astype(np.float32)
            dv[bi, hi] = (att[bi, hi].T @ dout[bi, hi]).astype(np.float32)
            # Backward through the causal softmax (row-wise Jacobian).
            rowdot = (datt * att[bi, hi]).sum(axis=1, dtype=np.float32, keepdims=True)
            dpreatt = (att[bi, hi] * (datt - rowdot) * scale).astype(np.float32)
            # Backward through the scores.
            dq[bi, hi] = (dpreatt @ k[bi, hi]).astype(np.float32)
            dk[bi, hi] = (dpreatt.T @ q[bi, hi]).astype(np.float32)
    return [dq, dk, dv]

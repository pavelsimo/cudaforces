"""Reference for attention-forward: multi-head causal self-attention."""

import math
from typing import Any

import numpy as np
import numpy.typing as npt

from ..types import RefCase

FloatArray = npt.NDArray[np.float32]

# fp32 accumulation order differs between GPU and reference.
RTOL = 1e-3


def tests() -> list[RefCase]:
    cases = [
        RefCase(
            {
                "B": 1,
                "NH": 1,
                "T": 2,
                "HS": 1,
                "q": np.array([1.0, 1.0], dtype=np.float32),
                "k": np.array([0.0, 10.0], dtype=np.float32),
                "v": np.array([1.0, 2.0], dtype=np.float32),
            }
        )
    ]
    for seed, (b, nh, t, hs) in [(1, (1, 2, 16, 8)), (2, (2, 4, 64, 16))]:
        rng = np.random.default_rng(seed)
        n = b * nh * t * hs
        cases.append(
            RefCase(
                {
                    "B": b,
                    "NH": nh,
                    "T": t,
                    "HS": hs,
                    "q": rng.standard_normal(n).astype(np.float32),
                    "k": rng.standard_normal(n).astype(np.float32),
                    "v": rng.standard_normal(n).astype(np.float32),
                }
            )
        )
    return cases


def attention_forward(q: FloatArray, k: FloatArray, v: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Causal attention over (B,NH,T,HS) tensors; returns (att, out) in float32."""
    b, nh, t, hs = q.shape
    scale = np.float32(1.0 / math.sqrt(hs))
    att = np.zeros((b, nh, t, t), dtype=np.float32)
    out = np.zeros_like(q)
    for bi in range(b):
        for hi in range(nh):
            preatt = (q[bi, hi] @ k[bi, hi].T).astype(np.float32) * scale
            for t1 in range(t):
                row = preatt[t1, : t1 + 1]
                e = np.exp((row - row.max()).astype(np.float32)).astype(np.float32)
                att[bi, hi, t1, : t1 + 1] = e / e.sum(dtype=np.float32)
            out[bi, hi] = (att[bi, hi] @ v[bi, hi]).astype(np.float32)
    return att, out


def solve(inputs: dict[str, Any]) -> list[FloatArray]:
    b, nh, t, hs = inputs["B"], inputs["NH"], inputs["T"], inputs["HS"]
    q = inputs["q"].reshape(b, nh, t, hs).astype(np.float32)
    k = inputs["k"].reshape(b, nh, t, hs).astype(np.float32)
    v = inputs["v"].reshape(b, nh, t, hs).astype(np.float32)
    _, out = attention_forward(q, k, v)
    return [out]

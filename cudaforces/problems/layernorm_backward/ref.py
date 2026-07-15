"""Reference for layernorm-backward: dinp, dweight, dbias from cached mean/rstd.

Mirrors the standard CPU reference: norm = (inp - mean) * rstd, dnorm = weight * dout,
dinp = (dnorm - mean(dnorm) - norm * mean(dnorm * norm)) * rstd per row;
dweight/dbias reduce over all (b, t) rows.
"""

from typing import Any

import numpy as np
import numpy.typing as npt

from ..types import RefCase

FloatArray = npt.NDArray[np.float32]

EPS = np.float32(1e-5)

# Cross-row fp32 accumulation into dweight/dbias is order-dependent on the GPU.
RTOL = 1e-3


def _stats(inp: FloatArray, rows: int, c: int) -> tuple[FloatArray, FloatArray]:
    """mean/rstd consistent with the forward pass (population variance, eps=1e-5)."""
    x = inp.reshape(rows, c)
    mean: FloatArray = x.mean(axis=1, dtype=np.float32)
    centered: FloatArray = x - mean[:, None]
    var: FloatArray = np.mean(centered * centered, axis=1, dtype=np.float32)
    rstd: FloatArray = (1.0 / np.sqrt(var + EPS)).astype(np.float32)
    return mean.astype(np.float32), rstd


def _case(b: int, t: int, c: int, inp: FloatArray, dout: FloatArray, weight: FloatArray) -> RefCase:
    mean, rstd = _stats(inp, b * t, c)
    return RefCase(
        {
            "B": b,
            "T": t,
            "C": c,
            "dout": dout,
            "inp": inp,
            "weight": weight,
            "mean": mean,
            "rstd": rstd,
        }
    )


def tests() -> list[RefCase]:
    cases = [
        _case(
            1,
            1,
            2,
            inp=np.array([1.0, 3.0], dtype=np.float32),
            dout=np.array([1.0, 0.0], dtype=np.float32),
            weight=np.array([1.0, 1.0], dtype=np.float32),
        )
    ]
    for seed, (b, t, c) in [(1, (2, 32, 64)), (2, (2, 64, 256)), (3, (1, 16, 768))]:
        rng = np.random.default_rng(seed)
        cases.append(
            _case(
                b,
                t,
                c,
                inp=rng.standard_normal(b * t * c).astype(np.float32),
                dout=rng.standard_normal(b * t * c).astype(np.float32),
                weight=rng.uniform(0.5, 1.5, c).astype(np.float32),
            )
        )
    return cases


def solve(inputs: dict[str, Any]) -> list[FloatArray]:
    b, t, c = int(inputs["B"]), int(inputs["T"]), int(inputs["C"])
    rows = b * t
    dout: FloatArray = inputs["dout"].reshape(rows, c)
    inp: FloatArray = inputs["inp"].reshape(rows, c)
    weight: FloatArray = inputs["weight"]
    mean: FloatArray = inputs["mean"]
    rstd: FloatArray = inputs["rstd"]

    norm: FloatArray = (inp - mean[:, None]) * rstd[:, None]
    dnorm: FloatArray = weight[None, :] * dout
    dnorm_mean: FloatArray = dnorm.mean(axis=1, dtype=np.float32)
    dnorm_norm_mean: FloatArray = np.mean(dnorm * norm, axis=1, dtype=np.float32)
    dinp: FloatArray = (dnorm - dnorm_mean[:, None] - norm * dnorm_norm_mean[:, None]) * rstd[:, None]
    dweight: FloatArray = np.sum(norm * dout, axis=0, dtype=np.float32)
    dbias: FloatArray = dout.sum(axis=0, dtype=np.float32)
    return [dinp.astype(np.float32), dweight.astype(np.float32), dbias.astype(np.float32)]

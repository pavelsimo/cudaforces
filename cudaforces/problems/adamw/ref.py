"""Reference for adamw: one AdamW step over all parameters, in place.

m = b1*m + (1-b1)*g; v = b2*v + (1-b2)*g^2; bias-corrected mhat/vhat;
params -= lr * (mhat / (sqrt(vhat) + eps) + weight_decay * params).
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
                "num_parameters": 1,
                "lr": 0.1,
                "beta1": 0.9,
                "beta2": 0.999,
                "t": 1,
                "eps": 1e-8,
                "weight_decay": 0.0,
                "params": np.array([1.0], dtype=np.float32),
                "grads": np.array([0.5], dtype=np.float32),
                "m": np.array([0.0], dtype=np.float32),
                "v": np.array([0.0], dtype=np.float32),
            }
        )
    ]
    for seed, n, t in [(1, 1_000, 3), (2, 50_000, 10)]:
        rng = np.random.default_rng(seed)
        cases.append(
            RefCase(
                {
                    "num_parameters": n,
                    "lr": 3e-4,
                    "beta1": 0.9,
                    "beta2": 0.999,
                    "t": t,
                    "eps": 1e-8,
                    "weight_decay": 0.1,
                    "params": rng.standard_normal(n).astype(np.float32),
                    "grads": rng.standard_normal(n).astype(np.float32),
                    "m": (rng.standard_normal(n) * 0.01).astype(np.float32),
                    "v": (rng.random(n) * 0.01).astype(np.float32),
                }
            )
        )
    return cases


def solve(inputs: dict[str, Any]) -> list[FloatArray]:
    lr = np.float32(inputs["lr"])
    beta1 = np.float32(inputs["beta1"])
    beta2 = np.float32(inputs["beta2"])
    t = int(inputs["t"])
    eps = np.float32(inputs["eps"])
    weight_decay = np.float32(inputs["weight_decay"])
    params = inputs["params"].astype(np.float32)
    grads = inputs["grads"]
    m = beta1 * inputs["m"] + (np.float32(1.0) - beta1) * grads
    v = beta2 * inputs["v"] + (np.float32(1.0) - beta2) * grads * grads
    mhat = m / (np.float32(1.0) - beta1**t)
    vhat = v / (np.float32(1.0) - beta2**t)
    params = params - lr * (mhat / (np.sqrt(vhat, dtype=np.float32) + eps) + weight_decay * params)
    return [params.astype(np.float32), m.astype(np.float32), v.astype(np.float32)]

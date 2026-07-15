"""Reference for softened all-pairs gravitational acceleration with G = 1."""

from typing import Any

import numpy as np
import numpy.typing as npt

from ..types import RefCase

FloatArray = npt.NDArray[np.float32]

RTOL = 1e-3
ATOL = 1e-4


def tests() -> list[RefCase]:
    cases = [
        RefCase(
            {
                "N": 2,
                "softening": 0.5,
                "position": np.array([0.0, 0.0, 0.0, 2.0, 0.0, 0.0], dtype=np.float32),
                "mass": np.array([3.0, 5.0], dtype=np.float32),
            }
        ),
        RefCase(
            {
                "N": 1,
                "softening": 0.01,
                "position": np.array([1.0, -2.0, 3.0], dtype=np.float32),
                "mass": np.array([7.0], dtype=np.float32),
            }
        ),
        RefCase(
            {
                "N": 3,
                "softening": 0.25,
                "position": np.array(
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                    dtype=np.float32,
                ),
                "mass": np.array([1.0, 2.0, 4.0], dtype=np.float32),
            }
        ),
    ]
    rng = np.random.default_rng(101)
    n = 193
    cases.append(
        RefCase(
            {
                "N": n,
                "softening": 0.05,
                "position": rng.uniform(-2.0, 2.0, size=3 * n).astype(np.float32),
                "mass": rng.uniform(0.1, 3.0, size=n).astype(np.float32),
            }
        )
    )
    return cases


def solve(inputs: dict[str, Any]) -> list[FloatArray]:
    n = int(inputs["N"])
    position = inputs["position"].reshape(n, 3)
    mass = inputs["mass"]
    softening = np.float32(inputs["softening"])
    acceleration = np.zeros((n, 3), dtype=np.float32)
    for i in range(n):
        delta = position - position[i]
        r2 = np.sum(delta * delta, axis=1, dtype=np.float32) + softening * softening
        inv_r3 = np.float32(1.0) / (r2 * np.sqrt(r2, dtype=np.float32))
        inv_r3[i] = np.float32(0.0)
        acceleration[i] = np.sum(delta * (mass * inv_r3)[:, None], axis=0, dtype=np.float32)
    return [acceleration]

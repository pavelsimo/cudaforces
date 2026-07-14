"""Reference for matmul-backward-bias: dbias[o] = sum over rows of dout[:, o]."""

from typing import Any

import numpy as np
import numpy.typing as npt

from ..types import RefCase

FloatArray = npt.NDArray[np.float32]

# Long fp32 column sums: GPU reduction order differs from the reference.
RTOL = 1e-3


def tests() -> list[RefCase]:
    cases = [
        RefCase(
            {
                "BT": 3,
                "OC": 2,
                "dout": np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32),
            }
        )
    ]
    for seed, (bt, oc) in [(1, (1024, 128)), (2, (4096, 48))]:
        rng = np.random.default_rng(seed)
        cases.append(
            RefCase(
                {
                    "BT": bt,
                    "OC": oc,
                    "dout": rng.standard_normal(bt * oc).astype(np.float32),
                }
            )
        )
    return cases


def solve(inputs: dict[str, Any]) -> list[FloatArray]:
    bt, oc = inputs["BT"], inputs["OC"]
    dout = inputs["dout"].reshape(bt, oc).astype(np.float32)
    dbias: FloatArray = dout.sum(axis=0, dtype=np.float32)
    return [dbias]

"""Reference for adding one length-C row to every row of an R-by-C matrix."""

from typing import Any

import numpy as np
import numpy.typing as npt

from ..types import RefCase

FloatArray = npt.NDArray[np.float32]


def tests() -> list[RefCase]:
    cases = [
        RefCase(
            {
                "R": 2,
                "C": 3,
                "matrix": np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32),
                "row": np.array([10.0, 20.0, 30.0], dtype=np.float32),
            }
        ),
        RefCase(
            {
                "R": 1,
                "C": 1,
                "matrix": np.array([-2.0], dtype=np.float32),
                "row": np.array([2.0], dtype=np.float32),
            }
        ),
    ]
    for seed, r, c in [(31, 1, 257), (32, 33, 129)]:
        rng = np.random.default_rng(seed)
        cases.append(
            RefCase(
                {
                    "R": r,
                    "C": c,
                    "matrix": rng.standard_normal(r * c).astype(np.float32),
                    "row": rng.standard_normal(c).astype(np.float32),
                }
            )
        )
    return cases


def solve(inputs: dict[str, Any]) -> list[FloatArray]:
    r, c = int(inputs["R"]), int(inputs["C"])
    matrix = inputs["matrix"].reshape(r, c)
    out: FloatArray = matrix + inputs["row"].reshape(1, c)
    return [np.ascontiguousarray(out, dtype=np.float32)]

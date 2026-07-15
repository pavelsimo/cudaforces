"""Reference for the zero-exterior five-point stencil."""

from typing import Any

import numpy as np
import numpy.typing as npt

from ..types import RefCase

FloatArray = npt.NDArray[np.float32]

RTOL = 1e-5
ATOL = 1e-5


def tests() -> list[RefCase]:
    cases = [
        RefCase(
            {
                "H": 2,
                "W": 2,
                "inp": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
            }
        ),
        RefCase(
            {
                "H": 1,
                "W": 1,
                "inp": np.array([2.5], dtype=np.float32),
            }
        ),
        RefCase(
            {
                "H": 1,
                "W": 5,
                "inp": np.array([1.0, -1.0, 2.0, -2.0, 3.0], dtype=np.float32),
            }
        ),
    ]
    rng = np.random.default_rng(149)
    cases.append(
        RefCase(
            {
                "H": 353,
                "W": 521,
                "inp": rng.standard_normal(353 * 521).astype(np.float32),
            }
        )
    )
    return cases


def solve(inputs: dict[str, Any]) -> list[FloatArray]:
    h, w = int(inputs["H"]), int(inputs["W"])
    inp = inputs["inp"].reshape(h, w).astype(np.float32)
    padded = np.pad(inp, ((1, 1), (1, 1)))
    out = (np.float32(4.0) * inp - padded[:-2, 1:-1] - padded[2:, 1:-1] - padded[1:-1, :-2] - padded[1:-1, 2:]).astype(
        np.float32
    )
    return [out]

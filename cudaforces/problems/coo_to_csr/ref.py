"""Reference for canonical COO-to-CSR conversion."""

from typing import Any

import numpy as np
import numpy.typing as npt

from ..types import RefCase

FloatArray = npt.NDArray[np.float32]
IntArray = npt.NDArray[np.int32]

RTOL = 0.0
ATOL = 0.0


def tests() -> list[RefCase]:
    cases = [
        RefCase(
            {
                "R": 2,
                "C": 3,
                "NNZ": 3,
                "rows": np.array([1, 0, 1], dtype=np.int32),
                "cols": np.array([0, 2, 0], dtype=np.int32),
                "values": np.array([4.0, 3.0, 5.0], dtype=np.float32),
            }
        ),
        RefCase(
            {
                "R": 1,
                "C": 1,
                "NNZ": 1,
                "rows": np.array([0], dtype=np.int32),
                "cols": np.array([0], dtype=np.int32),
                "values": np.array([-1.5], dtype=np.float32),
            }
        ),
        RefCase(
            {
                "R": 8,
                "C": 7,
                "NNZ": 5,
                "rows": np.array([7, 7, 0, 7, 3], dtype=np.int32),
                "cols": np.array([6, 1, 2, 1, 4], dtype=np.int32),
                "values": np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32),
            }
        ),
    ]
    rng = np.random.default_rng(91)
    r, c, nnz = 257, 193, 6_001
    cases.append(
        RefCase(
            {
                "R": r,
                "C": c,
                "NNZ": nnz,
                "rows": rng.integers(0, r, size=nnz, dtype=np.int32),
                "cols": rng.integers(0, c, size=nnz, dtype=np.int32),
                "values": rng.standard_normal(nnz).astype(np.float32),
            }
        )
    )
    return cases


def solve(inputs: dict[str, Any]) -> list[IntArray | FloatArray]:
    rows = inputs["rows"]
    cols = inputs["cols"]
    original = np.arange(int(inputs["NNZ"]), dtype=np.int64)
    order = np.lexsort((original, cols, rows))
    sorted_rows = rows[order]
    counts = np.bincount(sorted_rows, minlength=int(inputs["R"]))
    row_ptr = np.empty(int(inputs["R"]) + 1, dtype=np.int32)
    row_ptr[0] = 0
    np.cumsum(counts, out=row_ptr[1:])
    col_out = np.ascontiguousarray(cols[order], dtype=np.int32)
    values_out = np.ascontiguousarray(inputs["values"][order], dtype=np.float32)
    return [row_ptr, col_out, values_out]

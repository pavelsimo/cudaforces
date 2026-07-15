"""Reference for sparse matrix-vector multiplication in CSR format."""

from typing import Any

import numpy as np
import numpy.typing as npt

from ..types import RefCase

FloatArray = npt.NDArray[np.float32]

RTOL = 1e-4
ATOL = 1e-5


def tests() -> list[RefCase]:
    cases = [
        RefCase(
            {
                "R": 3,
                "C": 3,
                "NNZ": 3,
                "row_ptr": np.array([0, 2, 2, 3], dtype=np.int32),
                "col_idx": np.array([0, 2, 1], dtype=np.int32),
                "values": np.array([2.0, 3.0, 4.0], dtype=np.float32),
                "x": np.array([5.0, 6.0, 7.0], dtype=np.float32),
            }
        ),
        RefCase(
            {
                "R": 1,
                "C": 1,
                "NNZ": 1,
                "row_ptr": np.array([0, 1], dtype=np.int32),
                "col_idx": np.array([0], dtype=np.int32),
                "values": np.array([-2.0], dtype=np.float32),
                "x": np.array([3.0], dtype=np.float32),
            }
        ),
        RefCase(
            {
                "R": 5,
                "C": 4,
                "NNZ": 3,
                "row_ptr": np.array([0, 0, 2, 2, 2, 3], dtype=np.int32),
                "col_idx": np.array([0, 3, 2], dtype=np.int32),
                "values": np.array([1.0, -2.0, 4.0], dtype=np.float32),
                "x": np.array([2.0, 3.0, -1.0, 0.5], dtype=np.float32),
            }
        ),
    ]
    rng = np.random.default_rng(157)
    r, c, nnz = 4_096, 8_192, 110_000
    counts = rng.multinomial(nnz, np.full(r, 1.0 / r))
    row_ptr = np.concatenate(([0], np.cumsum(counts))).astype(np.int32)
    cases.append(
        RefCase(
            {
                "R": r,
                "C": c,
                "NNZ": nnz,
                "row_ptr": row_ptr,
                "col_idx": rng.integers(0, c, size=nnz, dtype=np.int32),
                "values": rng.standard_normal(nnz).astype(np.float32),
                "x": rng.standard_normal(c).astype(np.float32),
            }
        )
    )
    return cases


def solve(inputs: dict[str, Any]) -> list[FloatArray]:
    r = int(inputs["R"])
    row_ptr = inputs["row_ptr"].astype(np.int32)
    col_idx = inputs["col_idx"].astype(np.int32)
    values: FloatArray = inputs["values"].astype(np.float32)
    x: FloatArray = inputs["x"].astype(np.float32)
    out = np.zeros(r, dtype=np.float32)
    for row in range(r):
        start, end = int(row_ptr[row]), int(row_ptr[row + 1])
        out[row] = np.sum(values[start:end] * x[col_idx[start:end]], dtype=np.float32)
    return [out]

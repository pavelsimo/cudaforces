"""Reference for directed breadth-first search over a CSR graph."""

from collections import deque
from typing import Any

import numpy as np
import numpy.typing as npt

from ..types import RefCase

IntArray = npt.NDArray[np.int32]

RTOL = 0.0
ATOL = 0.0


def _csr(vertex_count: int, edges: list[tuple[int, int]]) -> tuple[IntArray, IntArray]:
    adjacency: list[list[int]] = [[] for _ in range(vertex_count)]
    for source, target in edges:
        adjacency[source].append(target)
    row_ptr = np.zeros(vertex_count + 1, dtype=np.int32)
    row_ptr[1:] = np.cumsum([len(neighbors) for neighbors in adjacency], dtype=np.int32)
    col_idx = np.array([target for neighbors in adjacency for target in neighbors], dtype=np.int32)
    return row_ptr, col_idx


def tests() -> list[RefCase]:
    worked_edges = [(0, 1), (0, 2), (2, 3)]
    worked_row_ptr, worked_col_idx = _csr(4, worked_edges)

    self_loop_row_ptr, self_loop_col_idx = _csr(2, [(0, 0)])

    cyclic_edges = [(0, 1), (1, 2), (2, 0), (2, 3), (3, 3), (4, 5)]
    cyclic_row_ptr, cyclic_col_idx = _csr(6, cyclic_edges)

    vertex_count = 50_000
    large_edges = [(vertex, vertex + 1) for vertex in range(vertex_count - 1)]
    large_edges.extend((vertex, vertex + 2) for vertex in range(0, vertex_count - 2, 3))
    large_row_ptr, large_col_idx = _csr(vertex_count, large_edges)

    return [
        RefCase(
            {
                "V": 4,
                "E": len(worked_edges),
                "source": 0,
                "row_ptr": worked_row_ptr,
                "col_idx": worked_col_idx,
            }
        ),
        RefCase(
            {
                "V": 2,
                "E": 1,
                "source": 0,
                "row_ptr": self_loop_row_ptr,
                "col_idx": self_loop_col_idx,
            }
        ),
        RefCase(
            {
                "V": 6,
                "E": len(cyclic_edges),
                "source": 2,
                "row_ptr": cyclic_row_ptr,
                "col_idx": cyclic_col_idx,
            }
        ),
        RefCase(
            {
                "V": vertex_count,
                "E": len(large_edges),
                "source": 0,
                "row_ptr": large_row_ptr,
                "col_idx": large_col_idx,
            }
        ),
    ]


def solve(inputs: dict[str, Any]) -> list[IntArray]:
    vertex_count = inputs["V"]
    source = inputs["source"]
    row_ptr: IntArray = inputs["row_ptr"]
    col_idx: IntArray = inputs["col_idx"]
    distance = np.full(vertex_count, -1, dtype=np.int32)
    distance[source] = 0
    frontier = deque([source])
    while frontier:
        vertex = frontier.popleft()
        for edge in range(int(row_ptr[vertex]), int(row_ptr[vertex + 1])):
            neighbor = int(col_idx[edge])
            if distance[neighbor] == -1:
                distance[neighbor] = distance[vertex] + 1
                frontier.append(neighbor)
    return [distance]

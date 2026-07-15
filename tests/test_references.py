"""Sanity checks for the NumPy reference implementations."""

import numpy as np
import pytest

from cudaforces import problems

ALL_SLUGS = [p.slug for p in problems.all_problems()]
NEW_SLUGS = set(ALL_SLUGS[:30])


@pytest.mark.parametrize("slug", ALL_SLUGS)
def test_reference_produces_output(slug: str) -> None:
    ref = problems.ref_module(slug)
    cases = ref.tests()
    minimum = 4 if slug in NEW_SLUGS else 2
    assert len(cases) >= minimum, "need the worked example plus edge and random tests"
    for case in cases:
        total_values = sum(np.asarray(v).size for v in case.inputs.values() if isinstance(v, np.ndarray))
        assert total_values <= 250_000, f"{slug}: test too large ({total_values} values)"
        outputs = ref.solve(case.inputs)
        assert len(outputs) >= 1
        assert any(np.asarray(arr).size > 0 for arr in outputs)
        for arr in outputs:
            values = np.asarray(arr)
            assert np.all(np.isfinite(values.astype(np.float64)))


def _example_outputs(slug: str) -> list[np.ndarray]:
    ref = problems.ref_module(slug)
    return [np.asarray(a).ravel() for a in ref.solve(ref.tests()[0].inputs)]


def test_gelu_forward_known_values() -> None:
    (out,) = _example_outputs("gelu-forward")
    np.testing.assert_allclose(out, [-0.1588, 0.0, 0.8412, 1.9546], atol=1e-4)


def test_softmax_forward_known_values() -> None:
    (out,) = _example_outputs("softmax-forward")
    np.testing.assert_allclose(out, [0.0900, 0.2447, 0.6652], atol=1e-4)


def test_layernorm_forward_known_values() -> None:
    out, mean, rstd = _example_outputs("layernorm-forward")
    np.testing.assert_allclose(out, [-1.2247, 0.0, 1.2247], atol=1e-3)
    np.testing.assert_allclose(mean, [2.0], atol=1e-6)
    np.testing.assert_allclose(rstd, [1.2247], atol=1e-3)


def test_crossentropy_forward_known_values() -> None:
    (losses,) = _example_outputs("crossentropy-forward")
    np.testing.assert_allclose(losses, [0.6931, 0.1054], atol=1e-4)


def test_adamw_known_values() -> None:
    params, m, v = _example_outputs("adamw")
    np.testing.assert_allclose(params, [0.9], atol=1e-6)
    np.testing.assert_allclose(m, [0.05], atol=1e-7)
    np.testing.assert_allclose(v, [0.00025], rtol=1e-4)


def test_attention_backward_single_position() -> None:
    dq, dk, dv = _example_outputs("attention-backward")
    np.testing.assert_allclose(dq, [0.0], atol=1e-7)
    np.testing.assert_allclose(dk, [0.0], atol=1e-7)
    np.testing.assert_allclose(dv, [1.0], atol=1e-7)


def test_foundation_reduction_and_selection_rules() -> None:
    argmax = problems.ref_module("argmax-reduction")
    value, index = argmax.solve({"N": 5, "inp": np.array([1, 8, 3, 8, 2], dtype=np.float32)})
    np.testing.assert_array_equal(value, [8])
    np.testing.assert_array_equal(index, [1])

    segmented = problems.ref_module("segmented-sum")
    (sums,) = segmented.solve(
        {
            "N": 3,
            "S": 4,
            "values": np.array([1, 2, 3], dtype=np.float32),
            "offsets": np.array([0, 0, 2, 2, 3], dtype=np.int32),
        }
    )
    np.testing.assert_array_equal(sums, [0, 3, 0, 3])

    compact = problems.ref_module("stream-compaction")
    count, kept = compact.solve({"N": 3, "inp": np.array([0, -1, -2], dtype=np.float32)})
    np.testing.assert_array_equal(count, [0])
    assert kept.size == 0


def test_foundation_atomic_and_ordering_rules() -> None:
    histogram = problems.ref_module("integer-histogram")
    (bins,) = histogram.solve({"N": 5, "B": 4, "inp": np.array([3, 3, 3, 3, 0], dtype=np.int32)})
    np.testing.assert_array_equal(bins, [1, 0, 0, 4])

    scatter = problems.ref_module("scatter-add")
    (out,) = scatter.solve(
        {
            "N": 4,
            "M": 3,
            "values": np.array([1, 2, 3, 4], dtype=np.float32),
            "indices": np.array([0, 2, 0, 2], dtype=np.int32),
        }
    )
    np.testing.assert_array_equal(out, [4, 0, 6])

    rle = problems.ref_module("run-length-encoding")
    runs, unique, counts = rle.solve({"N": 6, "inp": np.array([2, 2, 2, 5, 5, 1], dtype=np.int32)})
    np.testing.assert_array_equal(runs, [3])
    np.testing.assert_array_equal(unique, [2, 5, 1])
    np.testing.assert_array_equal(counts, [3, 2, 1])


def test_foundation_sorting_stability_and_ties() -> None:
    radix = problems.ref_module("radix-digit-pass")
    keys, values = radix.solve(
        {
            "N": 4,
            "shift": 0,
            "keys": np.array([258, 1, 514, 2], dtype=np.int32),
            "values": np.array([0, 1, 2, 3], dtype=np.int32),
        }
    )
    np.testing.assert_array_equal(keys, [1, 258, 514, 2])
    np.testing.assert_array_equal(values, [1, 0, 2, 3])

    top_k = problems.ref_module("top-k-selection")
    selected, indices = top_k.solve({"N": 5, "K": 3, "inp": np.array([4, 9, 2, 9, 7], dtype=np.float32)})
    np.testing.assert_array_equal(selected, [9, 9, 7])
    np.testing.assert_array_equal(indices, [1, 3, 4])


def test_foundation_neighborhood_boundaries() -> None:
    convolution = problems.ref_module("convolution-1d")
    (out,) = convolution.solve(
        {
            "N": 4,
            "K": 3,
            "inp": np.array([1, 2, 3, 4], dtype=np.float32),
            "kernel": np.array([1, 0, -1], dtype=np.float32),
        }
    )
    np.testing.assert_array_equal(out, [-2, -2, -2, 3])

    stencil = problems.ref_module("five-point-stencil")
    (out,) = stencil.solve({"H": 2, "W": 2, "inp": np.array([1, 2, 3, 4], dtype=np.float32)})
    np.testing.assert_array_equal(out.reshape(2, 2), [[-1, 3], [7, 11]])


def test_foundation_sparse_graph_and_simulation_rules() -> None:
    spmv = problems.ref_module("csr-spmv")
    (out,) = spmv.solve(
        {
            "R": 3,
            "C": 2,
            "NNZ": 1,
            "row_ptr": np.array([0, 0, 1, 1], dtype=np.int32),
            "col_idx": np.array([1], dtype=np.int32),
            "values": np.array([3], dtype=np.float32),
            "x": np.array([2, 4], dtype=np.float32),
        }
    )
    np.testing.assert_array_equal(out, [0, 12, 0])

    coo = problems.ref_module("coo-to-csr")
    row_ptr, columns, values = coo.solve(
        {
            "R": 2,
            "C": 3,
            "NNZ": 4,
            "rows": np.array([1, 0, 1, 1], dtype=np.int32),
            "cols": np.array([0, 2, 0, 0], dtype=np.int32),
            "values": np.array([4, 3, 5, 6], dtype=np.float32),
        }
    )
    np.testing.assert_array_equal(row_ptr, [0, 1, 4])
    np.testing.assert_array_equal(columns, [2, 0, 0, 0])
    np.testing.assert_array_equal(values, [3, 4, 5, 6])

    bfs = problems.ref_module("breadth-first-search")
    (distance,) = bfs.solve(
        {
            "V": 4,
            "E": 4,
            "source": 0,
            "row_ptr": np.array([0, 1, 2, 3, 4], dtype=np.int32),
            "col_idx": np.array([1, 2, 0, 3], dtype=np.int32),
        }
    )
    np.testing.assert_array_equal(distance, [0, 1, 2, -1])

    nbody = problems.ref_module("n-body-acceleration")
    (acceleration,) = nbody.solve(
        {
            "N": 2,
            "softening": 0.1,
            "position": np.array([-1, 0, 0, 1, 0, 0], dtype=np.float32),
            "mass": np.array([1, 1], dtype=np.float32),
        }
    )
    np.testing.assert_allclose(acceleration[0], -acceleration[1], atol=1e-7)

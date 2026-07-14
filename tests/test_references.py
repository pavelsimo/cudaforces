"""Sanity checks for the NumPy reference implementations."""

import numpy as np
import pytest

from cudaforces import problems

ALL_SLUGS = [p.slug for p in problems.all_problems()]


@pytest.mark.parametrize("slug", ALL_SLUGS)
def test_reference_produces_output(slug: str) -> None:
    ref = problems.ref_module(slug)
    cases = ref.tests()
    assert len(cases) >= 2, "need the worked example plus random tests"
    for case in cases:
        total_values = sum(np.asarray(v).size for v in case.inputs.values() if isinstance(v, np.ndarray))
        assert total_values <= 250_000, f"{slug}: test too large ({total_values} values)"
    outputs = ref.solve(cases[0].inputs)
    assert len(outputs) >= 1
    for arr in outputs:
        squeezed = np.asarray(arr)
        assert squeezed.size > 0
        assert np.all(np.isfinite(squeezed.astype(np.float64)))


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

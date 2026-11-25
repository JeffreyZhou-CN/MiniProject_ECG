"""Checks for the baseline LMS canceller (src/baseline.py)."""

import numpy as np
import pytest

from src.baseline import lms_filter
from src.data import PATHWAY, make_ecg


def test_hand_computed_example_from_guide():
    """Hand-computable example from the course guide: d = r = [1, 1], M = 1, mu = 0.5."""
    
    e, w = lms_filter([1.0, 1.0], [1.0, 1.0], M=1, mu=0.5)
    np.testing.assert_allclose(e, [1.0, 0.5])
    np.testing.assert_allclose(w, [0.75])


def test_hand_computed_two_taps():
    """Hand-computable example with M = 2, checking the delay line."""
    
    e, w = lms_filter([1.0, 2.0], [1.0, 1.0], M=2, mu=0.5)
    np.testing.assert_allclose(e, [1.0, 1.5])
    np.testing.assert_allclose(w, [1.25, 0.75])


def test_zero_reference_leaves_signal_unchanged():
    """Edge case: with a zero reference there is nothing to learn, so e = d and w stays 0."""
   
    d = np.random.default_rng(0).normal(size=100)
    e, w = lms_filter(d, np.zeros(100), M=4, mu=0.1)
    np.testing.assert_array_equal(e, d)
    np.testing.assert_array_equal(w, np.zeros(4))


def test_zero_step_size_does_not_adapt():
    """Edge case: mu = 0 means no adaptation, so e = d and w stays 0."""
    
    rng = np.random.default_rng(1)
    d, r = rng.normal(size=100), rng.normal(size=100)
    e, w = lms_filter(d, r, M=3, mu=0.0)
    np.testing.assert_array_equal(e, d)
    np.testing.assert_array_equal(w, np.zeros(3))


def test_converges_to_true_pathway_without_ecg():
    """Property: with no ECG, the weights converge to the true pathway (padded with zeros)."""
   
    rng = np.random.default_rng(2)
    r = rng.normal(size=20_000)
    d = np.convolve(r, PATHWAY)[:len(r)]
    _, w = lms_filter(d, r, M=5, mu=0.01)
    np.testing.assert_allclose(w, np.r_[PATHWAY, 0.0, 0.0], atol=1e-3)


def test_reduces_interference_on_synthetic_ecg():
    """Property: on a synthetic recording the output is much closer to the clean ECG than the input."""
   
    clean, measured, reference = make_ecg(0)
    e, _ = lms_filter(measured, reference, M=3, mu=0.05)
    warmup = 250
    mse_in = np.mean((measured[warmup:] - clean[warmup:]) ** 2)
    mse_out = np.mean((e[warmup:] - clean[warmup:]) ** 2)
    assert mse_out < 0.1 * mse_in


def test_shapes_finite_and_inputs_not_mutated():
    """Output shapes are correct, values are finite and the input arrays are not modified."""
   
    rng = np.random.default_rng(3)
    d, r = rng.normal(size=200), rng.normal(size=200)
    d_copy, r_copy = d.copy(), r.copy()
    e, w = lms_filter(d, r, M=4, mu=0.05)
    assert e.shape == (200,) and w.shape == (4,)
    assert np.all(np.isfinite(e)) and np.all(np.isfinite(w))
    np.testing.assert_array_equal(d, d_copy)
    np.testing.assert_array_equal(r, r_copy)


@pytest.mark.parametrize("d, r, M, mu", [
    ([1.0, 2.0], [1.0], 1, 0.1),   # lengths differ
    ([1.0], [1.0], 0, 0.1),        # M < 1
    ([1.0], [1.0], 1.5, 0.1),      # M not an integer
    ([1.0], [1.0], 1, -0.1),       # negative step size
])

def test_invalid_arguments_raise(d, r, M, mu):
    """Edge case: invalid inputs raise ValueError."""

    with pytest.raises(ValueError):
        lms_filter(d, r, M, mu)
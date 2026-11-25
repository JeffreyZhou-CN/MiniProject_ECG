"""Checks for the modified NLMS canceller (src/modified.py)."""

import numpy as np
import pytest

from src.baseline import lms_filter
from src.data import PATHWAY
from src.modified import nlms_filter

def test_hand_computed_example_and_lms_overshoot():
    """Hand-computable example: d = [1, 1], r = [2, 2], M = 1, mu = 0.5, eps = 0."""
    # Energy x·x = 4, effective step 0.125. NLMS: e = [1, 0.5], w = 0.375.
    e, w = nlms_filter([1.0, 1.0], [2.0, 2.0], M=1, mu=0.5, eps=0.0)
    np.testing.assert_allclose(e, [1.0, 0.5])
    np.testing.assert_allclose(w, [0.375])
    # With the same inputs LMS overshoots: e = [1, -1], w = 0.
    e_lms, w_lms = lms_filter([1.0, 1.0], [2.0, 2.0], M=1, mu=0.5)
    np.testing.assert_allclose(e_lms, [1.0, -1.0])
    np.testing.assert_allclose(w_lms, [0.0])


@pytest.mark.parametrize("c", [0.1, 10.0, 1000.0])
def test_output_invariant_to_reference_scale(c):
    """Property: with eps = 0, scaling r by c leaves e unchanged and divides w by c."""
   
    rng = np.random.default_rng(7)
    d, r = rng.normal(size=1000), rng.normal(size=1000)
    e1, w1 = nlms_filter(d, r, M=4, mu=0.3, eps=0.0)
    e2, w2 = nlms_filter(d, c * r, M=4, mu=0.3, eps=0.0)
    np.testing.assert_allclose(e2, e1, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(c * w2, w1, rtol=1e-9, atol=1e-9)


def test_ablation_without_energy_equals_baseline():
    """Ablation: normalize=False without a fixed energy reproduces the baseline LMS exactly."""
   
    rng = np.random.default_rng(8)
    d, r = rng.normal(size=500), rng.normal(size=500)
    e, w = nlms_filter(d, r, M=3, mu=0.02, normalize=False)
    e_lms, w_lms = lms_filter(d, r, M=3, mu=0.02)
    np.testing.assert_array_equal(e, e_lms)
    np.testing.assert_array_equal(w, w_lms)


def test_ablation_with_fixed_energy_equals_rescaled_lms():
    """Ablation: a fixed energy E gives LMS with step mu / (eps + E)."""
   
    rng = np.random.default_rng(9)
    d, r = rng.normal(size=500), rng.normal(size=500)
    mu, eps, energy = 0.4, 1e-6, 3.0
    e, w = nlms_filter(d, r, M=3, mu=mu, eps=eps, normalize=False, reference_energy=energy)
    e_lms, w_lms = lms_filter(d, r, M=3, mu=mu / (eps + energy))
    np.testing.assert_allclose(e, e_lms, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(w, w_lms, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("eps", [0.0, 1e-6])
def test_zero_reference_leaves_signal_unchanged(eps):
    """Edge case: a zero reference gives e = d and w = 0, without division by zero."""
  
    d = np.random.default_rng(0).normal(size=100)
    e, w = nlms_filter(d, np.zeros(100), M=4, mu=0.5, eps=eps)
    np.testing.assert_array_equal(e, d)
    np.testing.assert_array_equal(w, np.zeros(4))


def test_converges_to_true_pathway_without_ecg():
    """Property: with no ECG, the weights converge to the true pathway."""
  
    rng = np.random.default_rng(2)
    r = rng.normal(size=5_000)
    d = np.convolve(r, PATHWAY)[:len(r)]
    _, w = nlms_filter(d, r, M=5, mu=0.5)
    np.testing.assert_allclose(w, np.r_[PATHWAY, 0.0, 0.0], atol=1e-3)


def test_stable_at_large_reference_where_lms_diverges():
    """Property: at a 100x reference amplitude NLMS stays stable, LMS with mu = 0.05 diverges."""

    rng = np.random.default_rng(3)
    r = 100.0 * rng.normal(size=2_000)
    d = np.convolve(r, PATHWAY)[:len(r)] + 0.01 * rng.normal(size=len(r))
    e_nlms, _ = nlms_filter(d, r, M=3, mu=0.5)
    assert np.all(np.isfinite(e_nlms))
    assert np.mean(e_nlms[-500:] ** 2) < 1e-2 * np.mean(d ** 2)
    # errstate silences the overflow warnings produced when LMS diverges.
    with np.errstate(all="ignore"):
        e_lms, _ = lms_filter(d, r, M=3, mu=0.05)
    assert not np.all(np.isfinite(e_lms)) or np.max(np.abs(e_lms)) > 1e6


def test_shapes_finite_and_inputs_not_mutated():
    """Output shapes are correct, values are finite and the inputs are not modified."""

    # Output shapes are correct, values are finite and the input arrays are unchanged.
    rng = np.random.default_rng(4)
    d, r = rng.normal(size=200), rng.normal(size=200)
    d_copy, r_copy = d.copy(), r.copy()
    e, w = nlms_filter(d, r, M=4, mu=0.2)
    assert e.shape == (200,) and w.shape == (4,)
    assert np.all(np.isfinite(e)) and np.all(np.isfinite(w))
    np.testing.assert_array_equal(d, d_copy)
    np.testing.assert_array_equal(r, r_copy)


@pytest.mark.parametrize("kwargs", [
    dict(mu=-0.1),                          # negative step size
    dict(mu=0.1, eps=-1e-6),                # negative eps
    dict(mu=0.1, reference_energy=1.0),     # fixed energy needs normalize=False
])
def test_invalid_arguments_raise(kwargs):
    """Edge case: invalid arguments raise ValueError."""

    # Edge case: a negative step, a negative eps, or a fixed energy in NLMS mode must raise.
    with pytest.raises(ValueError):
        nlms_filter([1.0, 2.0], [1.0, 1.0], M=1, **kwargs)
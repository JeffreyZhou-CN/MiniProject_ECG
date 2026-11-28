"""Checks for the experiment building blocks (src/experiment.py)."""

import numpy as np
import pytest

from src.baseline import lms_filter
from src.data import make_ecg
from src.experiment import run_method, select_parameters
from src.modified import nlms_filter


def _signals():
    """Small random test signals (d, r), fixed by a seed."""
   
    rng = np.random.default_rng(0)
    return rng.normal(size=300), rng.normal(size=300)


def test_run_method_matches_direct_calls():
    """Each method name gives exactly the same output as calling its filter directly."""
    
    d, r = _signals()
    np.testing.assert_array_equal(run_method("lms", d, r, {"M": 3, "mu": 0.02}),
                                  lms_filter(d, r, 3, 0.02)[0])
    np.testing.assert_array_equal(run_method("nlms", d, r, {"M": 3, "mu": 0.2}, eps=1e-6),
                                  nlms_filter(d, r, 3, 0.2, eps=1e-6)[0])
    np.testing.assert_array_equal(
        run_method("nlms_ablation", d, r, {"M": 3, "mu": 0.2, "energy": 3.0}, eps=1e-6),
        nlms_filter(d, r, 3, 0.2, eps=1e-6, normalize=False, reference_energy=3.0)[0])


def test_run_method_unknown_name_raises():
    """Edge case: an unknown method name raises ValueError."""
  
    d, r = _signals()
    with pytest.raises(ValueError):
        run_method("nmls", d, r, {"M": 3, "mu": 0.1})



def test_select_parameters_returns_best_of_table():
    """The selected (M, mu) is the row with the highest mean SNR, and every grid point is listed."""

    recordings = [make_ecg(seed) for seed in (100, 101)]
    best, table = select_parameters("lms", [2, 3], [0.01, 0.05], recordings, warmup=250)
    assert len(table) == 4
    top = max(table, key=lambda row: row["mean_val_snr_db"])
    assert (best["M"], best["mu"]) == (top["M"], top["mu"])
    assert best["score"] == top["mean_val_snr_db"]


def test_select_parameters_never_picks_a_diverging_step():
    """A step size that makes LMS diverge scores -inf and is not selected."""

    recordings = [make_ecg(100)]
    best, table = select_parameters("lms", [3], [0.05, 5.0], recordings, warmup=250)
    diverged = [row for row in table if row["mu"] == 5.0][0]
    assert diverged["mean_val_snr_db"] == -np.inf
    assert best["mu"] == 0.05
"""Checks for the experiment building blocks (src/experiment.py)."""

import numpy as np
import pytest

from src.baseline import lms_filter
from src.experiment import run_method
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
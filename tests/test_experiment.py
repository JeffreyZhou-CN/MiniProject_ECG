"""Checks for the experiment building blocks (src/experiment.py)."""

import numpy as np
import pytest

from src.baseline import lms_filter
from src.data import make_ecg
from src.experiment import (check_disjoint, evaluate_case, expand_seeds, mean_input_energy,run_method, select_parameters)
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


def test_mean_input_energy_hand_computed():
    """Hand example: r = [1, 2, 3], M = 2 gives energies [1, 5, 13] and mean 19/3."""
   
    assert mean_input_energy([[1.0, 2.0, 3.0]], M=2) == pytest.approx(19 / 3)


def test_mean_input_energy_matches_filter_loop():
    """Property: agrees with x·x computed sample by sample with the filters' delay line."""
    
    references = [np.random.default_rng(seed).normal(size=200) for seed in (0, 1)]
    M = 3
    loop_energies = []
    for r in references:
        x = np.zeros(M)
        for value in r:
            x[1:] = x[:-1]
            x[0] = value
            loop_energies.append(x @ x)
    assert mean_input_energy(references, M) == pytest.approx(np.mean(loop_energies))



def test_evaluate_case_reports_all_metrics_consistently():
    """All metrics are present, the gain equals output minus input SNR, and filtering helps."""
    # All metrics are present; the gain equals output SNR minus input SNR; LMS does
    # improve the SNR on this recording.
    clean, measured, reference = make_ecg(100)
    result = evaluate_case("lms", clean, measured, reference, {"M": 3, "mu": 0.05}, warmup=250)
    assert set(result) == {"snr_out_db", "snr_in_db", "snr_gain_db", "mse", "peak_error",
                           "snr_warmup_db", "snr_post_change_db", "diverged"}
    assert result["snr_gain_db"] == pytest.approx(result["snr_out_db"] - result["snr_in_db"])
    assert result["snr_out_db"] > result["snr_in_db"]
    assert result["diverged"] is False
    assert np.isnan(result["snr_post_change_db"])


def test_evaluate_case_flags_divergence():
    """A diverging LMS is flagged and scored as SNR = -inf."""
    # A diverging LMS (mu = 5) is flagged and its output SNR is -inf.
    clean, measured, reference = make_ecg(100)
    result = evaluate_case("lms", clean, measured, reference, {"M": 3, "mu": 5.0}, warmup=250)
    assert result["diverged"] is True
    assert result["snr_out_db"] == -np.inf


def test_evaluate_case_post_change_window():
    """With a pathway change, the first-second SNR after the change is a finite number."""
    # With a pathway change at 6 s (sample 1500), the SNR in the first second after the
    # change is a finite number.
    clean, measured, reference = make_ecg(100, pathway_after=[-0.2, 0.3, 0.1], change_time=6.0)
    result = evaluate_case("nlms", clean, measured, reference, {"M": 3, "mu": 0.1},
                           warmup=250, change_index=1500)
    assert np.isfinite(result["snr_post_change_db"])



def test_expand_seeds():
    """{"start": 100, "count": 3} expands to [100, 101, 102]."""
    # Hand example of expanding a seed specification.
    assert expand_seeds({"start": 100, "count": 3}) == [100, 101, 102]


def test_check_disjoint_accepts_project_splits_and_rejects_overlap():
    """The project's splits are disjoint; an overlapping split raises ValueError."""
    # The project's three seed groups (0-19, 100-109, 1000-1019) do not overlap and pass;
    # if the validation set accidentally reuses test seeds, it must raise.
    development = expand_seeds({"start": 0, "count": 20})
    validation = expand_seeds({"start": 100, "count": 10})
    test = expand_seeds({"start": 1000, "count": 20})
    check_disjoint(development, validation, test)
    with pytest.raises(ValueError):
        check_disjoint(development, expand_seeds({"start": 1005, "count": 10}), test)
"""Checks for the configuration handling in run_experiments.py."""

import copy

import numpy as np
import pytest

from run_experiments import (ROOT, evaluate_test_cases, load_config, select_all_parameters, summarise, validate_config)


@pytest.mark.parametrize("name", ["smoke", "final"])
def test_project_configs_are_valid(name):
    """Both saved configurations pass the fairness checks."""
    load_config(ROOT / "configs" / f"{name}.json")


def test_overlapping_seeds_are_rejected():
    """A validation split that reuses test seeds is rejected."""
    config = copy.deepcopy(load_config(ROOT / "configs" / "final.json"))
    config["validation"]["seeds"] = {"start": 1000, "count": 10}
    with pytest.raises(ValueError):
        validate_config(config)


def test_unequal_tuning_budgets_are_rejected():
    """Grids with different numbers of step sizes for LMS and NLMS are rejected."""

    config = copy.deepcopy(load_config(ROOT / "configs" / "final.json"))
    config["grid"]["nlms_mu"] = config["grid"]["nlms_mu"][:-1]
    with pytest.raises(ValueError):
        validate_config(config)



def test_evaluate_test_cases_covers_every_case():
    """One row per condition x seed x method; post-change SNR only for the pathway condition."""

    config = copy.deepcopy(load_config(ROOT / "configs" / "smoke.json"))
    config["test"]["seeds"] = {"start": 1000, "count": 2}
    params, _ = select_all_parameters(config)
    rows = evaluate_test_cases(config, params)
    n_conditions = len(config["test"]["conditions"])
    assert len(rows) == n_conditions * 2 * 3
    for row in rows:
        has_change = row["condition"] == "pathway_change"
        assert has_change == (row["snr_post_change_db"] == row["snr_post_change_db"])
        # NaN is not equal to itself, so x == x is False exactly when x is NaN.



def _row(method, seed, snr, diverged=False):
    """A minimal fake result row for testing summarise."""
   
    return {"condition": "c", "seed": seed, "method": method, "snr_out_db": snr,
            "snr_in_db": 0.0, "peak_error": 0.1, "snr_warmup_db": 5.0,
            "snr_post_change_db": np.nan, "diverged": diverged}


def test_summarise_hand_computed():
    """Hand example: NLMS [12, 14], LMS [10, 11], ablation [11, 13] over two seeds."""
   
    # NLMS mean 13, LMS mean 10.5; NLMS - LMS differences [2, 3], mean 2.5;
    # NLMS - ablation differences [1, 1], mean 1.
    rows = [_row("nlms", 2, 14.0), _row("nlms", 1, 12.0),
            _row("lms", 1, 10.0), _row("lms", 2, 11.0),
            _row("nlms_ablation", 1, 11.0), _row("nlms_ablation", 2, 13.0)]
    summary, paired = summarise(rows, ["c"])
    means = {row["method"]: row["snr_out_db"] for row in summary}
    assert means["nlms"] == pytest.approx(13.0) and means["lms"] == pytest.approx(10.5)
    snr = {row["comparison"]: row for row in paired if row["metric"] == "snr_out_db"}
    assert snr["nlms - lms"]["mean"] == pytest.approx(2.5)
    assert snr["nlms - nlms_ablation"]["mean"] == pytest.approx(1.0)
    
    # Note that the NLMS rows are deliberately given in reverse seed order, to check that
    # sorting still pairs them correctly.


def test_summarise_counts_divergence_and_excludes_it():
    """A diverged LMS case is counted and excluded from the LMS mean and the paired interval."""
    
    rows = [_row("nlms", 1, 12.0), _row("nlms", 2, 14.0),
            _row("lms", 1, 10.0), _row("lms", 2, -np.inf, diverged=True),
            _row("nlms_ablation", 1, 11.0), _row("nlms_ablation", 2, 13.0)]
    summary, paired = summarise(rows, ["c"])
    lms = next(row for row in summary if row["method"] == "lms")
    assert lms["n_diverged"] == 1 and lms["snr_out_db"] == pytest.approx(10.0)
    pair = next(row for row in paired
                if row["comparison"] == "nlms - lms" and row["metric"] == "snr_out_db")
    assert pair["n"] == 1 and pair["n_excluded"] == 1
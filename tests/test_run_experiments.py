"""Checks for the configuration handling in run_experiments.py."""

import copy

import pytest

from run_experiments import ROOT, evaluate_test_cases, load_config, select_all_parameters, validate_config


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
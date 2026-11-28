"""Checks for the configuration handling in run_experiments.py."""

import copy

import pytest

from run_experiments import ROOT, load_config, validate_config


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
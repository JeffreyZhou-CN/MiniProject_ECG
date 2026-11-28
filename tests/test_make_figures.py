"""Checks for the figure script (make_figures.py)."""


import numpy as np

from make_figures import (ROOT, error_power, figure_adaptation, figure_example, figure_failures, figure_paired)


def test_figure_paired_writes_a_png(tmp_path):
    """Figure 2 is created from the committed final results."""
    # Generate figure 2 from the committed final results and check that the image file
    # exists and is not empty.
    # tmp_path is a temporary folder provided by pytest and cleaned up afterwards, so the
    # project is left untouched.
    figure_paired(ROOT / "results" / "final", tmp_path)
    output = tmp_path / "fig2_paired_snr.png"
    assert output.exists() and output.stat().st_size > 10_000



def test_figure_example_writes_a_png(tmp_path):
    """Figure 1 is created from the saved configuration and parameters."""
    # Generate figure 1 from the saved configuration and parameters and check that the
    # image file exists and is not empty.
    figure_example(ROOT / "results" / "final", tmp_path)
    output = tmp_path / "fig1_example.png"
    assert output.exists() and output.stat().st_size > 10_000



def test_error_power_hand_computed():
    """Hand example: error [2, 0, 0, 0], k = 2 samples gives [4, 2, 0, 0]."""

    power = error_power(clean=np.zeros(4), estimate=[2.0, 0.0, 0.0, 0.0], fs=10, smooth_seconds=0.2)
    np.testing.assert_allclose(power, [4.0, 2.0, 0.0, 0.0])


def test_figure_adaptation_writes_a_png(tmp_path):
    """Figure 3 is created from the saved configuration and parameters."""

    figure_adaptation(ROOT / "results" / "final", tmp_path)
    output = tmp_path / "fig3_adaptation.png"
    assert output.exists() and output.stat().st_size > 10_000


def test_figure_failures_writes_a_png(tmp_path):
    """Figure 4 is created from the saved results, configuration and parameters."""

    figure_failures(ROOT / "results" / "final", tmp_path)
    output = tmp_path / "fig4_failure_cases.png"
    assert output.exists() and output.stat().st_size > 10_000
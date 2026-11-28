"""Checks for the figure script (make_figures.py)."""


from make_figures import ROOT, figure_paired


def test_figure_paired_writes_a_png(tmp_path):
    """Figure 2 is created from the committed final results."""
    # Generate figure 2 from the committed final results and check that the image file
    # exists and is not empty.
    # tmp_path is a temporary folder provided by pytest and cleaned up afterwards, so the
    # project is left untouched.
    figure_paired(ROOT / "results" / "final", tmp_path)
    output = tmp_path / "fig2_paired_snr.png"
    assert output.exists() and output.stat().st_size > 10_000
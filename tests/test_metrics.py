"""Checks for the evaluation metrics (src/metrics.py)."""

import numpy as np
import pytest

from src.data import make_ecg
from src.metrics import mse, paired_difference, peak_amplitude_error, r_peak_indices, snr_db
from scipy.stats import t as student_t


def test_hand_computed_mse_and_snr():
    """Hand-computable example: s = [1, 1], s_hat = [1.1, 0.9] gives MSE 0.01 and SNR 20 dB."""

    assert mse([1.0, 1.0], [1.1, 0.9]) == pytest.approx(0.01)
    assert snr_db([1.0, 1.0], [1.1, 0.9]) == pytest.approx(20.0)


def test_warmup_samples_are_excluded():
    """Samples before start are ignored, so a large error there does not matter."""
    
    clean = [1.0, 1.0, 1.0]
    estimate = [100.0, 1.1, 0.9]
    assert mse(clean, estimate, start=1) == pytest.approx(0.01)
    assert snr_db(clean, estimate, start=1) == pytest.approx(20.0)


def test_exact_and_diverged_estimates():
    """Limiting cases: an exact estimate gives SNR +inf; a non-finite one gives -inf."""
    
    clean = np.array([1.0, -2.0, 3.0])
    assert snr_db(clean, clean) == np.inf
    assert mse(clean, clean) == 0.0
    diverged = np.array([1.0, np.inf, np.nan])
    assert snr_db(clean, diverged) == -np.inf
    assert mse(clean, diverged) == np.inf


def test_input_snr_matches_definition_on_synthetic_ecg():
    """The input SNR (estimate = measured signal) agrees with the formula written out directly."""
    
    clean, measured, _ = make_ecg(0)
    s, d = clean[250:], measured[250:]
    expected = 10 * np.log10(np.sum(s ** 2) / np.sum((d - s) ** 2))
    assert snr_db(clean, measured, start=250) == pytest.approx(expected)


@pytest.mark.parametrize("clean, estimate, start", [
    ([1.0, 2.0], [1.0], 0),        # lengths differ
    ([1.0, 2.0], [1.0, 2.0], 2),   # start outside the signal
    ([[1.0]], [[1.0]], 0),         # not 1-D
])
def test_invalid_arguments_raise(clean, estimate, start):
    """Edge case: mismatched shapes or an invalid start raise ValueError."""

    with pytest.raises(ValueError):
        snr_db(clean, estimate, start=start)



def test_r_peaks_found_once_per_beat():
    """Property: one R peak per beat (beats every 0.75-1.05 s over 12 s), each of height about 1."""
   
    clean, _, _ = make_ecg(0)
    peaks = r_peak_indices(clean)
    assert 10 <= len(peaks) <= 16
    np.testing.assert_allclose(clean[peaks], 1.0, atol=0.05)


def test_peak_error_hand_checked():
    """Exact estimate gives 0; halving the signal gives the mean of half the peak heights."""
   
    clean, _, _ = make_ecg(0)
    peaks = r_peak_indices(clean)
    assert peak_amplitude_error(clean, clean) == 0.0
    assert peak_amplitude_error(clean, 0.5 * clean) == pytest.approx(np.mean(0.5 * clean[peaks]))


def test_peak_error_limiting_cases():
    """Limiting cases: no peak after start gives nan; a non-finite estimate gives inf."""
   
    clean, _, _ = make_ecg(0)
    assert np.isnan(peak_amplitude_error(clean, clean, start=len(clean) - 1))
    assert peak_amplitude_error(clean, np.full_like(clean, np.inf)) == np.inf



def test_paired_difference_hand_computed():
    """Hand example: differences [2, 3, 2] give mean 7/3 and a t-interval computed by hand."""

    result = paired_difference([3.0, 5.0, 4.0], [1.0, 2.0, 2.0])
    half = student_t.ppf(0.975, 2) * np.sqrt(1 / 3) / np.sqrt(3)
    assert result["n"] == 3 and result["n_excluded"] == 0
    assert result["mean"] == pytest.approx(7 / 3)
    assert result["sd"] == pytest.approx(np.sqrt(1 / 3))
    assert result["ci_low"] == pytest.approx(7 / 3 - half)
    assert result["ci_high"] == pytest.approx(7 / 3 + half)
    assert result["n_a_greater"] == 3


def test_paired_difference_excludes_diverged_pairs():
    """A pair with -inf is excluded from the mean but still counted as a > b."""
    
    result = paired_difference([10.0, 12.0, 11.0], [8.0, -np.inf, 9.0])
    assert result["n"] == 2 and result["n_excluded"] == 1
    assert result["mean"] == pytest.approx(2.0)
    assert result["n_a_greater"] == 3


def test_paired_difference_single_pair_has_no_interval():
    """Limiting case: with one pair the mean exists but the interval cannot be computed."""

    result = paired_difference([5.0], [3.0])
    assert result["mean"] == pytest.approx(2.0)
    assert np.isnan(result["ci_low"]) and np.isnan(result["ci_high"])
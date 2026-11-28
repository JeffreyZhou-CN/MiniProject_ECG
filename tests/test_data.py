"""Checks for the synthetic data generator (src/data.py)."""

import numpy as np
import pytest

from src.data import make_ecg


def test_shapes_and_finite_values():
    """All three signals have N = seconds * fs samples and contain no NaN or inf."""

    clean, measured, reference = make_ecg(0, seconds=12, fs=250)
    for signal in (clean, measured, reference):
        assert signal.shape == (3000,)
        assert np.all(np.isfinite(signal))


def test_same_seed_gives_same_recording():
    """The same seed reproduces the recording exactly (reproducibility)."""

    first = make_ecg(3)
    again = make_ecg(3)
    for a, b in zip(first, again):
        np.testing.assert_array_equal(a, b)


def test_different_seeds_give_different_recordings():
    """Different seeds give independent recordings."""
 
    _, measured_a, _ = make_ecg(3)
    _, measured_b, _ = make_ecg(4)
    assert not np.array_equal(measured_a, measured_b)


def test_scale_multiplies_reference_but_not_clean_ecg():
    """Property: scale multiplies the reference exactly and leaves the clean ECG unchanged."""

    clean_1, _, reference_1 = make_ecg(0, scale=1.0)
    clean_3, _, reference_3 = make_ecg(0, scale=3.0)
    np.testing.assert_allclose(reference_3, 3.0 * reference_1)
    np.testing.assert_array_equal(clean_3, clean_1)


def test_r_peaks_have_unit_amplitude():
    """Property: the largest feature (R wave) has amplitude close to 1."""
    
    clean, _, _ = make_ecg(0)
    assert clean.max() == pytest.approx(1.0, abs=0.05)


def test_invalid_arguments_raise():
    """Edge case: non-positive duration or sample rate raises ValueError."""
   
    with pytest.raises(ValueError):
        make_ecg(0, seconds=0)
    with pytest.raises(ValueError):
        make_ecg(0, fs=-250)



def test_leakage_only_changes_reference():
    """Stress option: leakage adds leakage * clean to the reference and changes nothing else."""

    clean, measured, reference = make_ecg(5)
    clean_l, measured_l, reference_l = make_ecg(5, leakage=0.2)
    np.testing.assert_array_equal(clean_l, clean)
    np.testing.assert_array_equal(measured_l, measured)
    np.testing.assert_allclose(reference_l, reference + 0.2 * clean)


def test_pathway_change_only_affects_later_samples():
    """Stress option: before change_time the recording is identical; after it, d changes."""

    clean, measured, reference = make_ecg(6)
    clean_c, measured_c, reference_c = make_ecg(6, pathway_after=[-0.2, 0.3, 0.1], change_time=6.0)
    np.testing.assert_array_equal(clean_c, clean)
    np.testing.assert_array_equal(reference_c, reference)
    np.testing.assert_array_equal(measured_c[:1500], measured[:1500])
    assert not np.allclose(measured_c[1500:], measured[1500:])


def test_invalid_stress_options_raise():
    """Edge case: pathway_after without change_time, or a change outside the recording."""
   
    with pytest.raises(ValueError):
        make_ecg(0, pathway_after=[1.0])
    with pytest.raises(ValueError):
        make_ecg(0, pathway_after=[1.0], change_time=20.0)       
"""Evaluation metrics: MSE and SNR (dB) over the samples after a warm-up."""

import numpy as np

from scipy.signal import find_peaks


def _segment(clean, estimate, start):
    """Return clean[start:] and estimate[start:] as float arrays after checking shapes."""

    clean = np.asarray(clean, dtype=float)
    estimate = np.asarray(estimate, dtype=float)
    if clean.ndim != 1 or clean.shape != estimate.shape:
        raise ValueError("clean and estimate must be 1-D arrays of the same length")
    if not 0 <= start < len(clean):
        raise ValueError("start must lie inside the signal")
    return clean[start:], estimate[start:]


def mse(clean, estimate, start=0):
    """Mean squared error over samples [start, N); inf if the estimate is not finite.
    Parameters
    ----------
    clean : array_like, shape (N,) (Clean ECG s, the ground truth.)
    estimate : array_like, shape (N,) (Estimate s_hat: the filter output e, or the input d.)
    start : int (First evaluated sample; earlier samples are the warm-up.)

    Returns
    -------
    float (Mean of (s_hat - s)^2; smaller is better.)
    """
    # Mean squared error, smaller is better. Returns inf if the estimate contains NaN or
    # inf (a diverged filter).
    s, s_hat = _segment(clean, estimate, start)
    if not np.all(np.isfinite(s_hat)):
        return np.inf
    return float(np.mean((s_hat - s) ** 2))


def snr_db(clean, estimate, start=0):
    """Signal-to-noise ratio of the estimate in dB over samples [start, N).
    Returns +inf for an exact estimate and -inf for a non-finite (diverged) estimate.
   
    Parameters
    ----------
    clean : array_like, shape (N,) (Clean ECG s, the ground truth.)
    estimate : array_like, shape (N,) (Estimate s_hat.)
    start : int (First evaluated sample.)

    Returns
    -------
    float (10 * log10(sum s^2 / sum (s_hat - s)^2) in dB; larger is better.)
    """
    s, s_hat = _segment(clean, estimate, start)
    if not np.all(np.isfinite(s_hat)):
        return -np.inf
    with np.errstate(over="ignore"):
        error_energy = float(np.sum((s_hat - s) ** 2))
    # errstate silences overflow warnings when squaring huge values; an overflow gives inf
    if not np.isfinite(error_energy):
        return -np.inf
    if error_energy == 0.0:
        return np.inf
    return float(10.0 * np.log10(np.sum(s ** 2) / error_energy))



def r_peak_indices(clean, start=0, fs=250):
    """Indices of R peaks in the clean ECG at or after ``start``.

    Peaks must exceed 0.5 and be at least 0.3 s apart, so only R waves are found.

    Parameters
    ----------
    clean : array_like, shape (N,) (Clean ECG s; R waves have amplitude about 1.)
    start : int (Ignore peaks before this sample, i.e. in the warm-up.)
    fs : float (Sample rate in Hz, used for the minimum spacing.)

    Returns
    -------
    ndarray of int (Sample indices of the R peaks.)
    """
    
    peaks, _ = find_peaks(np.asarray(clean, dtype=float), height=0.5,
                          distance=max(1, int(0.3 * fs)))
    return peaks[peaks >= start]


def peak_amplitude_error(clean, estimate, start=0, fs=250):
    """Mean absolute error of the estimate at the true R-peak positions.

    Parameters
    ----------
    clean : array_like, shape (N,) (Clean ECG s.)
    estimate : array_like, shape (N,) (Estimate s_hat.)
    start : int (Only peaks at or after this sample are used.)
    fs : float (Sample rate in Hz.)

    Returns
    -------
    float (Mean |s_hat - s| at R peaks; smaller is better. inf if the estimate is not
        finite there, nan if no peak lies after start.)
    """
    
    clean = np.asarray(clean, dtype=float)
    estimate = np.asarray(estimate, dtype=float)
    peaks = r_peak_indices(clean, start=start, fs=fs)
    if len(peaks) == 0:
        return np.nan
    values = estimate[peaks]
    if not np.all(np.isfinite(values)):
        return np.inf
    return float(np.mean(np.abs(values - clean[peaks])))
"""Synthetic ECG-like signal, reference and interference generator (course guide, Track B)."""

import numpy as np

#: Interference pathway (unknown to the filters).
PATHWAY = np.array([0.35, -0.15, 0.08])

#: Gaussian pulses (amplitude, offset from beat time [s], width [s]) for P, Q, R, S, T.
PULSES = (
    (0.12, -0.18, 0.035),  # P wave
    (-0.15, -0.035, 0.012),  # Q wave
    (1.0, 0.0, 0.016),  # R wave
    (-0.25, 0.035, 0.014),  # S wave
    (0.30, 0.24, 0.065),  # T wave
)

#def make_ecg(seed, seconds=12, fs=250, scale=1.0):
def make_ecg(seed, seconds=12, fs=250, scale=1.0,pathway_after=None, change_time=None, leakage=0.0):
    """
    Generate one synthetic recording: d = s + v + noise, v = pathway * r.
    Parameters
    ----------
    seed : int (Seed for ``numpy.random.default_rng``; fixes the whole recording.)
    seconds : float (Recording length in seconds; N = int(seconds * fs) samples.)
    fs : float (Sample rate in Hz.)
    scale : float (Amplitude of the reference (and therefore of the interference).)

    pathway_after : array_like or None (Stress case: pathway used from change_time onwards.)
    change_time : float or None (Time in s at which the pathway changes; inside the recording.)
    leakage : float (Stress case: fraction of the clean ECG added to the returned reference.)
    Returns
    -------
    clean : ndarray, shape (N,)  (Clean ECG-like signal s; ground truth)
    measured : ndarray, shape (N,) (Measured signal d = s + v + 0.02 * Gaussian noise.)
    reference : ndarray, shape (N,) (Reference measurement r.)
    """
    if seconds <= 0 or fs <= 0:
        raise ValueError("seconds and fs must be positive")
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * fs)) / fs

    clean = np.zeros_like(t)
    interval = rng.uniform(0.75, 1.05)
    for beat in np.arange(0.6, seconds - 0.4, interval):
        for amplitude, offset, width in PULSES:
            clean += amplitude * np.exp(-0.5 * ((t - beat - offset) / width) ** 2)

    phase = rng.uniform(0, 2 * np.pi)
    reference = scale * (np.sin(2 * np.pi * 50 * t + phase) + 0.2 * rng.normal(size=len(t)))

    # interference = np.convolve(reference, PATHWAY, mode="full")[:len(t)]
    # measured = clean + interference + 0.02 * rng.normal(size=len(t))
    # return clean, measured, reference

    interference = np.convolve(reference, PATHWAY, mode="full")[:len(t)]
    if (pathway_after is None) != (change_time is None):
        raise ValueError("pathway_after and change_time must be given together")
    if pathway_after is not None:
        if not 0 < change_time < seconds:
            raise ValueError("change_time must lie inside the recording")
        later = np.convolve(reference, np.asarray(pathway_after, dtype=float), mode="full")[:len(t)]
        interference = np.where(t >= change_time, later, interference)
    # Pathway change: from change_time on, the interference uses the new pathway;
    # earlier samples are unchanged.
    # np.where(condition, A, B) takes A where the condition holds and B elsewhere.

    measured = clean + interference + 0.02 * rng.normal(size=len(t))
    return clean, measured, reference + leakage * clean
    # ECG leakage: the ECG is added only to the returned reference; the interference in
    # the measured signal is still generated from the ECG-free reference.
    # Neither option draws extra random numbers, so the same seed gives the same ECG and noise.
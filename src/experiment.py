"""Experiment building blocks: run the methods, select parameters, evaluate recordings."""

import numpy as np

from src.metrics import snr_db
from src.baseline import lms_filter
from src.modified import nlms_filter

#: Names of the compared methods.
METHODS = ("lms", "nlms", "nlms_ablation")


def run_method(method, measured, reference, params, eps=1e-6):
    """Run one method by name on one recording and return its output e.

    Parameters
    ----------
    method : str ("lms", "nlms" or "nlms_ablation".)
    measured : array_like, shape (N,) (Measured signal d.)
    reference : array_like, shape (N,) (Reference signal r.)
    params : dict ({"M": int, "mu": float}; the ablation also needs "energy": float.)
    eps : float (NLMS regulariser, used by "nlms" and "nlms_ablation".)

    Returns
    -------
    e : ndarray, shape (N,) (Cleaned ECG estimate; may be non-finite if LMS diverges.)
    """
    # Call the filter that matches the name, so all three methods get identical inputs.

    # LMS divergence produces overflow warnings; they are silenced here because
    # divergence is scored as SNR = -inf.
    with np.errstate(over="ignore", invalid="ignore"):
        if method == "lms":
            e, _ = lms_filter(measured, reference, params["M"], params["mu"])
        elif method == "nlms":
            e, _ = nlms_filter(measured, reference, params["M"], params["mu"], eps=eps)
        elif method == "nlms_ablation":
            e, _ = nlms_filter(measured, reference, params["M"], params["mu"], eps=eps,
                               normalize=False, reference_energy=params["energy"])
        else:
            raise ValueError(f"unknown method {method!r}")
    return e



def select_parameters(method, filter_lengths, step_sizes, recordings, warmup, eps=1e-6):
    """Grid search on validation recordings; criterion = mean output SNR after warm-up.

    Parameters
    ----------
    method : str ("lms" or "nlms".)
    filter_lengths : sequence of int (Candidate filter lengths M.)
    step_sizes : sequence of float (Candidate step sizes mu.)
    recordings : list of (clean, measured, reference) tuples (Validation recordings only.)
    warmup : int (Number of initial samples excluded from scoring.)
    eps : float (NLMS regulariser.)

    Returns
    -------
    best : dict ({"M", "mu", "score"} of the highest mean validation SNR.)
    table : list of dict (One row per (M, mu) with its mean validation SNR, for saving.)
    """
    # Grid search: for each (M, mu) pair, compute the output SNR on every validation
    # recording and use the mean as its score; the highest score wins.
    # Only validation recordings may be passed in; if any recording diverges the mean is
    # -inf, so that pair cannot be chosen.
    best = None
    table = []
    for M in filter_lengths:
        for mu in step_sizes:
            scores = [snr_db(clean, run_method(method, measured, reference,
                                               {"M": M, "mu": mu}, eps), start=warmup)
                      for clean, measured, reference in recordings]
            score = float(np.mean(scores))
            table.append({"method": method, "M": M, "mu": mu, "mean_val_snr_db": score})
            if best is None or score > best["score"]:
                best = {"M": int(M), "mu": float(mu), "score": score}
    return best, table




def mean_input_energy(references, M):
    """Mean of x_n^T x_n over all samples of the given reference signals.

    x_n holds the current and previous M - 1 reference samples (zeros before the start),
    exactly as inside the filters.

    Parameters
    ----------
    references : list of array_like, each shape (N,) (Reference signals, validation only.)
    M : int (Filter length used by the NLMS method.)

    Returns
    -------
    float (Average input-vector energy E, used by the ablation step mu / (eps + E).)
    """
    # Compute the average input energy x·x over all samples, for use by the ablation.
    # x·x is the sum of the last M values of r², i.e. r² convolved with M ones, truncated
    # to N samples.
    energies = [np.convolve(np.asarray(r, dtype=float) ** 2, np.ones(M))[:len(r)]
                for r in references]
    return float(np.mean(np.concatenate(energies)))
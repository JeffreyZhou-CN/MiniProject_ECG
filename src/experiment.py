"""Experiment building blocks: run the methods, select parameters, evaluate recordings."""

import time
import numpy as np

from src.metrics import mse, peak_amplitude_error, snr_db
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




def evaluate_case(method, clean, measured, reference, params, warmup, fs=250, eps=1e-6,
                  change_index=None):
    """Run one method on one recording and return all metrics as a dict.

    Parameters
    ----------
    method : str ("lms", "nlms" or "nlms_ablation".)
    clean, measured, reference : ndarray, shape (N,) (One recording; clean is used only for scoring.)
    params : dict (Method parameters, as for run_method.)
    warmup : int (Samples excluded from the main scores; also the window for snr_warmup_db.)
    fs : float (Sample rate in Hz; the post-change window is fs samples, i.e. 1 s.)
    eps : float (NLMS regulariser.)
    change_index : int or None (Sample at which the pathway changes, if it does.)

    Returns
    -------
    dict (snr_out_db, snr_in_db, snr_gain_db, mse, peak_error, snr_warmup_db,
        snr_post_change_db, diverged.)
    """
    # Run one method on one recording and return all metrics in a dict. clean is used
    # only for scoring.
    e = run_method(method, measured, reference, params, eps)
    snr_in = snr_db(clean, measured, start=warmup)
    snr_out = snr_db(clean, e, start=warmup)
    # Input SNR uses the measured signal as the estimate (no filtering); output SNR uses
    # the filter output.

    post_change = np.nan
    if change_index is not None:
        window = slice(change_index, change_index + int(fs))
        post_change = snr_db(clean[window], e[window])
    # SNR in the first second after a pathway change, measuring re-adaptation; NaN when
    # there is no change.

    return {
        "snr_out_db": snr_out,
        "snr_in_db": snr_in,
        "snr_gain_db": snr_out - snr_in,
        "mse": mse(clean, e, start=warmup),
        "peak_error": peak_amplitude_error(clean, e, start=warmup, fs=fs),
        "snr_warmup_db": snr_db(clean[:warmup], e[:warmup]),
        "snr_post_change_db": post_change,
        "diverged": bool(not np.all(np.isfinite(e))),
    }
    # snr_warmup_db is the SNR during the first second (the warm-up), measuring how
    # quickly the filter starts to adapt.




def expand_seeds(spec):
    """Seeds from a config entry {"start": a, "count": k}, i.e. [a, a+1, ..., a+k-1].

    Parameters
    ----------
    spec : dict ({"start": int, "count": int}.)

    Returns
    -------
    list of int (The seeds, one per independent ECG recording.)
    """
    # Expand a config entry {"start": a, "count": k} into the seed list [a, a+1, ..., a+k-1].
    return list(range(int(spec["start"]), int(spec["start"]) + int(spec["count"])))


def check_disjoint(*seed_lists):
    """Raise ValueError if any seed appears in more than one split (data-leakage guard).

    Parameters
    ----------
    *seed_lists : lists of int (e.g. development, validation and test seeds.)
    """
    # Check that the seed groups do not overlap; raise if any seed appears in two groups,
    # to prevent data leakage.
    seen = set()
    for seeds in seed_lists:
        overlap = seen.intersection(seeds)
        if overlap:
            raise ValueError(f"seeds used in more than one split: {sorted(overlap)}")
        seen.update(seeds)



def time_method(method, recordings, params, eps=1e-6, repeats=5):
    """Wall-clock time per recording: one untimed warm-up call, then repeated timed calls.

    Only the filter is timed; signal generation is excluded and neither filter has a
    separate setup step.

    Parameters
    ----------
    method : str ("lms", "nlms" or "nlms_ablation".)
    recordings : list of (clean, measured, reference) tuples (Recordings to time on.)
    params : dict (Method parameters, as for run_method.)
    eps : float (NLMS regulariser.)
    repeats : int (Timed calls per recording.)

    Returns
    -------
    dict (median_s, q25_s, q75_s: median and quartiles in seconds; n_timed: number of timed calls.)
    """
   
    times = []
    for _, measured, reference in recordings:
        run_method(method, measured, reference, params, eps)
        for _ in range(repeats):
            start = time.perf_counter()
            run_method(method, measured, reference, params, eps)
            times.append(time.perf_counter() - start)
    times = np.asarray(times)
    return {"median_s": float(np.median(times)),
            "q25_s": float(np.percentile(times, 25)),
            "q75_s": float(np.percentile(times, 75)),
            "n_timed": int(times.size)}
"""Experiment building blocks: run the methods, select parameters, evaluate recordings."""

import numpy as np

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
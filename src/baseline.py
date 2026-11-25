"""Baseline least-mean-squares (LMS) reference-noise canceller"""

import numpy as np

def lms_filter(d, r, M, mu):
    """Cancel reference-correlated interference from d with an LMS adaptive filter.

    At sample n:  x_n = [r[n], ..., r[n-M+1]],  y_n = w^T x_n,  e_n = d[n] - y_n,
    then w <- w + mu * e_n * x_n.  Weights start at zero.

    Parameters
    ----------
    d : array_like, shape (N,) (Measured signal: ECG + interference + noise.)
    r : array_like, shape (N,) (Reference measurement correlated with the interference.)
    M : int (Filter length, M >= 1.)
    mu : float (Step size, mu >= 0; too large a value makes the filter diverge.)

    Returns
    -------
    e : ndarray, shape (N,) (Residual d - y, i.e. the cleaned ECG estimate.)
    w : ndarray, shape (M,) (Final filter weights.)
    """
    
    d = np.asarray(d, dtype=float)
    r = np.asarray(r, dtype=float)

    if d.ndim != 1 or d.shape != r.shape:
        raise ValueError("d and r must be 1-D arrays of the same length")
    if int(M) != M or M < 1:
        raise ValueError("filter length M must be a positive integer")
    if mu < 0:
        raise ValueError("step size mu must be non-negative")
    M = int(M)

    w = np.zeros(M)          # filter weights, w_0 = 0
    x = np.zeros(M)          # delay line [r[n], r[n-1], ..., r[n-M+1]]
    e = np.empty(len(d))     # output (cleaned ECG estimate)
   
    for n in range(len(d)):
        x[1:] = x[:-1]       # shift older samples down the delay line
        x[0] = r[n]          # newest reference sample at the front

        e[n] = d[n] - w @ x  # predict the interference and subtract it
        w += mu * e[n] * x   # LMS update
    return e, w
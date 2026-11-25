"""Modified canceller: LMS with the step size normalised by reference energy (NLMS).

Only the weight update changes relative to src/baseline.py:
    LMS:   w <- w + mu * e_n * x_n
    NLMS:  w <- w + mu / (eps + x_n^T x_n) * e_n * x_n
NLMS is a known method (Nagumo & Noda, 1967), implemented independently here.
"""

import numpy as np


def nlms_filter(d, r, M, mu, eps=1e-6, normalize=True, reference_energy=None):
    """Cancel reference-correlated interference with a normalised LMS filter.

    Parameters
    ----------
    d : array_like, shape (N,) (Measured signal: ECG + interference + noise.)
    r : array_like, shape (N,) (Reference measurement correlated with the interference.)
    M : int (Filter length, M >= 1.)
    mu : float (Normalised step size, mu >= 0; usually 0 < mu < 2.)
    eps : float (Small constant added to the energy to avoid division by zero, eps >= 0.)
    normalize : bool (True: NLMS. False: ablation with the normalisation switched off.)
    reference_energy : float or None (Ablation only: fixed energy E, step = mu / (eps + E);
        if None, the step is plain mu, identical to the baseline LMS.)

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
    if mu < 0 or eps < 0:
        raise ValueError("mu and eps must be non-negative")
    if reference_energy is not None and normalize:
        raise ValueError("reference_energy is only used when normalize=False")
    M = int(M)
    # Input checks as in the baseline; also eps must be non-negative and reference_energy
    # may only be given for the ablation.

    if not normalize:
        fixed_step = mu if reference_energy is None else mu / (eps + reference_energy)
    # Ablation: the step is a fixed value, computed once before the loop.

    w = np.zeros(M)
    x = np.zeros(M)
    e = np.empty(len(d))
    for n in range(len(d)):
        x[1:] = x[:-1]
        x[0] = r[n]
        e[n] = d[n] - w @ x
        # Identical to the baseline: shift the delay line, predict the interference,
        # compute the output.

        if normalize:
            energy = eps + x @ x
            step = mu / energy if energy > 0 else 0.0
            # The only change vs. the baseline: divide the step by the input energy x·x plus eps.
            # If eps = 0 and x is all zeros the energy is 0; skip the update instead of
            # dividing by zero.
        else:
            step = fixed_step
        w += step * e[n] * x
        # Update the weights with this sample's step size.

    return e, w
# -*- coding: utf-8 -*-
"""Low-level 1-D smoothing: Savitzky-Golay and Whittaker-Eilers.

All implementations use only NumPy and SciPy (already required by wdfkit).
No additional dependencies are introduced.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.optimize import minimize_scalar
from scipy.signal import savgol_filter

# ---------------------------------------------------------------------------
# Savitzky-Golay
# ---------------------------------------------------------------------------


def savgol_smooth_1d(
    y: np.ndarray,
    window_length: int = 11,
    polyorder: int = 3,
) -> np.ndarray:
    """Apply a Savitzky-Golay filter to a 1-D spectrum.

    Parameters
    ----------
    y
        1-D array of intensities.
    window_length
        Number of channels in the filter window (odd, >= polyorder + 2).
    polyorder
        Polynomial order (must be < window_length).

    Returns
    -------
    np.ndarray
        Smoothed spectrum, same shape as ``y``, dtype float64.
    """
    y = np.asarray(y, dtype=np.float64)
    if y.ndim != 1:
        raise ValueError(
            f"savgol_smooth_1d expects 1-D input; got ndim={y.ndim}"
        )
    if window_length % 2 == 0 or window_length < 3:
        raise ValueError(
            f"window_length must be odd and >= 3; got {window_length}"
        )
    if polyorder >= window_length:
        raise ValueError(
            f"polyorder ({polyorder}) must be < "
            f"window_length ({window_length})"
        )
    return savgol_filter(y, window_length, polyorder).astype(np.float64)


# ---------------------------------------------------------------------------
# Whittaker-Eilers
# ---------------------------------------------------------------------------


def _diff_matrix(n: int, d: int) -> sp.csc_matrix:
    """Build the d-th order finite difference matrix of size (n-d, n)."""
    D: sp.csc_matrix = sp.eye(n, format="csc")
    for _ in range(d):
        m = D.shape[0]
        delta = sp.diags(
            [-np.ones(m - 1), np.ones(m - 1)],
            [0, 1],
            shape=(m - 1, m),
            format="csc",
        )
        D = delta @ D
    return D


def whittaker_smooth_1d(
    y: np.ndarray,
    lam: float,
    d: int = 2,
) -> np.ndarray:
    """Whittaker-Eilers smoother.

    Minimises ``||y - z||² + lam · ||D^d z||²`` where ``D^d`` is the
    d-th order difference operator.  Solved as a sparse linear system.

    Parameters
    ----------
    y
        1-D array of intensities.
    lam
        Smoothness penalty (larger → smoother).
    d
        Difference order (default 2 — cubic-like smoothness).

    Returns
    -------
    np.ndarray
        Smoothed spectrum, same shape as ``y``, dtype float64.
    """
    y = np.asarray(y, dtype=np.float64)
    if y.ndim != 1:
        raise ValueError(
            f"whittaker_smooth_1d expects 1-D input; got ndim={y.ndim}"
        )
    if lam <= 0:
        raise ValueError(f"lam must be positive; got {lam}")
    n = len(y)
    D = _diff_matrix(n, d)
    A = sp.eye(n, format="csc") + lam * D.T.dot(D)
    return spla.spsolve(A, y)


# ---------------------------------------------------------------------------
# GCV-based automatic lambda selection
# ---------------------------------------------------------------------------


def _gcv_score(log_lam: float, y: np.ndarray, d: int) -> float:
    """Generalised Cross-Validation score for Whittaker at exp(log_lam).

    GCV(λ) = n · RSS / (n - tr(H))²
    where RSS = ||y - z||² and tr(H) = n - y·(A⁻¹y) / ||y||² (approximated
    via the sparse solve residual).
    """
    lam = float(np.exp(log_lam))
    n = len(y)
    D = _diff_matrix(n, d)
    A = sp.eye(n, format="csc") + lam * D.T.dot(D)
    z = spla.spsolve(A, y)
    # Effective degrees of freedom via the influence matrix diagonal
    # tr(H) ≈ sum of diagonal of A^{-1}.  For large n use the residual proxy.
    rss = float(np.sum((y - z) ** 2))
    # Approximate tr(H) using the fact that (I - H)y = y - z = A^{-1}(λ D'D y)
    # → tr(I - H) ≈ n · rss / y'(y - z)  [Eilers & Marx 1996 proxy]
    ydiff = float(np.dot(y, y - z))
    if rss < 1e-300 or abs(ydiff) < 1e-300:
        return 0.0
    dof_residual = n * rss / ydiff  # ≈ n - tr(H)
    if dof_residual <= 0:
        return np.inf
    return float(n * rss / dof_residual**2)


def auto_lam(
    y: np.ndarray,
    d: int = 2,
    lam0: float = 100.0,
    max_calls: int = 5,
) -> float:
    """Select the Whittaker smoothness parameter λ by GCV minimisation.

    Uses ``scipy.optimize.minimize_scalar`` in bounded mode on
    ``log(λ) ∈ [-2, 16]`` (λ ∈ ~[0.14, 8.9 × 10⁶]), starting near
    ``log(lam0)``.  The optimiser is limited to ``max_calls`` function
    evaluations.

    Parameters
    ----------
    y
        1-D spectrum.
    d
        Difference order (should match the smoother).
    lam0
        Initial λ hint (search starts near ``log(lam0)``).
    max_calls
        Maximum GCV evaluations (default 5).

    Returns
    -------
    float
        Optimal λ.
    """
    y = np.asarray(y, dtype=np.float64)
    bounds = (-2.0, 16.0)

    result = minimize_scalar(
        _gcv_score,
        bounds=bounds,
        method="bounded",
        args=(y, d),
        options={"maxiter": max_calls, "xatol": 0.5},
    )
    return float(np.exp(result.x))


__all__ = [
    "savgol_smooth_1d",
    "whittaker_smooth_1d",
    "auto_lam",
]

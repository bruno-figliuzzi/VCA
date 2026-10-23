import numpy as np
from typing import Tuple, Optional

# -------------------------- Noise estimation ------------------------------------------
def estimate_noise(
    Y: np.ndarray,
    eps: float = 1e-8
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Local linear prediction noise estimation (HYSIME implementation).

    Each band is predicted from every other band via the closed-form
    least-squares regression coefficients implied by the inverse sample
    correlation matrix; the residual of that prediction is the estimated
    noise for that band.

    References
    ----------

    - Bioucas-Dias, J. M., & Nascimento, J. M. P. (2008).
      Hyperspectral subspace identification.
      IEEE TGRS.

    Parameters
    ----------
    Y : (L, N)
        Hyperspectral data (L bands, N pixels)
    eps : float
        Numerical stability term

    Returns
    -------
    W : (L, N)
        Estimated noise matrix
    RW : (L, L)
        Diagonal noise correlation matrix
    """
    if not isinstance(Y, np.ndarray):
        raise ValueError("Y must be a numpy array")

    if Y.ndim != 2:
        raise ValueError("Y must be 2D (L, N)")

    L, N = Y.shape
    if L < 3:
        raise ValueError("Need at least 3 bands")

    # Data Covariance and pseudo-inverse
    RY = (Y @ Y.T) / N
    RYi = np.linalg.pinv(RY)

    # Extract the diagonal of the inverse matrix
    diag_RYi = np.diag(RYi).reshape(-1, 1)
    W = (RYi / diag_RYi) @ Y

    # Noise correlation
    RW = (W @ W.T) / N
    return W, RW


def estimate_snr(Y: np.ndarray,
    Ym: np.ndarray,
    Yp: np.ndarray,
    eps: float = 1e-12
) -> float:
    """
    Estimate the signal-to-noise ratio (SNR) of hyperspectral data in decibels.

    This function computes an empirical SNR estimate based on the total
    observed signal energy and the projected signal energy in a reduced
    subspace.

    Parameters
    ----------
    Y : np.ndarray
        Original data matrix of shape (L, N), where:
        - L is the number of spectral bands (channels)
        - N is the number of pixels (samples)

    Ym : np.ndarray
        Mean spectrum of ``Y`` (typically the mean used to center the data
        before computing ``Yp``), of shape (L, 1). Its squared norm adds
        the DC (mean) component back into the total signal power estimate.

    Yp : np.ndarray
        Projected data matrix in a reduced subspace of shape (p, N),
        where p is the reduced dimensionality.

    eps : float, default=1e-12
        Small numerical constant used to avoid division by zero and
        improve numerical stability.

    Returns
    -------
    float
        Estimated signal-to-noise ratio in decibels (dB). Returns
        ``+inf`` when the data contains no detectable out-of-subspace
        (noise) power.
    """
    L, N = Y.shape
    P_y = np.sum(Y**2) / N
    P_x = np.sum(Yp**2) / N + np.sum(Ym**2)
    p = Yp.shape[0]

    denom = P_y - P_x
    if denom < eps:
        return float(np.inf)

    num = max(P_x - (p / L) * P_y, eps)
    snr_est = 10 * np.log10(num / denom)
    return float(snr_est)


# -------------------------- HySime implementation -------------------------------------
def hysime(
    Y: np.ndarray,
    W: Optional[np.ndarray] = None,
    RW: Optional[np.ndarray] = None,
    eps: float = 1e-10,
    center: bool = False
) -> Tuple[int, np.ndarray]:
    """
    HySime: Hyperspectral signal subspace estimation.

    References
    ----------

    - Bioucas-Dias, J. M., & Nascimento, J. M. P. (2008).
      Hyperspectral subspace identification.
      IEEE TGRS.

    Parameters
    ----------
    Y:  ndarray (L x N)
        Hyperspectral data (L bands, N pixels).
    W:  ndarray (L x N), optional
        Noise matrix. Must be provided together with ``RW`` (or not at
        all, in which case both are estimated from ``Y`` via
        :func:`estimate_noise`).
    RW: ndarray (L x L), optional
        Noise correlation matrix. Must be provided together with ``W``.
    eps: float(default: 1e-10)
        Noise conditionning parameter
    center: bool (default: False)
        Whether or not to center the hyperspectral dataset

    Returns
    -------
    kf : int
        Estimated signal subspace dimension
    Ek : ndarray (L x kf)
        Eigenvectors spanning the signal subspace, ordered from most to
        least confidently identified as signal (increasing cost, Eq. 22
        of the reference).
    """

    Y = np.asarray(Y)

    if Y.size == 0:
        raise ValueError("Dataset is empty")

    L, N = Y.shape

    if (W is None) != (RW is None):
        raise ValueError(
            "W and RW must be provided together, or both omitted."
        )

    if W is None:
        W, RW = estimate_noise(Y)

    if Y.shape != W.shape:
        raise ValueError("Noise matrix shape does not match dataset shape")

    if RW.shape != (L, L):
        raise ValueError("The shape of the noise correlation matrix"
            f"should be ({L}, {L})")

    # Noise regularization
    RW = RW + np.trace(RW)/L * eps * np.eye(L)

    # Mean centering
    if center:
        mean_Y = np.mean(Y, axis=1, keepdims=True)
        Yc = Y - mean_Y
    else:
        Yc = Y

    # Center noise consistently
    if center:
        mean_W = np.mean(W, axis=1, keepdims=True)
        Wc = W - mean_W
    else:
        Wc = W

    # Signal Estimation
    X = Yc - Wc

    # Sample and estimated correlation matrices
    RY = (Yc @ Yc.T) / N
    RX = (X @ X.T) / N
    RY = (RY + RY.T) / 2
    RX = (RX + RX.T) / 2

    # Eigen-decomposition : returns the eigenvalues in ascending order
    D, E = np.linalg.eigh(RX)

    # Eq. (23-24)
    PY = np.diag(E.T @ RY @ E)
    PW = np.diag(E.T @ RW @ E)

    # Eq. (22)
    cost = 2*PW - PY

    # Select the components with negative cost, ordered from most to
    # least confidently identified as signal (increasing cost).
    kf = int(np.sum(cost < 0))
    if kf == 0:
        return 0, np.empty((L, 0))

    sort_idx = np.argsort(cost)[:kf]
    Ek = E[:, sort_idx]
    return kf, Ek

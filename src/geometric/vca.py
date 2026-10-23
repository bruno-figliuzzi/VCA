from __future__ import annotations
import numpy as np
from typing import Optional
from .base import GeometricUnmixing
from ..projection import PCA, SVD
from ..projection.hysime import estimate_snr as _estimate_snr

EPS = 1e-12
MAX_ITER = 20

def estimate_snr(
    Y: np.ndarray,
    p: int,
) -> float:
    """
    Estimate the signal-to-noise ratio (SNR) of a set of hyperspectral
    observations.

    Parameters
    ----------
    Y : np.ndarray of shape (n_bands, n_pixels)
        Hyperspectral data matrix
    p : int
        Dimension of the signal subspace.

    Returns
    -------
    float
        Estimated signal-to-noise ratio in decibels (dB).
    """
    Y = np.asarray(Y)
    pca = PCA(p)
    Y_proj = pca.fit_transform(Y)
    return _estimate_snr(Y, pca.mean_, Y_proj, eps=EPS)


class VCA(GeometricUnmixing):
    """
    Vertex Component Analysis (VCA) for hyperspectral endmember extraction.

    VCA extracts spectrally pure pixels (endmembers) from hyperspectral data
    under the linear mixing model.

    The algorithm operates in a reduced signal subspace and identifies
    vertices of a simplex formed by the data projection.

    Algorithm overview
    ------------------

    1. Subspace estimation:
        The data is projected onto a p-dimensional signal subspace using
        a truncated SVD followed by a simplex normalization. The projected data 
        is normalized to enforce simplex structure, making the abundance model
        approximately sum-to-one.

    2. Iterative vertex extraction:
        - A random direction is drawn uniformly in [0, 1]^p (as in the
          reference MATLAB code)
        - It is orthogonalized with respect to already selected vertices
          (the first one with respect to the last axis)
        - The pixel maximizing projection magnitude is selected as an endmember

    3. Reconstruction:
        The endmembers are the selected pixels projected onto the signal
        subspace (``Y_rec[:, indices_]``), as in the reference MATLAB code
        (``Ae = Rp(:, indice)``) and HySUPP: the projection removes the
        noise outside the subspace. The raw pixels are ``Y[:, indices_]``.

    References
    ----------
    - Nascimento, J. M. P., & Bioucas-Dias, J. M. (2004).
      Vertex Component Analysis: A fast algorithm to unmix hyperspectral data.
      IEEE Transactions on Geoscience and Remote Sensing.

    - Bioucas-Dias, J. M., & Nascimento, J. M. P. (2008).
      Hyperspectral subspace identification.
      IEEE Transactions on Geoscience and Remote Sensing.

    Attributes
    ----------        
    n_endmembers : int
        Number of endmembers (signal subspace dimension).

    rng : np.random.Generator
        Local random numbers generator for reproducible stochastic operations.
                
    endmembers_ : np.ndarray of shape (n_bands, p)
        The extracted endmember signatures: the selected pure pixels,
        projected onto the estimated signal subspace (denoised).
        
    snr_: float
        Estimated signal-to-noise ratio of the input data, in decibels (dB).
        
    indices_ : np.ndarray of shape (n_endmembers,)
        Indices of the selected endmember pixels in the input data matrix `Y`.
        Each entry corresponds to the column index of an extracted endmember
        spectrum. The raw pixels are `Y[:, indices_]`; `endmembers_` holds
        their projection onto the signal subspace.
        
    Parameters
    ----------
    n_endmembers : int
        Number of endmembers (signal subspace dimension).
        
    rng : np.random.Generator
        Local random numbers generator for reproducible stochastic operations.
    """
    def __init__(self, 
        n_endmembers: int,
        rng: Optional[np.random.Generator] = None
    ) -> None:
    
        super().__init__(n_endmembers=n_endmembers, rng=rng)

    def _project(
        self,
        Y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Project hyperspectral data onto a signal subspace and normalize it
        for vertex component analysis.

        This method performs:
        1. Mean removal from spectral data
        2. SVD-based estimation of a p-dimensional signal subspace
        3. Projection of data into the subspace
        4. SNR-based decision to select a normalization strategy

        For high SNR scenarios, data is projected using the full observation
        matrix and normalized using simplex-like scaling.
        For low SNR scenarios, a reduced subspace is used and an augmented
        representation is constructed to improve robustness.

        Parameters
        ----------
        Y : np.ndarray of shape (n_bands, n_pixels)
            Input hyperspectral data matrix.

        Returns
        -------
        Y_proj : np.ndarray of shape (p, n_pixels)
            Normalized projected data used for endmember extraction, where
            p = n_endmembers. In the low SNR case this is obtained from a
            `(p - 1)`-dimensional PCA projection augmented with one
            constant row; in the high SNR case it is a `p`-dimensional
            SVD projection directly.
        Y_rec : np.ndarray of shape (n_bands, n_pixels)
            Reconstructed data from the projected subspace representation.
        """
        
        _, n_pixels = Y.shape
        p = self.n_endmembers
        
        # SNR Estimation
        snr = estimate_snr(Y, p)
        snr_th = 15 + 10 * np.log10(p)
        self.snr_ = snr
        
        # Projection (low SNR)
        if snr < snr_th:
        
            pca = PCA(p - 1)
            Y_proj = pca.fit_transform(Y)
            Y_rec = pca.reconstruct(Y_proj)
            c = np.linalg.norm(Y_proj, axis=0).max()
            Y_proj = np.vstack((Y_proj, c * np.ones((1, n_pixels))))
        
        # Projection (high SNR)    
        else:
            
            svd = SVD(p)
            Y_proj = svd.fit_transform(Y)
            Y_rec = svd.reconstruct(Y_proj)
            u = np.mean(Y_proj, axis=1, keepdims=True)
            den = u.T @ Y_proj
            small = np.abs(den) < EPS
            den[small] = np.where(den[small] < 0, -EPS, EPS)
            Y_proj = Y_proj / den

        return Y_proj, Y_rec
        

    def fit(self, 
        Y: np.ndarray
    ) -> "VCA":
        """
        Fit the Vertex Component Analysis (VCA) model to hyperspectral data.

        Parameters
        ----------
        Y : np.ndarray of shape (n_bands, n_pixels)
            Hyperspectral data matrix.

        Returns
        -------
        self : VCA
            Fitted estimator with attributes:
            - ``indices_``: indices of selected endmember pixels
            - ``endmembers_``: extracted spectral signatures (selected
              pixels projected onto the signal subspace)
            - ``snr_``: estimated signal-to-noise ratio.
        """   
        Y = self._validate_dataset(Y, self.n_endmembers)
        Y = np.asarray(Y, dtype=np.float64)

        Y_proj, Y_rec = self._project(Y)
        p = self.n_endmembers

        # Matrix whose columns are the selected projected endmembers.
        # As in the reference MATLAB code, it starts with A[p-1, 0] = 1 so
        # that the first direction is orthogonal to the last axis.
        A = np.zeros((p, p))
        A[p - 1, 0] = 1.0

        indices = np.empty(p, dtype=int)

        for i in range(p):

            # Uniform random direction, orthogonal to the columns of A.
            P = A @ np.linalg.pinv(A)
            for _ in range(MAX_ITER):

                w = self.rng.random((p, 1))
                f = w - P @ w

                norm = np.linalg.norm(f)
                if norm > EPS:
                    break
            else:
                raise RuntimeError(
                    "Failed to generate an independent search direction."
                )

            f /= norm

            # Select the vertex maximizing the projection magnitude.
            v = f.T @ Y_proj
            idx = int(np.argmax(np.abs(v)))

            indices[i] = idx
            A[:, i] = Y_proj[:, idx]

        self.indices_ = indices
        # As in the reference MATLAB code (Ae = Rp(:, indice)) and HySUPP:
        # the selected pixels are returned denoised, i.e. projected onto
        # the signal subspace, rather than as raw pixels.
        self.endmembers_ = Y_rec[:, indices]
        return self
        

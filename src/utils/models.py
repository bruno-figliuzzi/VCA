from abc import ABC, abstractmethod
import numpy as np
from typing import Optional

class MixingModel(ABC):
    """
    Abstract base class for hyperspectral mixing models.
    """
    @abstractmethod
    def mix(
        self,
        endmembers: np.ndarray,
        abundances: np.ndarray,
    ) -> np.ndarray:
        """
        Compute mixed hyperspectral observations from endmembers and abundances.

        Parameters
        ----------
        endmembers : ndarray
            Endmember representation used for mixing. Supported shapes:

            - ``(n_bands, n_endmembers)``: constant endmember matrix
              shared across all pixels.
            - ``(n_pixels, n_bands, n_endmembers)``: pixel-wise varying
              endmember tensor.

        abundances : ndarray of shape (n_pixels, n_endmembers)
            Abundance matrix where each row contains the fractional
            contribution of each endmember for a given pixel.

        Returns
        -------
        ndarray of shape (n_pixels, n_bands)
            Mixed hyperspectral image (observed pixel spectra).
        """
        raise NotImplementedError


def _ensure_pixelwise_endmembers(
    endmembers: np.ndarray, 
    n_pixels: int
) -> np.ndarray:
    """
    Ensure a consistent pixel-wise representation of endmembers.

    Standardizes endmember input into a 3D tensor, with shape 
    (n_pixels, n_bands, n_endmembers)

    It supports both constant and spatially varying endmember models.

    Parameters
    ----------
    endmembers : ndarray
        Endmember array in one of the following formats:

        - (n_bands, n_endmembers):
            Constant endmember matrix applied to all pixels.
        - (n_pixels, n_bands, n_endmembers):
            Pixel-specific endmember tensor.

    n_pixels : int
        Number of pixels expected in the output representation.

    Returns
    -------
    ndarray of shape (n_pixels, n_bands, n_endmembers)
        Broadcasted or validated pixel-wise endmember tensor.
    """
    if endmembers.ndim == 2:
        return np.broadcast_to(
            endmembers,
            (n_pixels, endmembers.shape[0], endmembers.shape[1])
        )

    if endmembers.ndim == 3:
        if endmembers.shape[0] != n_pixels:
            raise ValueError(
                f"Expected {n_pixels} pixels in endmembers, got {endmembers.shape[0]}"
            )
        return endmembers

    raise ValueError(f"Invalid endmember shape: {endmembers.shape}")


class LinearMixingModel(MixingModel):
    """
    Linear Spectral Mixing Model (LMM).

    This model assumes that each pixel spectrum is a convex
    combination of endmember spectra.
    """
    def mix(
        self,
        endmembers: np.ndarray,
        abundances: np.ndarray,
    ) -> np.ndarray:
        """
        Perform linear mixing of endmembers and abundances.

        Parameters
        ----------
        endmembers : ndarray
            Either:
            - (n_bands, n_endmembers)
            - (n_pixels, n_bands, n_endmembers)

        abundances : ndarray of shape (n_pixels, n_endmembers)
            Abundance fractions per pixel.

        Returns
        -------
        ndarray of shape (n_bands, n_pixels)
            Linearly mixed hyperspectral image.
        """
        n_endmembers, n_pixels = abundances.shape
        M = _ensure_pixelwise_endmembers(endmembers, n_pixels)
        n_bands = M.shape[1]

        if M.shape[2] != n_endmembers:
            raise ValueError("Endmember/abundance dimension mismatch")

        X = np.zeros((n_bands, n_pixels), dtype=float)

        for p in range(n_pixels):
        
            X[:, p] = M[p] @ abundances[:, p]

        return X


class FanBilinearMixingModel(MixingModel):
    """
    Fan Bilinear Mixing Model.

    Nonlinear extension of the linear mixing model that
    accounts for pairwise interactions between endmembers.
    """
    def mix(
        self,
        endmembers: np.ndarray,
        abundances: np.ndarray,
    ) -> np.ndarray:
        """
        Perform Fan bilinear mixing.

        Parameters
        ----------
        endmembers : ndarray
            Either:
            - (n_bands, n_endmembers)
            - (n_pixels, n_bands, n_endmembers)

        abundances : ndarray of shape (n_pixels, n_endmembers)
            Abundance fractions per pixel.

        Returns
        -------
        ndarray of shape (n_bands, n_pixels)
            Nonlinearly mixed hyperspectral image.
        """
        n_pixels, n_endmembers = abundances.shape
        M = _ensure_pixelwise_endmembers(endmembers, n_pixels)
        n_bands = M.shape[1]

        if M.shape[2] != n_endmembers:
            raise ValueError("Dimension mismatch between endmembers and abundances")

        X = np.zeros((n_bands, n_pixels), dtype=float)
        for p in range(n_pixels):

            Mp = M[p]
            a = abundances[p, :]

            # Linear contribution
            x = Mp @ a

            # Bilinear interaction terms
            for i in range(n_endmembers):
                for j in range(i + 1, n_endmembers):
                    interaction = Mp[:, i] * Mp[:, j]
                    x += a[i] * a[j] * interaction

            X[:, p] = x

        return X

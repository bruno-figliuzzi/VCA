from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class OutlierModel(ABC):
    """
    Abstract base class for pixel outlier injection models.

    Outlier models select a subset of pixels in a hyperspectral observation
    matrix and corrupt them so that they violate the mixing model assumed
    by the rest of the generation pipeline (e.g. the linear mixing model),
    simulating acquisition artifacts or anomalous materials.

    Notes
    -----
    Implementations operate on data arranged as ``(n_bands, n_pixels)`` and
    must preserve this shape.
    """
    @abstractmethod
    def add_outliers(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Inject outlier pixels into a hyperspectral dataset.

        Parameters
        ----------
        X : np.ndarray, shape (n_bands, n_pixels)
            Clean hyperspectral data.

        Returns
        -------
        X_out : np.ndarray, shape (n_bands, n_pixels)
            Data with outlier pixels injected. Same shape as ``X``.
        outlier_mask : np.ndarray of bool, shape (n_pixels,)
            ``True`` for pixels that were replaced by an outlier.
        """
        raise NotImplementedError

    def __call__(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.add_outliers(X)


class ForeignSpectrumOutliers(OutlierModel):
    """
    Replace a fraction of pixels with foreign spectra.

    A random subset of pixels is replaced by spectra drawn from a
    user-provided pool of candidate signatures that are not part of the
    endmember set used to generate the scene (e.g. other entries from a
    spectral library). This simulates the presence of a rare, anomalous
    material that is not accounted for by the mixing model.

    Parameters
    ----------
    candidate_spectra : np.ndarray, shape (n_bands, n_candidates)
        Pool of foreign spectra to draw outliers from.
    fraction : float, default=0.02
        Fraction of pixels to replace by an outlier. Must lie in (0, 1].
    replace : bool, default=True
        Whether candidate spectra can be reused across different outlier
        pixels. Set to ``False`` to enforce distinct spectra, which
        requires ``candidate_spectra`` to contain at least as many columns
        as the number of outlier pixels.
    rng : numpy.random.Generator, optional
        Random number generator used to select outlier pixels and spectra.
        If omitted, a new default generator is created.
    """
    def __init__(
        self,
        candidate_spectra: np.ndarray,
        fraction: float = 0.02,
        replace: bool = True,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        candidate_spectra = np.asarray(candidate_spectra, dtype=float)
        if candidate_spectra.ndim != 2:
            raise ValueError(
                "candidate_spectra must be a 2D array (n_bands, n_candidates)."
            )
        if not (0.0 < fraction <= 1.0):
            raise ValueError("fraction must lie in (0, 1].")

        self.candidate_spectra = candidate_spectra
        self.fraction = fraction
        self.replace = replace
        self.rng = rng or np.random.default_rng()

    def add_outliers(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Replace a random fraction of pixels with foreign spectra.

        Parameters
        ----------
        X : np.ndarray, shape (n_bands, n_pixels)
            Clean hyperspectral data.

        Returns
        -------
        X_out : np.ndarray, shape (n_bands, n_pixels)
            Data with outlier pixels replaced by spectra drawn from
            ``candidate_spectra``.
        outlier_mask : np.ndarray of bool, shape (n_pixels,)
            ``True`` for pixels that were replaced by a foreign spectrum.

        Raises
        ------
        ValueError
            If ``candidate_spectra`` does not have the same number of bands
            as ``X``, or if ``replace=False`` and there are not enough
            candidate spectra to cover all outlier pixels.
        """
        X = np.asarray(X, dtype=float)
        n_bands, n_pixels = X.shape

        if self.candidate_spectra.shape[0] != n_bands:
            raise ValueError(
                f"candidate_spectra has {self.candidate_spectra.shape[0]} bands, "
                f"expected {n_bands}."
            )

        n_outliers = max(1, round(self.fraction * n_pixels))
        n_candidates = self.candidate_spectra.shape[1]

        if not self.replace and n_outliers > n_candidates:
            raise ValueError(
                f"Cannot draw {n_outliers} distinct outlier spectra from "
                f"{n_candidates} candidates without replacement."
            )

        pixel_idx = self.rng.choice(n_pixels, size=n_outliers, replace=False)
        spectra_idx = self.rng.choice(
            n_candidates, size=n_outliers, replace=self.replace
        )

        X_out = X.copy()
        X_out[:, pixel_idx] = self.candidate_spectra[:, spectra_idx]

        outlier_mask = np.zeros(n_pixels, dtype=bool)
        outlier_mask[pixel_idx] = True

        return X_out, outlier_mask

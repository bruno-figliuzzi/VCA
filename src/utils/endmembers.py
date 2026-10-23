from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


# ----------------------------------- Base class ---------------------------------------
class EndmemberGenerator(ABC):
    """
    Abstract base class for endmember generators.

    Endmember generators are responsible for producing spectral
    signatures (endmember matrices) used in hyperspectral
    unmixing.
    """
    @abstractmethod
    def generate(
        self,
        n_pixels: int,
        n_bands: int,
        n_endmembers: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate endmember matrix.

        Parameters
        ----------
        n_pixels : int
            Number of pixels in the image.

        n_bands : int
            Number of spectral bands.

        n_endmembers : int
            Number of endmembers.

        Returns
        -------
        endmembers : np.ndarray of shape (n_bands, n_endmembers)
            Reference endmember signatures.

        per_pixel_endmembers : np.ndarray of shape (n_pixels, n_bands, n_endmembers)
            Endmember tensor containing the spectral signatures used
            for each pixel. For constant models, the same endmembers
            are repeated across all pixels; for variable models, the
            signatures may vary spatially.
        """
        raise NotImplementedError


# -------------------------- Constant endmembers ---------------------------------------
class ConstantEndmemberGenerator(EndmemberGenerator):
    """
    Generator for constant endmembers shared across all pixels.
    
    Parameters
    ----------
    endmembers : np.ndarray of shape (n_bands, n_endmembers)
        Endmember matrix defining spectral signatures.
    
    Attributes
    ----------
    endmembers : np.ndarray of shape (n_bands, n_endmembers)
        Fixed spectral signatures used for all pixels.
    """
    def __init__(self, endmembers: np.ndarray):
        self.endmembers = np.asarray(endmembers, dtype=float)

    def generate(
        self,
        n_pixels: int,
        n_bands: int,
        n_endmembers: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns a constant endmember tensor broadcast over pixels.

        Parameters
        ----------
        n_pixels : int
            Number of pixels in the image.

        n_bands : int
            Number of spectral bands.

        n_endmembers : int
            Number of endmembers.

        Returns
        -------
        endmembers : np.ndarray of shape (n_bands, n_endmembers)
            Endmember matrix.

        per_pixel_endmembers : np.ndarray of shape (n_pixels, n_bands, n_endmembers)
            Repeated endmember matrix for each pixel
        """
        if self.endmembers.shape != (n_bands, n_endmembers):
            raise ValueError(
                f"Expected shape {(n_bands, n_endmembers)}, "
                f"got {self.endmembers.shape}"
            )
        
        per_pixel_endmembers = np.broadcast_to(
            self.endmembers,
            (n_pixels, n_bands, n_endmembers),
        ).copy()
        
        return self.endmembers, per_pixel_endmembers


# ----------------------------- Endmember bundles --------------------------------------
class BundleEndmemberGenerator(EndmemberGenerator):
    """
    Generator using endmember bundles.

    Each material (endmember) is represented by a discrete set
    of spectral variants (a bundle). For each pixel, one spectrum
    per bundle is randomly selected.
    The reference endmember for each bundle is the mean endmember in the 
    bundle.

    Attributes
    ----------        
    bundles : list of np.ndarray
        List of length n_endmembers.
        Each element has shape (n_variants_k, n_bands).

    rng : np.random.Generator
        Random number generator used for sampling.
    """
    def __init__(
        self,
        bundles: list[np.ndarray],
        rng: Optional[np.random.Generator] = None,
    ):
        """
        Parameters
        ----------
        bundles : list of np.ndarray
            Each entry corresponds to one endmember and contains
            multiple spectral variants of shape
            (n_elements, n_bands).

        rng : np.random.Generator, optional
            Random number generator for reproducibility.
        """
        self.bundles = [np.asarray(b, dtype=float) for b in bundles]
        self.rng = rng or np.random.default_rng()
        self.n_endmembers, self.n_bands = self._validate_bundles()
        
    
    def _validate_bundles(self) -> tuple[int, int]:
        """
        Validate bundle structure.

        Returns
        -------
        n_endmembers : int
            Number of bundles.

        n_bands : int
            Common number of spectral bands.
        """
        if not self.bundles:
            raise ValueError("At least one bundle must be provided.")

        n_bands = None

        for k, bundle in enumerate(self.bundles):

            if bundle.ndim != 2:
                raise ValueError(
                    f"Bundle {k} must be a 2D array "
                    f"(n_variants, n_bands), got shape {bundle.shape}."
                )

            if bundle.shape[0] == 0:
                raise ValueError(
                    f"Bundle {k} contains no spectral variants."
                )

            if n_bands is None:
                n_bands = bundle.shape[1]
            
            elif bundle.shape[1] != n_bands:
                raise ValueError(
                    f"All bundles must have the same number of bands. "
                    f"Bundle 0 has {n_bands} bands, "
                    f"bundle {k} has {bundle.shape[1]}."
                )

            if not np.all(np.isfinite(bundle)):
                raise ValueError(
                    f"Bundle {k} contains NaN or infinite values."
                )

        return len(self.bundles), n_bands
        

    def generate(
        self,
        n_pixels: int,
        n_bands: int,
        n_endmembers: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate a spatially varying endmember tensor by sampling
        from bundles.

        Parameters
        ----------
        n_pixels : int
            Number of pixels.

        n_bands : int
            Number of spectral bands.

        n_endmembers : int
            Number of endmembers (must match number of bundles).

        Returns
        -------
        endmembers : np.ndarray of shape (n_bands, n_endmembers)
            Reference Endmember matrix.

        per_pixel_endmembers : np.ndarray of shape (n_pixels, n_bands, n_endmembers)
            Endmember matrix for each pixel. For each pixel, one endmember
            per bundle is selected randomly.
        """
        
        if n_endmembers != self.n_endmembers:
            raise ValueError(
                f"Expected {self.n_endmembers} endmembers, "
                f"got {n_endmembers}."
            )

        if n_bands != self.n_bands:
            raise ValueError(
                f"Expected {self.n_bands} bands, got {n_bands}."
            ) 
    
        per_pixel_endmembers = np.zeros(
            (n_pixels, n_bands, n_endmembers), dtype=float
        )
        endmembers = np.zeros((n_bands, n_endmembers), dtype=float)

        for k, bundle in enumerate(self.bundles):

            idx = self.rng.integers(
                0,
                bundle.shape[0],
                size=n_pixels,
            )

            per_pixel_endmembers[:, :, k] = bundle[idx, :]
            endmembers[:, k] = np.mean(bundle, axis=0)

        return endmembers, per_pixel_endmembers


# ------------------- Variable endmembers (continuous variability) ---------------------
class EndmemberVariabilityModel(ABC):
    """
    Abstract class for modeling spectral variability.

    This class defines how a reference spectrum is transformed
    into multiple spatially varying realizations.
    """
    @abstractmethod
    def sample(
        self,
        base_endmembers: np.ndarray,
        n_samples: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Generate perturbed spectral samples.

        Parameters
        ----------
        base_endmembers : np.ndarray of shape (n_bands,)
            Reference spectrum.

        n_samples : int
            Number of samples to generate.

        rng : np.random.Generator
            Random number generator.

        Returns
        -------
        np.ndarray of shape (n_samples, n_bands)
            Generated spectral variants.
        """
        raise NotImplementedError


class GaussianVariability(EndmemberVariabilityModel):
    """
    Multivariate Gaussian perturbation model for spectral variability.

    Attributes
    ----------
    covariance : np.ndarray
        Covariance matrix of shape (n_bands, n_bands).
    """
    def __init__(self, covariance: np.ndarray):
        """
        Parameters
        ----------
        covariance : np.ndarray
            Covariance matrix of shape (n_bands, n_bands).
        """
        covariance = np.asarray(covariance, dtype=float)

        if covariance.ndim != 2:
            raise ValueError(
                f"`covariance` must be 2D, got shape {covariance.shape}"
            )

        if covariance.shape[0] != covariance.shape[1]:
            raise ValueError(
                "`covariance` must be square."
            )

        if not np.allclose(covariance, covariance.T):
            raise ValueError(
                "`covariance` must be symmetric."
            )

        eigvals = np.linalg.eigvalsh(covariance)
        if np.any(eigvals < -1e-10):
            raise ValueError(
                "`covariance` must be positive semidefinite."
            )

        self.covariance = covariance

    def sample(
        self,
        base: np.ndarray,
        n_samples: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Generate Gaussian-perturbed spectral samples.

        Parameters
        ----------
        base : np.ndarray of shape (n_bands,)
            Reference spectrum.

        n_samples : int
            Number of samples.

        rng : np.random.Generator
            Random generator.

        Returns
        -------
        np.ndarray of shape (n_samples, n_bands)
            Perturbed spectral samples.
        """
        base = np.asarray(base, dtype=float)

        if base.ndim != 1:
            raise ValueError(
                f"`base` must be 1D (n_bands,), got shape {base.shape}"
            )

        n_bands = base.shape[0]

        if self.covariance.shape != (n_bands, n_bands):
            raise ValueError(
                f"`covariance` must have shape "
                f"({n_bands}, {n_bands}), "
                f"got {self.covariance.shape}"
            )

        noise = rng.multivariate_normal(
            mean=np.zeros(n_bands),
            cov=self.covariance,
            size=n_samples,
        )

        return base + noise


class VariableEndmemberGenerator(EndmemberGenerator):
    """
    Generator for continuously varying endmembers.

    Each pixel has a distinct realization of each endmember
    generated from a variability model.

    Attributes
    ----------
    base_endmembers : np.ndarray of shape (n_bands, n_endmembers)
        Reference endmember matrix.

    models : list of EndmemberVariabilityModel
        One variability model per endmember.

    rng : np.random.Generator
        Random number generator.
    """
    def __init__(
        self,
        base_endmembers: np.ndarray,
        variability_models: list[EndmemberVariabilityModel],
        rng: Optional[np.random.Generator] = None,
    ):
        """
        Parameters
        ----------
        base_endmembers : np.ndarray of shape (n_bands, n_endmembers)
            Reference spectral signatures.

        variability_models : list of EndmemberVariabilityModel
            One model per endmember defining how it varies.

        rng : np.random.Generator, optional
            Random number generator.
        """
        self.base_endmembers = np.asarray(base_endmembers, dtype=float)
        self.models = variability_models
        self.rng = rng or np.random.default_rng()

    def generate(
        self,
        n_pixels: int,
        n_bands: int,
        n_endmembers: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate spatially varying endmembers.

        Parameters
        ----------
        n_pixels : int
            Number of pixels.

        n_bands : int
            Number of spectral bands.

        n_endmembers : int
            Number of endmembers.

        Returns
        -------
        endmembers : np.ndarray of shape (n_bands, n_endmembers)
            Reference Endmember matrix.

        per_pixel_endmembers : np.ndarray of shape (n_pixels, n_bands, n_endmembers)
            Endmember matrix for each pixel.
        """
        if self.base_endmembers.shape != (n_bands, n_endmembers):
            raise ValueError("Base endmember shape mismatch")

        if len(self.models) != n_endmembers:
            raise ValueError("Variability model count mismatch")

        per_pixel_endmembers = np.zeros(
            (n_pixels, n_bands, n_endmembers), dtype=float
        )

        for k in range(n_endmembers):
            samples = self.models[k].sample(
                self.base_endmembers[:, k],
                n_pixels,
                self.rng,
            )
            per_pixel_endmembers[:, :, k] = samples

        return self.base_endmembers, per_pixel_endmembers 

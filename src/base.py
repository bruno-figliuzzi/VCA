from abc import ABC, abstractmethod
import numpy as np
from typing import Optional


class BlindUnmixing(ABC):
    """
    Abstract base class for blind hyperspectral unmixing algorithms.

    This class provides a template shared by every family of blind
    unmixing methods (e.g. geometric endmember extraction, autoencoder
    based unmixing, ...): algorithms that estimate endmembers - and,
    depending on the method, abundances - directly from hyperspectral
    observations, without prior knowledge of either. It validates the
    input dataset and defines common attributes used by derived models.

    Parameters
    ----------
    n_endmembers : int
        Estimated number of endmembers (signal subspace dimension).

    rng : np.random.Generator
        Local random numbers generator for reproducible stochastic operations.
    """
    def __init__(
        self,
        n_endmembers: int,
        rng: np.random.Generator | None = None
    ) -> None:

        if n_endmembers <= 0:
            raise ValueError("n_endmembers must be positive.")

        self.n_endmembers = n_endmembers

        # Set random seed for reproducibility
        self.rng = rng or np.random.default_rng()

    @staticmethod
    def _validate_dataset(
        Y: np.ndarray,
        n_endmembers: int,
    ) -> np.ndarray:
        """
        Validate the hyperspectral dataset

        Parameters
        ----------
        Y : np.ndarray of shape (n_bands, n_pixels)
            Hyperspectral data matrix
        n_endmembers: int
            Number of endmembers
            
        Return
        ------
        np.ndarray
            Hyperspectral data matrix as a numpy array (n_bands, n_pixels)
        """
        Y = np.asarray(Y)
        if Y.ndim != 2:
            raise ValueError("Y must be a 2D array of shape (n_bands, n_pixels)")

        n_bands, n_pixels = Y.shape

        if n_bands < n_endmembers:
            raise ValueError(
                f"The number of spectral bands {n_bands} should be larger than "
                f"the number {n_endmembers} of endmembers."
            )

        if n_endmembers >= n_pixels:
            raise ValueError(
                f"The number of pixels {n_pixels} should be larger than "
                f"the number {n_endmembers} of endmembers."
            )

        return Y

    @staticmethod
    def _validate_endmembers_init(
        M_init: np.ndarray,
        n_bands: int,
        n_endmembers: int,
        nonnegative: bool = False,
    ) -> np.ndarray:
        """
        Validate user-provided initial endmembers.

        Parameters
        ----------
        M_init : np.ndarray of shape (n_bands, n_endmembers)
            Initial endmember matrix, one spectrum per column.
        n_bands : int
            Number of spectral bands of the data.
        n_endmembers : int
            Number of endmembers of the model.
        nonnegative : bool, default=False
            Whether every entry must be non-negative (NMF algorithms).

        Returns
        -------
        np.ndarray of shape (n_bands, n_endmembers)
            A float64 copy of ``M_init``.

        Raises
        ------
        ValueError
            If ``M_init`` has the wrong shape, contains non-finite values,
            or negative values when ``nonnegative`` is True.
        """
        M_init = np.array(M_init, dtype=np.float64)

        if M_init.shape != (n_bands, n_endmembers):
            raise ValueError(
                f"endmembers_init must have shape ({n_bands}, {n_endmembers}) "
                f"(n_bands, n_endmembers), got {M_init.shape}."
            )

        if not np.all(np.isfinite(M_init)):
            raise ValueError("endmembers_init contains non-finite values.")

        if nonnegative and np.any(M_init < 0):
            raise ValueError("endmembers_init must be non-negative.")

        return M_init

    @abstractmethod
    def fit(
        self,
        Y: np.ndarray,
        *args,
        **kwargs,
    ) -> "BlindUnmixing":
        """
        Estimate endmembers (and, depending on the algorithm, abundances)
        from the hyperspectral data.

        Parameters
        ----------
        Y : np.ndarray of shape (n_bands, n_pixels)
            Hyperspectral data matrix

        Must be implemented by subclasses.
        """
        raise NotImplementedError
        
        
class SparseUnmixing(ABC):
    """
    Abstract base class for sparse hyperspectral unmixing algorithms.

    This class provides a template shared by every family of sparse
    unmixing methods (e.g. SunSAL, CLS ...): algorithms that estimate 
    endmembers and abundances based on a previously available endmembers 
    library. It validates the library, the input dataset and defines 
    common attributes used by derived models.

    Parameters
    ----------
    n_endmembers : int
        Estimated number of endmembers (signal subspace dimension).
        
    library: np.ndarray
        Endmembers library

    rng : np.random.Generator
        Local random numbers generator for reproducible stochastic operations.
    """
    def __init__(
        self,
        n_endmembers: int,
        library: np.ndarray,
        rng: np.random.Generator | None = None
    ) -> None:

        if n_endmembers <= 0:
            raise ValueError("n_endmembers must be positive.")
            
        self.n_endmembers = n_endmembers
        
        self.library = self._validate_library(library, n_endmembers)

        # Set random seed for reproducibility
        self.rng = rng or np.random.default_rng()

    @staticmethod
    def _validate_dataset(
        Y: np.ndarray,
        n_endmembers: int,
    ) -> np.ndarray:
        """
        Validate the hyperspectral dataset

        Parameters
        ----------
        Y : np.ndarray of shape (n_bands, n_pixels)
            Hyperspectral data matrix
        n_endmembers: int
            Number of endmembers
            
        Return
        ------
        np.ndarray
            Hyperspectral data matrix as a numpy array (n_bands, n_pixels)
        """
        Y = np.asarray(Y)
        if Y.ndim != 2:
            raise ValueError("Y must be a 2D array of shape (n_bands, n_pixels)")

        n_bands, n_pixels = Y.shape

        if n_bands < n_endmembers:
            raise ValueError(
                f"The number of spectral bands {n_bands} should be larger than "
                f"the number {n_endmembers} of endmembers."
            )

        if n_endmembers >= n_pixels:
            raise ValueError(
                f"The number of pixels {n_pixels} should be larger than "
                f"the number {n_endmembers} of endmembers."
            )

        return Y


    @staticmethod
    def _validate_library(
        library: np.ndarray,
        n_endmembers: int,
    ) -> np.ndarray:
        """
        Validate the endmembers library

        Parameters
        ----------
        library : np.ndarray of shape (n_bands, n_spectra)
            Endmember library
            
        n_endmembers: int
            Number of endmembers
            
        Return
        ------
        np.ndarray
            Endmember library as a numpy array (n_bands, n_spectra)
        """
        
        library = np.asarray(library)
        if library.ndim != 2:
            raise ValueError("library must be a 2D array of shape (n_bands, n_spectra)")

        n_bands, n_spectra = library.shape

        if n_bands < n_endmembers:
            raise ValueError(
                f"The number of spectral bands {n_bands} should be larger than "
                f"the number {n_endmembers} of endmembers."
            )

        if n_endmembers >= n_spectra:
            raise ValueError(
                f"The number of spectra {n_spectra} in the library should be larger "
                f"than the number {n_endmembers} of endmembers."
            )

        return library
        

    @abstractmethod
    def fit(
        self,
        Y: np.ndarray,
        *args,
        **kwargs,
    ) -> "SparseUnmixing":
        """
        Estimate endmembers (and, depending on the algorithm, abundances)
        from the hyperspectral data.

        Parameters
        ----------
        Y : np.ndarray of shape (n_bands, n_pixels)
            Hyperspectral data matrix

        Must be implemented by subclasses.
        """
        raise NotImplementedError
        

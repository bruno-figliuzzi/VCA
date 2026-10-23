from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
from .dirichlet import DirichletMixture


# ----------------------------------- Base class ---------------------------------------
class AbundanceGenerator(ABC):
    """
    Abstract base class for abundance generators.

    Abundance generators are responsible for producing abundance vectors
    satisfying the simplex constraints commonly used in hyperspectral
    unmixing.
    """
    @abstractmethod
    def generate(
        self,
        n_pixels: int,
        n_endmembers: int,
        min_purity: float | np.ndarray | None = None,
        max_purity: float | np.ndarray | None = None
    ) -> np.ndarray:
        """
        Generate abundance vectors.

        Parameters
        ----------
        n_pixels : int
            Number of pixels to generate.

        n_endmembers : int
            Number of endmembers.
            
        min_purity : float or ndarray of shape (n_endmembers,), optional
            Elementwise lower bounds on the abundance for each endmembers. 
            If a scalar is provided, the same lower bound is applied to 
            all dimensions.

        max_purity : float or ndarray of shape (n_endmembers,), optional
            Elementwise upper bounds on the abundance for each endmembers. 
            If a scalar is provided, the same upper bound is applied to 
            all dimensions.        

        Returns
        -------
        ndarray of shape (n_endmembers, n_pixels)
            Generated abundance matrix.
        """
        raise NotImplementedError


# -------------------- Mixture of Dirichlet distributions-------------------------------
class DirichletAbundanceGenerator(AbundanceGenerator):
    """
    Abundance generator using a mixture of Dirichlet distribution.
    
    Parameters
    ----------
    distr: DirichletMixture
        Instance of the class DirichletMixture representing the mixture of 
        Dirichlet distributions.
    rng: 
    """
    def __init__(
        self,
        distr: DirichletMixture,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
    
        self.distr = distr
        self.rng = rng or np.random.default_rng()
        

    def generate(
        self,
        n_pixels: int,
        n_endmembers: int,
        min_purity: float | np.ndarray | None = None,
        max_purity: float | np.ndarray | None = None
    ) -> np.ndarray:
    
        if self.distr.dim != n_endmembers:
            raise ValueError(
                f"The number of endmembers should match the dimension "
                f"{self.distr.dim} of the mixture of Dirichlet distribution."
            )
        
        samples = self.distr.sample(
            n_pixels,
            lower_bounds = min_purity,
            upper_bounds = max_purity,
            rng = self.rng
        )
        return samples


# ------------------------------ Custom abundances -------------------------------------
class CustomAbundanceGenerator(AbundanceGenerator):
    """
    Wrap a user-provided abundance matrix.

    Parameters
    ----------
    abundances : ndarray of shape (n_endmembers, n_pixels)
    """
    def __init__(
        self,
        abundances: np.ndarray,
    ) -> None:
    
        self._validate_simplex(np.asarray(abundances, dtype=float))
        self.abundances = np.asarray(
            abundances,
            dtype=float,
        )

    def generate(
        self,
        n_pixels: int,
        n_endmembers: int,
        min_purity: float | np.ndarray | None = None,
        max_purity: float | np.ndarray | None = None
    ) -> np.ndarray:

        if self.abundances.shape != (
            n_endmembers,
            n_pixels,
        ):
            raise ValueError(
                f"Shape mismatch: expected {(n_endmembers, n_pixels)}, "
                f"got {self.abundances.shape}"
            )

        return self.abundances.copy()
        
        
    def _validate_simplex(self, abundances: np.ndarray) -> None:
        """
        Validate simplex constraints:
        - non-negative
        - rows sum to 1
        
        Parameters
        ----------
        abundances: np.ndarray
            Specified abundance array
        """

        if np.any(abundances < 0):
            raise ValueError("Abundances must be non-negative.")

        row_sums = abundances.sum(axis=0)
        if not np.allclose(row_sums, 1.0, atol=1e-4):
            raise ValueError("Columns of abundances must sum to 1.")

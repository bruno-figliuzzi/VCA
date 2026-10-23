import warnings
from abc import abstractmethod
import numpy as np

from ..base import BlindUnmixing


class GeometricUnmixing(BlindUnmixing):
    """
    Abstract base class for geometric unmixing algorithms.

    This class provides a template for geometric hyperspectral unmixing methods
    (e.g. N-FINDR, VCA, etc.)

    Parameters
    ----------
    n_endmembers : int
        Estimated number of endmembers (signal subspace dimension).

    rng : np.random.Generator
        Local random numbers generator for reproducible stochastic operations.

    Attributes
    ----------
    endmembers_ : np.ndarray of shape (n_bands, n_endmembers)
        Estimated endmembers, set by `fit`.

    abundances_ : np.ndarray of shape (n_endmembers, n_pixels)
        Abundances of the estimated endmembers, set by
        `estimate_abundances` (geometric algorithms only estimate the
        endmembers).

    abundances_init_ : np.ndarray of shape (n_endmembers, n_pixels)
        Abundances of the initial endmembers (``endmembers_init_``), set by
        `estimate_abundances` for the algorithms that store their
        initialization.
    """

    @abstractmethod
    def fit(self,
        Y: np.ndarray
    ) -> "GeometricUnmixing":
        """
        Extract endmembers from the hyperspectral data.

        Parameters
        ----------
        Y : np.ndarray of shape (n_bands, n_pixels)
            Hyperspectral data matrix

        Must be implemented by subclasses.
        """
        raise NotImplementedError

    def estimate_abundances(
        self,
        Y: np.ndarray,
        solver=None,
    ) -> "GeometricUnmixing":
        """
        Estimate the abundances of the fitted endmembers.

        Sets ``abundances_`` and, when the model stores its initialization
        (``endmembers_init_``), ``abundances_init_``.

        Parameters
        ----------
        Y : np.ndarray of shape (n_bands, n_pixels)
            Hyperspectral data matrix, at the same scale as the data passed
            to `fit`.

        solver : AbundanceSolver, optional
            Solver exposing ``solve(Y, M)``. Defaults to fully constrained
            least squares (`abundances.fcls.FCLS`).

        Returns
        -------
        self : GeometricUnmixing
        """
        if not hasattr(self, "endmembers_"):
            raise RuntimeError("The model must be fitted before estimating abundances.")

        if solver is None:
            from ..abundances.fcls import FCLS  # lazy: avoids importing torch with the module
            solver = FCLS()

        self.abundances_ = solver.solve(Y, self.endmembers_)
        M_init = getattr(self, "endmembers_init_", None)
        if M_init is not None:
            self.abundances_init_ = solver.solve(Y, M_init)
        return self

    @property
    def endmembers(self) -> np.ndarray:
        """Deprecated alias of ``endmembers_``."""
        warnings.warn(
            "`endmembers` is deprecated, use `endmembers_` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.endmembers_

    @endmembers.setter
    def endmembers(self, value: np.ndarray) -> None:
        warnings.warn(
            "`endmembers` is deprecated, use `endmembers_` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.endmembers_ = value

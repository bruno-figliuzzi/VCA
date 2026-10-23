from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypeAlias

import numpy as np
import torch
from torch import Tensor

ArrayLike: TypeAlias = np.ndarray | Tensor


class AbundanceSolver(ABC):
    """
    Base class for abundance estimation algorithms.

    Abundance solvers estimate the abundance matrix associated with a
    hyperspectral image for a fixed endmember matrix.

    Parameters
    ----------
    dtype : torch.dtype, default=torch.float32
        Floating-point dtype used for all internal computations. During
        validation, all inputs are converted to this dtype.

    device : str, default="cpu"
        Device on which computations are performed.
        
    Notes
    -----
    The public API accepts both NumPy arrays and PyTorch tensors. Inputs
    are automatically converted to PyTorch tensors during validation,
    allowing implementations to rely exclusively on PyTorch operations.

    All inputs are converted to the solver's computation dtype specified by
    ``dtype``.
    """
    def __init__(
        self, 
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ):
        self.dtype = dtype
        
        device = torch.device(device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable.")
        self.device = device        
    
    def _as_tensor(self, X: ArrayLike) -> Tensor:
        """
        Convert an array-like input to a tensor, copying it first if it is
        a non-writable NumPy array (`torch.as_tensor` would otherwise wrap
        that read-only memory directly, which triggers a PyTorch warning).
        """
        if isinstance(X, np.ndarray) and not X.flags.writeable:
            X = X.copy()
        return torch.as_tensor(X, dtype=self.dtype, device=self.device)

    def _validate_inputs(
        self,
        Y: ArrayLike,
        M: ArrayLike,
        A_init: ArrayLike | None = None,
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        """
        Validate and convert the input data.
        """
        Y = self._as_tensor(Y)
        M = self._as_tensor(M)

        if A_init is not None:
            A_init = self._as_tensor(A_init)

        # Checks that all tensors have proper dimensions and shapes
        if Y.ndim != 2:
            raise ValueError(
                "Y must have shape (n_bands, n_pixels)."
            )

        if M.ndim != 2:
            raise ValueError(
                "M must have shape (n_bands, n_endmembers)."
            )

        if A_init is not None and A_init.ndim != 2:
            raise ValueError(
                "A_init must have shape (n_endmembers, n_pixels)."
            )

        if Y.shape[0] != M.shape[0]:
            raise ValueError(
                f"Band mismatch: Y has {Y.shape[0]} bands "
                f"whereas M has {M.shape[0]}."
            )

        if A_init is not None and A_init.shape != (M.shape[1], Y.shape[1]):
            raise ValueError(
                f"A_init must have shape {(M.shape[1], Y.shape[1])}, "
                f"got {tuple(A_init.shape)}."
            )

        if not torch.isfinite(Y).all().item():
            raise ValueError(
                "Y contains NaN or infinite values."
            )

        if not torch.isfinite(M).all().item():
            raise ValueError(
                "M contains NaN or infinite values."
            )

        if A_init is not None and not torch.isfinite(A_init).all().item():
            raise ValueError(
                "A_init contains NaN or infinite values."
            )

        return Y, M, A_init


    def solve_torch(
        self,
        Y: ArrayLike,
        M: ArrayLike,
        A_init: ArrayLike | None = None,
    ) -> Tensor:
        """
        Estimate abundance maps.

        Parameters
        ----------
        Y : ndarray or Tensor of shape (n_bands, n_pixels)
            Hyperspectral observations.

        M : ndarray or Tensor of shape (n_bands, n_endmembers)
            Endmember matrix.

        A_init : ndarray or Tensor of shape (n_endmembers, n_pixels), optional
            Initial abundance estimate used to warm-start the optimization.
            
        Returns
        -------
        Tensor of shape (n_endmembers, n_pixels)
            Estimated abundance matrix.
        """
        Y, M, A_init = self._validate_inputs(Y, M, A_init)
        return self._solve(Y, M, A_init)
    
        
    @abstractmethod
    def _solve(
        self,
        Y: Tensor,
        M: Tensor,
        A_init: Tensor | None = None,
    ) -> Tensor:
        """
        Internal implementation of the abundance estimation solver.
        Inputs are already validated and converted to torch tensors. 
        """
        raise NotImplementedError
        
        
    def solve(
        self,
        Y: ArrayLike,
        M: ArrayLike,
        A_init: ArrayLike | None = None,
    ) -> np.ndarray:
        """
        Estimate abundance maps and return a NumPy array.
        
        Parameters
        ----------
        Y : ndarray or Tensor of shape (n_bands, n_pixels)
            Hyperspectral observations.

        M : ndarray or Tensor of shape (n_bands, n_endmembers)
            Endmember matrix.
            
        A_init : ndarray or Tensor of shape (n_endmembers, n_pixels), optional
            Initial abundance estimate used to warm-start the optimization.
            
        Returns
        -------
        numpy array of shape (n_endmembers, n_pixels)
            Estimated abundance matrix.
        """
        return self.solve_torch(Y, M, A_init).detach().cpu().numpy()

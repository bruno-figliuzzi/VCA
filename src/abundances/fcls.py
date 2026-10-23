from __future__ import annotations
import math
import torch
from torch import Tensor
from .base import AbundanceSolver, ArrayLike


class FCLS(AbundanceSolver):
    """
    Fully Constrained Least Squares (FCLS).

    This solver estimates abundance vectors by minimizing the
    reconstruction error under the abundance non-negativity and
    sum-to-one constraints using the FISTA algorithm.

    Parameters
    ----------
    max_iter : int, default=500
        Maximum number of FISTA iterations for each mini-batch.

    tol : float, default=1e-6
        Convergence tolerance based on the maximum absolute change
        between successive abundance estimates.

    batch_size : int or None, default=None
        Number of pixels processed simultaneously. If ``None``, the
        entire dataset is processed in a single batch.

    dtype : torch.dtype, default=torch.float32
        Floating-point dtype used for all internal computations. During
        validation, all inputs are converted to this dtype.

    device : str, default="cpu"
        Device on which computations are performed.
    """
    def __init__(
        self,
        max_iter: int = 500,
        tol: float = 1e-6,
        batch_size: int | None = None,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ) -> None:
    
        super().__init__(dtype, device)

        if batch_size is not None and batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        if max_iter <= 0:
            raise ValueError("max_iter must be positive.")

        if tol <= 0:
            raise ValueError("tol must be positive.")

        self.batch_size = batch_size
        self.max_iter = max_iter
        self.tol = tol
        

    @staticmethod
    def _project_simplex(
        x: Tensor,
    ) -> Tensor:
        """
        Project each column onto the probability simplex.

        Parameters
        ----------
        x : Tensor of shape (n_endmembers, n_pixels)

        Returns
        -------
        Tensor
            Projected abundances with the same shape as ``x``.
        """
        v = x.transpose(0, 1)
        u, _ = torch.sort(
            v,
            dim=1,
            descending=True,
        )

        cssv = torch.cumsum(u, dim=1) - 1
        ind = torch.arange(
            1,
            v.shape[1] + 1,
            device=v.device,
            dtype=v.dtype,
        )

        cond = u - cssv / ind > 0
        rho = cond.sum(dim=1, keepdim=True) - 1
        theta = cssv.gather(1, rho) / (rho + 1)
        return torch.clamp(v - theta, min=0.0).T
        

    def _solve_batch(
        self,
        MtM: Tensor,
        MtY: Tensor,
        step: float,
        A_init: Tensor | None = None,
    ) -> Tensor:
        """
        Solve the Fully Constrained Least Squares (FCLS) problem for one mini-batch
        using the FISTA algorithm.

        Parameters
        ----------
        MtM : Tensor of shape (n_endmembers, n_endmembers)
            Precomputed matrix product $M^T M$.
        MtY : Tensor of shape (n_endmembers, n_pixels)
            Precomputed matrix product $M^T Y$ for the batch.
        step : float
            Step size (inverse of the Lipschitz constant) for the gradient step.
        A_init : Tensor of shape (n_endmembers, n_pixels), optional
            Initial abundance estimate for warm-starting.

        Returns
        -------
        Tensor of shape (n_endmembers, n_pixels)
            Optimized abundance estimates for the batch satisfying simplex constraints.
        """
        p, n_pixels = MtY.shape

        if A_init is None:
            A = torch.full(
                (p, n_pixels),
                1.0 / p,
                device=self.device,
                dtype=MtY.dtype,
            )
        else:
            A = self._project_simplex(A_init)

        Z = A.clone()
        t = 1.0
        
        with torch.no_grad():
            for _ in range(self.max_iter):

                A_old = A
                grad = MtM @ Z - MtY
                A = self._project_simplex(Z - step * grad)

                if torch.max(torch.abs(A - A_old)) < self.tol:
                    break

                t_new = (1.0 + math.sqrt(1.0 + 4.0 * t * t)) / 2.0
                Z = A + ((t - 1.0) / t_new) * (A - A_old)
                t = t_new

        return A


    def _solve(
        self,
        Y: Tensor,
        M: Tensor,
        A_init: Tensor | None = None,
    ) -> Tensor:
        """
        Internal FISTA implementation.

        Inputs are already validated and reside on the configured
        device with the configured dtype.
        """       
        _, n_pixels = Y.shape
        p = M.shape[1]
        batch_size = n_pixels if self.batch_size is None else self.batch_size

        Mt = M.T
        MtM = Mt @ M

        lipschitz = torch.linalg.norm(M, ord=2).square()
        step = 1.0 / lipschitz.item()

        A = torch.empty(
            (p, n_pixels),
            device=self.device,
            dtype=Y.dtype,
        )
        A_old = A.detach().clone()

        for start in range(0, n_pixels, batch_size):

            stop = min(start + batch_size, n_pixels)
            MtY = Mt @ Y[:, start:stop]

            A[:, start:stop] = self._solve_batch(
                MtM=MtM,
                MtY=MtY,
                step=step,
                A_init=None if A_init is None else A_init[:, start:stop],
            )
        return A


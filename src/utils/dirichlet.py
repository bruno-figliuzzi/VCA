import numpy as np
from scipy.special import gammaln
import numbers

class DirichletMixture:
    """
    Mixture of Dirichlet distributions.

    A Dirichlet mixture models probability distributions over the probability simplex.
    Each component is a Dirichlet distribution parameterized by a given
    concentration vector
   
    Parameters
    ----------
    dim : int
        Dimensionality of the simplex (number of categories).

    alphas : list of np.ndarray, shape (dim,)
        List of Dirichlet concentration parameters for each mixture component.
        Each parameter must be strictly positive.

    weights : np.ndarray, shape (n_components,), optional
        Mixture weights for components.
        Must be non-negative and sum to 1.
        If None, uniform weights are used.
    """
    def __init__(
        self,
        dim: int,
        alphas: list[np.ndarray],
        weights: np.ndarray | None = None,
    ):
        if dim < 1:
            raise ValueError("dim must be >= 1")

        if not alphas:
            raise ValueError("alphas must be a non-empty list")

        self.dim = dim
        self.alphas = []
        self._log_norm_const = []

        # Parameters
        for a in alphas:
        
            a = np.asarray(a, dtype=float)
            if a.ndim != 1:
                raise ValueError("Each alpha must be a 1D array")
            if a.shape[0] != dim:
                raise ValueError("Each alpha must have shape (D,)")
            if np.any(~np.isfinite(a)):
                raise ValueError("alpha contains non-finite values")
            if np.any(a <= 0):
                raise ValueError("Dirichlet parameters must be strictly positive")

            self.alphas.append(a)

            # cache normalization constant of Dirichlet
            alpha0 = np.sum(a)
            self._log_norm_const.append(
                gammaln(alpha0) - np.sum(gammaln(a))
            )

        self.n_components = len(self.alphas)
        self._log_norm_const = np.asarray(self._log_norm_const)

        # weights
        if weights is None:
            self.weights = np.ones(self.n_components) / self.n_components
        else:
            weights = np.asarray(weights, dtype=float)

            if weights.ndim != 1:
                raise ValueError("weights must be 1D")
            if len(weights) != self.n_components:
                raise ValueError("weights must match number of components")
            if np.any(weights < 0):
                raise ValueError("weights must be non-negative")
            if not np.isclose(weights.sum(), 1.0):
                raise ValueError("weights must sum to 1")

            self.weights = weights

        self._log_weights = np.log(self.weights)

    # ---------------------------------------------------------------------
    # Dirichlet log density
    # ---------------------------------------------------------------------
    def dirichlet_logpdf(self, x: np.ndarray) -> np.ndarray:
        """
        Compute log-density of each Dirichlet component at x.

        Parameters
        ----------
        x : np.ndarray, shape (dim,)
            Point on the probability simplex.

        Returns
        -------
        np.ndarray, shape (n_components,)
            Log-density under each Dirichlet component.

        Notes
        -----
        - Returns -inf for invalid (non-simplex) inputs with negative values.
        - Requires Σ x_i = 1 (within tolerance).
        """
        x = np.asarray(x, dtype=float)

        if x.shape != (self.dim,):
            raise ValueError(f"x must have shape ({self.dim},)")

        if np.any(x < 0):
            raise ValueError("Negative values not allowed")

        if not np.isclose(np.sum(x), 1.0, atol=1e-6):
            raise ValueError("x must lie on the probability simplex")

        alphas = np.stack(self.alphas)  # (n_components, dim)

        # safe log(x)
        logx = np.zeros_like(x)
        mask = x > 0
        logx[mask] = np.log(x[mask])

        out = self._log_norm_const + ((alphas - 1.0) @ logx)

        # boundary correction (vectorized logic)
        zero_mask = (x == 0)

        if np.any(zero_mask):
            a_zero = alphas[:, zero_mask]

            out[np.any(a_zero < 1.0, axis=1)] = np.inf
            out[np.any(a_zero > 1.0, axis=1)] = -np.inf

        return out

    # ---------------------------------------------------------------------
    # Log density of mixture
    # ---------------------------------------------------------------------
    def log_density(self, x: np.ndarray) -> float | np.ndarray:
        """
        Compute log p(x) under the Dirichlet mixture.

        Supports both single and batched inputs.

        Parameters
        ----------
        x : np.ndarray
            Shape:
                - (dim,) → single sample
                - (N, dim) → batch of samples

        Returns
        -------
        float or np.ndarray
            - scalar if input is (dim,)
            - array of shape (N,) if input is (N, dim)

        Raises
        ------
        ValueError
            If input shape is invalid or points are not on simplex.

        Notes
        -----
        Uses numerically stable log-sum-exp over mixture components.
        """
        x = np.asarray(x, dtype=float)
        single = (x.ndim == 1)

        if single:
            x = x[None, :]

        if x.shape[1] != self.dim:
            raise ValueError("Invalid input shape")

        if np.any(x < 0):
            raise ValueError("Negative values not allowed")

        if not np.allclose(np.sum(x, axis=1), 1.0, atol=1e-6):
            raise ValueError("All rows must lie on simplex")

        logx = np.where(x > 0, np.log(x), -np.inf)
        A = np.stack(self.alphas)
        log_probs = self._log_weights + self._log_norm_const + logx @ (A - 1).T
            
        m = np.max(log_probs, axis=1, keepdims=True)
        out = m + np.log(np.sum(np.exp(log_probs - m), axis=1, keepdims=True))
        out = out.ravel()
        return float(out[0]) if single else out

    # ---------------------------------------------------------------------
    # Density
    # ---------------------------------------------------------------------
    def density(self, x: np.ndarray) -> float | np.ndarray:
        """
        Compute probability density p(x).

        Parameters
        ----------
        x : np.ndarray
            Input samples.

        Returns
        -------
        float or np.ndarray
            Density values.

        Notes
        -----
        This function exponentiates log_density and may underflow for extreme values.
        """
        return np.exp(self.log_density(x))

    # ---------------------------------------------------------------------
    # Sampling
    # ---------------------------------------------------------------------
    def _sample_unconstrained(
        self,
        n_samples: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Draw samples from the unconstrained Dirichlet mixture distribution.
        """
        components = rng.choice(
            self.n_components,
            size=n_samples,
            p=self.weights,
        )

        out = np.empty((n_samples, self.dim), dtype=float)

        for k in range(self.n_components):
            idx = np.where(components == k)[0]

            if len(idx) == 0:
                continue

            gamma_samples = rng.gamma(
                shape=self.alphas[k],
                scale=1.0,
                size=(len(idx), self.dim),
            )

            out[idx] = (
                gamma_samples
                / gamma_samples.sum(axis=1, keepdims=True)
            )

        return out


    def sample(
        self,
        n_samples: int = 1,
        lower_bounds: float | np.ndarray | None = None,
        upper_bounds: float | np.ndarray | None = None,
        max_generated: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """
        Draw samples from the Dirichlet mixture distribution subject to
        per-dimension bounds.

        Parameters
        ----------
        n_samples : int, default=1
            Number of samples to draw.

        lower_bounds : float or ndarray of shape (dim,), optional
            Elementwise lower bounds. If a scalar is provided, the same
            lower bound is applied to all dimensions.

        upper_bounds : float or ndarray of shape (dim,), optional
            Elementwise upper bounds. If a scalar is provided, the same
            upper bound is applied to all dimensions.

        max_generated : int, optional
            Maximum number of proposal samples allowed in rejection sampling.
            If None, defaults to 1000 * n_samples.

            This parameter controls the trade-off between runtime and the
            probability of successfully obtaining the requested number of
            valid samples under constraints. If the limit is reached before
            sufficient accepted samples are obtained, a RuntimeError is raised.
    
        rng : np.random.Generator, optional
            Random number generator for reproducibility.

        Returns
        -------
        np.ndarray of shape (dim, n_samples)
            Samples satisfying

                lower_bounds <= x <= upper_bounds

            elementwise.
        """
        if n_samples <= 0:
            raise ValueError("n_samples must be positive")

        rng = rng or np.random.default_rng()

        if lower_bounds is None:
            lower_bounds = np.zeros(self.dim)
            
        elif isinstance(lower_bounds, numbers.Real):
            lower_bounds = lower_bounds * np.ones(self.dim)
        
        if upper_bounds is None:
            upper_bounds = np.ones(self.dim)
            
        elif isinstance(upper_bounds, numbers.Real):
            upper_bounds = upper_bounds * np.ones(self.dim)
            
        lower_bounds = np.asarray(lower_bounds, dtype=float)
        upper_bounds = np.asarray(upper_bounds, dtype=float)

        if lower_bounds.shape != (self.dim,):
            raise ValueError(
                f"lower_bounds must have shape ({self.dim},)"
            )

        if upper_bounds.shape != (self.dim,):
            raise ValueError(
                f"upper_bounds must have shape ({self.dim},)"
            )

        if np.any(lower_bounds < 0):
            raise ValueError(
                "lower_bounds must be non-negative"
            )

        if np.any(upper_bounds > 1):
            raise ValueError(
                "upper_bounds must be <= 1"
            )

        if np.any(lower_bounds > upper_bounds):
            raise ValueError(
                "lower_bounds must be <= upper_bounds elementwise"
            )

        # Feasibility checks on the simplex
        if lower_bounds.sum() > 1:
            raise ValueError(
                "Infeasible constraints: sum(lower_bounds) > 1"
            )

        if upper_bounds.sum() < 1:
            raise ValueError(
                "Infeasible constraints: sum(upper_bounds) < 1"
            )
            
        # No constraints
        if (
            np.all(lower_bounds == 0)
            and np.all(upper_bounds == 1)
        ):
            return self._sample_unconstrained(n_samples, rng).T

        accepted = []
        batch_size = max(100, 4 * n_samples)
        if max_generated is None:
            max_generated = 1000 * n_samples
        
        generated = 0   
        while len(accepted) < n_samples:

            candidates = self._sample_unconstrained(
                batch_size,
                rng,
            )
            mask = np.all(
                (candidates >= lower_bounds) & (candidates <= upper_bounds),
                axis=1,
            )
            accepted.extend(candidates[mask])
            generated += len(candidates)

            if generated > max_generated:
                raise RuntimeError(
                    f"Unable to obtain enough samples after {generated} generations. "
                     "Acceptance rate appears too low."
                )
                 
        return np.asarray(accepted[:n_samples]).T


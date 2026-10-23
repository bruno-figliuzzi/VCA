from abc import ABC, abstractmethod
from typing import Optional
import numpy as np

class NoiseModel(ABC):
    """
    Abstract base class for noise models.

    Noise models are responsible for perturbing a clean hyperspectral
    dataset to simulate sensor imperfections and acquisition noise.

    Implementations should preserve the shape of the input array.

    Notes
    -----
    The input data is assumed to be arranged as

        (..., n_bands)

    where the last dimension n_bands corresponds to spectral bands.
    """
    @abstractmethod
    def add_noise(self, X: np.ndarray) -> np.ndarray:
        """
        Apply noise to a hyperspectral dataset.

        Parameters
        ----------
        X : np.ndarray
            Clean hyperspectral data.

        Returns
        -------
        np.ndarray
            Noisy hyperspectral data with the same shape as ``X``.
        """
        ...

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return self.add_noise(X)
        
           
class GaussianNoise(NoiseModel):
    """
    Additive white Gaussian noise.

    Noise can be specified either directly through its standard
    deviation or indirectly through a target signal-to-noise ratio
    (SNR). Exactly one of ``scale`` or ``snr_db`` must be provided.

    Parameters
    ----------
    scale : float, optional
        Standard deviation of the additive Gaussian noise.
        Must be non-negative.

    snr_db : float, optional
        Desired signal-to-noise ratio in decibels. When provided,
        the noise standard deviation is computed from the input data
        at each call to :meth:`add_noise` according to

        .. math::

            \\mathrm{SNR}_{dB}
            =
            10 \\log_{10}
            \\left(
                \\frac{P_{signal}}
                     {P_{noise}}
            \\right),

        where :math:`P_{signal}` and :math:`P_{noise}` denote the
        average signal and noise powers, respectively.

    rng : numpy.random.Generator, optional
        Random number generator used to sample the noise. If omitted,
        a new default generator is created.

    Raises
    ------
    ValueError
        If neither or both of ``scale`` and ``snr_db`` are provided.
        
    Notes
    -----
    When ``snr_db`` is used, the signal power is estimated from all values
    in ``X``. The resulting noise level therefore corresponds to a global
    dataset-wide SNR rather than a per-band or per-pixel SNR.
    """
    def __init__(
        self,
        scale: float | None = None,
        snr_db: float | None = None,
        rng: Optional[np.random.Generator] = None,
    ):
        if (scale is None) == (snr_db is None):
            raise ValueError(
                "Exactly one of scale or snr_db must be provided."
            )
            
        if scale is not None and scale < 0:
            raise ValueError(
                "scale must be non-negative."
            )

        self.scale = scale
        self.snr_db = snr_db
        self.rng = rng or np.random.default_rng()
         
    def add_noise(self, X: np.ndarray) -> np.ndarray:
        """
        Add Gaussian noise to a hyperspectral dataset.

        If ``snr_db`` was specified at construction time, the noise
        standard deviation is computed from the power of ``X`` before
        noise is generated.
        """
        X = np.asarray(X, dtype=float)
        
        if self.snr_db is not None:
            signal_power = np.mean(X**2)
            noise_power = signal_power / (10**(self.snr_db / 10))
            scale = np.sqrt(noise_power)
        else:
            scale = self.scale

        noise = self.rng.normal(
            0.0,
            scale,
            X.shape,
        )
        return X + noise


class BandwiseGaussianNoise(NoiseModel):
    """
    Additive Gaussian noise with band-dependent variance.

    Each spectral band receives independent Gaussian noise with its own
    standard deviation.

    Parameters
    ----------
    std : ndarray of shape (n_bands,)
        Standard deviation for each spectral band.
    rng : numpy.random.Generator, optional
        Random number generator.

    Notes
    -----
    This model is useful for simulating sensors whose noise level
    varies across wavelengths.
    """

    def __init__(
        self,
        std: np.ndarray,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
    
        std = np.asarray(std, dtype=float)
        if np.any(std <= 0):
            raise ValueError(
                "The standard deviation must be positive for each band."
            )
        if std.ndim != 1:
            raise ValueError(
                "std must be a 1D array."
            )
        self.std = std
        self.rng = rng or np.random.default_rng()

    def add_noise(self, X: np.ndarray) -> np.ndarray:
    
        X = np.asarray(X, dtype=float)
        
        n_bands = X.shape[-1]
        if n_bands != self.std.size:
            raise ValueError(
                f"The number of spectral bands {n_bands} of the hyperspectral "
                f"observations should match the number {self.std.size} of spectral "
                f"bands of the noise."
            )

        noise = self.rng.normal(
            loc=0.0,
            scale=self.std,
            size=X.shape,
        )
        return X + noise


class SignalDependentGaussianNoise(NoiseModel):
    """
    Signal-dependent Gaussian noise.

    The variance depends on the signal intensity:

    .. math::

        \\sigma^2(x) = \\alpha x + \\beta

    Parameters
    ----------
    alpha : float
        Signal-dependent component.
    beta : float
        Constant variance component.
    rng : numpy.random.Generator, optional
        Random number generator.

    Notes
    -----
    This model approximates the behaviour of imaging sensors,
    where brighter pixels exhibit larger variance.
    """
    def __init__(
        self,
        alpha: float,
        beta: float,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
    
        alpha = float(alpha)
        beta = float(beta)
        if alpha < 0:
            raise ValueError(
                "The signal-dependent component must be non-negative."
            )
        if beta < 0:
            raise ValueError(
                "The constant variance component must be non-negative."
            )
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.rng = rng or np.random.default_rng()

    def add_noise(self, X: np.ndarray) -> np.ndarray:
    
        X = np.asarray(X, dtype=float)
        variance = np.maximum(
            self.alpha * np.maximum(X, 0.0) + self.beta,
            0.0,
        )
        std = np.sqrt(variance)

        noise = self.rng.normal(
            loc=0.0,
            scale=std,
            size=X.shape,
        )

        return X + noise


class PoissonNoise(NoiseModel):
    """
    Poisson photon-counting noise.

    This model simulates shot noise arising from photon counting
    statistics. Input spectra are first interpreted as expected photon
    counts after scaling.

    Parameters
    ----------
    scale : float, default=1000.0
        Conversion factor between reflectance values and expected
        photon counts.

    Notes
    -----
    Larger values of ``scale`` correspond to higher photon counts and
    therefore lower relative noise levels.

    The output is rescaled to the original reflectance domain.
    """
    def __init__(self, 
        scale: float = 1000.0,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
    
        self.scale = float(scale)
        if scale <= 0:
            raise ValueError(
                "Scale should be strictly positive."
            )
        self.rng = rng or np.random.default_rng()
        
        
    def add_noise(self, X: np.ndarray) -> np.ndarray:
    
        X = np.asarray(X, dtype=float)
        counts = np.clip(X, 0.0, None) * self.scale
        noisy_counts = self.rng.poisson(counts)
        return noisy_counts / self.scale

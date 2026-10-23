from abc import ABC, abstractmethod
import numpy as np

# -------------------------- Abstract projection class ---------------------------------
class Projection(ABC):
    """
    Abstract base class for linear and affine projection models.

    This class defines a unified interface for dimensionality reduction
    methods used in hyperspectral data analysis, including PCA, truncated
    SVD, and affine subspace projection models.

    All subclasses must implement a consistent API for:
    - fitting a projection model from data
    - projecting data into a low-dimensional representation
    - reconstructing data back into the original space
    - performing fit + transform in a single step

    The design is specialized for spectral data matrices of shape (L, N), where:

        L = number of spectral bands (features)
        N = number of pixels / samples

    Attributes
    ----------
    fitted : bool
        Indicates whether the model has been trained using `fit()`.
    """
    def __init__(
        self
    ) -> None:
        self.fitted = False

    @abstractmethod
    def fit(self, Y: np.ndarray) -> "Projection":
        """
        Learns the projection basis from input data.
        """
        ...  

    @abstractmethod
    def transform(self, Y: np.ndarray) -> np.ndarray:
        """
        Projects input data into the learned low-dimensional subspace.
        """
        ...
        
    @abstractmethod
    def reconstruct(self, Y_proj: np.ndarray) -> np.ndarray:
        """
        Reconstructs data from the low-dimensional representation.
        """
        ...
        
    def fit_transform(self, Y: np.ndarray) -> np.ndarray:
        """
        Convenience method that fits the model and returns projected data.
        """
        self.fit(Y)
        return self.transform(Y)    

    
# -------------------------- Principal component analysis ------------------------------
class PCA(Projection):
    """
    Perform Principal Component Analysis (PCA) on a hyperspectral data matrix.

    PCA is computed using Singular Value Decomposition (SVD) of the
    mean-centered data matrix. The leading ``p - 1`` singular vectors are
    retained to define a low-dimensional signal subspace.
    
    Attributes
    ----------
    
    mean_: np.ndarray
        Mean vector of input data (L x 1).

    U_: np.ndarray
        PCA basis vectors (L x p).

    S_: np.ndarray
        Singular values corresponding to PCA components (p,).

    Parameters
    ----------
    p : int
        Dimension of the low-dimensional signal subspace.
    """
    def __init__(self, 
        p: int
    ):
        super().__init__()
        self.p = p
        self.mean_ = None
        self.U_ = None
        self.S_ = None

    def fit(self, Y: np.ndarray) -> "PCA":
        """
        Fit the affine projection model from data.

        This method computes:
        - Data mean
        - PCA basis via SVD

        Parameters
        ----------
        Y : np.ndarray of shape (n_bands, n_pixels)
            Input data matrix with n_bands features and n_pixels samples.

        Returns
        -------
        self : PCA
            Fitted estimator.
        """
        
        Y = np.asarray(Y)
        if Y.ndim != 2:
            raise ValueError("Y must be 2D with shape (n_bands x n_pixels)")
        n_bands, _ = Y.shape
        
        if not (1 <= self.p <= n_bands):
            raise ValueError(f"p must satisfy 1 <= p <= {n_bands}")

        self.mean_ = np.mean(Y, axis=1, keepdims=True)
        Yc = Y - self.mean_

        U, S, _ = np.linalg.svd(Yc, full_matrices=False)

        self.U_ = U[:, :self.p]
        self.S_ = S[:self.p]  
        self.fitted = True                            
        return self
        

    def transform(self, Y: np.ndarray) -> np.ndarray:
        """
        Project data into the learned affine subspace.

        Parameters
        ----------
        Y : np.ndarray of shape (L, N)
            Input data matrix.

        Returns
        -------
        np.ndarray of shape (p, N)
            Affine-projected (and optionally whitened) representation.
        """
        if not self.fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        Y = np.asarray(Y)
        if Y.ndim != 2:
            raise ValueError("Y must be 2D (L x N)")

        Yc = Y - self.mean_
        Y_proj = self.U_.T @ Yc
        return Y_proj


    def reconstruct(self, Y_proj: np.ndarray) -> np.ndarray:
        """
        Reconstruct data back into the original space.

        Parameters
        ----------
        Y_proj : np.ndarray of shape (p, N)
            Affine-subspace representation.

        Returns
        -------
        np.ndarray of shape (L, N)
            Reconstructed approximation in original space.
        """
        if not self.fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        Y_proj = np.asarray(Y_proj)
        Yc = self.U_ @ Y_proj
        return Yc + self.mean_
        

# -------------------------- Singular Value Decomposition ------------------------------
class SVD(Projection):
    """
    Compute a truncated Singular Value Decomposition (SVD) of a hyperspectral 
    data matrix. 
    
    This function computes a rank-``p`` approximation of the input matrix using
    the leading left singular vectors of Y. The decomposition is performed 
    directly on the data matrix without mean-centering.
    
    Attributes
    ----------

    U_: np.ndarray
        PCA basis vectors (L x p).
        
    S_: np.ndarray
        Singular values corresponding to PCA components (p,).

    Parameters
    ----------
    p : int
        Estimated number of endmembers. p dimensions are used for the SVD.
    """
    def __init__(self, 
        p: int
    ):
        super().__init__()
        self.p = p
        self.U_ = None
        self.S_ = None

    def fit(self, Y: np.ndarray) -> "SVD":
        """
        Fit the affine projection model from data.

        This method computes:
        - Data mean
        - PCA basis via SVD

        Parameters
        ----------
        Y : np.ndarray of shape (L, N)
            Input data matrix with L features and N samples.

        Returns
        -------
        self : SVD
            Fitted estimator.
        """
        
        Y = np.asarray(Y)
        if Y.ndim != 2:
            raise ValueError("Y must be 2D (L x N)")
        L, _ = Y.shape
        
        if not (1 <= self.p <= L):
            raise ValueError(f"p must satisfy 1 <= p <= {L}")

        U, S, _ = np.linalg.svd(Y, full_matrices=False)
        self.U_ = U[:, :self.p]
        self.S_ = S[:self.p]
        self.fitted = True                         
        return self
               
    def transform(self, Y: np.ndarray) -> np.ndarray:
        """
        Project data into the learned affine subspace.

        Parameters
        ----------
        Y : np.ndarray of shape (L, N)
            Input data matrix.

        Returns
        -------
        np.ndarray of shape (p, N)
            Affine-projected (and optionally whitened) representation.
        """
        if not self.fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        Y = np.asarray(Y)
        if Y.ndim != 2:
            raise ValueError("Y must be 2D (L x N)")
            
        Y_proj = self.U_.T @ Y
        return Y_proj
        

    def reconstruct(self, Y_proj: np.ndarray) -> np.ndarray:
        """
        Reconstruct data back into the original space.
        
        This is NOT full SVD reconstruction, but a projection back into the
        learned left-singular subspace.

        Parameters
        ----------
        Y_proj : np.ndarray of shape (p, N)
            Affine-subspace representation.

        Returns
        -------
        np.ndarray of shape (L, N)
            Reconstructed approximation in original space.
        """
        if not self.fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        Y_proj = np.asarray(Y_proj)
        Y = self.U_ @ Y_proj
        return Y
        

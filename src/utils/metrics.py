import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Dict
from scipy.optimize import linear_sum_assignment
from ..projection.projection import PCA


# ----------------------------- Min/Max normalization-----------------------------------
def minmax_normalize(
    M: np.ndarray,
    eps: float = 1e-12
) -> np.ndarray:
    """
    Apply column-wise min-max normalization.

    Each column is shifted so that its minimum value becomes zero and
    scaled so that its maximum value becomes one:

        M_norm[:, j] = (M[:, j] - min(M[:, j])) / max(M[:, j] - min(M[:, j]))
        
    The input array is not modified.

    Parameters
    ----------
    M : np.ndarray, shape (n_bands, n_endmembers)
        Input matrix whose columns represent endmember spectra.
    eps : float, default=1e-12
        Small constant added to the denominator to avoid division by zero
        for constant-valued columns.

    Returns
    -------
    np.ndarray, shape (n_bands, n_endmembers)
        Column-wise min-max normalized matrix with values in the interval
        [0, 1].
    """
    M_norm = M.astype(float, copy=True)
    M_norm -= M_norm.min(axis=0, keepdims=True)
    M_norm /= M_norm.max(axis=0, keepdims=True) + eps
    return M_norm


# ----------------------------- Endmember metrics --------------------------------------
class EndmemberMetrics:
    """
    Computes pairwise similarity and distance metrics between endmember spectra.

    This class provides vectorized implementations of common hyperspectral
    evaluation metrics used to compare estimated and ground-truth endmembers.

    Notes
    -----
    All methods operate on matrices of shape (n_bands, n_endmembers) where:
    - n_bands is the number of spectral bands 
    - n_endmembers is the number of endmembers.
    """

    @staticmethod
    def spectral_angle_distance(
        M_est: np.ndarray,
        M_gt: np.ndarray,
        eps: float = 1e-12,
    ) -> np.ndarray:
        """
        Compute Spectral Angle Distance (SAD) between two sets of spectra.

        SAD measures angular dissimilarity between spectral signatures and
        is invariant to scale.

        Parameters
        ----------
        M_est : np.ndarray, shape (n_bands, n_endmembers)
            Estimated endmembers.
        M_gt : np.ndarray, shape (n_bands, n_endmembers)
            Reference (true) endmembers.
        eps : float, optional
            Numerical stability constant.

        Returns
        -------
        np.ndarray, shape (n_endmembers, n_endmembers)
            Pairwise SAD matrix.
        """
        if M_est.shape != M_gt.shape:
            raise ValueError(f"Shape mismatch: {M_est.shape} vs {M_gt.shape}")
            
        A_n = M_est / (np.linalg.norm(M_est, axis=0, keepdims=True) + eps)
        B_n = M_gt / (np.linalg.norm(M_gt, axis=0, keepdims=True) + eps)

        cos_sim = np.clip(B_n.T @ A_n, -1.0 + eps, 1.0 - eps)
        return np.arccos(cos_sim)
        

    @staticmethod
    def mse(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Compute mean squared error between spectral signatures.

        Parameters
        ----------
        A : np.ndarray, shape (n_bands, n_endmembers)
        B : np.ndarray, shape (n_bands, n_endmembers)

        Returns
        -------
        np.ndarray, shape (n_endmembers, n_endmembers)
            Pairwise normalized squared Euclidean distances.
        """
        return (
            np.sum(A**2, axis=0)[None, :]
            + np.sum(B**2, axis=0)[:, None]
            - 2 * (B.T @ A)
            ) / A.shape[0]
            
    
    @staticmethod
    def standardized_cross_correlation(
        A: np.ndarray,
        B: np.ndarray,
        eps: float = 1e-12
    ) -> np.ndarray:
        """
        Compute standardized cross-correlation between spectra.

        Parameters
        ----------
        A : np.ndarray, shape (n_bands, n_endmembers)
        B : np.ndarray, shape (n_bands, n_endmembers)
        eps : float
            Stability constant for variance normalization.

        Returns
        -------
        np.ndarray, shape (n_endmembers, n_endmembers)
            Cross-correlation matrix.
        """
        A0 = A - A.mean(axis=0, keepdims=True)
        B0 = B - B.mean(axis=0, keepdims=True)
        A0 /= (A0.std(axis=0, keepdims=True) + eps)
        B0 /= (B0.std(axis=0, keepdims=True) + eps)
        # Population std (ddof=0) above, so divide by n: SCC lies in [-1, 1]
        return (B0.T @ A0) / A.shape[0]


class EndmemberMatcher:
    """
    Hungarian algorithm-based optimal assignment.

    This class solves an assignment problem that aligns estimated
    endmembers with ground-truth references by minimizing a cost matrix
    (e.g., SAD).

    Methods
    -------
    match(cost_matrix)
        Computes optimal permutation matrix
    """
    @staticmethod
    def match(cost_matrix: np.ndarray) -> np.ndarray:
        """
        Compute optimal assignment (permutation) matrix.

        Parameters
        ----------
        cost_matrix : np.ndarray, shape (n_endmembers, n_endmembers)
            Pairwise cost between true and estimated endmembers.

        Returns
        -------
        np.ndarray, shape (n_endmembers, n_endmembers)
            Permutation matrix P such that M_aligned = M @ P.
        """
        if cost_matrix.shape[0] != cost_matrix.shape[1]:
            raise ValueError("Cost matrix must be square")
            
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        K = cost_matrix.shape[0]
        P = np.zeros((K, K), dtype=int)
        P[col_ind, row_ind] = 1
        return P
        

class EndmemberAligner:
    """
    Align estimated endmembers with the ground truth.
    """
    def align(
        self,
        M_est: np.ndarray,
        M_gt: np.ndarray,
        alignment_metric: str = "SAD",
        normalize: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Align the estimated endmembers with the ground truth

        Parameters
        ----------
        M_est : np.ndarray, shape (n_bands, n_endmembers)
            Estimated endmembers.
        M_gt : np.ndarray, shape (n_bands, n_endmembers)
            Ground truth endmembers.
        alignment_metric: str (default: "SAD")
            Metric used for aligning the endmembers
            Possible choices include spectral angle distance ("SAD"),
            mean squared error ("MSE") or standardized cross-correlation ("SCC")
        normalize: bool (default: False)
            Whether or not to normalize the estimated and ground truth endmembers
            (column-wise min-max normalization)
            
        Returns
        -------
        M_aligned : np.ndarray, shape (n_bands, n_endmembers)
            Aligned estimated endmembers.
        P : np.ndarray, shape (n_endmembers, n_endmembers)
            Permutation matrix such that:
                M_aligned = M_est @ P 
        """
        if normalize:
            M_gt = minmax_normalize(M_gt)
            M_est = minmax_normalize(M_est)
            
        metric_map = {
            "SAD": EndmemberMetrics.spectral_angle_distance,
            "MSE": EndmemberMetrics.mse,
            "SCC": lambda a,b: 1 - EndmemberMetrics.standardized_cross_correlation(a,b),
        }

        try:
            cost_fn = metric_map[alignment_metric]
        except KeyError:
            raise ValueError(
                f"Unknown alignment metric '{alignment_metric}'. "
                f"Choose one of {list(metric_map)}."
            )

        cost = cost_fn(M_est, M_gt)   
        P = EndmemberMatcher.match(cost)
        return M_est @ P, P
        
              
class EndmemberEvaluation:
    """
    High-level pipeline for evaluating endmember estimation methods.

    This class combines:

    - optimal assignment
    - alignment of estimated endmembers
    - metrics computation

    Parameters
    ----------   
    M_gt : np.ndarray, shape (n_bands, n_endmembers)
        Ground-truth endmembers.
    normalize: bool (default: False)
        Whether or not to normalize the estimated and ground truth endmembers
        (column-wise min-max normalization)

    Attributes
    ----------
    M_gt : np.ndarray, shape (n_bands, n_endmembers)
        Ground-truth endmembers used as reference for alignment and
        evaluation. If ``normalize=True``, this matrix is stored after
        column-wise min-max normalization.

    normalize : bool
        Indicates whether column-wise min-max normalization is applied
        before evaluation.

    aligner : EndmemberAligner
        Object responsible for computing the optimal permutation that
        aligns estimated endmembers with the ground truth.

    P : np.ndarray or None, shape (n_endmembers, n_endmembers)
        Permutation matrix obtained during the last call to
        :meth:`evaluate`. The matrix satisfies:

            M_aligned = M_est @ P

        and can be reused to align abundance matrices:

            A_aligned = P.T @ A_est

        ``P`` is ``None`` until the first evaluation is performed. 
    """
    def __init__(self, 
        M_gt: np.ndarray, 
        normalize:bool = False,
    ):
        self.aligner = EndmemberAligner()
        if normalize:
            M_gt = minmax_normalize(M_gt)
        self.M_gt = M_gt
        self.normalize = normalize
        self.P = None
        
        
    def evaluate(self, 
        M_est: np.ndarray, 
        alignment_metric: str="SAD"
    ) -> Dict[str, np.ndarray]:
        """
        Evaluate the estimated endmembers against the ground truth.

        The estimated endmembers are first aligned with the ground truth
        using the specified alignment metric. After alignment, evaluation
        metrics are computed pairwise between corresponding endmembers.
    
        Parameters
        ----------
        M_est : np.ndarray, shape (n_bands, n_endmembers)
            Estimated endmembers.
        alignment_metric: str (default: "SAD")
            Metric used for aligning the endmembers
            Possible choices include spectral angle distance ("SAD"),
            mean squared error ("MSE") or standardized cross-correlation ("SCC")
            
        Returns
        -------
        metrics : Dict[str, np.ndarray]
            Dictionary containing the evaluation metrics for each aligned
            endmember pair.

            Keys include:
            - "SAD": Spectral angle distances
            - "MSE": Mean squared errors
            - "SCC": Standardized cross-correlations

            Each value is a 1D array of shape (n_endmembers,).
        """            
        if M_est.shape != self.M_gt.shape:
            raise ValueError(f"Shape mismatch: {M_est.shape} vs {self.M_gt.shape}")
        
        if(self.normalize):
            M_est = minmax_normalize(M_est)
            
        M_aligned, P = self.aligner.align(
            M_est, 
            self.M_gt,
            alignment_metric=alignment_metric,
            normalize = False,
        )
        self.P = P
        
        sad = np.diag(EndmemberMetrics.spectral_angle_distance(
            M_aligned, self.M_gt)
        )
        mse = np.diag(EndmemberMetrics.mse(
            M_aligned, self.M_gt)
        )
        scc = np.diag(EndmemberMetrics.standardized_cross_correlation(
            M_aligned, self.M_gt)
        )
        
        metrics = {
            'SAD': sad, 
            'MSE': mse, 
            'SCC': scc 
        }
        
        return metrics


# ---------------------------- Abundances metrics --------------------------------------
class AbundancesMetrics:
    """
    Computes evaluation metrics between estimated and ground-truth abundance
    matrices.

    All methods operate on abundance matrices of shape:

        ``(n_endmembers, n_pixels)``

    where:
    - ``n_endmembers`` is the number of material classes
    - ``n_pixels`` is the number of pixels

    Notes    M_gt,
    gt_kwargs=None,
    est_kwargs=None,
    -----
    Estimated abundances should be aligned with the ground truth before
    computing metrics:

        A_aligned = P.T @ A_est
    """
    @staticmethod
    def mse(
        A_est: np.ndarray,
        A_gt: np.ndarray,
    ) -> float:
        """
        Compute mean squared error.

        Parameters
        ----------
        A_est : np.ndarray
            Estimated abundance matrix.
        A_gt : np.ndarray
            Ground-truth abundance matrix.

        Returns
        -------
        float
            Mean squared error.
        """
        return np.mean((A_est - A_gt) ** 2)


    @staticmethod
    def rmse(
        A_est: np.ndarray,
        A_gt: np.ndarray,
    ) -> float:
        """
        Compute root mean squared error.

        Returns
        -------
        float
            Root mean squared error.
        """
        return np.sqrt(
            AbundancesMetrics.mse(A_est, A_gt)
        )


    @staticmethod
    def mae(
        A_est: np.ndarray,
        A_gt: np.ndarray,
    ) -> float:
        """
        Compute mean absolute error.

        Returns
        -------
        float
            Mean absolute error.
        """
        return np.mean(
            np.abs(A_est - A_gt)
        )


    @staticmethod
    def spectral_angle(
        A_est: np.ndarray,
        A_gt: np.ndarray,
        eps: float = 1e-12,
    ) -> float:
        """
        Compute abundance spectral angle mapper (SAM).

        The angle is computed independently for each pixel between
        abundance vectors.

        Returns
        -------
        float
            Mean spectral angle in radians.
        """

        dot = np.sum(
            A_est * A_gt,
            axis=0
        )

        norm = (
            np.linalg.norm(A_est, axis=0)
            *
            np.linalg.norm(A_gt, axis=0)
            + eps
        )

        cos_angle = np.clip(
            dot / norm,
            -1.0,
            1.0,
        )

        return np.mean(
            np.arccos(cos_angle)
        )


    @staticmethod
    def correlation(
        A_est: np.ndarray,
        A_gt: np.ndarray,
        eps: float = 1e-12,
    ) -> np.ndarray:
        """
        Compute abundance correlation coefficient per endmember.

        Returns
        -------
        np.ndarray, shape (n_endmembers,)
            Pearson correlation coefficient for each abundance map.
        """

        A0 = (
            A_est
            - A_est.mean(axis=1, keepdims=True)
        )

        B0 = (
            A_gt
            - A_gt.mean(axis=1, keepdims=True)
        )

        numerator = np.sum(
            A0 * B0,
            axis=1
        )

        denominator = (
            np.linalg.norm(A0, axis=1)
            *
            np.linalg.norm(B0, axis=1)
            + eps
        )

        return numerator / denominator

    
class AbundancesAligner:
    """
    Align abundance maps according to an endmember permutation matrix.

    The permutation matrix is obtained during endmember alignment.
    If endmembers are aligned as:

        M_aligned = M_est @ P

    then abundances are aligned as:

        A_aligned = P.T @ A_est
    """
    @staticmethod
    def align(
        A_est: np.ndarray,
        P: np.ndarray
    ) -> np.ndarray:
        """
        Apply endmember permutation to abundance matrix.

        Parameters
        ----------
        A_est : np.ndarray, shape (n_endmembers, n_pixels)
            Estimated abundance matrix.

        P : np.ndarray, shape (n_endmembers, n_endmembers)
            Permutation matrix obtained from endmember matching.

        Returns
        -------
        np.ndarray
            Aligned abundance matrix.

        Raises
        ------
        ValueError
            If dimensions are inconsistent.
        """

        if A_est.ndim != 2:
            raise ValueError(
                "A_est must be a 2D matrix (n_endmembers, n_pixels)"
            )
        K = A_est.shape[0]

        if P.shape != (K, K):
            raise ValueError(
                f"Permutation matrix shape {P.shape} incompatible "
                f"with abundance matrix {A_est.shape}"
            )
        return P.T @ A_est
        
        
class AbundancesEvaluation:
    """
    High-level pipeline for evaluating abundance estimation.

    Parameters
    ----------
    A_gt : np.ndarray, shape (K, N)
        Ground-truth abundance matrix.

    Attributes
    ----------
    A_gt : np.ndarray
        Ground-truth abundances.

    P : np.ndarray or None
        Permutation matrix used for abundance alignment.
    """

    def __init__(
        self,
        A_gt: np.ndarray,
    ):
        self.A_gt = A_gt


    def evaluate(
        self,
        A_est: np.ndarray,
        P: np.ndarray = None,
    ) -> Dict[str, np.ndarray]:
        """
        Evaluate estimated abundances.

        Parameters
        ----------
        A_est : np.ndarray
            Estimated abundance matrix.

        P : np.ndarray, optional
            Endmember permutation matrix obtained from
            EndmemberAligner.

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary containing abundance metrics.
        """

        if A_est.shape != self.A_gt.shape:
            raise ValueError(
                f"Shape mismatch: {A_est.shape} vs {self.A_gt.shape}"
            )

        if P is not None:
            A_est = P.T @ A_est


        metrics = {
            "MSE": AbundancesMetrics.mse(
                A_est,
                self.A_gt,
            ),

            "RMSE": AbundancesMetrics.rmse(
                A_est,
                self.A_gt,
            ),

            "MAE": AbundancesMetrics.mae(
                A_est,
                self.A_gt,
            ),

            "SAM": AbundancesMetrics.spectral_angle(
                A_est,
                self.A_gt,
            ),

            "SCC": AbundancesMetrics.correlation(
                A_est,
                self.A_gt,
            ),
        }

        return metrics


# ---------------------------- Comparison tables ----------------------------------------
class ComparisonTable:
    """
    Build and style comparison tables from per-algorithm evaluation metrics.

    This class turns a dictionary of per-algorithm evaluation metrics, as
    returned by :class:`EndmemberEvaluation` or :class:`AbundancesEvaluation`,
    into a :class:`pandas.DataFrame` suitable for display or export, and
    provides helpers to render it for notebook display or export it to
    LaTeX, bolding the best-performing algorithm for each column.
    """

    @staticmethod
    def build(
        metrics: Dict[str, Dict[str, np.ndarray]],
        metric: str,
        index: Optional[list] = None,
        add_mean: bool = True,
    ) -> pd.DataFrame:
        """
        Assemble a comparison table for a single metric across algorithms.

        Parameters
        ----------
        metrics : Dict[str, Dict[str, np.ndarray]]
            Dictionary mapping algorithm names to the evaluation metrics
            dictionary returned by :class:`EndmemberEvaluation.evaluate` or
            :class:`AbundancesEvaluation.evaluate`.
            
        metric : str
            Key of the metric to extract from each algorithm's metrics
            dictionary (e.g. "SAD", "MSE", "SCC").
            
        index : list, optional
            Labels for the metric values (e.g. endmember names). If
            ``None``, default integer labels are used.
            
        add_mean : bool, default=True
            Whether to append a "Mean" column averaging the metric across
            ``index``.

        Returns
        -------
        pd.DataFrame, shape (n_algorithms, len(index) [+ 1])
            Table with one row per algorithm and one column per entry of
            ``index``, plus an optional "Mean" column.
        """
        
        data = {name: np.atleast_1d(m[metric]) for name, m in metrics.items()}
        table = pd.DataFrame(data, index=index).T

        if add_mean:
            table["Mean"] = table.mean(axis=1)

        return table
        

    @staticmethod
    def style(
        table: pd.DataFrame,
        precision: int = 4,
        best: str = "min",
        subset: Optional[list] = None,
    ):
        """
        Apply styling to a comparison table.

        Formats values and renders the best-performing algorithm in bold
        for each column.

        Parameters
        ----------
        table : pd.DataFrame
            Comparison table, as returned by :meth:`build`.
        precision : int, default=4
            Number of decimal places used to format values.
        best : str, default="min"
            Whether lower ("min") or higher ("max") values are better.
        subset : list, optional
            Columns to bold the best value of. If ``None``, all columns of
            ``table`` are used.

        Returns
        -------
        pandas.io.formats.style.Styler
            Styled table, ready to be passed to ``display()`` in a notebook.
        """
        
        subset = subset if subset is not None else table.columns.tolist()

        if best not in ("min", "max"):
            raise ValueError(f"Unknown value for `best`: {best!r}."
                f" Choose 'min' or 'max'.")

        def _bold_best(column: pd.Series) -> list[str]:
            target = column.min() if best == "min" else column.max()
            return ["font-weight: bold" if v == target else "" for v in column]

        styled = (
            table.style
                .format(f"{{:.{precision}f}}")
                .apply(_bold_best, axis=0, subset=subset)
        )

        return styled
        

    @staticmethod
    def to_latex(
        table: pd.DataFrame,
        precision: int = 4,
        best: str = "min",
        subset: Optional[list] = None,
        caption: Optional[str] = None,
        label: Optional[str] = None,
    ) -> str:
        """
        Export a comparison table to LaTeX.

        The best-performing algorithm for each column is rendered in bold,
        matching the styling applied by :meth:`style`.

        Parameters
        ----------
        table : pd.DataFrame
            Comparison table, as returned by :meth:`build`.
        precision : int, default=4
            Number of decimal places used to format values.
        best : str, default="min"
            Whether lower ("min") or higher ("max") values are better.
        subset : list, optional
            Columns to bold the best value of. If ``None``, all columns of
            ``table`` are used.
        caption : str, optional
            Table caption.
        label : str, optional
            Label used for cross-referencing (e.g. ``"tab:sad"``).

        Returns
        -------
        str
            LaTeX source for the table, using booktabs-style rules.
            Requires the ``booktabs`` package in the LaTeX preamble.
        """
        
        styled = ComparisonTable.style(
            table,
            precision=precision,
            best=best,
            subset=subset,
        )

        return styled.to_latex(
            caption=caption,
            label=label,
            hrules=True,
            convert_css=True,
        )



# ---------------------------- Unmixing evaluation --------------------------------------
class UnmixingEvaluation:
    """
    Evaluate and compare several unmixing estimates, possibly over several
    random seeds, against the same ground truth.

    Each estimate is added under a name (e.g. ``"VCA (init)"``, ``"ICE"``)
    and, optionally, a seed. Every (name, seed) run is aligned with the
    ground truth once, when it is added: its endmembers are matched to the
    reference endmembers by the Hungarian algorithm on the spectral angle
    distance (SAD), and the same permutation is applied to its abundances.
    All metrics are then computed and stored, so that tables and figures
    are only views of the stored results.

    Endmember metrics, per endmember:

    - ``"SAD"``: spectral angle distance (radians), on the raw spectra;
    - ``"SCC"``: standardized cross-correlation (higher is better);
    - ``"MSE"``: mean squared error on the raw spectra. It assumes that the
      estimates are at the scale of the reference endmembers, which is not
      the case for some real datasets (e.g. Samson, Urban);
    - ``"MSE (normalized)"``: mean squared error after a column-wise
      min-max normalization of the estimated and reference spectra.

    Abundance metrics (only when ground-truth abundances are given), over
    the whole image: ``"MSE"``, ``"RMSE"``, ``"MAE"``, ``"SAM"`` (mean
    per-pixel spectral angle between abundance vectors, radians) and
    ``"SCC"`` (Pearson correlation of the abundance maps, averaged over the
    endmembers; higher is better).

    When an estimate has several seeds, the tables aggregate them in two
    steps: the per-endmember metrics of each seed, and their mean over the
    endmembers, are first computed for each run; each column is then
    summarized over the seeds (median by default). The "Mean" column is
    therefore the median over the seeds of the per-seed mean, as in the
    usual benchmarking protocol, and not the mean of the medians.

    Parameters
    ----------
    M_gt : np.ndarray, shape (n_bands, n_endmembers)
        Ground-truth endmembers.

    A_gt : np.ndarray, shape (n_endmembers, n_pixels), optional
        Ground-truth abundances. If ``None``, abundance metrics and plots
        are not available.

    names : list[str], optional
        Names of the endmembers, used as table columns and plot titles.
        If ``None``, ``"EM 1"``, ``"EM 2"``, ... are used.

    Attributes
    ----------
    M_gt : np.ndarray, shape (n_bands, n_endmembers)
        Ground-truth endmembers.

    A_gt : np.ndarray or None, shape (n_endmembers, n_pixels)
        Ground-truth abundances.

    names : list[str]
        Names of the endmembers.

    endmember_results : pd.DataFrame
        Long table of the endmember metrics, with columns ``estimate``,
        ``seed``, ``endmember``, ``metric`` and ``value``.

    abundance_results : pd.DataFrame
        Long table of the abundance metrics, with columns ``estimate``,
        ``seed``, ``metric`` and ``value``.

    Examples
    --------
    >>> report = UnmixingEvaluation(HSI.endmembers, HSI.abundances, names=HSI.names)
    >>> for seed in range(5):
    ...     model = ICE(n_endmembers=3, init="vca", rng=np.random.default_rng(seed)).fit(Y)
    ...     report.add_model("ICE", model, scale=scale_factor, seed=seed)
    >>> report.display()
    >>> report.plot_endmembers(HSI.wavelengths)
    """

    ENDMEMBER_METRICS = ("SAD", "SCC", "MSE", "MSE (normalized)")
    ABUNDANCE_METRICS = ("MSE", "RMSE", "MAE", "SAM", "SCC")

    # Metrics for which higher values are better
    _HIGHER_IS_BETTER = {"SCC"}

    _INIT_NAMES = {"vca": "VCA (init)", "random": "Random (init)", "custom": "Custom (init)"}

    def __init__(
        self,
        M_gt: np.ndarray,
        A_gt: Optional[np.ndarray] = None,
        names: Optional[list] = None,
    ) -> None:

        M_gt = np.asarray(M_gt, dtype=float)
        if M_gt.ndim != 2:
            raise ValueError("M_gt must be a 2D matrix (n_bands, n_endmembers).")

        n_endmembers = M_gt.shape[1]

        if A_gt is not None:
            A_gt = np.asarray(A_gt, dtype=float)
            if A_gt.ndim != 2 or A_gt.shape[0] != n_endmembers:
                raise ValueError(
                    f"A_gt must have shape (n_endmembers={n_endmembers}, n_pixels), "
                    f"got {A_gt.shape}."
                )

        if names is None:
            names = [f"EM {k + 1}" for k in range(n_endmembers)]
        names = [str(name) for name in names]
        if len(names) != n_endmembers:
            raise ValueError(
                f"Expected {n_endmembers} endmember names, got {len(names)}."
            )

        self.M_gt = M_gt
        self.A_gt = A_gt
        self.names = names

        self._M_gt_normalized = minmax_normalize(M_gt)
        self._aligner = EndmemberAligner()

        # {(estimate, seed): {"M": aligned endmembers, "A": aligned abundances or None, "P": permutation}}
        self._runs: Dict[tuple, dict] = {}
        self._endmember_records: list = []
        self._abundance_records: list = []


    # ------------------------------------------------------------------ adding runs
    def add(
        self,
        name: str,
        M: np.ndarray,
        A: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
    ) -> "UnmixingEvaluation":
        """
        Align one estimate with the ground truth and compute its metrics.

        Parameters
        ----------
        name : str
            Name of the estimate (row label in the tables).

        M : np.ndarray, shape (n_bands, n_endmembers)
            Estimated endmembers, at the scale of the observations used to
            build ``M_gt`` (e.g. multiplied back by the normalization factor
            of the data).

        A : np.ndarray, shape (n_endmembers, n_pixels), optional
            Estimated abundances. Abundance metrics are computed only if
            both ``A`` and the ground-truth abundances are available.

        seed : int, optional
            Seed of the run. An estimate is either evaluated on a single
            run (``seed=None``) or on several seeds, not both.

        Returns
        -------
        self : UnmixingEvaluation
            The evaluation itself, so that calls can be chained.

        Raises
        ------
        ValueError
            If the (name, seed) run was already added, if seeded and
            unseeded runs are mixed for the same estimate, or if ``M`` or
            ``A`` does not have the expected shape (for instance when an
            algorithm pruned some endmembers).
        """
        key = (name, seed)
        if key in self._runs:
            raise ValueError(f"The run {self._run_label(name, seed)} was already added.")

        existing_seeds = self.seeds(name)
        if existing_seeds and ((seed is None) != (existing_seeds[0] is None)):
            raise ValueError(
                f"'{name}' mixes seeded and unseeded runs: either give a seed "
                "to every run of an estimate, or to none."
            )

        M = np.asarray(M, dtype=float)
        if M.shape != self.M_gt.shape:
            raise ValueError(
                f"Estimated endmembers of {self._run_label(name, seed)} have shape "
                f"{M.shape}, expected {self.M_gt.shape} (same number of endmembers "
                "as the ground truth)."
            )

        if A is not None:
            A = np.asarray(A, dtype=float)
            if A.ndim != 2 or A.shape[0] != self.M_gt.shape[1]:
                raise ValueError(
                    f"Estimated abundances of {self._run_label(name, seed)} have shape "
                    f"{A.shape}, expected ({self.M_gt.shape[1]}, n_pixels)."
                )
            if self.A_gt is not None and A.shape != self.A_gt.shape:
                raise ValueError(
                    f"Estimated abundances of {self._run_label(name, seed)} have shape "
                    f"{A.shape}, expected {self.A_gt.shape}."
                )

        M_aligned, P = self._aligner.align(M, self.M_gt, alignment_metric="SAD")
        A_aligned = AbundancesAligner.align(A, P) if A is not None else None

        endmember_values = {
            "SAD": np.diag(EndmemberMetrics.spectral_angle_distance(M_aligned, self.M_gt)),
            "SCC": np.diag(EndmemberMetrics.standardized_cross_correlation(M_aligned, self.M_gt)),
            "MSE": np.diag(EndmemberMetrics.mse(M_aligned, self.M_gt)),
            "MSE (normalized)": np.diag(
                EndmemberMetrics.mse(minmax_normalize(M_aligned), self._M_gt_normalized)
            ),
        }
        for metric, values in endmember_values.items():
            for endmember, value in zip(self.names, values):
                self._endmember_records.append({
                    "estimate": name, "seed": seed, "endmember": endmember,
                    "metric": metric, "value": float(value),
                })

        if A_aligned is not None and self.A_gt is not None:
            abundance_values = {
                "MSE": AbundancesMetrics.mse(A_aligned, self.A_gt),
                "RMSE": AbundancesMetrics.rmse(A_aligned, self.A_gt),
                "MAE": AbundancesMetrics.mae(A_aligned, self.A_gt),
                "SAM": AbundancesMetrics.spectral_angle(A_aligned, self.A_gt),
                "SCC": np.mean(AbundancesMetrics.correlation(A_aligned, self.A_gt)),
            }
            for metric, value in abundance_values.items():
                self._abundance_records.append({
                    "estimate": name, "seed": seed, "metric": metric, "value": float(value),
                })

        self._runs[key] = {"M": M_aligned, "A": A_aligned, "P": P}
        return self


    def add_model(
        self,
        name: str,
        model,
        scale: float = 1.0,
        seed: Optional[int] = None,
        init_name: Optional[str] = "auto",
    ) -> "UnmixingEvaluation":
        """
        Add a fitted model and, optionally, its initialization.

        The estimate is read from ``model.endmembers_`` and, when present,
        ``model.abundances_``. If the model stores its initialization
        (``endmembers_init_``, and optionally ``abundances_init_``, as the
        NMF algorithms do), it is added as a second estimate.

        Parameters
        ----------
        name : str
            Name of the estimate.

        model : object
            Fitted model exposing ``endmembers_`` and optionally
            ``abundances_``, ``endmembers_init_`` and ``abundances_init_``.

        scale : float, default=1.0
            Factor applied to the endmembers (estimate and initialization)
            to bring them back to the scale of the ground truth, e.g. the
            factor returned by ``HyperspectralData.normalized_observations``.

        seed : int, optional
            Seed of the run.

        init_name : str or None, default="auto"
            Name under which the initialization is added. ``"auto"`` uses
            ``"VCA (init)"``, ``"Random (init)"`` or ``"Custom (init)"``
            (endmembers passed to ``fit``) according to ``model.init_``; ``None`` does not add the initialization. When
            several models share the same initialization (same name and
            seed), it is added only once.

        Returns
        -------
        self : UnmixingEvaluation
            The evaluation itself, so that calls can be chained.

        Raises
        ------
        ValueError
            If a different initialization was already added under the same
            name and seed.
        """
        M_init = getattr(model, "endmembers_init_", None)

        if init_name is not None and M_init is not None:
            if init_name == "auto":
                init = getattr(model, "init_", getattr(model, "init", None))
                init_name = self._INIT_NAMES.get(init, f"{init} (init)")

            M_init = scale * np.asarray(M_init, dtype=float)
            A_init = getattr(model, "abundances_init_", None)

            if (init_name, seed) in self._runs:
                # Same initialization shared by several models: keep the first one
                previous = self._runs[(init_name, seed)]["M"]
                M_aligned, _ = self._aligner.align(M_init, self.M_gt, alignment_metric="SAD")
                if not np.allclose(previous, M_aligned):
                    raise ValueError(
                        f"A different initialization was already added as "
                        f"{self._run_label(init_name, seed)}: pass another init_name."
                    )
            else:
                self.add(init_name, M_init, A_init, seed=seed)

        A = getattr(model, "abundances_", None)
        return self.add(name, scale * np.asarray(model.endmembers_, dtype=float), A, seed=seed)


    # ------------------------------------------------------------------ results
    @property
    def endmember_results(self) -> pd.DataFrame:
        return pd.DataFrame(
            self._endmember_records,
            columns=["estimate", "seed", "endmember", "metric", "value"],
        )

    @property
    def abundance_results(self) -> pd.DataFrame:
        return pd.DataFrame(
            self._abundance_records,
            columns=["estimate", "seed", "metric", "value"],
        )

    @property
    def estimates(self) -> list:
        """Names of the estimates, in the order in which they were added."""
        return list(dict.fromkeys(name for name, _ in self._runs))

    def seeds(self, name: str) -> list:
        """Seeds of the runs of an estimate, sorted (``[None]`` for a single unseeded run)."""
        seeds = [seed for estimate, seed in self._runs if estimate == name]
        return sorted(seeds, key=lambda seed: (seed is None, seed if seed is not None else 0))

    def aligned(self, name: str, seed: Optional[int] = None) -> tuple:
        """
        Return the aligned endmembers and abundances of one run.

        Parameters
        ----------
        name : str
            Name of the estimate.
        seed : int, optional
            Seed of the run.

        Returns
        -------
        M : np.ndarray, shape (n_bands, n_endmembers)
            Endmembers, with column ``k`` matched to ground-truth endmember ``k``.
        A : np.ndarray or None, shape (n_endmembers, n_pixels)
            Abundances aligned with the same permutation, if available.
        """
        run = self._get_run(name, seed)
        return run["M"], run["A"]


    def run_values(
        self,
        metric: str = "SAD",
        target: str = "endmembers",
    ) -> Dict[str, Dict[Optional[int], float]]:
        """
        Value of a metric for every run: the mean over the endmembers for an
        endmember metric, the image-level value for an abundance metric.

        Parameters
        ----------
        metric : str, default="SAD"
            Name of the metric.
        target : {"endmembers", "abundances"}, default="endmembers"
            Whether ``metric`` is an endmember or an abundance metric.

        Returns
        -------
        Dict[str, Dict[Optional[int], float]]
            ``{estimate: {seed: value}}``, in the order of addition.
        """
        self._check_metric(metric, target)
        records = (
            self._endmember_records if target == "endmembers"
            else self._abundance_records
        )

        values: Dict[str, Dict[Optional[int], list]] = {}
        for record in records:
            if record["metric"] == metric:
                values.setdefault(record["estimate"], {}).setdefault(record["seed"], []).append(record["value"])

        return {
            name: {seed: float(np.mean(v)) for seed, v in sorted(
                runs.items(), key=lambda item: (item[0] is None, item[0] if item[0] is not None else 0)
            )}
            for name, runs in values.items()
        }


    def table(
        self,
        metric: str = "SAD",
        aggregate: str = "median",
        add_mean: bool = True,
    ) -> pd.DataFrame:
        """
        Table of an endmember metric: one row per estimate, one column per
        endmember, plus an optional "Mean" column.

        Parameters
        ----------
        metric : str, default="SAD"
            One of :attr:`ENDMEMBER_METRICS`.
        aggregate : str, default="median"
            Statistic used to summarize the seeds of each estimate
            (``"median"``, ``"mean"``, ``"std"``, ``"min"``, ``"max"``).
            Estimates with a single run are not affected (except by
            ``"std"``, which gives NaN).
        add_mean : bool, default=True
            Whether to add a "Mean" column: the statistic over the seeds of
            the per-seed mean over the endmembers.

        Returns
        -------
        pd.DataFrame
            Table indexed by the estimate names.
        """
        self._check_metric(metric, "endmembers")
        results = self.endmember_results
        results = results[results["metric"] == metric]

        rows = {}
        for name in self.estimates:
            runs = results[results["estimate"] == name]
            per_seed = runs.pivot_table(
                index=runs["seed"].map(repr), columns="endmember", values="value", sort=False,
            )[self.names]
            if add_mean:
                per_seed["Mean"] = per_seed.mean(axis=1)
            rows[name] = per_seed.agg(aggregate)

        table = pd.DataFrame(rows).T
        table.columns.name = None
        return table


    def abundance_table(self, aggregate: str = "median") -> pd.DataFrame:
        """
        Table of the abundance metrics: one row per estimate with
        abundances, one column per metric.

        Parameters
        ----------
        aggregate : str, default="median"
            Statistic used to summarize the seeds of each estimate.

        Returns
        -------
        pd.DataFrame
            Table indexed by the estimate names, with the columns of
            :attr:`ABUNDANCE_METRICS`. Empty if no abundances were evaluated.
        """
        results = self.abundance_results
        rows = {}
        for name in self.estimates:
            runs = results[results["estimate"] == name]
            if runs.empty:
                continue
            per_seed = runs.pivot_table(
                index=runs["seed"].map(repr), columns="metric", values="value", sort=False,
            )[list(self.ABUNDANCE_METRICS)]
            rows[name] = per_seed.agg(aggregate)

        return pd.DataFrame(rows, index=list(self.ABUNDANCE_METRICS)).T


    def per_seed(
        self,
        metric: str = "SAD",
        target: str = "endmembers",
    ) -> pd.DataFrame:
        """
        Value of a metric for every seed: one row per seed, one column per
        estimate (mean over the endmembers for an endmember metric).

        Parameters
        ----------
        metric : str, default="SAD"
            Name of the metric.
        target : {"endmembers", "abundances"}, default="endmembers"
            Whether ``metric`` is an endmember or an abundance metric.

        Returns
        -------
        pd.DataFrame
            Table indexed by the seeds. Estimates evaluated on a single
            unseeded run appear in a row labelled ``None``.
        """
        values = self.run_values(metric, target)
        seeds = sorted(
            {seed for runs in values.values() for seed in runs},
            key=lambda seed: (seed is None, seed if seed is not None else 0),
        )
        # Built by hand: pandas would turn a None seed into NaN and the seeds into floats
        return pd.DataFrame(
            {name: [runs.get(seed, np.nan) for seed in seeds] for name, runs in values.items()},
            index=pd.Index(seeds, dtype=object, name="seed"),
        )


    def summary(
        self,
        metric: str = "SAD",
        target: str = "endmembers",
    ) -> pd.DataFrame:
        """
        Mean, standard deviation and median over the seeds of a metric
        (mean over the endmembers for an endmember metric), per estimate.

        Parameters
        ----------
        metric : str, default="SAD"
            Name of the metric.
        target : {"endmembers", "abundances"}, default="endmembers"
            Whether ``metric`` is an endmember or an abundance metric.

        Returns
        -------
        pd.DataFrame
            Table indexed by the estimate names, with columns ``mean``,
            ``std``, ``median`` and ``n_seeds``.
        """
        values = self.run_values(metric, target)
        rows = {}
        for name, runs in values.items():
            v = np.array(list(runs.values()))
            rows[name] = {
                "mean": v.mean(),
                "std": v.std(ddof=1) if len(v) > 1 else np.nan,
                "median": np.median(v),
                "n_seeds": len(v),
            }
        return pd.DataFrame(rows).T.astype({"n_seeds": int})


    def median_seed(
        self,
        name: str,
        metric: str = "SAD",
    ) -> Optional[int]:
        """
        Seed of the run that represents the median behaviour of an estimate:
        the run whose mean metric over the endmembers is closest to the
        median over the seeds (the smallest seed in case of a tie).

        Parameters
        ----------
        name : str
            Name of the estimate.
        metric : str, default="SAD"
            Endmember metric used to rank the runs.

        Returns
        -------
        int or None
            The median seed (``None`` for an estimate with a single unseeded run).
        """
        values = self.run_values(metric, "endmembers")
        if name not in values:
            raise KeyError(f"Unknown estimate '{name}'. Known estimates: {self.estimates}.")

        runs = values[name]
        median = np.median(list(runs.values()))
        best_seed, best_gap = None, np.inf
        for seed, value in runs.items():  # sorted by seed, so ties keep the smallest one
            gap = abs(value - median)
            if gap < best_gap:
                best_seed, best_gap = seed, gap
        return best_seed


    # ------------------------------------------------------------------ display
    def display(
        self,
        aggregate: str = "median",
        precision: int = 4,
    ) -> None:
        """
        Display the endmember tables (SAD, normalized MSE, MSE, SCC), the
        abundance table and, when an estimate has several seeds, the mean
        SAD of every seed and its summary over the seeds.

        Parameters
        ----------
        aggregate : str, default="median"
            Statistic used to summarize the seeds of each estimate.
        precision : int, default=4
            Number of decimal places.
        """
        show = _display_function()
        multi_seed = any(len(self.seeds(name)) > 1 for name in self.estimates)
        over_seeds = f" - {aggregate} over seeds" if multi_seed else ""
        columns = self.names + ["Mean"]

        titles = {
            "SAD": "Spectral Angle Distance (SAD)",
            "MSE (normalized)": "Mean Squared Error (MSE) on min-max normalized spectra",
            "MSE": "Mean Squared Error (MSE) on raw spectra",
            "SCC": "Standardized cross-correlation (SCC)",
        }
        for metric, title in titles.items():
            best = "max" if metric in self._HIGHER_IS_BETTER else "min"
            print(f"{title}{over_seeds} - {'higher' if best == 'max' else 'lower'} is better")
            show(ComparisonTable.style(
                self.table(metric, aggregate), precision=precision, best=best, subset=columns,
            ))

        abundances = self.abundance_table(aggregate)
        if not abundances.empty:
            print(f"Abundance errors{over_seeds} - lower is better (SCC: higher is better)")
            show(_style_mixed(abundances, precision, self._HIGHER_IS_BETTER))

        if multi_seed:
            print("Mean SAD of every seed - lower is better")
            show(self.per_seed("SAD").style.format(precision=precision, na_rep="-"))
            print("Mean SAD over the seeds")
            show(self.summary("SAD").style.format(precision=precision, na_rep="-"))


    def plot_endmembers(
        self,
        wavelengths: np.ndarray,
        seed: Optional[int] = None,
        band: Optional[str] = "iqr",
        normalize: bool = True,
        estimates: Optional[list] = None,
        title: Optional[str] = None,
        gt_kwargs: Optional[dict] = None,
        est_kwargs: Optional[dict] = None,
        band_kwargs: Optional[dict] = None,
        figsize: Optional[tuple] = None,
    ):
        """
        Plot the estimated endmembers against the ground truth, one row per
        estimate.

        Each row shows a real run: the given ``seed``, or by default the
        median seed of the estimate (see :meth:`median_seed`), named in the
        row label. For estimates with several seeds, a shaded band shows
        the spread of the aligned spectra over the seeds.

        Parameters
        ----------
        wavelengths : np.ndarray, shape (n_bands,)
            Wavelengths of the spectral bands.
        seed : int, optional
            Seed to plot for every estimate. If ``None``, each estimate
            uses its median seed.
        band : {"iqr", "std", "minmax"} or None, default="iqr"
            Spread shown around the spectra of estimates with several
            seeds: interquartile range, mean ± one standard deviation, or
            minimum-maximum envelope. ``None`` draws no band.
        normalize : bool, default=True
            Whether to apply a column-wise min-max normalization to every
            spectrum (each seed separately) before plotting and computing
            the band.
        estimates : list[str], optional
            Estimates to plot, in this order. All of them by default.
        title : str, optional
            Figure title.
        gt_kwargs, est_kwargs, band_kwargs : dict, optional
            Keyword arguments forwarded to
            :meth:`~utils.plotting.EndmemberPlotter.plot_aligned`.
        figsize : tuple, optional
            Figure size in inches.

        Returns
        -------
        fig : matplotlib.figure.Figure
        axes : ndarray of matplotlib.axes.Axes, shape (n_estimates, n_endmembers)
        """
        from .plotting import EndmemberPlotter

        if band not in (None, "iqr", "std", "minmax"):
            raise ValueError(f"Unknown band {band!r}. Choose 'iqr', 'std', 'minmax' or None.")

        prepare = minmax_normalize if normalize else (lambda M: M)
        spectra, bands = {}, {}

        for name in self._select_estimates(estimates):
            run_seed = self._resolve_seed(name, seed)
            label = self._run_label(name, run_seed)
            spectra[label] = prepare(self._get_run(name, run_seed)["M"])

            seeds = self.seeds(name)
            if seed is None and band is not None and len(seeds) > 1:
                stack = np.stack([prepare(self._runs[(name, s)]["M"]) for s in seeds])
                if band == "iqr":
                    bands[label] = (np.percentile(stack, 25, axis=0), np.percentile(stack, 75, axis=0))
                elif band == "std":
                    mean, std = stack.mean(axis=0), stack.std(axis=0, ddof=1)
                    bands[label] = (mean - std, mean + std)
                else:
                    bands[label] = (stack.min(axis=0), stack.max(axis=0))

        plotter = EndmemberPlotter(np.asarray(wavelengths))
        return plotter.plot_aligned(
            spectra,
            prepare(self.M_gt),
            bands=bands,
            names=self.names,
            gt_kwargs={"color": "red", **(gt_kwargs or {})},
            est_kwargs={"color": "orange", **(est_kwargs or {})},
            band_kwargs=band_kwargs,
            title=title,
            figsize=figsize,
        )


    def plot_abundances(
        self,
        image_shape: tuple,
        seed: Optional[int] = None,
        estimates: Optional[list] = None,
        title: Optional[str] = None,
        cmap: str = "viridis",
        figsize: Optional[tuple] = None,
    ):
        """
        Plot the ground-truth abundance maps and the aligned abundance maps
        of every estimate that has abundances, one row per estimate.

        Each row shows a real run: the given ``seed``, or by default the
        median seed of the estimate, named in the row label.

        Parameters
        ----------
        image_shape : tuple[int, int]
            Spatial dimensions ``(height, width)`` of the image.
        seed : int, optional
            Seed to plot for every estimate. If ``None``, each estimate
            uses its median seed.
        estimates : list[str], optional
            Estimates to plot, in this order. All the estimates with
            abundances by default.
        title : str, optional
            Figure title.
        cmap : str, default="viridis"
            Colormap of the maps.
        figsize : tuple, optional
            Figure size in inches.

        Returns
        -------
        fig : matplotlib.figure.Figure
        axes : ndarray of matplotlib.axes.Axes, shape (n_estimates + 1, n_endmembers)
        """
        from .plotting import AbundancesMapPlotter

        if self.A_gt is None:
            raise ValueError("No ground-truth abundances were given.")

        maps = {}
        for name in self._select_estimates(estimates, with_abundances=True):
            run_seed = self._resolve_seed(name, seed)
            maps[self._run_label(name, run_seed)] = self._get_run(name, run_seed)["A"]

        if not maps:
            raise ValueError("No estimate with abundances to plot.")

        return AbundancesMapPlotter(tuple(image_shape)).plot_comparison(
            maps, self.A_gt, names=self.names, cmap=cmap, title=title, figsize=figsize,
        )


    def plot_abundance_spread(
        self,
        image_shape: tuple,
        statistic: str = "iqr",
        estimates: Optional[list] = None,
        title: Optional[str] = None,
        cmap: str = "magma",
        figsize: Optional[tuple] = None,
    ):
        """
        Plot, for every estimate with abundances and several seeds, the
        per-pixel spread of its aligned abundances over the seeds. It shows
        where the estimate changes from one seed to another.

        Parameters
        ----------
        image_shape : tuple[int, int]
            Spatial dimensions ``(height, width)`` of the image.
        statistic : {"iqr", "std"}, default="iqr"
            Interquartile range or standard deviation over the seeds.
        estimates : list[str], optional
            Estimates to plot. All the estimates with abundances and
            several seeds by default.
        title : str, optional
            Figure title.
        cmap : str, default="magma"
            Colormap of the maps.
        figsize : tuple, optional
            Figure size in inches.

        Returns
        -------
        fig : matplotlib.figure.Figure
        axes : ndarray of matplotlib.axes.Axes, shape (n_estimates, n_endmembers)
        """
        from .plotting import AbundancesMapPlotter

        if statistic not in ("iqr", "std"):
            raise ValueError(f"Unknown statistic {statistic!r}. Choose 'iqr' or 'std'.")

        maps = {}
        for name in self._select_estimates(estimates, with_abundances=True):
            seeds = self.seeds(name)
            if len(seeds) < 2:
                continue
            stack = np.stack([self._runs[(name, s)]["A"] for s in seeds])
            if statistic == "iqr":
                spread = np.percentile(stack, 75, axis=0) - np.percentile(stack, 25, axis=0)
            else:
                spread = stack.std(axis=0, ddof=1)
            maps[f"{name} ({len(seeds)} seeds)"] = spread

        if not maps:
            raise ValueError("No estimate with abundances and several seeds to plot.")

        label = "Interquartile range" if statistic == "iqr" else "Standard deviation"
        return AbundancesMapPlotter(tuple(image_shape)).plot_maps(
            maps, names=self.names, cmap=cmap, figsize=figsize,
            title=title or f"Abundance spread over seeds ({label.lower()})",
            colorbar_label=label,
        )


    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _run_label(name: str, seed: Optional[int]) -> str:
        return name if seed is None else f"{name} (seed {seed})"

    def _get_run(self, name: str, seed: Optional[int]) -> dict:
        try:
            return self._runs[(name, seed)]
        except KeyError:
            raise KeyError(
                f"Unknown run {self._run_label(name, seed)}. "
                f"Seeds of '{name}': {self.seeds(name)}."
            ) from None

    def _resolve_seed(self, name: str, seed: Optional[int]) -> Optional[int]:
        seeds = self.seeds(name)
        if seeds == [None]:
            return None
        return self.median_seed(name) if seed is None else seed

    def _select_estimates(self, estimates: Optional[list], with_abundances: bool = False) -> list:
        names = self.estimates if estimates is None else list(estimates)
        for name in names:
            if name not in self.estimates:
                raise KeyError(f"Unknown estimate '{name}'. Known estimates: {self.estimates}.")
        if with_abundances:
            names = [
                name for name in names
                if all(self._runs[(name, s)]["A"] is not None for s in self.seeds(name))
            ]
        return names

    def _check_metric(self, metric: str, target: str) -> None:
        if target == "endmembers":
            known = self.ENDMEMBER_METRICS
        elif target == "abundances":
            known = self.ABUNDANCE_METRICS
        else:
            raise ValueError(f"Unknown target {target!r}. Choose 'endmembers' or 'abundances'.")
        if metric not in known:
            raise ValueError(f"Unknown {target} metric {metric!r}. Choose one of {list(known)}.")


def _display_function():
    """Return IPython's ``display`` in a notebook, ``print`` otherwise."""
    try:
        from IPython.display import display
        return display
    except ImportError:  # pragma: no cover - IPython is always there in notebooks
        return print


def _style_mixed(table: pd.DataFrame, precision: int, higher_is_better: set):
    """Style a table whose columns are different metrics, bolding the best value of each."""
    def _bold_best(column: pd.Series) -> list:
        target = column.max() if column.name in higher_is_better else column.min()
        return ["font-weight: bold" if v == target else "" for v in column]

    return table.style.format(f"{{:.{precision}f}}").apply(_bold_best, axis=0)


def check_against_reference(
    evaluation: UnmixingEvaluation,
    estimate: str,
    reference,
    factor: float = 1.10,
    statistic: str = "median",
    metric: str = "SAD",
) -> Dict[str, object]:
    """
    Check whether an estimate is similar to or better than a reference, i.e.
    whether its metric (mean over the endmembers, summarized over the seeds)
    is at most ``factor`` times that of the reference.

    Parameters
    ----------
    evaluation : UnmixingEvaluation
        Evaluation holding the estimate (and the reference, if given by name).
    estimate : str
        Name of the estimate to check.
    reference : str or float
        Name of a reference estimate of ``evaluation``, summarized with the
        same statistic, or a fixed reference value (e.g. the median SAD of
        VCA over a larger set of seeds).
    factor : float, default=1.10
        Tolerance: the threshold is ``factor`` times the reference value.
    statistic : {"median", "mean"}, default="median"
        Statistic used to summarize the seeds.
    metric : str, default="SAD"
        Endmember metric, for which lower values are better.

    Returns
    -------
    dict
        ``{"value", "reference", "threshold", "within"}``, where ``within``
        is ``True`` when ``value <= threshold``.
    """
    if statistic not in ("median", "mean"):
        raise ValueError(f"Unknown statistic {statistic!r}. Choose 'median' or 'mean'.")
    if metric in UnmixingEvaluation._HIGHER_IS_BETTER:
        raise ValueError(f"check_against_reference expects a metric where lower is better, got {metric!r}.")

    summary = evaluation.summary(metric)

    def _value(name: str) -> float:
        if name not in summary.index:
            raise KeyError(f"Unknown estimate '{name}'. Known estimates: {list(summary.index)}.")
        return float(summary.loc[name, statistic])

    value = _value(estimate)
    reference_value = _value(reference) if isinstance(reference, str) else float(reference)
    threshold = factor * reference_value

    return {
        "value": value,
        "reference": reference_value,
        "threshold": threshold,
        "within": bool(value <= threshold),
    }

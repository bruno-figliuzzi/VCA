import numpy as np
import matplotlib.pyplot as plt
from .dataset import HyperspectralData
from ..projection.projection import PCA
from .metrics import minmax_normalize, EndmemberAligner


# --------------------------------------------------------------------------------------
# Class used to display hyperspectral observations
# --------------------------------------------------------------------------------------
class HyperspectralPlotter:
    """
    Visualization utilities for HyperspectralData.

    Provides methods to visualize

    - endmember spectra,
    - spectral observations,
    - PCA projections,
    - hyperspectral image bands,
    - reconstructed images, and
    - abundance maps.
    """
    def __init__(self, data: HyperspectralData) -> None:
        self.data = data

    # --------------------------------- Endmembers -------------------------------------
    def plot_endmembers(self,
        linewidth: float = 1.5,
        max_cols: int = 5,
        figsize: tuple[int, int] | None = None,
        title: str | None = None,
    ) -> tuple[plt.Figure, np.ndarray]:
        """
        Plot endmembers as K subplots arranged in a grid.

        Parameters
        ----------
        linewidth : float
            Line width of spectra.

        max_cols : int, default=5
            Maximum number of subplot columns. If there are more endmembers
            than ``max_cols``, additional rows are added.

        figsize : tuple[int, int]
            Figure size.

        title : str | None
            Global figure title.

        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure containing the endmember plots.

        axes : ndarray of matplotlib.axes.Axes
            Array of subplot axes, one for each endmember.
        """

        if self.data.endmembers is None:
            raise ValueError("endmembers are not available")

        M = self.data.endmembers
        names = self.data.names
        n_bands, n_endmembers = M.shape

        x = (self.data.wavelengths if self.data.wavelengths is not None
            else np.arange(n_bands)
        )

        x_label = ("Wavelengths" if self.data.wavelengths is not None
            else "Spectral band"
        )

        n_cols = min(n_endmembers, max_cols)
        n_rows = int(np.ceil(n_endmembers / n_cols))

        if figsize is None:
            figsize = (4*n_cols, 4*n_rows)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize,
            sharex=True, sharey=True, squeeze=False)
        axes = axes.ravel()

        for k in range(n_endmembers):
            axes[k].plot(x, M[:, k], linewidth=linewidth)
            subplot_title = names[k] if names is not None else f"EM {k+1}"
            axes[k].set_title(subplot_title)
            axes[k].grid(True, alpha=0.3)

            if k // n_cols == n_rows - 1:
                axes[k].set_xlabel(x_label)

            if k % n_cols == 0:
                axes[k].set_ylabel("Reflectance")

        # Hide unused axes
        for ax in axes[n_endmembers:]:
            ax.axis("off")

        if title:
            fig.suptitle(title)

        fig.tight_layout()
        return fig, axes

    # --------------------------------- Observations------------------------------------
    def plot_observations(self,
        indices: np.ndarray,
        linewidth: float = 1.0,
        figsize: tuple[int, int] = (4, 4),
        title: str = "",
    ) -> tuple[plt.Figure, plt.Axes]:
        """
        Plot hyperspectral observations on a single axis.

        Parameters
        ----------
        indices : np.ndarray
            Indices of the observation to plot

        linewidth : float
            Line width.

        figsize : tuple[int, int]
            Figure size.

        title : str | None
            Plot title.

        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure containing the plot.

        ax : matplotlib.axes.Axes
            Axes on which the data are drawn.
        """
        Y = self.data.observations
        n_bands, _ = Y.shape
        
        x = (self.data.wavelengths if self.data.wavelengths is not None
            else np.arange(n_bands)
        )
        
        x_label = ("Wavelengths" if self.data.wavelengths is not None 
            else "Spectral band"
        )

        fig, ax = plt.subplots(figsize=figsize)

        indices = np.asarray(indices)
        for i in indices:
            ax.plot(x, Y[:, i], linewidth=linewidth)

        ax.set_xlabel(x_label)
        ax.set_ylabel("Reflectance")
        ax.set_title(title)
        fig.tight_layout()
        return fig, ax

   
    # ------------------- Reduced dimension representation -------------------------
    def plot_reduced_observations(self,
        figsize: tuple[int, int] = (4, 4),
        title: str = "",
    ) -> tuple[plt.Figure, plt.Axes]:
        """
        Scatterplot of the hyperspectral observations.
          
        The hyperspectral observations are projected onto a subspace of
        dimension 2 for visualization.

        Parameters
        ----------
        figsize : tuple[int, int]
            Figure size.
            
        title : str | None
            Plot title.

        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure containing the plot.

        ax : matplotlib.axes.Axes
            Axes on which the data are drawn.
            
        Notes
        -----
        This visualization is only relevant when the subspace spanned
        by the endmembers has dimension 2.
        """
        Y = self.data.observations
            
        pca = PCA(p=3)
        Y_reduced = pca.fit_transform(Y)
            
        fig, ax = plt.subplots(figsize=figsize)
        ax.scatter(Y_reduced[0, :], Y_reduced[1, :])
        ax.set_title(title)
        fig.tight_layout()
        return fig, ax
                  

    # --------------------------------- HSI Image --------------------------------------
    def plot_hsi_image(self,
        band_idx: int,
        title: str = "",
        figsize: tuple[int, int] = (10, 5),
    ) -> tuple[plt.Figure, plt.Axes]:
        """
        Display the hyperspectral image for the selected band

        Parameters
        ----------
        band_idx : int
            Index of the spectral band to plot

        title : str | None
            Plot title.

        figsize : tuple[int, int]
            Figure size.

        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure containing the plot.

        ax : matplotlib.axes.Axes
            Axes on which the data are drawn.
        """
        HSI_image = self.data.observations_image
        if band_idx < 0 or band_idx >= self.data.n_bands:
            raise ValueError(
                f"The provided index of the spectral band must remain between "
                f"0 and {self.data.n_bands}"
            )

        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        im = ax.imshow(HSI_image[band_idx, :, :])
        ax.set_title(title)
        ax.axis("off")
        fig.colorbar(im, ax=ax)
        return fig, ax


    # --------------------------------- HSI Image --------------------------------------
    def plot_reconstruction(self,
        band_idx: int,
        title: str | None = None,
        figsize: tuple[int, int] = (10, 5),
    ) -> tuple[plt.Figure, np.ndarray]:
        """
        Display the hyperspectral image for the selected band along with its 
        reconstruction.

        Parameters
        ----------
        band_idx : int
            Index of the spectral band to plot

        title : str | None
            Plot title.

        figsize : tuple[int, int]
            Figure size.
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure containing the plot.

        axes : ndarray of matplotlib.axes.Axes
            Array containing the axes for the original image and its reconstruction.
        """
        HSI_image = self.data.observations_image
        if band_idx < 0 or band_idx >= self.data.n_bands:
            raise ValueError(
                f"The provided index of the spectral band must remain between "
                f"0 and {self.data.n_bands}"
            )
           
        reconstruction = self.data.reconstruct().reshape(HSI_image.shape)
       
        fig, axes = plt.subplots(1, 2, figsize=figsize, 
            sharey=True, squeeze=False)
        axes = axes.ravel()
        
        if title is not None:
            fig.suptitle(title)
            
        im = axes[0].imshow(HSI_image[band_idx, :, :])
        axes[0].set_title(f"Spectral band #{band_idx}")
        axes[0].axis("off")
        
        axes[1].imshow(reconstruction[band_idx, :, :])
        axes[1].set_title("Reconstruction")
        axes[1].axis("off")
        fig.colorbar(im, ax=axes, shrink=0.8)
        
        return fig, axes
        
    # --------------------------------- Abundances--------------------------------------
    def plot_abundances(self,
        title: str | None = None,
        figsize: tuple[int, int] | None = None,
    ) -> tuple[plt.Figure, np.ndarray]:
        """
        Display the hyperspectral image for the selected band

        Parameters
        ----------
        figsize : tuple[int, int]
            Figure size.

        title : str | None
            Plot title.

        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure containing the plot.

        axes : ndarray of matplotlib.axes.Axes
            Array of subplot axes, one for each endmember.
        """
        A = self.data.abundances
        names = self.data.names
        n_endmembers = self.data.n_endmembers
        height, width = self.data.image_shape
        
        A = A.reshape((n_endmembers, height, width))
        
        if figsize is None:
            figsize = (4*n_endmembers, n_endmembers)
            
        fig, axes = plt.subplots(
            1, n_endmembers,
            figsize=figsize,
            sharey=True,
            squeeze=False,
            constrained_layout=True,
        )
        axes = axes.ravel()

        im = None
        for k in range(n_endmembers):
            im = axes[k].imshow(A[k, :, :], vmin=0., vmax=1.)
            subplot_title = names[k] if names is not None else f"EM {k+1}"
            axes[k].set_title(subplot_title)
            axes[k].axis("off")

        fig.colorbar(im, ax=axes)
        if title:
            fig.suptitle(title)

        return fig, axes
 
      
# --------------------------------------------------------------------------------------
# Classes used to display unmixing results
# --------------------------------------------------------------------------------------
class EndmemberPlotter:
    """
    Visualization utilities for comparing estimated and ground-truth endmembers.

    This class provides spectral comparison plots for qualitative evaluation
    of hyperspectral unmixing algorithms. Before plotting, the estimated
    endmembers are automatically aligned with the ground truth using an
    :class:`EndmemberAligner`, ensuring that corresponding spectra are
    displayed together.

    Parameters
    ----------
    wavelengths : np.ndarray, shape (n_bands,)
        Wavelength values (or spectral band indices) associated with each
        spectral band.

    max_cols : int, default=5
        Maximum number of subplot columns.
    """

    def __init__(
        self,
        wavelengths: np.ndarray,
        max_cols: int = 5,
    ) -> None:
        self.wavelengths = wavelengths
        self.max_cols = max_cols
        self.aligner = EndmemberAligner()

    def plot(
        self,
        M_est: np.ndarray,
        M_gt: np.ndarray,
        normalize: bool = False,
        names: list[str] | None = None,
        gt_kwargs: dict | None = None,
        est_kwargs: dict | None = None,
        title: str | None = None,
        figsize: tuple[float, float] | None = None,
    ) -> tuple[plt.Figure, np.ndarray]:
        """
        Plot estimated and ground-truth endmember spectra.

        The estimated endmembers are first aligned with the ground truth using
        :class:`EndmemberAligner`. Optionally, both matrices are normalized
        using column-wise min-max normalization before plotting.

        Parameters
        ----------
        M_est : np.ndarray, shape (n_bands, n_endmembers)
            Estimated endmember matrix.

        M_gt : np.ndarray, shape (n_bands, n_endmembers)
            Ground-truth endmember matrix.

        normalize : bool, default=False
            If ``True``, apply column-wise min-max normalization to both
            matrices before plotting.

        names : list[str] | None, default=None
            Names associated with each endmember. If ``None``, generic names
            ("EM 1", "EM 2", ...) are used.

        gt_kwargs : dict | None, default=None
            Additional keyword arguments passed to
            :meth:`matplotlib.axes.Axes.plot` for the ground-truth spectra.

        est_kwargs : dict | None, default=None
            Additional keyword arguments passed to
            :meth:`matplotlib.axes.Axes.plot` for the estimated spectra.

        title : str | None, default=None
            Figure title.

        figsize : tuple[float, float] | None, default=None
            Figure size in inches. If ``None``, a suitable size is computed
            automatically from the number of endmembers.

        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure containing the comparison plots.

        axes : ndarray of matplotlib.axes.Axes
            Flattened array of subplot axes.

        Raises
        ------
        ValueError
            If ``M_est`` and ``M_gt`` do not have the same shape.

        ValueError
            If the wavelength vector length does not match the number of
            spectral bands.

        ValueError
            If the number of provided endmember names does not match the
            number of endmembers.
        """
        if M_est.shape != M_gt.shape:
            raise ValueError(
                "M_est and M_gt must have the same shape."
            )

        n_bands, n_endmembers = M_est.shape

        if self.wavelengths.shape[0] != n_bands:
            raise ValueError(
                "The wavelength vector must contain one value per spectral band."
            )

        if names is not None and len(names) != n_endmembers:
            raise ValueError(
                "The number of endmember names must match the number of endmembers."
            )

        if normalize:
            M_est = minmax_normalize(M_est)
            M_gt = minmax_normalize(M_gt)

        M_est, _ = self.aligner.align(M_est, M_gt)

        gt_kwargs = {
            "linewidth": 2,
            **(gt_kwargs or {}),
        }

        est_kwargs = {
            "linewidth": 2,
            "linestyle": "--",
            **(est_kwargs or {}),
        }

        n_cols = min(n_endmembers, self.max_cols)
        n_rows = int(np.ceil(n_endmembers / n_cols))

        if figsize is None:
            figsize = (4 * n_cols, 3.5 * n_rows)

        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=figsize,
            sharex=True,
            sharey=True,
            squeeze=False,
        )

        axes = axes.ravel()

        for k in range(n_endmembers):
            ax = axes[k]

            ax.plot(
                self.wavelengths,
                M_gt[:, k],
                label="Ground truth",
                **gt_kwargs,
            )

            ax.plot(
                self.wavelengths,
                M_est[:, k],
                label="Estimated",
                **est_kwargs,
            )

            subplot_title = (
                names[k]
                if names is not None
                else f"EM {k + 1}"
            )

            ax.set_title(subplot_title)
            ax.set_xlabel("Wavelength")
            ax.grid(True, alpha=0.3)

            if k % n_cols == 0:
                ax.set_ylabel("Reflectance")

        # Hide unused axes
        for ax in axes[n_endmembers:]:
            ax.axis("off")

        axes[0].legend()

        if title is not None:
            fig.suptitle(title)
            fig.tight_layout(rect=[0, 0, 1, 0.96])
        else:
            fig.tight_layout()

        return fig, axes

    def plot_comparison(
        self,
        estimates: dict[str, np.ndarray],
        M_gt: np.ndarray,
        normalize: bool = False,
        names: list[str] | None = None,
        gt_kwargs: dict | None = None,
        est_kwargs: dict | None = None,
        title: str | None = None,
        figsize: tuple[float, float] | None = None,
    ) -> tuple[plt.Figure, np.ndarray]:
        """
        Plot ground-truth endmembers together with several estimates.

        The comparison is laid out as a grid with one row per algorithm in
        ``estimates`` and one column per endmember. Each subplot overlays
        the ground-truth spectrum with the corresponding algorithm's
        estimate. Every estimate is aligned with the ground truth
        independently, using :class:`EndmemberAligner`, before plotting.

        Parameters
        ----------
        estimates : dict[str, np.ndarray]
            Dictionary mapping algorithm names to estimated endmember
            matrices, each of shape (n_bands, n_endmembers).

        M_gt : np.ndarray, shape (n_bands, n_endmembers)
            Ground-truth endmember matrix.

        normalize : bool, default=False
            If ``True``, apply column-wise min-max normalization to the
            ground truth and to each estimate before plotting.

        names : list[str] | None, default=None
            Names associated with each endmember. If ``None``, generic names
            ("EM 1", "EM 2", ...) are used.

        gt_kwargs : dict | None, default=None
            Additional keyword arguments passed to
            :meth:`matplotlib.axes.Axes.plot` for the ground-truth spectra.

        est_kwargs : dict | None, default=None
            Additional keyword arguments passed to
            :meth:`matplotlib.axes.Axes.plot` for the estimated spectra.

        title : str | None, default=None
            Figure title.

        figsize : tuple[float, float] | None, default=None
            Figure size in inches. If ``None``, a suitable size is computed
            automatically from the number of algorithms and endmembers.

        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure containing the comparison plots.

        axes : ndarray of matplotlib.axes.Axes, shape (n_algorithms, n_endmembers)
            Grid of subplot axes, one row per algorithm and one column per
            endmember.

        Raises
        ------
        ValueError
            If ``estimates`` is empty.

        ValueError
            If any estimate in ``estimates`` does not have the same shape
            as ``M_gt``.

        ValueError
            If the wavelength vector length does not match the number of
            spectral bands.

        ValueError
            If the number of provided endmember names does not match the
            number of endmembers.
        """
        if not estimates:
            raise ValueError("`estimates` must contain at least one algorithm.")

        for name, M_est in estimates.items():
            if M_est.shape != M_gt.shape:
                raise ValueError(
                    f"Estimated endmembers for '{name}' must have the same "
                    f"shape as M_gt."
                )

        n_bands, n_endmembers = M_gt.shape
        n_algorithms = len(estimates)

        if self.wavelengths.shape[0] != n_bands:
            raise ValueError(
                "The wavelength vector must contain one value per spectral band."
            )

        if names is not None and len(names) != n_endmembers:
            raise ValueError(
                "The number of endmember names must match the number of endmembers."
            )

        M_gt_plot = minmax_normalize(M_gt) if normalize else M_gt

        aligned_estimates = {}
        for name, M_est in estimates.items():
            M_est_plot = minmax_normalize(M_est) if normalize else M_est
            M_aligned, _ = self.aligner.align(M_est_plot, M_gt_plot, normalize=False)
            aligned_estimates[name] = M_aligned

        return self.plot_aligned(
            aligned_estimates,
            M_gt_plot,
            names=names,
            gt_kwargs=gt_kwargs,
            est_kwargs=est_kwargs,
            title=title,
            figsize=figsize,
        )

    def plot_aligned(
        self,
        estimates: dict[str, np.ndarray],
        M_gt: np.ndarray,
        bands: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
        names: list[str] | None = None,
        gt_kwargs: dict | None = None,
        est_kwargs: dict | None = None,
        band_kwargs: dict | None = None,
        title: str | None = None,
        figsize: tuple[float, float] | None = None,
    ) -> tuple[plt.Figure, np.ndarray]:
        """
        Plot ground-truth endmembers together with several estimates that
        are already aligned with them.

        The layout is the same as :meth:`plot_comparison` (one row per
        estimate, one column per endmember), but no alignment or
        normalization is applied: column ``k`` of every estimate is drawn
        against column ``k`` of ``M_gt``. An optional shaded band can be
        drawn around each estimate, e.g. the spread of the estimates over
        several random seeds.

        Parameters
        ----------
        estimates : dict[str, np.ndarray]
            Dictionary mapping row labels to aligned endmember matrices,
            each of shape (n_bands, n_endmembers).

        M_gt : np.ndarray, shape (n_bands, n_endmembers)
            Ground-truth endmember matrix.

        bands : dict[str, tuple[np.ndarray, np.ndarray]] | None, default=None
            Optional lower and upper bounds of the shaded band, each of
            shape (n_bands, n_endmembers), keyed like ``estimates``.
            Estimates without an entry are drawn without a band.

        names : list[str] | None, default=None
            Names associated with each endmember. If ``None``, generic names
            ("EM 1", "EM 2", ...) are used.

        gt_kwargs : dict | None, default=None
            Additional keyword arguments passed to
            :meth:`matplotlib.axes.Axes.plot` for the ground-truth spectra.

        est_kwargs : dict | None, default=None
            Additional keyword arguments passed to
            :meth:`matplotlib.axes.Axes.plot` for the estimated spectra.

        band_kwargs : dict | None, default=None
            Additional keyword arguments passed to
            :meth:`matplotlib.axes.Axes.fill_between` for the bands.

        title : str | None, default=None
            Figure title.

        figsize : tuple[float, float] | None, default=None
            Figure size in inches. If ``None``, a suitable size is computed
            automatically from the number of estimates and endmembers.

        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure containing the comparison plots.

        axes : ndarray of matplotlib.axes.Axes, shape (n_estimates, n_endmembers)
            Grid of subplot axes, one row per estimate and one column per
            endmember.

        Raises
        ------
        ValueError
            If ``estimates`` is empty, if an estimate or a band does not
            have the same shape as ``M_gt``, if the wavelength vector length
            does not match the number of spectral bands, or if the number of
            names does not match the number of endmembers.
        """
        if not estimates:
            raise ValueError("`estimates` must contain at least one algorithm.")

        bands = bands or {}

        for name, M_est in estimates.items():
            if M_est.shape != M_gt.shape:
                raise ValueError(
                    f"Estimated endmembers for '{name}' must have the same "
                    f"shape as M_gt."
                )

        for name, (lower, upper) in bands.items():
            if lower.shape != M_gt.shape or upper.shape != M_gt.shape:
                raise ValueError(
                    f"The band of '{name}' must have the same shape as M_gt."
                )

        n_bands, n_endmembers = M_gt.shape
        n_algorithms = len(estimates)

        if self.wavelengths.shape[0] != n_bands:
            raise ValueError(
                "The wavelength vector must contain one value per spectral band."
            )

        if names is not None and len(names) != n_endmembers:
            raise ValueError(
                "The number of endmember names must match the number of endmembers."
            )

        gt_kwargs = {
            "linewidth": 2,
            "color": "black",
            **(gt_kwargs or {}),
        }

        est_kwargs = {
            "linewidth": 1.5,
            "linestyle": "--",
            **(est_kwargs or {}),
        }

        band_kwargs = {
            "alpha": 0.25,
            "linewidth": 0,
            **(band_kwargs or {}),
        }
        # The band takes the color of the estimate unless told otherwise
        if "color" not in band_kwargs and "color" in est_kwargs:
            band_kwargs["color"] = est_kwargs["color"]

        if figsize is None:
            figsize = (4 * n_endmembers, 3 * n_algorithms)

        fig, axes = plt.subplots(
            n_algorithms,
            n_endmembers,
            figsize=figsize,
            sharex=True,
            sharey=True,
            squeeze=False,
        )

        for row, (name, M_aligned) in enumerate(estimates.items()):
            for k in range(n_endmembers):
                ax = axes[row, k]

                ax.plot(
                    self.wavelengths,
                    M_gt[:, k],
                    label="Ground truth",
                    **gt_kwargs,
                )

                ax.plot(
                    self.wavelengths,
                    M_aligned[:, k],
                    label="Estimated",
                    **est_kwargs,
                )

                if name in bands:
                    lower, upper = bands[name]
                    ax.fill_between(
                        self.wavelengths,
                        lower[:, k],
                        upper[:, k],
                        label="Spread over seeds",
                        **band_kwargs,
                    )

                ax.grid(True, alpha=0.3)

                if row == 0:
                    subplot_title = (
                        names[k]
                        if names is not None
                        else f"EM {k + 1}"
                    )
                    ax.set_title(subplot_title)

                if row == n_algorithms - 1:
                    ax.set_xlabel("Wavelength")

                if k == 0:
                    ax.set_ylabel(f"{name}\nReflectance")

        # The legend of the first row with a band, so that the band is listed
        legend_row = next(
            (row for row, name in enumerate(estimates) if name in bands), 0
        )
        axes[legend_row, 0].legend()

        if title is not None:
            fig.suptitle(title)
            fig.tight_layout(rect=[0, 0, 1, 0.96])
        else:
            fig.tight_layout()

        return fig, axes


class AbundancesMapPlotter:
    """
    Visualization utilities for comparing estimated and ground-truth
    abundance maps.

    This class displays the ground-truth abundance maps, the estimated
    abundance maps, and the corresponding absolute error maps for each
    endmember.

    Parameters
    ----------
    image_shape : tuple[int, int]
        Spatial dimensions of the hyperspectral image as
        ``(height, width)``.
    """

    def __init__(self, image_shape: tuple[int, int]) -> None:
        self.image_shape = image_shape

    def plot_maps(
        self,
        maps: dict[str, np.ndarray],
        names: list[str] | None = None,
        max_cols: int = 5,
        cmap: str = "magma",
        title: str | None = None,
        colorbar_label: str | None = None,
        figsize: tuple[float, float] | None = None,
        imshow_kwargs: dict | None = None,
    ) -> tuple[plt.Figure, np.ndarray]:
        """
        Plot one row of per-endmember maps for each entry of ``maps``.

        Unlike :meth:`plot_comparison`, no ground-truth row is added: this
        is meant for derived maps, such as the spread of the abundances
        over several random seeds. All maps share the same color scale.

        Parameters
        ----------
        maps : dict[str, np.ndarray]
            Dictionary mapping row labels to matrices of shape
            (n_endmembers, n_pixels).

        names : list[str] | None, default=None
            Names of the endmembers. If ``None``, generic names
            ("EM 1", "EM 2", ...) are used.

        max_cols : int, default=5
            Maximum number of endmembers displayed.

        cmap : str, default="magma"
            Colormap used for the maps.

        title : str | None, default=None
            Figure title.

        colorbar_label : str | None, default=None
            Label of the shared colorbar.

        figsize : tuple[float, float] | None, default=None
            Figure size in inches. If ``None``, an appropriate size is
            computed automatically.

        imshow_kwargs : dict | None, default=None
            Additional keyword arguments forwarded to
            :meth:`matplotlib.axes.Axes.imshow`.

        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure containing the maps.

        axes : ndarray of matplotlib.axes.Axes, shape (n_rows, n_display)
            Grid of subplot axes, one row per entry of ``maps``.

        Raises
        ------
        ValueError
            If ``maps`` is empty, if the matrices do not all have the same
            shape, if the number of pixels is inconsistent with
            ``image_shape``, or if the number of names does not match the
            number of endmembers.
        """
        if not maps:
            raise ValueError("`maps` must contain at least one entry.")

        shapes = {values.shape for values in maps.values()}
        if len(shapes) != 1:
            raise ValueError("All maps must have the same shape.")

        n_endmembers, n_pixels = shapes.pop()
        height, width = self.image_shape

        if n_pixels != height * width:
            raise ValueError(
                "The maps are incompatible with image_shape."
            )

        if names is not None and len(names) != n_endmembers:
            raise ValueError(
                "The number of endmember names must match the number of endmembers."
            )

        n_display = min(n_endmembers, max_cols)
        n_rows = len(maps)

        imshow_kwargs = imshow_kwargs or {}

        vmin = min(values.min() for values in maps.values())
        vmax = max(values.max() for values in maps.values())

        if figsize is None:
            figsize = (4 * n_display, 3 * n_rows)

        fig, axes = plt.subplots(
            n_rows,
            n_display,
            figsize=figsize,
            squeeze=False,
            constrained_layout=True,
        )

        im = None

        for row, (label, values) in enumerate(maps.items()):
            for k in range(n_display):
                im = axes[row, k].imshow(
                    values[k].reshape(height, width),
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    **imshow_kwargs,
                )
                axes[row, k].set_xticks([])
                axes[row, k].set_yticks([])

                if row == 0:
                    axes[row, k].set_title(
                        names[k] if names is not None else f"EM {k + 1}"
                    )

            axes[row, 0].set_ylabel(label)

        fig.colorbar(
            im,
            ax=axes,
            shrink=0.8,
            location="right",
            label=colorbar_label,
        )

        if title is not None:
            fig.suptitle(title)

        return fig, axes

    def plot(
        self,
        A_est: np.ndarray,
        A_gt: np.ndarray,
        names: list[str] | None = None,
        max_cols: int = 5,
        cmap: str = "viridis",
        error_cmap: str = "magma",
        title: str | None = None,
        figsize: tuple[float, float] | None = None,
        imshow_kwargs: dict | None = None,
    ) -> tuple[plt.Figure, np.ndarray]:
        """
        Plot ground-truth, estimated and error abundance maps.

        Each column corresponds to one endmember. The three rows display:

        1. Ground-truth abundance map.
        2. Estimated abundance map.
        3. Absolute error map.

        Parameters
        ----------
        A_est : np.ndarray, shape (n_endmembers, n_pixels)
            Estimated abundance matrix.

        A_gt : np.ndarray, shape (n_endmembers, n_pixels)
            Ground-truth abundance matrix.

        names : list[str] | None, default=None
            Names of the endmembers. If ``None``, generic names
            ("EM 1", "EM 2", ...) are used.

        max_cols : int, default=5
            Maximum number of endmembers displayed.

        cmap : str, default="viridis"
            Colormap used for the abundance maps.

        error_cmap : str, default="magma"
            Colormap used for the absolute error maps.

        title : str | None, default=None
            Figure title.

        figsize : tuple[float, float] | None, default=None
            Figure size in inches. If ``None``, an appropriate size is
            computed automatically.

        imshow_kwargs : dict | None, default=None
            Additional keyword arguments forwarded to
            :meth:`matplotlib.axes.Axes.imshow`.

        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure containing the abundance maps.

        axes : ndarray of matplotlib.axes.Axes
            Array of subplot axes.

        Raises
        ------
        ValueError
            If ``A_est`` and ``A_gt`` do not have the same shape.

        ValueError
            If the number of pixels is inconsistent with ``image_shape``.

        ValueError
            If the number of provided names does not match the number of
            endmembers.
        """
        if A_est.shape != A_gt.shape:
            raise ValueError(
                "A_est and A_gt must have the same shape."
            )

        n_endmembers, n_pixels = A_est.shape
        height, width = self.image_shape

        if n_pixels != height * width:
            raise ValueError(
                "The abundance matrices are incompatible with image_shape."
            )

        if names is not None and len(names) != n_endmembers:
            raise ValueError(
                "The number of endmember names must match the number of endmembers."
            )

        n_display = min(n_endmembers, max_cols)

        if figsize is None:
            figsize = (4 * n_display, 9)

        imshow_kwargs = imshow_kwargs or {}

        vmin = min(A_gt.min(), A_est.min())
        vmax = max(A_gt.max(), A_est.max())

        fig, axes = plt.subplots(
            3,
            n_display,
            figsize=figsize,
            squeeze=False,
            constrained_layout=True,
        )

        im_gt = None
        im_err = None

        for k in range(n_display):
            gt = A_gt[k].reshape(height, width)
            est = A_est[k].reshape(height, width)
            err = np.abs(est - gt)

            subplot_title = (
                names[k]
                if names is not None
                else f"EM {k + 1}"
            )

            im_gt = axes[0, k].imshow(
                gt,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                **imshow_kwargs,
            )
            axes[0, k].set_title(subplot_title)
            axes[0, k].axis("off")

            axes[1, k].imshow(
                est,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                **imshow_kwargs,
            )
            axes[1, k].axis("off")

            im_err = axes[2, k].imshow(
                err,
                cmap=error_cmap,
                **imshow_kwargs,
            )
            axes[2, k].axis("off")

        axes[0, 0].set_ylabel("Ground truth")
        axes[1, 0].set_ylabel("Estimated")
        axes[2, 0].set_ylabel("Absolute error")

        fig.colorbar(
            im_gt,
            ax=axes[0, :],
            shrink=0.8,
            location="right",
        )

        fig.colorbar(
            im_gt,
            ax=axes[1, :],
            shrink=0.8,
            location="right",
        )
        
        fig.colorbar(
            im_err,
            ax=axes[2, :],
            shrink=0.8,
            location="right",
        )

        if title is not None:
            fig.suptitle(title)

        return fig, axes

    def plot_comparison(
        self,
        estimates: dict[str, np.ndarray],
        A_gt: np.ndarray,
        names: list[str] | None = None,
        max_cols: int = 5,
        cmap: str = "viridis",
        title: str | None = None,
        figsize: tuple[float, float] | None = None,
        imshow_kwargs: dict | None = None,
    ) -> tuple[plt.Figure, np.ndarray]:
        """
        Plot ground-truth abundance maps together with several estimates.

        The comparison is laid out as a grid with an additional top row
        for the ground-truth maps, followed by one row per algorithm in
        ``estimates``, and one column per endmember.

        Parameters
        ----------
        estimates : dict[str, np.ndarray]
            Dictionary mapping algorithm names to estimated abundance
            matrices, each of shape (n_endmembers, n_pixels). Estimates
            are expected to already be aligned with ``A_gt`` (e.g. using
            :class:`AbundancesAligner`).

        A_gt : np.ndarray, shape (n_endmembers, n_pixels)
            Ground-truth abundance matrix.

        names : list[str] | None, default=None
            Names of the endmembers. If ``None``, generic names
            ("EM 1", "EM 2", ...) are used.

        max_cols : int, default=5
            Maximum number of endmembers displayed.

        cmap : str, default="viridis"
            Colormap used for the abundance maps.

        title : str | None, default=None
            Figure title.

        figsize : tuple[float, float] | None, default=None
            Figure size in inches. If ``None``, an appropriate size is
            computed automatically from the number of algorithms and
            endmembers.

        imshow_kwargs : dict | None, default=None
            Additional keyword arguments forwarded to
            :meth:`matplotlib.axes.Axes.imshow`.

        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure containing the abundance maps.

        axes : ndarray of matplotlib.axes.Axes, shape (n_algorithms + 1, n_display)
            Grid of subplot axes. Row 0 holds the ground-truth maps;
            each subsequent row holds one algorithm's estimated maps, in
            the order of ``estimates``.

        Raises
        ------
        ValueError
            If ``estimates`` is empty.

        ValueError
            If any estimate in ``estimates`` does not have the same shape
            as ``A_gt``.

        ValueError
            If the number of pixels is inconsistent with ``image_shape``.

        ValueError
            If the number of provided names does not match the number of
            endmembers.
        """
        if not estimates:
            raise ValueError("`estimates` must contain at least one algorithm.")

        for name, A_est in estimates.items():
            if A_est.shape != A_gt.shape:
                raise ValueError(
                    f"Estimated abundances for '{name}' must have the same "
                    f"shape as A_gt."
                )

        n_endmembers, n_pixels = A_gt.shape
        height, width = self.image_shape

        if n_pixels != height * width:
            raise ValueError(
                "The abundance matrices are incompatible with image_shape."
            )

        if names is not None and len(names) != n_endmembers:
            raise ValueError(
                "The number of endmember names must match the number of endmembers."
            )

        n_display = min(n_endmembers, max_cols)
        n_algorithms = len(estimates)
        n_rows = n_algorithms + 1

        imshow_kwargs = imshow_kwargs or {}

        vmin = A_gt.min()
        vmax = A_gt.max()
        for A_est in estimates.values():
            vmin = min(vmin, A_est.min())
            vmax = max(vmax, A_est.max())

        if figsize is None:
            figsize = (4 * n_display, 3 * n_rows)

        fig, axes = plt.subplots(
            n_rows,
            n_display,
            figsize=figsize,
            squeeze=False,
            constrained_layout=True,
        )

        im = None

        for k in range(n_display):
            subplot_title = (
                names[k]
                if names is not None
                else f"EM {k + 1}"
            )

            im = axes[0, k].imshow(
                A_gt[k].reshape(height, width),
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                **imshow_kwargs,
            )
            axes[0, k].set_title(subplot_title)
            axes[0, k].set_xticks([])
            axes[0, k].set_yticks([])

        axes[0, 0].set_ylabel("Ground truth")

        for row, (name, A_est) in enumerate(estimates.items(), start=1):
            for k in range(n_display):
                im = axes[row, k].imshow(
                    A_est[k].reshape(height, width),
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    **imshow_kwargs,
                )
                axes[row, k].set_xticks([])
                axes[row, k].set_yticks([])

            axes[row, 0].set_ylabel(name)

        fig.colorbar(
            im,
            ax=axes,
            shrink=0.8,
            location="right",
        )

        if title is not None:
            fig.suptitle(title)

        return fig, axes


class SimplexPlotter:
    """
    Visualization utilities for displaying endmembers in a reduced-dimensional
    representation of the hyperspectral data.

    The observation matrix is projected onto the first two principal components
    using PCA. Estimated, ground-truth, and optionally initial endmembers are
    projected into the same space and overlaid on the observation cloud.

    This visualization is useful for qualitatively assessing the geometry of
    the estimated simplex.
    """

    def plot(
        self,
        Y: np.ndarray,
        M_est: np.ndarray,
        M_gt: np.ndarray,
        M_init: np.ndarray | None = None,
        title: str | None = None,
        figsize: tuple[float, float] = (6, 6),
        observation_kwargs: dict | None = None,
        estimated_kwargs: dict | None = None,
        ground_truth_kwargs: dict | None = None,
        initial_kwargs: dict | None = None,
        show_legend: bool = True,
    ) -> tuple[plt.Figure, plt.Axes]:
        """
        Plot the endmember simplex in a two-dimensional PCA subspace.

        The PCA model is fitted on the observation matrix and then used to
        project the observations and the different endmember matrices into
        the same reduced-dimensional space.

        Parameters
        ----------
        Y : np.ndarray, shape (n_bands, n_pixels)
            Hyperspectral observation matrix.

        M_est : np.ndarray, shape (n_bands, n_endmembers)
            Estimated endmember matrix.

        M_gt : np.ndarray, shape (n_bands, n_endmembers)
            Ground-truth endmember matrix.

        M_init : np.ndarray | None, shape (n_bands, n_endmembers), default=None
            Initial endmember matrix.

        title : str | None, default=None
            Figure title.

        figsize : tuple[float, float], default=(6, 6)
            Figure size in inches.

        observation_kwargs : dict | None, default=None
            Additional keyword arguments passed to ``Axes.scatter`` for the
            observations.

        estimated_kwargs : dict | None, default=None
            Additional keyword arguments passed to ``Axes.scatter`` for the
            estimated endmembers.

        ground_truth_kwargs : dict | None, default=None
            Additional keyword arguments passed to ``Axes.scatter`` for the
            ground-truth endmembers.

        initial_kwargs : dict | None, default=None
            Additional keyword arguments passed to ``Axes.scatter`` for the
            initial endmembers.

        show_legend : bool, default=True
            Whether to display the legend.

        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure containing the simplex visualization.

        ax : matplotlib.axes.Axes
            Axes containing the scatter plot.

        Raises
        ------
        ValueError
            If the matrices do not have compatible shapes.
        """
        if Y.ndim != 2:
            raise ValueError("Y must be a two-dimensional array.")

        n_bands, _ = Y.shape

        for name, M in {
            "M_est": M_est,
            "M_gt": M_gt,
        }.items():
            if M.ndim != 2:
                raise ValueError(f"{name} must be a two-dimensional array.")
            if M.shape[0] != n_bands:
                raise ValueError(
                    f"{name} must have {n_bands} rows."
                )

        if M_est.shape != M_gt.shape:
            raise ValueError(
                "M_est and M_gt must have identical shapes."
            )

        if M_init is not None:
            if M_init.shape != M_est.shape:
                raise ValueError(
                    "M_init must have the same shape as M_est."
                )

        observation_kwargs = observation_kwargs or {}
        estimated_kwargs = estimated_kwargs or {}
        ground_truth_kwargs = ground_truth_kwargs or {}
        initial_kwargs = initial_kwargs or {}

        pca = PCA(p=3)

        Y_red = pca.fit_transform(Y)
        M_est_red = pca.transform(M_est)
        M_gt_red = pca.transform(M_gt)

        if M_init is not None:
            M_init_red = pca.transform(M_init)

        fig, ax = plt.subplots(figsize=figsize)

        ax.scatter(
            Y_red[0],
            Y_red[1],
            s=5,
            alpha=0.3,
            label="Observations",
            **observation_kwargs,
        )

        ax.scatter(
            M_est_red[0],
            M_est_red[1],
            marker="o",
            s=80,
            label="Estimated",
            **estimated_kwargs,
        )

        ax.scatter(
            M_gt_red[0],
            M_gt_red[1],
            marker="x",
            s=100,
            linewidths=2,
            label="Ground truth",
            **ground_truth_kwargs,
        )

        if M_init is not None:
            ax.scatter(
                M_init_red[0],
                M_init_red[1],
                marker="^",
                s=80,
                label="Initial",
                **initial_kwargs,
            )

        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")

        if title is not None:
            ax.set_title(title)

        if show_legend:
            ax.legend()

        ax.set_aspect("equal")

        fig.tight_layout()

        return fig, ax         

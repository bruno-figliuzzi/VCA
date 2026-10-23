from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import numpy as np

def _validate_inputs(
    endmembers: np.ndarray,
    wavelengths: Optional[np.ndarray] = None,
    endmember_names: Optional[tuple[str]] = None,
):
    """
    Validate spectral library inputs.

    Parameters
    ----------
    endmembers : np.ndarray
        Endmember matrix of shape ``(n_bands, n_spectra)``, where ``n_bands`` is the 
        number of spectral bands and ``n_spectra`` is the number of endmembers.
        
    wavelengths : np.ndarray or None
        Optional wavelength axis of shape ``(n_bands,)``. If provided, its
        length must match the number of rows in ``endmembers``.

    endmember_names : tuple of str or None, optional
        Optional collection of endmember names. If provided, its length
        must match the number of endmembers ``n_spectra``.
    """
    # Endmembers
    if endmembers.ndim != 2:
        raise ValueError("endmembers must be a 2D array (n_bands, n_spectra)")  
    n_bands, n_spectra = endmembers.shape

    # Wavelengths
    if wavelengths is not None:
        if wavelengths.ndim != 1:
            raise ValueError("wavelengths must be a 1D array (n_bands,)")  

        if wavelengths.size != n_bands:
            raise ValueError(
                f"wavelengths shape mismatch: expected n_bands={n_bands}"
                f", got {wavelengths.size}"
            )
 
    # Endmembers names
    if endmember_names is not None:

        if len(endmember_names) != endmembers.shape[1]:
            raise ValueError(
                f"expected {endmembers.shape[1]} names, "
                f"got {len(endmember_names)}"
            )


def _validate_bundle_ids(
    bundle_ids: np.ndarray,
    n_spectra: int,
    bundle_names: Optional[tuple[str, ...]] = None,
):
    """
    Validate bundle membership inputs.

    Parameters
    ----------
    bundle_ids : np.ndarray
        1D array of shape ``(n_spectra,)`` assigning each endmember to a
        bundle. Must contain contiguous integers starting at 0.

    n_spectra : int
        Expected number of endmembers (columns of the underlying library).

    bundle_names : tuple of str or None, optional
        Optional collection of bundle names. If provided, its length must
        match the number of unique bundles.
    """
    if bundle_ids.ndim != 1:
        raise ValueError("bundle_ids must be a 1D array (n_spectra,)")

    if bundle_ids.size != n_spectra:
        raise ValueError(
            f"bundle_ids shape mismatch: expected n_spectra={n_spectra}"
            f", got {bundle_ids.size}"
        )

    if not np.issubdtype(bundle_ids.dtype, np.integer):
        raise ValueError("bundle_ids must contain integer values")

    if bundle_ids.size and bundle_ids.min() < 0:
        raise ValueError("bundle_ids must be non-negative")

    n_bundles = int(bundle_ids.max()) + 1 if bundle_ids.size else 0
    unique_ids = np.unique(bundle_ids)
    if not np.array_equal(unique_ids, np.arange(n_bundles)):
        raise ValueError(
            "bundle_ids must use contiguous integers starting at 0 "
            f"(0..{n_bundles - 1}), got unique values {unique_ids.tolist()}"
        )

    if bundle_names is not None and len(bundle_names) != n_bundles:
        raise ValueError(
            f"expected {n_bundles} bundle names, got {len(bundle_names)}"
        )


# ------------------------------ Endmembers library -----------------------------------
@dataclass(frozen=True)
class SpectralLibrary:
    """
    A container for hyperspectral endmember libraries with optional wavelength metadata.

    This class represents a spectral library as a set of endmember spectra organized
    in a matrix form, where each column corresponds to an endmember spectrum and each
    row corresponds to a wavelength channel.

    The library optionally stores wavelength information and human-readable names for
    each endmember. It provides utilities for validation, serialization, and spectral
    subset selection based on spectral angle distance (SAD).

    Parameters
    ----------
    wavelengths : np.ndarray or None, shape (n_bands,), optional
        1D array of wavelength values corresponding to spectral bands.
        If provided, must match the number of rows in `endmembers`.

    endmembers : np.ndarray, shape (n_bands, n_spectra)
        2D array of spectral signatures where each column represents an endmember
        and each row corresponds to a wavelength channel.

    endmember_names : tuple of str, optional
        Optional names for each endmember. If not provided, default names
        of the form "endmember #i" are generated.

    wavelength_unit : str, default="nm"
        Unit of the wavelength axis (e.g., "nm", "µm").

    Attributes
    ----------
    n_bands : int
        Number of wavelength channels (rows in `endmembers`).

    n_spectra : int
        Number of endmembers (columns in `endmembers`).

    names : tuple of str
        Resolved names of endmembers (user-defined or autogenerated).
    """
    endmembers: np.ndarray
    wavelengths: np.ndarray | None = None
    endmember_names: Optional[tuple[str, ...]] = None
    wavelength_unit: str = "nm"

    def __post_init__(self):
    
        _validate_inputs(
            self.endmembers,
            self.wavelengths,
            self.endmember_names
        )
        
        # enforce immutability of arrays
        for arr in [
            self.wavelengths,
            self.endmembers
        ]:
            if arr is not None:
                arr.setflags(write=False)
                                                  
    # ------------------------------- Properties ---------------------------------------
    @property
    def n_bands(self) -> int:
        """
        Number of spectral bands (wavelength channels).

        Returns
        -------
        int
            Number of rows in the endmember library.
        """
        return self.endmembers.shape[0]

    @property
    def n_spectra(self) -> int:
        """
        Number of endmembers in the spectral library.

        Returns
        -------
        int
            Number of columns in the endmember library.
        """
        return self.endmembers.shape[1]

    @property
    def names(self) -> Optional[tuple[str, ...]]:
        """
        Names of the endmembers in the spectral library.

        Returns
        -------
        tuple of str or None
            If `endmember_names` is provided, returns it directly.
            Otherwise returns autogenerated names of the form:
            "endmember #1", "endmember #2", ..., "endmember #n_spectra".
        """
        if self.endmember_names is not None:
            return self.endmember_names

        return tuple(
            f"endmember #{k+1}"
            for k in range(self.n_spectra)
        )

    # ---------------------------- Serialization ---------------------------------------
    def to_numpy_dict(self) -> Dict[str, Any]:
        """
        NPZ-compatible dictionary.
        
        Convert the spectral library to a dictionary compatible with
        NumPy serialization (e.g. ``np.savez``).

        Returns
        -------
        dict
            Dictionary containing the spectral library data and metadata:

            - ``"wavelengths"`` : ndarray or object scalar
                Wavelength axis of shape ``(n_bands,)``. If no wavelength
                information is available, a scalar object array containing
                `None`` is returned.
            - ``"endmembers"`` : ndarray
                Endmember matrix of shape ``(n_bands, n_spectra)``.
            - ``"names"`` : ndarray of dtype object
                Endmember names of shape ``(n_spectra,)``.
            - ``"wavelength_unit"`` : object scalar
                Unit associated with the wavelength axis.

        Notes
        -----
        - None values are preserved using 0-d object arrays.
        """
        def encode_optional(x):
            if x is None:
                return np.array(None, dtype=object)
            return x

        return {
            "wavelengths": encode_optional(self.wavelengths),
            "endmembers": self.endmembers,
            "endmember_names": np.array(self.names, dtype=object),
            "wavelength_unit": np.array(self.wavelength_unit, dtype=object),
        }
        
    def save_npz(self, path: str) -> None:
        """
        Save the spectral library to a compressed NumPy archive.

        Parameters
        ----------
        path : str
            Path to the output ``.npz`` file.
        """
        np.savez_compressed(
            path,
            **self.to_numpy_dict(),
        )
        
    def to_json(self) -> Dict[str, Any]:
        """
        Convert the spectral library to a JSON-serializable dictionary.

        NumPy arrays are converted to nested Python lists and metadata is
        represented using standard JSON-compatible types.

        Returns
        -------
        dict
            Dictionary containing the spectral library data and metadata:

            - ``"wavelengths"`` : list of float or None
                Wavelength axis of length ``n_bands``. If no wavelength
                information is available, ``None`` is returned.
            - ``"endmembers"`` : list of list of float
                Endmember matrix of shape ``(n_bands, n_spectra)`` represented as a
                nested list.
            - ``"names"`` : list of str
                Endmember names of length ``n_spectra``.
            - ``"wavelength_unit"`` : str
                Unit associated with the wavelength axis.

        Notes
        -----
        - The returned dictionary contains only JSON-compatible types
          (lists, strings, numbers, booleans, and ``None``).
        - The output can be serialized directly using ``json.dump`` or
          ``json.dumps``.
        """
        def convert(x):
            return x.tolist() if x is not None else None

        return {
            "wavelengths": convert(self.wavelengths),
            "endmembers": convert(self.endmembers),
            "endmember_names": list(self.names),
            "wavelength_unit": self.wavelength_unit,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "SpectralLibrary":
        """
        Create a spectral library from a dictionary.

        Parameters
        ----------
        d : dict
            Dictionary produced by :meth:`to_json` or containing equivalent
            fields.

        Returns
        -------
        SpectralLibrary
            Reconstructed spectral library.
        """
        wavelengths = d.get("wavelengths", None)
        wavelengths = np.asarray(wavelengths) if wavelengths is not None else None
        endmembers = np.asarray(d["endmembers"], dtype=float)

        endmember_names = d.get("endmember_names")
        if endmember_names is not None:
            endmember_names = tuple(endmember_names)

        wavelength_unit = d.get("wavelength_unit", "nm")

        _validate_inputs(
            endmembers,
            wavelengths,
            endmember_names
        )

        return SpectralLibrary(
            wavelengths=wavelengths,
            endmembers=endmembers,
            endmember_names=endmember_names,
            wavelength_unit=wavelength_unit,
        )

    @staticmethod
    def load_npz(path: str) -> "SpectralLibrary":
        """
        Load a spectral library from a NumPy archive.

        Parameters
        ----------
        path : str
            Path to the input ``.npz`` file.

        Returns
        -------
        SpectralLibrary
            Loaded spectral library.
        """
        def decode(x):
            if isinstance(x, np.ndarray) and x.dtype == object and x.shape == ():
                return x.item()
            return x


        with np.load(path, allow_pickle=True) as d:

            wavelengths = decode(d["wavelengths"])
            wavelengths=np.asarray(wavelengths) if wavelengths is not None else None
            endmembers = np.asarray(decode(d["endmembers"]), dtype=float)

            endmember_names = decode(d["endmember_names"])
            if endmember_names is not None:
                endmember_names = tuple(endmember_names)

            wavelength_unit = decode(d["wavelength_unit"]) or "nm"

            _validate_inputs(
                endmembers,
                wavelengths,
                endmember_names
            )

            return SpectralLibrary(
                wavelengths=wavelengths,
                endmembers=endmembers,
                endmember_names=endmember_names,
                wavelength_unit=wavelength_unit,
            )
   
    # ---------------------------- Endmember selection---------------------------------- 
    def _compute_sad_matrix(self, eps: float = 1e-6) -> np.ndarray:
        """
        Compute the pairwise Spectral Angle Distance (SAD) matrix.
        
        Parameters
        ----------
        
        eps: float 
            Numerical stability constant (default: 1e-6)

        Returns
        -------
        ndarray, shape (n_spectra, n_spectra)
            Symmetric SAD matrix in radians.
        """
        X = np.asarray(self.endmembers, dtype=float)
        norms = np.linalg.norm(X, axis=0, keepdims=True)
        norms = np.maximum(norms, eps)
        Xn = X / norms
        cosines = np.clip(Xn.T @ Xn, -1.0, 1.0)
        return np.arccos(cosines)
        
        
    def select_endmembers(self,
        n_endmembers: int,
        min_sad: float = 0.0,
        max_sad: float = np.pi,
        eps: float = 1e-6,
        seed: Optional[int] = None,
        max_restarts: int = 1000,
    ) -> np.ndarray:
        """
        Select endmembers from the spectral library subject to pairwise
        spectral angle distance (SAD) constraints.

        A subset of ``n_endmembers`` spectra is randomly selected such that
        every pair of selected spectra satisfies

        ``min_sad <= SAD(x_i, x_j) <= max_sad``.

        When SAD constraints are active, the library is represented as a graph
        whose vertices correspond to spectra and whose edges connect pairs of
        spectra satisfying the SAD interval constraint. Selection is then
        performed by randomized clique growth with random restarts.

        Parameters
        ----------
        n_endmembers : int
            Number of endmembers to select.

        min_sad : float, default=0
            Minimum allowed spectral angle distance (radians) between any
            pair of selected endmembers.

        max_sad : float, default=pi
            Maximum allowed spectral angle distance (radians) between any
            pair of selected endmembers.

        eps : float, default=1e-6
            Numerical stability constant used during SAD computation.

        seed : int or None, optional
            Seed used to initialize the random number generator.

        max_restarts : int, default=1000
            Maximum number of randomized clique-growth attempts before
            declaring failure.

        Returns
        -------
        np.ndarray of shape (n_bands, n_spectra)
            Matrix whose columns are the selected endmember spectra.
        """
        if n_endmembers > self.n_spectra:
            raise ValueError(
                f"The library only contains {self.n_spectra} endmembers"
            )

        if min_sad > max_sad:
            raise ValueError(
                "min_sad cannot be larger than max_sad"
            )

        rng = np.random.default_rng(seed)

        # Unconstrained random sampling
        if min_sad == 0 and max_sad == np.pi:

            idx = rng.choice(
                self.n_spectra,
                size=n_endmembers,
                replace=False,
            )
            return self.endmembers[:, idx]


        # SAD-constrained selection
        sad_matrix = self._compute_sad_matrix(eps=eps)

        adjacency = (
            (sad_matrix >= min_sad) &
            (sad_matrix <= max_sad)
        )
        np.fill_diagonal(adjacency, False)

        neighbors = [
            set(np.flatnonzero(adjacency[k]))
            for k in range(self.n_spectra)
        ]

        valid_starts = np.flatnonzero(
            adjacency.sum(axis=1) >= n_endmembers - 1
        )

        if len(valid_starts) == 0:
            raise ValueError(
                f"No clique of size {n_endmembers} can exist "
                 "under the current SAD constraints."
            )

        for _ in range(max_restarts):

            v = int(rng.choice(valid_starts))

            clique = [v]
            candidates = neighbors[v].copy()

            while len(clique) < n_endmembers:

                if not candidates:
                    break

                u = int(rng.choice(tuple(candidates)))
                clique.append(u)
                
                candidates.intersection_update(neighbors[u])
                candidates.difference_update(clique)

            if len(clique) == n_endmembers:

                idx = np.array(sorted(clique))
                return self.endmembers[:, idx]

        raise ValueError(
            f"Failed to find a set of {n_endmembers} endmembers satisfying "
            f"the constraints after {max_restarts} random restarts."
        )


# -------------------------------- Bundle library --------------------------------------
@dataclass(frozen=True)
class BundleLibrary:
    """
    A container for endmember bundles: spectral libraries whose endmembers
    are grouped into classes (bundles) capturing intra-class spectral
    variability.

    Internally, a `BundleLibrary` stores a single flat `SpectralLibrary`
    (every bundle concatenated into one endmember matrix sharing one
    wavelength axis) together with a `bundle_ids` mask assigning each
    endmember column to a bundle. This keeps the full endmember matrix
    directly available for downstream use (unmixing, SAD computations)
    while still exposing a per-bundle view.

    Per-spectrum identity (``library.endmember_names``) and per-bundle
    identity (``bundle_names``) are independent: the former names an
    individual spectrum, the latter names the class it belongs to.

    Parameters
    ----------
    library : SpectralLibrary
        Flat spectral library containing every endmember of every bundle.

    bundle_ids : np.ndarray, shape (n_spectra,)
        Bundle membership of each endmember in `library`. Must contain
        contiguous integers starting at 0 (0..n_bundles-1).

    bundle_names : tuple of str, optional
        Optional name for each bundle. If not provided, default names of
        the form "bundle #i" are generated.

    Attributes
    ----------
    n_bands : int
        Number of wavelength channels.

    n_spectra : int
        Total number of endmembers across all bundles.

    n_bundles : int
        Number of bundles.

    bundle_labels : tuple of str
        Resolved names of bundles (user-defined or autogenerated).

    bundle_sizes : tuple of int
        Number of endmembers in each bundle.

    display_names : tuple of str
        Per-endmember display name composed of the bundle name and either
        the endmember's own name (if `library.endmember_names` is set) or
        a local per-bundle index.
    """
    library: SpectralLibrary
    bundle_ids: np.ndarray
    bundle_names: Optional[tuple[str, ...]] = None

    def __post_init__(self):

        bundle_ids = np.asarray(self.bundle_ids)
        _validate_bundle_ids(bundle_ids, self.library.n_spectra, self.bundle_names)

        bundle_ids.setflags(write=False)
        object.__setattr__(self, "bundle_ids", bundle_ids)

    # ------------------------------- Properties ---------------------------------------
    @property
    def n_bands(self) -> int:
        """
        Number of spectral bands (delegates to the underlying library).
        """
        return self.library.n_bands

    @property
    def n_spectra(self) -> int:
        """
        Total number of endmembers across all bundles.
        """
        return self.library.n_spectra

    @property
    def endmembers(self) -> np.ndarray:
        """
        Flat endmember matrix across all bundles (delegates to the underlying library).
        """
        return self.library.endmembers

    @property
    def wavelengths(self) -> Optional[np.ndarray]:
        """
        Wavelength axis (delegates to the underlying library).
        """
        return self.library.wavelengths

    @property
    def wavelength_unit(self) -> str:
        """
        Wavelength unit (delegates to the underlying library).
        """
        return self.library.wavelength_unit

    @property
    def names(self) -> tuple[str, ...]:
        """
        Per-endmember names (delegates to the underlying library).
        """
        return self.library.names

    @property
    def n_bundles(self) -> int:
        """
        Number of bundles.
        """
        return int(self.bundle_ids.max()) + 1

    @property
    def bundle_labels(self) -> tuple[str, ...]:
        """
        Names of the bundles.

        Returns
        -------
        tuple of str
            If `bundle_names` is provided, returns it directly. Otherwise
            returns autogenerated names of the form: "bundle #1", ...,
            "bundle #n_bundles".
        """
        if self.bundle_names is not None:
            return self.bundle_names

        return tuple(f"bundle #{k+1}" for k in range(self.n_bundles))

    @property
    def bundle_sizes(self) -> tuple[int, ...]:
        """Number of endmembers assigned to each bundle."""
        return tuple(
            int(np.sum(self.bundle_ids == k))
            for k in range(self.n_bundles)
        )

    @property
    def display_names(self) -> tuple[str, ...]:
        """
        Per-endmember display names combining bundle and spectrum identity.

        Returns
        -------
        tuple of str
            For each endmember: "<bundle name> / <endmember name>" if
            `library.endmember_names` was supplied, otherwise
            "<bundle name> #<local index>", where the local index counts
            spectra within their own bundle starting at 1.
        """
        labels = self.bundle_labels
        endmember_names = self.library.endmember_names

        if endmember_names is not None:
            return tuple(
                f"{labels[bid]} / {endmember_names[i]}"
                for i, bid in enumerate(self.bundle_ids)
            )

        local_idx = np.zeros(self.n_bundles, dtype=int)
        names = []
        for bid in self.bundle_ids:
            local_idx[bid] += 1
            names.append(f"{labels[bid]} #{local_idx[bid]}")
        return tuple(names)

    # ------------------------------ Construction ---------------------------------------
    @staticmethod
    def from_bundles(
        bundles: list[np.ndarray],
        wavelengths: Optional[np.ndarray] = None,
        bundle_names: Optional[tuple[str, ...]] = None,
        endmember_names: Optional[tuple[str, ...]] = None,
        wavelength_unit: str = "nm",
    ) -> "BundleLibrary":
        """
        Build a bundle library from a list of per-bundle endmember matrices.

        Parameters
        ----------
        bundles : list of np.ndarray
            List of ``n_bundles`` arrays, each of shape ``(n_bands, n_i)``
            where ``n_i`` is the number of spectra in that bundle. All
            bundles must share the same number of bands (matches the
            convention returned by
            :meth:`bundles.base.BundleExtractionBase.extract_bundles`).

        wavelengths : np.ndarray or None, optional
            Optional wavelength axis shared by every bundle.

        bundle_names : tuple of str, optional
            Optional name for each bundle.

        endmember_names : tuple of str, optional
            Optional name for each individual endmember, across all
            bundles (length must equal the total number of endmembers).

        wavelength_unit : str, default="nm"
            Unit of the wavelength axis.

        Returns
        -------
        BundleLibrary
        """
        if len(bundles) == 0:
            raise ValueError("At least one bundle must be provided.")

        arrays = [np.asarray(b, dtype=float) for b in bundles]

        for k, arr in enumerate(arrays):
            if arr.ndim != 2:
                raise ValueError(
                    f"Bundle {k} must be a 2D array (n_bands, n_i), "
                    f"got shape {arr.shape}."
                )

        n_bands = arrays[0].shape[0]
        for k, arr in enumerate(arrays):
            if arr.shape[0] != n_bands:
                raise ValueError(
                    "All bundles must have the same number of bands. "
                    f"Bundle 0 has {n_bands} bands, bundle {k} has {arr.shape[0]}."
                )

        endmembers = np.hstack(arrays)
        bundle_ids = np.concatenate([
            np.full(arr.shape[1], k, dtype=int)
            for k, arr in enumerate(arrays)
        ])

        library = SpectralLibrary(
            endmembers=endmembers,
            wavelengths=wavelengths,
            endmember_names=endmember_names,
            wavelength_unit=wavelength_unit,
        )

        return BundleLibrary(
            library=library,
            bundle_ids=bundle_ids,
            bundle_names=bundle_names,
        )

    def to_bundles(self) -> list[np.ndarray]:
        """
        Split the flat endmember matrix back into a list of per-bundle
        matrices (inverse of `from_bundles`).

        Returns
        -------
        list of np.ndarray
            List of ``n_bundles`` arrays, each of shape ``(n_bands, n_i)``.
        """
        return [
            self.library.endmembers[:, self.bundle_ids == k]
            for k in range(self.n_bundles)
        ]

    def get_bundle(self, k: int) -> SpectralLibrary:
        """
        Extract a single bundle as a standalone spectral library.

        Parameters
        ----------
        k : int
            Bundle id, in ``[0, n_bundles)``.

        Returns
        -------
        SpectralLibrary
            Library containing only the endmembers assigned to bundle ``k``.
        """
        if not (0 <= k < self.n_bundles):
            raise ValueError(f"bundle id {k} out of range [0, {self.n_bundles})")

        mask = self.bundle_ids == k

        endmember_names = None
        if self.library.endmember_names is not None:
            endmember_names = tuple(
                name for name, m in zip(self.library.endmember_names, mask) if m
            )

        return SpectralLibrary(
            endmembers=self.library.endmembers[:, mask],
            wavelengths=self.library.wavelengths,
            endmember_names=endmember_names,
            wavelength_unit=self.library.wavelength_unit,
        )

    # ------------------------------ Bundle sampling ------------------------------------
    def reference_endmembers(self) -> np.ndarray:
        """
        Compute one reference endmember per bundle (the bundle mean spectrum).

        Returns
        -------
        np.ndarray of shape (n_bands, n_bundles)
            Column ``k`` is the mean spectrum of bundle ``k``.
        """
        return np.stack(
            [
                self.library.endmembers[:, self.bundle_ids == k].mean(axis=1)
                for k in range(self.n_bundles)
            ],
            axis=1,
        )

    def sample_variant(self, k: int, seed: Optional[int] = None) -> np.ndarray:
        """
        Draw one random spectrum from a given bundle.

        Parameters
        ----------
        k : int
            Bundle id, in ``[0, n_bundles)``.

        seed : int or None, optional
            Seed used to initialize the random number generator.

        Returns
        -------
        np.ndarray of shape (n_bands,)
            Randomly selected spectrum from bundle ``k``.
        """
        if not (0 <= k < self.n_bundles):
            raise ValueError(f"bundle id {k} out of range [0, {self.n_bundles})")

        rng = np.random.default_rng(seed)
        idx = np.flatnonzero(self.bundle_ids == k)
        return self.library.endmembers[:, rng.choice(idx)]

    # ---------------------------- Serialization ---------------------------------------
    def to_numpy_dict(self) -> Dict[str, Any]:
        """
        NPZ-compatible dictionary.

        Returns
        -------
        dict
            Dictionary containing the underlying library's data/metadata
            (see :meth:`SpectralLibrary.to_numpy_dict`) plus:

            - ``"bundle_ids"`` : ndarray of shape ``(n_spectra,)``
            - ``"bundle_names"`` : ndarray of dtype object, shape ``(n_bundles,)``
        """
        d = self.library.to_numpy_dict()
        d["bundle_ids"] = self.bundle_ids
        d["bundle_names"] = np.array(self.bundle_labels, dtype=object)
        return d

    def save_npz(self, path: str) -> None:
        """
        Save the bundle library to a compressed NumPy archive.

        Parameters
        ----------
        path : str
            Path to the output ``.npz`` file.
        """
        np.savez_compressed(path, **self.to_numpy_dict())

    def to_json(self) -> Dict[str, Any]:
        """
        Convert the bundle library to a JSON-serializable dictionary.

        Returns
        -------
        dict
            Dictionary containing the underlying library's data/metadata
            (see :meth:`SpectralLibrary.to_json`) plus ``"bundle_ids"``
            (list of int) and ``"bundle_names"`` (list of str).
        """
        d = self.library.to_json()
        d["bundle_ids"] = np.asarray(self.bundle_ids).tolist()
        d["bundle_names"] = list(self.bundle_labels)
        return d

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "BundleLibrary":
        """
        Create a bundle library from a dictionary.

        Parameters
        ----------
        d : dict
            Dictionary produced by :meth:`to_json` or containing equivalent
            fields.

        Returns
        -------
        BundleLibrary
        """
        library = SpectralLibrary.from_dict(d)

        bundle_ids = np.asarray(d["bundle_ids"], dtype=int)
        bundle_names = d.get("bundle_names")
        if bundle_names is not None:
            bundle_names = tuple(bundle_names)

        _validate_bundle_ids(bundle_ids, library.n_spectra, bundle_names)

        return BundleLibrary(
            library=library,
            bundle_ids=bundle_ids,
            bundle_names=bundle_names,
        )

    @staticmethod
    def load_npz(path: str) -> "BundleLibrary":
        """
        Load a bundle library from a NumPy archive.

        Parameters
        ----------
        path : str
            Path to the input ``.npz`` file.

        Returns
        -------
        BundleLibrary
        """
        library = SpectralLibrary.load_npz(path)

        def decode(x):
            if isinstance(x, np.ndarray) and x.dtype == object and x.shape == ():
                return x.item()
            return x

        with np.load(path, allow_pickle=True) as d:
            bundle_ids = np.asarray(d["bundle_ids"], dtype=int)
            bundle_names = decode(d["bundle_names"])
            if bundle_names is not None:
                bundle_names = tuple(bundle_names)

        _validate_bundle_ids(bundle_ids, library.n_spectra, bundle_names)

        return BundleLibrary(
            library=library,
            bundle_ids=bundle_ids,
            bundle_names=bundle_names,
        )


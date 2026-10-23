import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Any, Literal
from scipy.io import loadmat, savemat

@dataclass(frozen=True)
class HyperspectralData:
    """
    Container for hyperspectral observations or unmixing results.

    This class stores hyperspectral data and optional outputs of unmixing
    algorithms and supports:
    
    - Raw datasets (observations only)
    - Endmember + abundance
    - Spatially varying endmembers

    Parameters
    ----------
    observations : np.ndarray
        Hyperspectral data of shape (n_bands, n_pixels), where n_bands is the number 
        of spectral bands and n_pixels is the number of pixels.
        
    wavelengths: np.ndarray, optional (n_bands,)
        Wavelengths 

    endmembers : np.ndarray, optional
        Endmember matrix of shape (n_bands, n_endmembers).

    abundances : np.ndarray, optional
        Abundance matrix of shape (n_endmembers, n_pixels).

    per_pixel_endmembers : np.ndarray, optional
        Spatially varying endmembers of shape (n_pixels, n_bands, n_endmembers).
        
    Attributes
    ----------
    n_bands : int
        Number of spectral bands.
    n_pixels : int
        Number of pixels.
    n_endmembers : Optional[int]
        Number of endmembers (if defined).
        
    Notes
    -----
    - Abundances require endmembers, but endmembers may be provided independently.
    - Abundances are defined with respect to endmembers and cannot exist independently.
    - Variable endmembers extend the linear model by introducing spatially varying 
      endmember signatures; they require both endmembers and abundances to be defined.
    """
    observations: np.ndarray
    wavelengths: Optional[np.ndarray] = None
    endmembers: Optional[np.ndarray] = None
    abundances: Optional[np.ndarray] = None
    per_pixel_endmembers: Optional[np.ndarray] = None
    image_shape: Optional[tuple[int, int]] = None
    endmember_names: Optional[tuple[str, ...]] = None
    wavelength_unit: str = "nm"

    # -------------------------- Validation --------------------------------------------
    def __post_init__(self):
    
        # enforce immutability of arrays
        for name in [
            "observations",
            "wavelengths",
            "endmembers",
            "abundances",
            "per_pixel_endmembers",
        ]:
            arr = getattr(self, name)

            if arr is not None:
                arr = np.array(arr, copy=True)
                arr.setflags(write=False)
                object.__setattr__(self, name, arr)
                
        self._validate()
                
        
    def _validate(self) -> None:
        """
        Validate consistency of all stored data.
        """
        self._validate_dtypes()
        n_bands, n_pixels = self._validate_observations()
        self._validate_wavelengths(n_bands)
        self._validate_endmembers(n_bands)
        self._validate_abundances(n_pixels)
        self._validate_image_shape(n_pixels)
        self._validate_endmember_names()
        self._validate_per_pixel_endmembers(n_bands, n_pixels)
        
        
    def _validate_observations(self) -> tuple[int, int]:
        """
        Validate observations array
        
        Returns
        ------
        tuple[int, int]
            shape of the observations array
        """
        if self.observations.ndim != 2:
            raise ValueError("observations must be a 2D array (n_bands, n_pixels)")

        return self.observations.shape


    def _validate_dtypes(self) -> None:
        """
        Validate numerical dtype of stored arrays.
        """
        arrays = {
            "observations": self.observations,
            "wavelengths": self.wavelengths,
            "endmembers": self.endmembers,
            "abundances": self.abundances,
            "per_pixel_endmembers": self.per_pixel_endmembers,
        }

        for name, arr in arrays.items():
            if arr is None:
                continue

            if not np.issubdtype(arr.dtype, np.number):
                raise TypeError(
                    f"{name} must contain numerical values, "
                    f"got dtype={arr.dtype}"
                )
            
                
    def _validate_wavelengths(self, n_bands: int) -> None:
        """
        Validate wavelengths array
        """
        if self.wavelengths is None:
            return

        if self.wavelengths.ndim != 1:
            raise ValueError("wavelengths must be a 1D array (n_bands,)")

        if self.wavelengths.size != n_bands:
            raise ValueError(
                f"wavelengths shape mismatch: expected n_bands={n_bands}, "
                f"got {self.wavelengths.size}"
            )
        
    def _validate_endmembers(self, n_bands: int) -> None:
        """
        Validate endmembers
        """
        if self.endmembers is None:
            return

        if self.endmembers.ndim != 2:
            raise ValueError("endmembers must be a 2D array (n_bands, n_endmembers)")

        L, _ = self.endmembers.shape

        if L != n_bands:
            raise ValueError(
                f"endmembers shape mismatch: expected n_bands={n_bands}, got {L}"
            )
            
        
    def _validate_abundances(self, n_pixels: int) -> None:
        """
        Validate abundances array
        """
        if self.abundances is None:
            return

        if self.endmembers is None:
            raise ValueError(
                "abundances cannot be provided without endmembers"
            )

        if self.abundances.ndim != 2:
            raise ValueError("abundances must be a 2D array (n_endmembers, n_pixels)")

        K, N = self.abundances.shape
        K_ref = self.endmembers.shape[1]

        if N != n_pixels:
            raise ValueError(
                f"abundances shape mismatch: expected n_pixels={n_pixels}, got {N}"
            )

        if K != K_ref:
            raise ValueError(
                f"abundances shape mismatch: expected n_endmembers={K_ref}, got {K}"
            )
        
    def _validate_image_shape(self, n_pixels: int) -> None:
        """
        Validate image shape
        """
        if self.image_shape is None:
            return

        if len(self.image_shape) != 2:
            raise ValueError(
                "image_shape must be a tuple (height, width)"
            )  

        height, width = self.image_shape

        if height <= 0 or width <= 0:
            raise ValueError(
                "image dimensions must be positive"
            )

        if height * width != n_pixels:
            raise ValueError(
                f"image_shape incompatible with n_pixels={n_pixels}: "
                f"{height} * {width} != {n_pixels}"
            )
        
    def _validate_endmember_names(self) -> None:
        """
        Validate endmember names
        """
        if self.endmember_names is None:
            return

        if self.endmembers is None:
            raise ValueError(
                "endmember_names require endmembers"
            )

        K = self.endmembers.shape[1]

        if len(self.endmember_names) != K:
            raise ValueError(
                f"expected {K} names, "
                f"got {len(self.endmember_names)}"
            )
            
    def _validate_per_pixel_endmembers(self, n_bands: int, n_pixels: int) -> None:
        """
        Validate variable endmembers
        """
        if self.per_pixel_endmembers is None:
            return

        if self.endmembers is None:
            raise ValueError(
                "per_pixel_endmembers require endmembers"
            )

        if self.abundances is None:
            raise ValueError(
                "per_pixel_endmembers require abundances"
            )

        if self.per_pixel_endmembers.ndim != 3:
            raise ValueError(
                "per_pixel_endmembers must be a 3D array with shape"
                "(n_pixels, n_bands, n_endmembers)"
            )

        N, L, K = self.per_pixel_endmembers.shape
        if L != n_bands:
            raise ValueError(
                f"per_pixel_endmembers mismatch: expected n_bands={n_bands}, got {L}"
            )

        if N != n_pixels:
            raise ValueError(
                f"per_pixel_endmembers mismatch: expected n_pixels={n_pixels}, got {N}"
            )

        if K != self.endmembers.shape[1]:
            raise ValueError(
                f"K mismatch: endmembers n_endmembers={self.endmembers.shape[1]}, "
                f"per_pixel_endmembers n_endmembers={K}"
            )
                                           
    # ------------------------------- Properties ---------------------------------------
    @property
    def n_bands(self) -> int:
        """
        Returns the spectral dimension
        """
        return self.observations.shape[0]
        
    @property
    def n_pixels(self) -> int:
        """
        Returns the number of observations
        """
        return self.observations.shape[1]

    @property
    def n_endmembers(self) -> Optional[int]:
        """
        Returns the number of endmembers
        """
        if self.endmembers is None:
            return None
        else:
            return self.endmembers.shape[1]
            
    @property
    def height(self) -> Optional[int]:
        """
        Returns the image height
        """
        if self.image_shape is None:
            return None
        
        return self.image_shape[0]

    @property
    def width(self) -> Optional[int]:
        """
        Returns the image width
        """
        if self.image_shape is None:
            return None
        
        return self.image_shape[1]
        
    @property
    def normalized_observations(self) -> tuple[np.ndarray, float]:
        """
        Return the globally normalized hyperspectral observations.

        The observations are converted to floating-point values and divided by
        the maximum value over the entire image cube, resulting in values in the
        range ``[0, 1]`` (assuming the original observations are non-negative).
 
        Returns
        -------
        tuple[np.ndarray, float]
            A tuple ``(normalized_observations, scale_factor)``, where

            - ``normalized_observations`` is a NumPy array of shape
              ``(n_bands, n_pixels)`` containing the normalized observations.
            - ``scale_factor`` is the maximum value used for normalization.

            The original observations can be recovered (up to floating-point
            precision) by multiplying the normalized observations by
            ``scale_factor``.
        """
        observations = self.observations.astype('float', copy=False)
        scale_factor = observations.max()

        if scale_factor <= 0:
            raise ValueError(
                "Cannot normalize observations with non-positive maximum"
            )

        return observations / scale_factor, scale_factor
                        
    @property
    def observations_image(self) -> np.ndarray:
        """
        Return observations as hyperspectral image cube (n_bands, height, width).
        
        Returns
        -------
        np.ndarray
            Observation image data of shape (n_bands, height, width), where 
            n_bands is the number of spectral bands and (height, width) the 
            spatial shape of the image.
        """
        if self.image_shape is None:
            raise ValueError("image_shape is not defined")

        height, width = self.image_shape
        return self.observations.reshape(self.n_bands, height, width)
        
    @property
    def abundance_maps(self) -> np.ndarray:
        """
        Return abundance maps as (n_endmembers, height, width).
        
        Returns
        -------
        np.ndarray
            Abundance maps of shape (n_endmembers, height, width)
        """
        if self.image_shape is None:
            raise ValueError("image_shape is not defined")

        if self.abundances is None:
            raise ValueError("abundances are not defined")

        height, width = self.image_shape
        return self.abundances.reshape(self.n_endmembers, height, width)
        
    @property
    def names(self) -> Optional[tuple[str, ...]]:
        """
        Return the endmember names
        """
        if self.endmembers is None:
            return None

        if self.endmember_names is not None:
            return self.endmember_names

        return tuple(
            f"endmember #{k+1}"
            for k in range(self.n_endmembers)
        )

    @property
    def mode(self) -> Literal["raw", "linear", "variable"]:
        """
        Identify the data representation mode.
        """
        if self.per_pixel_endmembers is not None:
            return "variable"
        if self.endmembers is not None:
            return "linear"
        return "raw"
                    
    # ------------------------- Reconstruction -----------------------------------------
    def reconstruct(self) -> np.ndarray:
        """
        Reconstruct hyperspectral observations from the current model.

        The reconstruction depends on the available representation:

        - If `per_pixel_endmembers` is provided, reconstruction uses spatially
          varying endmembers

        - If `endmembers` is provided, reconstruction uses a linear
          mixing model
        
        Abundances are required for both modes.
        Raw mode (no endmembers) cannot be reconstructed.

        Returns
        -------
        np.ndarray
            Reconstructed data of shape (n_bands, n_pixels)
        """       
        if self.abundances is None:
            raise ValueError(
                "Cannot reconstruct without abundances"
            )
            
        if self.per_pixel_endmembers is not None:
            return np.einsum(
                "nlk,kn->ln", 
                self.per_pixel_endmembers, 
                self.abundances
            )
            
        if self.endmembers is not None:
            return self.endmembers @ self.abundances
            
        raise ValueError("raw data cannot be reconstructed")
    
    # ---------------------------- Serialization ---------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """
        Export the dataset as a JSON-serializable dictionary.

        All NumPy arrays are converted to native Python lists, making the
        returned dictionary compatible with standard serialization tools
        such as ``json.dumps``.

        Returns
        -------
        dict
            Dictionary containing the complete state of the dataset,
            including spectral data, optional unmixing results, and
            associated metadata.

        Notes
        -----
        The output is fully self-contained and can be used to reconstruct
        an equivalent :class:`HyperspectralData` instance via
        :meth:`from_dict`.

        NumPy array types are not preserved and will be converted back
        during deserialization.
        """
        def convert(x):
            return x.tolist() if isinstance(x, np.ndarray) else x

        return {
            "observations": convert(self.observations),
            "wavelengths": convert(self.wavelengths),
            "wavelength_unit": self.wavelength_unit,
            "image_shape": self.image_shape,
            "endmember_names": self.endmember_names,
            "endmembers": convert(self.endmembers),
            "abundances": convert(self.abundances),
            "per_pixel_endmembers": convert(self.per_pixel_endmembers),
        }
        
    def to_mat(self, filename: str) -> None:
        """
        Save the dataset to a MATLAB ``.mat`` file.

        The dataset is exported using :func:`scipy.io.savemat`, preserving NumPy
        arrays without conversion. Metadata such as the wavelength unit,
        endmember names, and image shape are also stored.

        Arrays are stored with the following shapes:

        - ``observations``: ``(n_bands, n_pixels)``
        - ``wavelengths``: ``(n_bands,)``
        - ``endmembers``: ``(n_bands, n_endmembers)``
        - ``abundances``: ``(n_endmembers, n_pixels)``
        - ``per_pixel_endmembers``: ``(n_bands, n_endmembers, n_pixels)``

        Optional attributes whose value is ``None`` are omitted from the file.

        Parameters
        ----------
        filename : str
            Path to the output ``.mat`` file.
        """
        data = {
            "observations": self.observations,
            "wavelengths": self.wavelengths,
            "wavelength_unit": self.wavelength_unit,
            "image_shape": self.image_shape,
            "endmember_names": self.endmember_names,
            "endmembers": self.endmembers,
            "abundances": self.abundances,
            "per_pixel_endmembers": self.per_pixel_endmembers,
        }

        # Remove entries that are None
        data = {k: v for k, v in data.items() if v is not None}
        savemat(filename, data)
        
    @classmethod
    def from_mat(cls, filename: str) -> "HyperspectralData":
        """
        Load a dataset from a MATLAB ``.mat`` file.

        Parameters
        ----------
        filename : str
            Path to a MATLAB ``.mat`` file previously created with :meth:`to_mat`.

        Returns
        -------
        HyperspectralData
            Reconstructed dataset instance.
        """
        data = loadmat(filename, squeeze_me=True)

        def get(key):
            if key not in data:
                return None

            value = data[key]

            if isinstance(value, np.ndarray) and value.size == 0:
                return None

            return value

        image_shape = get("image_shape")
        if image_shape is not None:
            image_shape = tuple(np.asarray(image_shape, dtype=int))

        endmember_names = get("endmember_names")
        if endmember_names is not None:
            endmember_names = tuple(np.atleast_1d(endmember_names).tolist())

        wavelength_unit = get("wavelength_unit")
        if wavelength_unit is None:
            wavelength_unit = "nm"
        else:
            wavelength_unit = str(wavelength_unit)

        return cls(
            observations=np.asarray(data["observations"]),
            wavelengths=get("wavelengths"),
            wavelength_unit=wavelength_unit,
            image_shape=image_shape,
            endmember_names=endmember_names,
            endmembers=get("endmembers"),
            abundances=get("abundances"),
            per_pixel_endmembers=get("per_pixel_endmembers"),
        )
        
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "HyperspectralData":
        """
        Construct a dataset from a serialized dictionary.

        Parameters
        ----------
        d : dict
            Dictionary describing a hyperspectral dataset.

            Required key:

            - ``observations``: array-like of shape (n_bands, n_pixels)

            Optional keys:

            - ``wavelengths``: array-like of shape (n_bands,)
            - ``wavelength_unit``: str
            - ``image_shape``: tuple[int, int]
            - ``endmember_names``: sequence[str]
            - ``endmembers``: array-like of shape (n_bands, n_endmembers)
            - ``abundances``: array-like of shape (n_endmembers, n_pixels)
            - ``per_pixel_endmembers``: array-like of shape
              (n_pixels, n_bands, n_endmembers)

            The dictionary is typically produced by :meth:`to_dict`.

        Returns
        -------
        HyperspectralData
            Reconstructed dataset instance.

        Notes
        -----
        Array-like fields are converted to NumPy arrays. Optional fields
        may be omitted or set to ``None``.
        """
        def array_or_none(x):
            return None if x is None else np.asarray(x)

        image_shape = d.get("image_shape")
        if image_shape is not None:
            image_shape = tuple(image_shape)

        endmember_names = d.get("endmember_names")
        if endmember_names is not None:
            endmember_names = tuple(endmember_names)

        return cls(
            observations=array_or_none(d["observations"]),
            wavelengths=array_or_none(d.get("wavelengths")),
            wavelength_unit=d.get("wavelength_unit", "nm"),
            image_shape=image_shape,
            endmember_names=endmember_names,
            endmembers=array_or_none(d.get("endmembers")),
            abundances=array_or_none(d.get("abundances")),
            per_pixel_endmembers=array_or_none(d.get("per_pixel_endmembers")),
        )

    def save_npz(self, path: str) -> None:
        """
        Save the dataset to a compressed NumPy archive.

        Parameters
        ----------
        path : str
            Destination file path. The ``.npz`` extension is recommended.

        Notes
        -----
        Unlike :meth:`to_dict`, this method preserves NumPy arrays without
        conversion and is therefore more efficient for storing large
        hyperspectral datasets.

        The archive contains all numerical data and metadata required to
        reconstruct the dataset using :meth:`load_npz`.
        """
        np.savez_compressed(
            path,
            wavelengths=self.wavelengths,
            wavelength_unit=self.wavelength_unit,
            image_shape=self.image_shape,
            endmember_names=self.endmember_names,
            observations=self.observations,
            endmembers=self.endmembers,
            abundances=self.abundances,
            per_pixel_endmembers=self.per_pixel_endmembers,
        )

    @classmethod
    def load_npz(cls, path: str) -> "HyperspectralData":
        """
        Load a dataset from a compressed NumPy archive.

        Parameters
        ----------
        path : str
            Path to a file previously created with :meth:`save_npz`.

        Returns
        -------
        HyperspectralData
            Reconstructed dataset instance.

        Notes
        -----
        All stored metadata, including wavelength units, image shape,
        and endmember names, are restored automatically.

        See Also
        --------
        save_npz : Save a dataset to a compressed archive.
        """
        with np.load(path, allow_pickle=True) as d:

            def get_val(key):
            
                if key not in d:
                    return None
                val = d[key]
                if (
                    isinstance(val, np.ndarray) 
                    and val.dtype == object 
                    and val.shape == ()
                ):
                    return val.item()
                    
                return val

            image_shape = get_val("image_shape")
            if image_shape is not None:
                image_shape = tuple(image_shape)

            endmember_names = get_val("endmember_names")
            if endmember_names is not None:
                endmember_names = tuple(endmember_names)

            return cls(
                observations=d["observations"],
                wavelengths=get_val("wavelengths"),
                wavelength_unit=get_val("wavelength_unit") or "nm",
                image_shape=image_shape,
                endmember_names=endmember_names,
                endmembers=get_val("endmembers"),
                abundances=get_val("abundances"),
                per_pixel_endmembers=get_val("per_pixel_endmembers"),
            )
   

import re
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
from scipy.io import loadmat

from .dataset import HyperspectralData
from .metrics import EndmemberAligner

# Maps the canonical dataset name (matching `src/dataset/<name>.npz`) to the
# token used in reference file names under `benchmarks/ref/<algo>_<token>.mat`,
# for the (rare) cases where they differ.
DATASET_ALIASES = {
    "jasper_ridge": "jasper",
}

_REF_FILE_RE = re.compile(r"^(?P<algo>[A-Za-z0-9]+)_(?P<token>.+)\.mat$")


def _normalize_endmember_name(name: str) -> str:
    """
    Turn a dataset endmember name (e.g. "#2 Tree") into a lookup key ("tree").
    """
    name = re.sub(r"^#?\d+\s*", "", str(name)).strip().lower()
    return re.sub(r"\s+", "_", name)


class ReferenceEndmembers:
    """
    Nested-dict container for reference endmember estimates.

    Reference estimates (e.g. Matlab implementations of VCA/MVSA/SISAL,
    stored as ``.mat`` files under ``benchmarks/ref/``) are loaded, aligned
    to each dataset's ground-truth endmembers, and indexed by
    ``[dataset][algorithm][endmember_name]``:

        >>> ref_endmembers = ReferenceEndmembers.load("benchmarks/ref", "src/dataset")
        >>> ref_endmembers["samson"]["mvsa"]["tree"]
        array([...])  # shape (n_bands,)

    ``dataset`` and ``algorithm`` keys are lower-cased. ``endmember_name``
    keys match :attr:`HyperspectralData.names` with the leading index
    stripped and lower-cased (e.g. ``"#2 Tree"`` -> ``"tree"``).
    """

    def __init__(self, data: Dict[str, Dict[str, Dict[str, np.ndarray]]]):
        self._data = data

    # ------------------------------- Loading -------------------------------
    @classmethod
    def load(
        cls,
        ref_dir,
        dataset_dir,
        datasets: Optional[Iterable[str]] = None,
        alignment_metric: str = "SAD",
    ) -> "ReferenceEndmembers":
        """
        Discover and load reference endmember files.

        Parameters
        ----------
        ref_dir : path-like
            Directory containing ``<algorithm>_<dataset>.mat`` files, each
            with a single key ``"M"`` of shape (n_bands, n_endmembers).
        dataset_dir : path-like
            Directory containing the ``<dataset>.npz`` files (as produced by
            :meth:`HyperspectralData.save_npz`), used to fetch ground-truth
            endmembers for alignment.
        datasets : Iterable[str], optional
            If given, restrict loading to these canonical dataset names.
        alignment_metric : str, default="SAD"
            Metric used to align each reference estimate with the
            ground-truth endmembers (see :class:`EndmemberAligner`).

        Returns
        -------
        ReferenceEndmembers
        """
        ref_dir = Path(ref_dir)
        dataset_dir = Path(dataset_dir)
        aligner = EndmemberAligner()
        alias_lookup = {token: name for name, token in DATASET_ALIASES.items()}

        gt_cache: Dict[str, HyperspectralData] = {}
        data: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}

        for matfile in sorted(ref_dir.glob("*.mat")):
            match = _REF_FILE_RE.match(matfile.name)
            if match is None:
                continue

            algo = match.group("algo").lower()
            token = match.group("token").lower()
            dataset = alias_lookup.get(token, token)

            if datasets is not None and dataset not in datasets:
                continue

            if dataset not in gt_cache:
                npz_path = dataset_dir / f"{dataset}.npz"
                if not npz_path.exists():
                    continue
                gt_cache[dataset] = HyperspectralData.load_npz(str(npz_path))

            HSI = gt_cache[dataset]
            if HSI.endmembers is None:
                continue

            M_est = np.asarray(loadmat(matfile)["M"], dtype=float)
            if M_est.shape != HSI.endmembers.shape:
                continue

            M_aligned, _ = aligner.align(
                M_est, HSI.endmembers, alignment_metric=alignment_metric
            )

            per_name = {
                _normalize_endmember_name(name): M_aligned[:, k].copy()
                for k, name in enumerate(HSI.names)
            }
            data.setdefault(dataset, {})[algo] = per_name

        return cls(data)

    # ------------------------------- Access ---------------------------------
    def __getitem__(self, dataset: str):
        return self._data[dataset]

    def __contains__(self, dataset: str) -> bool:
        return dataset in self._data

    def __iter__(self):
        return iter(self._data)

    def __repr__(self) -> str:
        summary = {
            dataset: sorted(algos)
            for dataset, algos in self._data.items()
        }
        return f"ReferenceEndmembers({summary})"

    def datasets(self) -> list[str]:
        """Return the list of datasets with at least one reference available."""
        return sorted(self._data)

    def algorithms(self, dataset: str) -> list[str]:
        """Return the list of algorithms with a reference available for `dataset`."""
        return sorted(self._data.get(dataset, {}))

    def matrix(self, dataset: str, algorithm: str, names: Iterable[str]) -> np.ndarray:
        """
        Reassemble the aligned reference endmembers of `algorithm` on
        `dataset` into a matrix, ordering columns according to `names`
        (as returned by ``HyperspectralData.names``).

        Returns
        -------
        np.ndarray, shape (n_bands, len(names))
        """
        per_name = self._data[dataset][algorithm.lower()]
        columns = [per_name[_normalize_endmember_name(n)] for n in names]
        return np.stack(columns, axis=1)

# Unmixing hyperspectral

Outils Python pour le démélange spectral (*spectral unmixing*) d'images hyperspectrales :
chargement et visualisation de jeux de données de référence, extraction d'endmembers
(VCA), estimation des abondances (FCLS) et évaluation par rapport à la vérité terrain.

## Installation

Python ≥ 3.10. Dépendances :

```bash
pip install numpy scipy pandas matplotlib torch jupyter
```

Le projet n'a pas besoin d'être installé : les notebooks ajoutent la racine du dépôt au
`sys.path` et importent le paquet `src`.

## Notebooks

Les notebooks se lancent depuis la racine du dépôt (les chemins des données sont relatifs
à ce dossier) :

```bash
jupyter lab            # ou : jupyter nbconvert --to notebook --execute --inplace 01_datasets.ipynb
```

| Notebook | Contenu |
|---|---|
| `01_datasets.ipynb` | Présentation des jeux de données réels et affichage des endmembers et des cartes d'abondances de référence. |
| `02_VCA.ipynb` | Vertex Component Analysis sur Samson et Jasper Ridge : extraction des endmembers, abondances par FCLS, métriques (SAD, MSE, SCC…) et figures. |

## Organisation du code

```
src/
├── base.py              # Classes de base BlindUnmixing / SparseUnmixing
├── geometric/           # Méthodes géométriques
│   ├── base.py          #   GeometricUnmixing (fit + estimate_abundances)
│   └── vca.py           #   VCA
├── abundances/          # Solveurs d'abondances (PyTorch)
│   ├── base.py          #   AbundanceSolver
│   └── fcls.py          #   FCLS (moindres carrés totalement contraints, FISTA)
├── projection/          # PCA, SVD tronquée, HySime, estimation du SNR
├── utils/
│   ├── dataset.py       #   HyperspectralData : chargement/sauvegarde .npz, export .mat
│   ├── plotting.py      #   HyperspectralPlotter, EndmemberPlotter, AbundancesMapPlotter, SimplexPlotter
│   ├── metrics.py       #   Métriques, appariement (algorithme hongrois), UnmixingEvaluation
│   ├── library.py       #   SpectralLibrary, BundleLibrary
│   ├── reference.py     #   ReferenceEndmembers
│   ├── models.py        #   Modèles de mélange linéaire et bilinéaire (Fan)
│   ├── endmembers.py    #   Générateurs d'endmembers (constants, bundles, variabilité)
│   ├── abundances.py    #   Générateurs d'abondances (Dirichlet…)
│   ├── noise.py         #   Modèles de bruit (gaussien, par bande, Poisson…)
│   └── outliers.py      #   Génération de pixels aberrants
└── dataset/             # Jeux de données au format .npz
```

## Jeux de données

Tous les fichiers sont dans `src/dataset/`. Les observations sont stockées sous la forme
d'une matrice `(n_bandes, n_pixels)`.

| Fichier | Image | Bandes | Longueurs d'onde (nm) | Endmembers | Abondances de référence |
|---|---|---|---|---|---|
| `samson.npz` | 95 × 95 | 156 | 401–889 | 3 | oui |
| `jasper_ridge.npz` | 100 × 100 | 198 | 399–2457 | 4 | oui |
| `urban4.npz` | 307 × 307 | 162 | 440–2369 | 4 | oui |
| `urban5.npz` | 307 × 307 | 162 | 440–2369 | 5 | oui |
| `urban6.npz` | 307 × 307 | 162 | 440–2369 | 6 | oui |
| `apex.npz` | 110 × 110 | 285 | 413–2420 | 4 | oui |
| `cuprite.npz` | 250 × 190 | 188 | 429–2510 | 12 | non |
| `WDM.npz` | 319 × 292 | 191 | 400–2362 | 6 | oui |
| `synthetic1.npz`, `outliers.npz` | 1024 pixels | 224 | indices de bande | — | oui |
| `USGS_1995_Library.npz` | bibliothèque spectrale USGS | | | | |

## Exemple

```python
import numpy as np
from src.utils.dataset import HyperspectralData
from src.utils.metrics import UnmixingEvaluation
from src.geometric import VCA

HSI = HyperspectralData.load_npz("src/dataset/samson.npz")
Y, scale = HSI.normalized_observations
Y = np.maximum(Y, 0)

model = VCA(n_endmembers=HSI.n_endmembers, rng=np.random.default_rng(0))
model.fit(Y)                    # endmembers_ (L, K), indices_, snr_
model.estimate_abundances(Y)    # abundances_ (K, N), par FCLS

evaluation = UnmixingEvaluation(HSI.endmembers, HSI.abundances, names=HSI.names)
evaluation.add_model("VCA", model, scale=scale)
evaluation.display()
evaluation.plot_abundances(HSI.image_shape)
```

## Remarques

- Sur Jasper Ridge, VCA (graine 0) sélectionne deux spectres proches de l'eau et aucun
  pixel de route : l'endmember « Road » a un SAD d'environ 1,09 rad. C'est une limite de
  l'algorithme (hypothèse de pixel pur), pas un bug.
- WDM : les cartes d'abondances semblent mal nommées (la carte `#5 Trail` correspond
  visiblement à l'eau). Les spectres et les cartes sont cohérents entre eux ; seuls
  les noms sont à vérifier par rapport à la source d'origine.

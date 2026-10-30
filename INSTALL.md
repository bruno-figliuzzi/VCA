# Tutoriel : lancer les notebooks

Ce guide explique, pas à pas, comment installer Python, Jupyter et les librairies
nécessaires, puis comment ouvrir et exécuter les notebooks `01_datasets.ipynb` et
`02_VCA.ipynb`.

---

## 1. Ce qu'il faut installer

| Outil | Rôle |
|---|---|
| **Anaconda** (ou Miniconda) | Installe Python et gère des « environnements » isolés contenant les librairies. |
| **Jupyter** (JupyterLab / Notebook) | Interface web pour ouvrir et exécuter les notebooks `.ipynb`. |
| **Librairies Python** | `numpy`, `scipy`, `pandas`, `matplotlib` (calcul et figures) et `torch` (PyTorch, utilisé pour le calcul des abondances). |

Le fichier `environment.yml` à la racine du dépôt décrit tout cela : une seule commande
suffira à tout installer (étape 4).

---

## 2. Installer Anaconda

1. Aller sur <https://www.anaconda.com/download> et télécharger l'installateur
   correspondant à votre système (Windows, macOS ou Linux).
   - Alternative plus légère : **Miniconda** (<https://docs.anaconda.com/miniconda/>),
     qui contient seulement `conda` et Python. Les commandes ci-dessous sont identiques.
2. Lancer l'installateur :
   - **Windows** : double-cliquer sur le `.exe` et garder les options par défaut
     (installation « Just Me »).
   - **macOS** : double-cliquer sur le `.pkg`.
   - **Linux** : dans un terminal,
     ```bash
     bash ~/Téléchargements/Anaconda3-*-Linux-x86_64.sh
     ```
     Accepter la licence, garder le dossier proposé, puis répondre `yes` à la question
     sur l'initialisation de conda (`conda init`).
3. Ouvrir un terminal :
   - **Windows** : menu Démarrer → **Anaconda Prompt**.
   - **macOS / Linux** : un terminal classique (le fermer puis le rouvrir après
     l'installation).
4. Vérifier que tout fonctionne :
   ```bash
   conda --version
   ```
   Une ligne du type `conda 24.x.x` doit s'afficher.

> Toutes les commandes qui suivent se tapent dans ce terminal (Anaconda Prompt sous Windows).

---

## 3. Récupérer le code

Si vous avez reçu le dossier (clé USB, archive `.zip`…), décompressez-le simplement où
vous voulez. Sinon, avec `git` :

```bash
git clone <adresse-du-dépôt> unmixing
```

Puis placez-vous dans le dossier du projet, celui qui contient `README.md` et les
fichiers `.ipynb` :

```bash
cd chemin/vers/unmixing
```

Astuce : sous Windows, on peut taper `cd ` puis glisser-déposer le dossier dans la fenêtre
pour obtenir son chemin.

---

## 4. Créer l'environnement et installer les librairies

Depuis le dossier du projet :

```bash
conda env create -f environment.yml
```

Cette commande crée un environnement nommé `unmixing` contenant Python 3.13, Jupyter et
toutes les librairies. Elle peut prendre plusieurs minutes (PyTorch pèse plusieurs
centaines de Mo).

Activer ensuite l'environnement :

```bash
conda activate unmixing
```

Le début de la ligne du terminal doit maintenant afficher `(unmixing)`.
**Il faut refaire `conda activate unmixing` à chaque nouveau terminal.**

Vérification rapide :

```bash
python -c "import numpy, scipy, pandas, matplotlib, torch; print('OK')"
```

### Variante sans Anaconda (pip seul)

Avec un Python ≥ 3.10 déjà installé :

```bash
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install numpy scipy pandas matplotlib torch jupyterlab
```

### PyTorch plus léger (optionnel)

Les notebooks n'ont pas besoin de carte graphique. Sous Linux et Windows, on peut
installer la version « CPU seulement » de PyTorch, beaucoup plus légère :

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

---

## 5. Lancer Jupyter

**Important : Jupyter doit être lancé depuis le dossier du projet.** Les notebooks
cherchent le code (`src/`) et les données (`src/dataset/*.npz`) à partir du dossier
courant.

```bash
cd chemin/vers/unmixing
conda activate unmixing
jupyter lab
```

Le navigateur s'ouvre automatiquement sur JupyterLab (sinon, copier dans le navigateur
l'adresse `http://localhost:8888/lab?token=...` affichée dans le terminal).

On peut aussi utiliser l'interface classique, plus simple :

```bash
jupyter notebook
```

Laisser le terminal ouvert tant qu'on travaille : le fermer arrête Jupyter. Pour arrêter
Jupyter proprement, taper `Ctrl + C` dans le terminal.

## Récapitulatif

```bash
# Une seule fois
cd chemin/vers/unmixing
conda env create -f environment.yml

# À chaque session
cd chemin/vers/unmixing
conda activate unmixing
jupyter lab
```

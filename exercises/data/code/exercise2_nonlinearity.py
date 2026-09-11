"""Exercise 2 — Non-linearity in higher dimensions.

Gera dois datasets 5D (gaussianas deslocadas vs. cascas concêntricas),
produz as Figuras 4-5 em ``figures/`` e imprime as métricas (variância
explicada pela PCA e distância entre centros) usadas no relatório.

Uso (a partir da raiz do repositório):

    python docs/exercises/data/code/exercise2_nonlinearity.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

FIGURES = Path(__file__).resolve().parents[1] / "figures"
RNG = np.random.default_rng(42)
N_PER_CLASS = 500
N_DIMS = 5


# Dataset I - gaussianas deslocadas

MEAN_A = [0.0, 0.0, 0.0, 0.0, 0.0]
COV_A = [
    [1.0, 0.8, 0.1, 0.0, 0.0],
    [0.8, 1.0, 0.3, 0.0, 0.0],
    [0.1, 0.3, 1.0, 0.5, 0.0],
    [0.0, 0.0, 0.5, 1.0, 0.2],
    [0.0, 0.0, 0.0, 0.2, 1.0],
]
MEAN_B = [1.5, 1.5, 1.5, 1.5, 1.5]
COV_B = [
    [1.5, -0.7, 0.2, 0.0, 0.0],
    [-0.7, 1.5, 0.4, 0.0, 0.0],
    [0.2, 0.4, 1.5, 0.6, 0.0],
    [0.0, 0.0, 0.6, 1.5, 0.3],
    [0.0, 0.0, 0.0, 0.3, 1.5],
]


def generate_gaussians() -> tuple[np.ndarray, np.ndarray]:
    """500 amostras por classe, normal multivariada 5D (Classe A = 0, Classe B = 1)."""
    XA = RNG.multivariate_normal(MEAN_A, COV_A, size=N_PER_CLASS)
    XB = RNG.multivariate_normal(MEAN_B, COV_B, size=N_PER_CLASS)
    X = np.vstack([XA, XB])
    y = np.concatenate([np.zeros(N_PER_CLASS), np.ones(N_PER_CLASS)])
    return X, y


# Dataset II - cascas concêntricas

RADIUS_C = (2.0, 0.4) 
RADIUS_D = (5.0, 0.4)


def random_directions(n: int, dims: int = N_DIMS) -> np.ndarray:
    """Direções uniformes na esfera unitária: v ~ N(0, I), u = v / ||v||."""
    v = RNG.normal(size=(n, dims))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def generate_shells() -> tuple[np.ndarray, np.ndarray]:
    """500 amostras por classe, cascas radiais 5D (Classe C = 0, Classe D = 1)."""
    u_c = random_directions(N_PER_CLASS)
    rho_c = RNG.normal(*RADIUS_C, size=N_PER_CLASS)
    XC = rho_c[:, None] * u_c

    u_d = random_directions(N_PER_CLASS)
    rho_d = RNG.normal(*RADIUS_D, size=N_PER_CLASS)
    XD = rho_d[:, None] * u_d

    X = np.vstack([XC, XD])
    y = np.concatenate([np.zeros(N_PER_CLASS), np.ones(N_PER_CLASS)])
    return X, y


# PCA, distância entre centros e raio

def pca_2d(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Projeta X em 2D via PCA. Retorna (pontos_2d, variancia_explicada_2_componentes)."""
    pca = PCA(n_components=2, random_state=0)
    X2d = pca.fit_transform(X)
    return X2d, pca.explained_variance_ratio_


def center_distance(X: np.ndarray, y: np.ndarray) -> float:
    """||mu_1 - mu_2|| calculado em 5D, com médias amostrais de cada classe."""
    mu0 = X[y == 0].mean(axis=0)
    mu1 = X[y == 1].mean(axis=0)
    return float(np.linalg.norm(mu0 - mu1))


def radii(X: np.ndarray) -> np.ndarray:
    """||x|| de cada ponto (norma euclidiana das 5 features)."""
    return np.linalg.norm(X, axis=1)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)

    X1, y1 = generate_gaussians()
    X2, y2 = generate_shells()
    datasets = {
        "Dataset I (gaussianas)": (X1, y1, ("Classe A", "Classe B")),
        "Dataset II (cascas)": (X2, y2, ("Classe C", "Classe D")),
    }

    # Figura 4 - PCA 2D lado a lado + variância explicada
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    print("Variancia explicada pelos 2 primeiros componentes da PCA")
    for ax, (name, (X, y, labels)) in zip(axes, datasets.items()):
        X2d, var_ratio = pca_2d(X)
        for c, label in enumerate(labels):
            ax.scatter(*X2d[y == c].T, s=14, alpha=0.7, label=label)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(name)
        ax.legend(loc="best", fontsize=8)
        print(f"  {name:<24} PC1={var_ratio[0]:.3f}  PC2={var_ratio[1]:.3f}  "
              f"soma={var_ratio.sum():.3f}")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig04-pca-2d.png", dpi=150)
    plt.close(fig)

    # Figura 5 - histogramas de raio
    print("\nDistancia entre centros (5D)")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, (name, (X, y, labels)) in zip(axes, datasets.items()):
        dist = center_distance(X, y)
        print(f"  {name:<24} ||mu_1 - mu_2|| = {dist:.3f}")
        r = radii(X)
        for c, label in enumerate(labels):
            ax.hist(r[y == c], bins=30, alpha=0.6, label=label)
        ax.set_xlabel(r"$\|x\|$")
        ax.set_ylabel("contagem")
        ax.set_title(name)
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig05-radius-hist.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
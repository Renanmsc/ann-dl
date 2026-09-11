"""Exercise 1 — Exploring class separability in 2D.

Gera as 4 classes gaussianas do enunciado, salva a figura em ``figures/`` e
imprime as métricas que alimentam a tabela *Results summary* do relatório.

Uso (a partir da raiz do repositório):

    python docs/exercises/data/code/exercise1_point_clouds.py
"""

from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

FIGURES = Path(__file__).resolve().parents[1] / "figures"
RNG = np.random.default_rng(42)

CLASSES = {
    0: {"mean": [2.0, 3.0], "std": [0.8, 2.5]},
    1: {"mean": [5.0, 6.0], "std": [1.2, 1.9]},
    2: {"mean": [8.0, 1.0], "std": [0.9, 0.9]},
    3: {"mean": [15.0, 4.0], "std": [0.5, 2.0]},
}
N_PER_CLASS = 100
SCALES = (0.5, 1.0, 2.0, 4.0)


def generate(scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Amostra 100 pontos por classe, com os desvios multiplicados por ``scale``."""
    xs, ys = [], []
    for label, params in CLASSES.items():
        mean = np.asarray(params["mean"])
        std = np.asarray(params["std"]) * scale
        xs.append(RNG.normal(mean, std, size=(N_PER_CLASS, 2)))
        ys.append(np.full(N_PER_CLASS, label))
    return np.vstack(xs), np.concatenate(ys)


def separation_ratios() -> dict[tuple[int, int], float]:
    """r_ij = ||mu_i - mu_j|| / (sigma_bar_i + sigma_bar_j), para s = 1.

    Usa os parâmetros nominais do enunciado (não estimativas amostrais),
    já que r_ij é definido em cima de mu e sigma da distribuição.
    """
    sigma_bar = {c: np.asarray(params["std"]).mean() for c, params in CLASSES.items()}
    ratios = {}
    for i, j in combinations(CLASSES, 2):
        mean_i = np.asarray(CLASSES[i]["mean"])
        mean_j = np.asarray(CLASSES[j]["mean"])
        dist = np.linalg.norm(mean_i - mean_j)
        ratios[(i, j)] = dist / (sigma_bar[i] + sigma_bar[j])
    return ratios


def mixing_rate(X: np.ndarray, y: np.ndarray) -> float:
    """Fração de pontos cujo centro de classe mais próximo não é o da própria classe.

    Puramente geométrico: compara cada ponto às 4 médias fixas, sem treinar nada.
    """
    means = np.stack([np.asarray(CLASSES[c]["mean"]) for c in CLASSES])
    dists = np.linalg.norm(X[:, None, :] - means[None, :, :], axis=2)
    nearest = dists.argmin(axis=1)
    return float((nearest != y).mean())


def plot_clouds(ax: plt.Axes, X: np.ndarray, y: np.ndarray, title: str) -> None:
    """Desenha o scatter colorido por classe com o centro (média) marcado."""
    for c in CLASSES:
        ax.scatter(*X[y == c].T, s=14, alpha=0.75, label=f"Classe {c}")
        mean = np.asarray(CLASSES[c]["mean"])
        ax.scatter(
            *mean, marker="X", s=140, c="black",
            edgecolors="white", linewidths=1.2, zorder=5,
        )
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title(title)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)

    # Figura 1 - Scatter com centros marcados
    X1, y1 = generate(1.0)
    fig, ax = plt.subplots(figsize=(7, 5))
    plot_clouds(ax, X1, y1, "Nuvens de pontos gaussianas (s = 1.0)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig01-point-clouds.png", dpi=150)
    plt.close(fig)  # (2)!

    # Figura 2 - Grid 2x2, eixos compartilhados
    datasets = {s: generate(s) for s in SCALES}
    all_X = np.vstack([X for X, _ in datasets.values()])
    pad = 1.0
    xlim = (all_X[:, 0].min() - pad, all_X[:, 0].max() + pad)
    ylim = (all_X[:, 1].min() - pad, all_X[:, 1].max() + pad)

    fig, axes = plt.subplots(2, 2, figsize=(11, 9), sharex=True, sharey=True)
    for ax, s in zip(axes.flat, SCALES):
        Xs, ys = datasets[s]
        plot_clouds(ax, Xs, ys, f"s = {s}")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
    axes.flat[0].legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig02-scale-grid.png", dpi=150)
    plt.close(fig)

    # Razão de separação r_ij, apenas para s = 1
    ratios = separation_ratios()
    print("\nRazao de separacao r_ij (s = 1.0)")
    print(f"{'par (i,j)':<12}{'r_ij':>10}{'r_ij em s=2':>16}")
    for (i, j), r in sorted(ratios.items(), key=lambda kv: kv[1]):
        print(f"({i}, {j}){'':<6}{r:>10.3f}{r / 2:>16.3f}")
    min_pair, min_r = min(ratios.items(), key=lambda kv: kv[1])
    print(f"\nMenor r_ij: par {min_pair} = {min_r:.3f}  ->  em s=2 (r_ij escala com 1/s): {min_r / 2:.3f}")

    rates = {s: mixing_rate(*datasets[s]) for s in SCALES}
    print("\nTaxa de mistura por s")
    for s, rate in rates.items():
        print(f"s = {s:<5} mixing rate = {rate:.3f}")

    # Figura 3 - Mixing rate
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(SCALES, [rates[s] for s in SCALES], marker="o")
    ax.set_xlabel("fator de escala s")
    ax.set_ylabel("taxa de mistura")
    ax.set_title("Taxa de mistura vs. fator de escala")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig03-mixing-rate.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
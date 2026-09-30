"""Perceptron: Exercício 1 (dados separáveis) e Exercício 2 (dados sobrepostos).

Execute com:  python exercise.py
Gera as Figuras 1 a 6 em ../figures/ e results.json em ../figures/.
Apenas NumPy e Matplotlib. O perceptron é implementado do zero.
"""

import json
from pathlib import Path
from typing import Callable

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SEED = 42
LEARNING_RATE = 0.01
MAX_EPOCHS = 100
FIGURES_DIR = Path(__file__).resolve().parents[1] / "figures"


# ---------------------------------------------------------------------------
# Dados
# ---------------------------------------------------------------------------
def generate_separable_data(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Generate the two classes for Exercise 1."""
    class_0 = rng.multivariate_normal(
        mean=[1.5, 1.5], cov=[[0.5, 0.0], [0.0, 0.5]], size=1000
    )
    class_1 = rng.multivariate_normal(
        mean=[5.0, 5.0], cov=[[0.5, 0.0], [0.0, 0.5]], size=1000
    )
    features = np.vstack((class_0, class_1))
    labels = np.concatenate((np.zeros(1000, dtype=int), np.ones(1000, dtype=int)))
    return features, labels


def generate_overlapping_data(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Generate the two classes for Exercise 2."""
    covariance = [[1.5, 0.0], [0.0, 1.5]]
    class_0 = rng.multivariate_normal(mean=[3.0, 3.0], cov=covariance, size=1000)
    class_1 = rng.multivariate_normal(mean=[4.0, 4.0], cov=covariance, size=1000)
    features = np.vstack((class_0, class_1))
    labels = np.concatenate((np.zeros(1000, dtype=int), np.ones(1000, dtype=int)))
    return features, labels


# ---------------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------------
def step(values: np.ndarray) -> np.ndarray:
    """Apply the 0/1 step activation specified in the assignment."""
    return (values >= 0).astype(int)


def predict(features: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
    """Predict class labels for a batch of points."""
    return step(features @ weights + bias)


def accuracy(features: np.ndarray, labels: np.ndarray, weights: np.ndarray, bias: float) -> float:
    """Return the fraction of correctly classified points."""
    return float(np.mean(predict(features, weights, bias) == labels))


def _epoch_pass(
    features: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    bias: float,
    learning_rate: float,
    on_update: Callable[[np.ndarray, float], None] | None = None,
) -> tuple[np.ndarray, float, int]:
    """Uma época: percorre os pontos na ordem do dataset e aplica o perceptron.

    w <- w + eta (y - y_hat) x ;  b <- b + eta (y - y_hat)
    `on_update`, quando fornecido, é chamado após cada correção. Isso permite que
    o treino com pocket avalie cada novo par (w, b) sem duplicar a regra de
    atualização usada no Exercício 1.
    """
    updates = 0
    for x, y in zip(features, labels):
        y_hat = 1 if (weights @ x + bias) >= 0 else 0
        error = y - y_hat
        if error != 0:
            weights = weights + learning_rate * error * x
            bias = bias + learning_rate * error
            updates += 1
            if on_update is not None:
                on_update(weights, bias)
    return weights, bias, updates


def train_perceptron(
    features: np.ndarray,
    labels: np.ndarray,
    rng: np.random.Generator,
    learning_rate: float = LEARNING_RATE,
    max_epochs: int = MAX_EPOCHS,
    init_weights: np.ndarray | None = None,
) -> dict[str, object]:
    """Train the single-layer perceptron.

    Se init_weights for None, w ~ N(0, 0.01) usando rng; caso contrário usa a cópia
    fornecida (útil para comparar taxas com a mesma inicialização). b começa em 0.
    Os pontos são percorridos na ordem do dataset. Para se uma época completa
    terminar sem nenhuma atualização.
    """
    if init_weights is None:
        weights = rng.normal(0, 0.01, size=2)
    else:
        weights = np.array(init_weights, dtype=float)
    bias = 0.0
    history: list[float] = []
    updates_per_epoch: list[int] = []

    for _ in range(max_epochs):
        weights, bias, updates = _epoch_pass(features, labels, weights, bias, learning_rate)
        updates_per_epoch.append(updates)
        history.append(accuracy(features, labels, weights, bias))
        if updates == 0:
            break

    return {
        "weights": weights,
        "bias": bias,
        "history": history,
        "updates_per_epoch": updates_per_epoch,
        "epochs": len(history),
        "converged": updates_per_epoch[-1] == 0,
    }


def train_with_pocket(
    features: np.ndarray,
    labels: np.ndarray,
    rng: np.random.Generator,
    learning_rate: float = LEARNING_RATE,
    max_epochs: int = MAX_EPOCHS,
    init_weights: np.ndarray | None = None,
) -> dict[str, object]:
    """Mesmo treino do perceptron, guardando os melhores parâmetros (pocket).

    Depois de cada atualização, se a acurácia no conjunto completo superar a
    melhor já registrada, guarda cópias de w, b e a época (contada a partir de 1).
    """
    if init_weights is None:
        weights = rng.normal(0, 0.01, size=2)
    else:
        weights = np.array(init_weights, dtype=float)
    bias = 0.0
    history: list[float] = []
    best_history: list[float] = []
    updates_per_epoch: list[int] = []

    best_acc = accuracy(features, labels, weights, bias)
    best_w, best_b, best_epoch = weights.copy(), bias, 0

    for epoch in range(1, max_epochs + 1):
        def update_pocket(current_w: np.ndarray, current_b: float) -> None:
            nonlocal best_acc, best_w, best_b, best_epoch
            current_acc = accuracy(features, labels, current_w, current_b)
            if current_acc > best_acc:
                best_acc = current_acc
                best_w = current_w.copy()
                best_b = current_b
                best_epoch = epoch

        weights, bias, updates = _epoch_pass(
            features,
            labels,
            weights,
            bias,
            learning_rate,
            on_update=update_pocket,
        )
        acc = accuracy(features, labels, weights, bias)
        updates_per_epoch.append(updates)
        history.append(acc)
        best_history.append(best_acc)
        if updates == 0:
            break

    return {
        "weights": weights,
        "bias": bias,
        "final_accuracy": history[-1],
        "history": history,
        "best_history": best_history,
        "updates_per_epoch": updates_per_epoch,
        "epochs": len(history),
        "pocket_weights": best_w,
        "pocket_bias": best_b,
        "pocket_accuracy": best_acc,
        "pocket_epoch": best_epoch,
    }


# ---------------------------------------------------------------------------
# Gráficos
# ---------------------------------------------------------------------------
def plot_scatter(features: np.ndarray, labels: np.ndarray, title: str, filename: str) -> None:
    """Save a labeled scatter plot for one dataset."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for class_id, class_name in ((0, "Classe 0"), (1, "Classe 1")):
        points = features[labels == class_id]
        plt.scatter(points[:, 0], points[:, 1], label=class_name, alpha=0.65)
    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=160)
    plt.close()


def _draw_boundary(ax, features, w, b, label, style):
    """Desenha w.x + b = 0 no intervalo de x1 dos dados."""
    x1 = np.array([features[:, 0].min() - 0.5, features[:, 0].max() + 0.5])
    if abs(w[1]) > 1e-12:
        ax.plot(x1, -(w[0] * x1 + b) / w[1], style, lw=2, label=label)
    else:
        ax.axvline(-b / w[0], color=style[0], ls="--", lw=2, label=label)


def plot_boundaries(features, labels, params, title, filename):
    """Desenha cada fronteira e seus erros; comparações usam painéis separados."""
    n_panels = len(params)
    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=(7 * n_panels, 6),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for ax, (w, b, label, style) in zip(axes[0], params):
        for class_id, class_name in ((0, "Classe 0"), (1, "Classe 1")):
            pts = features[labels == class_id]
            ax.scatter(pts[:, 0], pts[:, 1], label=class_name, alpha=0.35, s=14)
        wrong = predict(features, w, b) != labels
        ax.scatter(
            features[wrong, 0],
            features[wrong, 1],
            marker="x",
            c="black",
            s=25,
            linewidths=0.8,
            label=f"Erros ({wrong.sum()})",
        )
        _draw_boundary(ax, features, w, b, "Fronteira de decisão", style)
        ax.set_xlim(features[:, 0].min() - 0.5, features[:, 0].max() + 0.5)
        ax.set_ylim(features[:, 1].min() - 0.5, features[:, 1].max() + 0.5)
        ax.set_title(f"{label} — acurácia {1 - wrong.mean():.2%}")
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.legend(loc="upper left", fontsize=8)
    if n_panels == 1:
        axes[0, 0].set_title(title)
    else:
        fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.96) if n_panels > 1 else None)
    fig.savefig(FIGURES_DIR / filename, dpi=160)
    plt.close(fig)


def plot_accuracy(curves, title, filename, ylim=None):
    """curves: lista de (valores por época, rótulo)."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for values, label in curves:
        ax.plot(range(1, len(values) + 1), values, marker="o", ms=3, label=label)
    ax.set_title(title)
    ax.set_xlabel("Época")
    ax.set_ylabel("Acurácia no conjunto completo")
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename, dpi=160)
    plt.close(fig)


def direction(w: np.ndarray) -> np.ndarray:
    return w / np.linalg.norm(w)


# ---------------------------------------------------------------------------
# Experimentos
# ---------------------------------------------------------------------------
def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)  # único gerador de todo o relatório
    res: dict[str, object] = {}

    # ----- Exercício 1 -----
    X1, y1 = generate_separable_data(rng)
    plot_scatter(X1, y1, "Figura 1: dados separáveis (Exercício 1)", "fig01-separable-data.png")

    w0 = rng.normal(0, 0.01, size=2)  # mesma inicialização para eta = 0.01 e eta = 1.0
    r1 = train_perceptron(X1, y1, rng, learning_rate=0.01, init_weights=w0)
    wrong1 = predict(X1, r1["weights"], r1["bias"]) != y1
    plot_boundaries(
        X1, y1, [(r1["weights"], r1["bias"], "perceptron η=0,01", "k-")],
        "Figura 2: fronteira de decisão e erros (Exercício 1)", "fig02-decision-boundary-ex1.png",
    )
    plot_accuracy(
        [(r1["history"], "η = 0,01")],
        "Figura 3: acurácia por época (Exercício 1)", "fig03-accuracy-ex1.png", ylim=(0.45, 1.02),
    )

    r1b = train_perceptron(X1, y1, rng, learning_rate=1.0, init_weights=w0)
    cos = float(direction(r1["weights"]) @ direction(r1b["weights"]))

    # Q3: w = 0, b = 0 com várias taxas: pesos proporcionais
    zeros = np.zeros(2)
    z_a = train_perceptron(X1, y1, rng, learning_rate=0.01, init_weights=zeros)
    z_b = train_perceptron(X1, y1, rng, learning_rate=1.0, init_weights=zeros)
    z_c = train_perceptron(X1, y1, rng, learning_rate=0.5, init_weights=zeros)

    res["ex1"] = {
        "w0": w0.tolist(),
        "w": r1["weights"].tolist(), "b": r1["bias"], "epochs": r1["epochs"],
        "converged": r1["converged"], "accuracy": r1["history"][-1],
        "history": r1["history"], "updates_per_epoch": r1["updates_per_epoch"],
        "n_wrong": int(wrong1.sum()), "total_updates": int(sum(r1["updates_per_epoch"])),
        "eta1": {
            "w": r1b["weights"].tolist(), "b": r1b["bias"], "epochs": r1b["epochs"],
            "accuracy": r1b["history"][-1], "updates_per_epoch": r1b["updates_per_epoch"],
            "total_updates": int(sum(r1b["updates_per_epoch"])),
            "dir": direction(r1b["weights"]).tolist(), "cos_with_eta001": cos,
        },
        "dir_eta001": direction(r1["weights"]).tolist(),
        "zero_init": {
            name: {
                "w": r["weights"].tolist(), "b": r["bias"], "epochs": r["epochs"],
                "acc": r["history"][-1], "updates": r["updates_per_epoch"],
            }
            for name, r in (("0.01", z_a), ("1.0", z_b), ("0.5", z_c))
        },
        "zero_ratio_w_1_over_001": (z_b["weights"] / z_a["weights"]).tolist(),
        "zero_ratio_b_1_over_001": z_b["bias"] / z_a["bias"],
    }

    # ----- Exercício 2 -----
    X2, y2 = generate_overlapping_data(rng)
    plot_scatter(X2, y2, "Figura 4: dados sobrepostos (Exercício 2)", "fig04-overlapping-data.png")

    r2 = train_with_pocket(X2, y2, rng, learning_rate=0.01, max_epochs=MAX_EPOCHS)
    plot_boundaries(
        X2, y2,
        [
            (r2["weights"], r2["bias"], "pesos finais", "k-"),
            (r2["pocket_weights"], r2["pocket_bias"], f"pocket (época {r2['pocket_epoch']})", "r--"),
        ],
        "Figura 5: fronteiras final e pocket (Exercício 2)", "fig05-boundaries-ex2.png",
    )
    plot_accuracy(
        [(r2["history"], "Pesos atuais"), (r2["best_history"], "Melhor acurácia (pocket)")],
        "Figura 6: acurácia por época (Exercício 2)", "fig06-accuracy-ex2.png", ylim=(0.4, 0.8),
    )

    # Análise Q1/Q3 do Ex. 2
    norms = np.linalg.norm(X2, axis=1)
    fin_pred = predict(X2, r2["weights"], r2["bias"])
    # Extras: taxa menor e mais épocas (w = 0 para tornar a comparação exata)
    ex_small = train_with_pocket(X2, y2, rng, learning_rate=0.001, max_epochs=100, init_weights=zeros)
    ex_ref = train_with_pocket(X2, y2, rng, learning_rate=0.01, max_epochs=100, init_weights=zeros)
    ex_long = train_with_pocket(X2, y2, rng, learning_rate=0.01, max_epochs=1000, init_weights=zeros)
    res["ex2"] = {
        "w": r2["weights"].tolist(), "b": r2["bias"], "final_accuracy": r2["final_accuracy"],
        "pocket_w": r2["pocket_weights"].tolist(), "pocket_b": r2["pocket_bias"],
        "pocket_accuracy": r2["pocket_accuracy"], "pocket_epoch": r2["pocket_epoch"],
        "epochs": r2["epochs"], "history": r2["history"], "best_history": r2["best_history"],
        "updates_per_epoch": r2["updates_per_epoch"],
        "mean_norm_x": float(norms.mean()), "median_norm_x": float(np.median(norms)),
        "final_frac_pred_1": float(fin_pred.mean()),
        "final_n_wrong": int((fin_pred != y2).sum()),
        "final_dist_origin": float(-r2["bias"] / np.linalg.norm(r2["weights"])),
        "pocket_dist_origin": float(-r2["pocket_bias"] / np.linalg.norm(r2["pocket_weights"])),
        "acc_min": float(min(r2["history"])), "acc_max": float(max(r2["history"])),
        "acc_std": float(np.std(r2["history"])),
        "extra": {
            name: {
                "pocket_accuracy": r["pocket_accuracy"], "pocket_epoch": r["pocket_epoch"],
                "final_accuracy": r["final_accuracy"], "epochs": r["epochs"],
                "w": r["weights"].tolist(), "b": r["bias"],
            }
            for name, r in (("eta0.001_100ep", ex_small), ("eta0.01_100ep", ex_ref), ("eta0.01_1000ep", ex_long))
        },
        "extra_dir_ratio": float(
            direction(ex_small["pocket_weights"]) @ direction(ex_ref["pocket_weights"])
        ),
    }

    with open(FIGURES_DIR / "results.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

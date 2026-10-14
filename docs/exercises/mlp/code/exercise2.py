"""Exercício 2: classificação binária que uma reta não resolve.

Contém o MLP escrito do zero (classe `MLP`), reutilizado sem alterações nos
Exercícios 3 e 4 (`from exercise2 import MLP`). Execute com:  python exercise2.py
Gera as Figuras 1 a 3 em ../figures/ e results_ex2.json.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

SEED = 42
FIGURES_DIR = Path(__file__).resolve().parents[1] / "figures"

HIDDEN_SIZES = [8]
LEARNING_RATE = 0.5
EPOCHS = 2000


def sigmoid(u: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-u))


def softmax(u: np.ndarray) -> np.ndarray:
    e = np.exp(u - u.max(axis=1, keepdims=True))  
    return e / e.sum(axis=1, keepdims=True)


def binary_cross_entropy(P: np.ndarray, Y: np.ndarray) -> float:
    P = np.clip(P, 1e-12, 1 - 1e-12)
    return float(-np.mean(Y * np.log(P) + (1 - Y) * np.log(1 - P)))


def categorical_cross_entropy(P: np.ndarray, Y: np.ndarray) -> float:
    P = np.clip(P, 1e-12, 1.0)
    return float(-np.mean(np.sum(Y * np.log(P), axis=1)))


# MLP do zero
class MLP:
    """MLP totalmente conectado com ativação tanh nas camadas ocultas.

    layer_sizes: [n_entrada, h1, ..., hk, n_saida]  (número arbitrário de camadas)
    output_activation: sigmoid (binário) ou softmax (multiclasse)
    loss: perda correspondente (entropia cruzada binária ou categórica)

    Para os dois pares (sigmoide + entropia cruzada binária) e (softmax + entropia
    cruzada categórica), dL/du na saída vale (P - Y) / N. Por isso a passagem
    reversa é exatamente a mesma nos dois casos.
    """

    def __init__(self, layer_sizes, rng, output_activation=sigmoid, loss=binary_cross_entropy):
        self.layer_sizes = list(layer_sizes)
        self.output_activation = output_activation
        self.loss = loss
        self.W, self.b = [], []
        for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            # inicialização de Xavier: N(0, 1/n_in)
            self.W.append(rng.normal(0.0, np.sqrt(1.0 / n_in), size=(n_in, n_out)))
            self.b.append(np.zeros(n_out))

    def n_params(self) -> int:
        return sum(W.size + b.size for W, b in zip(self.W, self.b))

    # ----- passagem direta -----
    def forward(self, X):
        """Retorna a lista de ativações [A0=X, A1, ..., AL=P]."""
        activations = [X]
        A = X
        for l in range(len(self.W)):
            U = A @ self.W[l] + self.b[l]
            A = np.tanh(U) if l < len(self.W) - 1 else self.output_activation(U)
            activations.append(A)
        return activations

    # ----- passagem reversa -----
    def backward(self, activations, Y):
        """Gradientes de L em relação a cada W e b, pela regra da cadeia."""
        N = Y.shape[0]
        grads_W = [None] * len(self.W)
        grads_b = [None] * len(self.W)
        delta = (activations[-1] - Y) / N         
        for l in reversed(range(len(self.W))):
            grads_W[l] = activations[l].T @ delta     
            grads_b[l] = delta.sum(axis=0)          
            if l > 0:
                delta = (delta @ self.W[l].T) * (1.0 - activations[l] ** 2)  
        return grads_W, grads_b

    def update(self, grads_W, grads_b, learning_rate):
        for l in range(len(self.W)):
            self.W[l] -= learning_rate * grads_W[l]
            self.b[l] -= learning_rate * grads_b[l]

    def fit(self, X, Y, learning_rate, epochs):
        """Gradiente descendente em lote completo. Retorna a perda de treino por época."""
        losses = []
        for _ in range(epochs):
            activations = self.forward(X)
            losses.append(self.loss(activations[-1], Y))
            grads_W, grads_b = self.backward(activations, Y)
            self.update(grads_W, grads_b, learning_rate)
        return losses

    def predict_proba(self, X):
        return self.forward(X)[-1]


def predict_binary(model: MLP, X: np.ndarray) -> np.ndarray:
    return (model.predict_proba(X)[:, 0] >= 0.5).astype(int)


# Dados e experimento
def make_data():
    centers = np.array([[0.0, 0.0], [3.0, 3.0], [-3.0, -3.0]])
    X, c = make_blobs(n_samples=[500, 250, 250], centers=centers, cluster_std=1.2, random_state=42)
    y = (c > 0).astype(int)  # classe 0: 1 grupo | classe 1: 2 grupos
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    X_train, X_test, y_train, y_test = make_data()

    # Figura 1: dispersão dos 1000 pontos
    X_all = np.vstack((X_train, X_test))
    y_all = np.concatenate((y_train, y_test))
    plt.figure(figsize=(6.5, 5.5))
    for cls, name in ((0, "Classe 0"), (1, "Classe 1")):
        pts = X_all[y_all == cls]
        plt.scatter(pts[:, 0], pts[:, 1], s=14, alpha=0.65, label=name)
    plt.title("Figura 1: dados do Exercício 2 (1000 pontos)")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig01-data-ex2.png", dpi=160)
    plt.close()

    # B: linha de base linear
    baseline = LogisticRegression().fit(X_train, y_train)
    baseline_acc = float(baseline.score(X_test, y_test))
    print(f"Linha de base (regressão logística), acurácia no teste: {baseline_acc:.4f}")

    # C: MLP
    sizes = [2] + HIDDEN_SIZES + [1]
    model = MLP(sizes, rng, output_activation=sigmoid, loss=binary_cross_entropy)
    losses = model.fit(X_train, y_train.reshape(-1, 1).astype(float), LEARNING_RATE, EPOCHS)
    train_acc = float(np.mean(predict_binary(model, X_train) == y_train))
    test_acc = float(np.mean(predict_binary(model, X_test) == y_test))
    print(f"MLP {sizes}: lr={LEARNING_RATE}, épocas={EPOCHS}, perda final={losses[-1]:.4f}")
    print(f"acurácia treino={train_acc:.4f}  teste={test_acc:.4f}  parâmetros={model.n_params()}")

    # Figura 2: perda de treino x época
    plt.figure(figsize=(7, 4.5))
    plt.plot(range(1, EPOCHS + 1), losses, label="Perda de treino (entropia cruzada binária)")
    plt.title("Figura 2: perda de treino por época (Exercício 2)")
    plt.xlabel("Época")
    plt.ylabel("Perda")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig02-loss-ex2.png", dpi=160)
    plt.close()

    # Figura 3: fronteira de decisão sobre os pontos de teste
    gx, gy = np.meshgrid(
        np.linspace(X_all[:, 0].min() - 1, X_all[:, 0].max() + 1, 400),
        np.linspace(X_all[:, 1].min() - 1, X_all[:, 1].max() + 1, 400),
    )
    grid = np.c_[gx.ravel(), gy.ravel()]
    zz = predict_binary(model, grid).reshape(gx.shape)
    plt.figure(figsize=(6.5, 5.5))
    plt.contourf(gx, gy, zz, levels=[-0.5, 0.5, 1.5], colors=["#9ecae1", "#fdd0a2"], alpha=0.6)
    for cls, name, color in ((0, "Classe 0 (teste)", "tab:blue"), (1, "Classe 1 (teste)", "tab:orange")):
        pts = X_test[y_test == cls]
        plt.scatter(pts[:, 0], pts[:, 1], s=16, color=color, edgecolor="k", linewidth=0.3, label=name)
    plt.title("Figura 3: fronteira de decisão do MLP (Exercício 2)")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig03-decision-boundary-ex2.png", dpi=160)
    plt.close()

    frac_class1 = float(zz.mean())
    base_pred = baseline.predict(X_test)
    results = {
        "baseline_acc": baseline_acc,
        "baseline_frac_pred_1": float(base_pred.mean()),
        "baseline_coef": baseline.coef_.ravel().tolist(),
        "baseline_intercept": float(baseline.intercept_[0]),
        "sizes": sizes, "lr": LEARNING_RATE, "epochs": EPOCHS,
        "n_params": model.n_params(),
        "final_loss": losses[-1], "loss_at": {str(e): losses[e - 1] for e in (1, 100, 500, 1000, 2000)},
        "train_acc": train_acc, "test_acc": test_acc,
        "frac_grid_class1": frac_class1,
        "n_train": int(len(y_train)), "n_test": int(len(y_test)),
        "test_class_counts": [int((y_test == 0).sum()), int((y_test == 1).sum())],
    }
    with open(FIGURES_DIR / "results_ex2.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()

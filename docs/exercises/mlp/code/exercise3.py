"""Exercício 3: multiclasse, a mesma rede.

Reutiliza literalmente a classe `MLP` do Exercício 2 (passagem direta, reversa e
atualização intocadas). Mudam apenas argumentos: tamanho da saída (3), ativação
de saída (softmax), perda (entropia cruzada categórica) e hiperparâmetros.
Execute com:  python exercise3.py
Gera as Figuras 4 e 5 em ../figures/ e results_ex3.json.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from exercise2 import MLP, categorical_cross_entropy, softmax

SEED = 42
FIGURES_DIR = Path(__file__).resolve().parents[1] / "figures"

# Hiperparâmetros (também usados no Exercício 4: mesmo orçamento de treino)
HIDDEN_SIZES = [16]
LEARNING_RATE = 0.5
EPOCHS = 2000
N_CLASSES = 3


def make_data():
    """Gera os dados, divide 80/20 estratificado e padroniza com estatísticas do treino."""
    X, y = make_classification(
        n_samples=1500, n_features=4, n_informative=4, n_redundant=0, n_repeated=0,
        n_classes=3, n_clusters_per_class=2, class_sep=1.0, random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    mean, std = X_train.mean(axis=0), X_train.std(axis=0)
    return (X_train - mean) / std, (X_test - mean) / std, y_train, y_test


def train_and_eval(hidden_sizes, init_seed, X_train, y_train, X_test, y_test,
                   learning_rate=LEARNING_RATE, epochs=EPOCHS):
    """Treina um MLP multiclasse e devolve (modelo, perdas por época, acurácia de teste)."""
    sizes = [X_train.shape[1]] + list(hidden_sizes) + [N_CLASSES]
    model = MLP(sizes, np.random.default_rng(init_seed),
                output_activation=softmax, loss=categorical_cross_entropy)
    Y_train = np.eye(N_CLASSES)[y_train]  # rótulos one-hot
    losses = model.fit(X_train, Y_train, learning_rate, epochs)
    pred = model.predict_proba(X_test).argmax(axis=1)
    return model, losses, float(np.mean(pred == y_test))


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    X_train, X_test, y_train, y_test = make_data()

    baseline = LogisticRegression(max_iter=1000).fit(X_train, y_train)
    baseline_acc = float(baseline.score(X_test, y_test))
    print(f"Linha de base (regressão logística): {baseline_acc:.4f}")

    model, losses, test_acc = train_and_eval(HIDDEN_SIZES, SEED, X_train, y_train, X_test, y_test)
    pred = model.predict_proba(X_test).argmax(axis=1)
    train_acc = float(np.mean(model.predict_proba(X_train).argmax(axis=1) == y_train))
    sizes = model.layer_sizes
    print(f"MLP {sizes}: lr={LEARNING_RATE}, épocas={EPOCHS}, perda final={losses[-1]:.4f}")
    print(f"acurácia treino={train_acc:.4f}  teste={test_acc:.4f}  parâmetros={model.n_params()}")

    # Figura 4: perda de treino x época
    plt.figure(figsize=(7, 4.5))
    plt.plot(range(1, EPOCHS + 1), losses, label="Perda de treino (entropia cruzada categórica)")
    plt.title("Figura 4: perda de treino por época (Exercício 3)")
    plt.xlabel("Época")
    plt.ylabel("Perda")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig04-loss-ex3.png", dpi=160)
    plt.close()

    # Figura 5: matriz de confusão no teste
    cm = confusion_matrix(y_test, pred)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=13)
    ax.set_xticks(range(N_CLASSES), [f"Classe {k}" for k in range(N_CLASSES)])
    ax.set_yticks(range(N_CLASSES), [f"Classe {k}" for k in range(N_CLASSES)])
    ax.set_title("Figura 5: matriz de confusão no teste (Exercício 3)")
    ax.set_xlabel("Classe prevista")
    ax.set_ylabel("Classe real")
    fig.colorbar(im, ax=ax, label="Número de amostras")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig05-confusion-ex3.png", dpi=160)
    plt.close(fig)

    # Pares mais confundidos (soma das duas direções)
    pairs = {}
    for i in range(N_CLASSES):
        for j in range(i + 1, N_CLASSES):
            pairs[f"{i}-{j}"] = int(cm[i, j] + cm[j, i])
    recall = (cm.diagonal() / cm.sum(axis=1)).tolist()
    base_cm = confusion_matrix(y_test, baseline.predict(X_test))
    results = {
        "sizes": sizes, "lr": LEARNING_RATE, "epochs": EPOCHS, "n_params": model.n_params(),
        "final_loss": losses[-1], "train_acc": train_acc, "test_acc": test_acc,
        "baseline_acc": baseline_acc, "confusion": cm.tolist(), "pair_errors": pairs,
        "recall_per_class": recall, "baseline_confusion": base_cm.tolist(),
        "test_class_counts": np.bincount(y_test).tolist(),
    }
    with open(FIGURES_DIR / "results_ex3.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(cm, pairs, recall)


if __name__ == "__main__":
    main()

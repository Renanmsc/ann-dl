"""Exercício 4: uma segunda camada oculta ajuda?

Comparação controlada com o Exercício 3: mesmos dados, mesma divisão, mesmas
épocas e mesma taxa de aprendizado. Muda apenas a profundidade (2 camadas
ocultas). Reutiliza `MLP` (Exercício 2) e `train_and_eval` (Exercício 3).
Execute com:  python exercise4.py
Gera a Figura 6 em ../figures/ e results_ex4.json.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from exercise3 import EPOCHS, HIDDEN_SIZES, LEARNING_RATE, make_data, train_and_eval

SEED = 42
FIGURES_DIR = Path(__file__).resolve().parents[1] / "figures"

SHALLOW = HIDDEN_SIZES
DEEP = [16, 16] 
INIT_SEEDS = [42, 43, 44, 45, 46]


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    X_train, X_test, y_train, y_test = make_data()

    runs = {"shallow": [], "deep": []}
    curves = {}
    n_params = {}
    for name, hidden in (("shallow", SHALLOW), ("deep", DEEP)):
        for seed in INIT_SEEDS:
            model, losses, acc = train_and_eval(hidden, seed, X_train, y_train, X_test, y_test)
            runs[name].append({"seed": seed, "final_loss": losses[-1], "test_acc": acc})
            curves.setdefault(name, []).append(losses)
            n_params[name] = model.n_params()
        arch = [X_train.shape[1]] + hidden + [3]
        print(name, arch, "parâmetros:", n_params[name])
        for r in runs[name]:
            print("  semente", r["seed"], f"perda={r['final_loss']:.4f}", f"acc={r['test_acc']:.4f}")

    def mean(name, key):
        return float(np.mean([r[key] for r in runs[name]]))

    def std(name, key):
        return float(np.std([r[key] for r in runs[name]]))

    # Figura 6: curvas de perda (semente 42 em destaque, média das sementes tracejada)
    plt.figure(figsize=(7.5, 4.8))
    for name, label, color in (("shallow", "Exercício 3: 1 camada oculta", "tab:blue"),
                               ("deep", "Exercício 4: 2 camadas ocultas", "tab:red")):
        arr = np.array(curves[name])
        plt.plot(range(1, EPOCHS + 1), arr[0], color=color, label=f"{label} (semente 42)")
        plt.plot(range(1, EPOCHS + 1), arr.mean(axis=0), color=color, ls="--", alpha=0.7,
                 label=f"{label} (média de {len(INIT_SEEDS)} sementes)")
    plt.title("Figura 6: perda de treino, rede rasa vs. rede profunda")
    plt.xlabel("Época")
    plt.ylabel("Perda de treino")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig06-loss-comparison-ex4.png", dpi=160)
    plt.close()

    results = {
        "init_seeds": INIT_SEEDS, "lr": LEARNING_RATE, "epochs": EPOCHS,
        "shallow_arch": [4] + SHALLOW + [3], "deep_arch": [4] + DEEP + [3],
        "n_params": n_params, "runs": runs,
        "mean": {k: {"final_loss": mean(k, "final_loss"), "test_acc": mean(k, "test_acc")} for k in runs},
        "std": {k: {"final_loss": std(k, "final_loss"), "test_acc": std(k, "test_acc")} for k in runs},
        "acc_diff_deep_minus_shallow": mean("deep", "test_acc") - mean("shallow", "test_acc"),
        "loss_at_epoch_mean": {
            k: {str(e): float(np.mean([c[e - 1] for c in curves[k]])) for e in (100, 500, 1000, 2000)}
            for k in curves
        },
    }
    with open(FIGURES_DIR / "results_ex4.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("média acurácia raso/profundo:", results["mean"]["shallow"]["test_acc"], results["mean"]["deep"]["test_acc"])
    print("diferença (profunda - rasa):", results["acc_diff_deep_minus_shallow"])


if __name__ == "__main__":
    main()

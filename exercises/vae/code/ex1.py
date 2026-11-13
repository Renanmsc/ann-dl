"""Exercício 1: um VAE no MNIST (d = 16). Gera as Figuras 1, 2 e 3."""
import json

import matplotlib
matplotlib.use("Agg")  # Salva as figuras sem depender de uma interface gráfica.
import matplotlib.pyplot as plt
import numpy as np
import torch

from vae import FIG_DIR, HERE, evaluate, load_data, tile, train_vae

EPOCHS = 20

data = load_data()
model, hist = train_vae(d=16, data=data, epochs=EPOCHS)
torch.save(model.state_dict(), HERE / "vae_d16.pt")  

test_rec, test_kl = evaluate(model, data["test"][0])
print(f"\nTESTE d=16 | reconstrução {test_rec:.2f} | KL {test_kl:.2f} | "
      f"-ELBO {test_rec + test_kl:.2f} (nats/imagem)")
print(f"KL de validação: época 1 = {hist['val_kl'][0]:.2f}, "
      f"época {EPOCHS} = {hist['val_kl'][-1]:.2f}")

with open(FIG_DIR / "results_ex1.json", "w") as f:
    json.dump({"test_rec": test_rec, "test_kl": test_kl,
               "test_neg_elbo": test_rec + test_kl,
               "val_kl_first": hist["val_kl"][0], "val_kl_last": hist["val_kl"][-1],
               "history": hist}, f, indent=2)

# Figura 1: reconstrução x época e KL x época
ep = np.arange(1, EPOCHS + 1)
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, key, title in [(axes[0], "rec", "Termo de reconstrução (BCE)"),
                       (axes[1], "kl", "Termo KL")]:
    ax.plot(ep, hist[f"train_{key}"], marker="o", ms=3, label="Treino")
    ax.plot(ep, hist[f"val_{key}"], marker="s", ms=3, label="Validação")
    ax.set(title=title, xlabel="Época", ylabel="nats por imagem")
    ax.grid(alpha=0.3)
    ax.legend()
fig.suptitle("Figura 1: os dois termos da perda por época (d = 16)")
fig.tight_layout()
fig.savefig(FIG_DIR / "fig1.png", dpi=150)
plt.close(fig)

# Figura 2: originais (cima) e reconstruções a partir de mu (baixo)
Xte, yte = data["test"]
idx = [(yte == k).nonzero(as_tuple=True)[0][0].item() for k in range(10)]
model.eval()
with torch.no_grad():
    mu, _ = model.encode(Xte[idx])
    rec = model.decode(mu)  # decodifica mu, não amostra z

fig, axes = plt.subplots(2, 10, figsize=(13, 3.2))
for k in range(10):
    for r, img in enumerate([Xte[idx[k]], rec[k]]):
        ax = axes[r, k]
        ax.imshow(img.view(28, 28), cmap="gray")
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0, k].set_title(f"dígito {k}", fontsize=9)
axes[0, 0].set_ylabel("Original")
axes[1, 0].set_ylabel("Recon. (μ)")
fig.suptitle("Figura 2: primeira imagem de teste de cada dígito e sua reconstrução")
fig.supxlabel("Dígito (uma coluna por classe)")
fig.tight_layout()
fig.savefig(FIG_DIR / "fig2.png", dpi=150)
plt.close(fig)

torch.manual_seed(0)
z = torch.randn(64, 16)
with torch.no_grad():
    samples = model.decode(z)

fig, ax = plt.subplots(figsize=(6, 6.4))
ax.imshow(tile(samples, 8, 8), cmap="gray")
ax.set_xticks([])
ax.set_yticks([])
ax.set_title("Figura 3: 64 amostras decodificadas de z ~ N(0, I), d = 16")
ax.set_xlabel("Colunas da grade (8 amostras por linha; semente 0)")
ax.set_ylabel("Linhas da grade (8 linhas)")
fig.tight_layout()
fig.savefig(FIG_DIR / "fig3.png", dpi=150)
plt.close(fig)

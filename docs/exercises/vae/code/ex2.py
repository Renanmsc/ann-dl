"""Exercício 2: um espaço latente que dá para ver (d = 2). Gera as Figuras 4, 5 e 6."""
import json

import matplotlib
matplotlib.use("Agg")  # Salva as figuras sem depender de uma interface gráfica.
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import norm

from vae import FIG_DIR, HERE, VAE, evaluate, load_data, tile, train_vae

EPOCHS = 20
A_DIGIT, B_DIGIT = 3, 8  # dígitos diferentes para a interpolação (item D)

data = load_data()
Xte, yte = data["test"]

# A. Treine um VAE 2-D (mesmo modelo, dados, sementes, beta e orçamento)
model2, hist2 = train_vae(d=2, data=data, epochs=EPOCHS)
rec2, kl2 = evaluate(model2, Xte)

# Modelo d = 16 do Exercício 1 (checkpoint salvo pelo ex1.py; retreina se faltar)
ckpt = HERE / "vae_d16.pt"
if ckpt.exists():
    model16 = VAE(16)
    model16.load_state_dict(torch.load(ckpt))
else:
    model16, _ = train_vae(d=16, data=data, epochs=EPOCHS)
rec16, kl16 = evaluate(model16, Xte)

print("\nTESTE (nats/imagem)")
print(f"  d=16 | reconstrução {rec16:7.2f} | KL {kl16:6.2f} | -ELBO {rec16 + kl16:7.2f}")
print(f"  d= 2 | reconstrução {rec2:7.2f} | KL {kl2:6.2f} | -ELBO {rec2 + kl2:7.2f}")

# B. Figura 4: mu do conjunto de teste, colorido por dígito
model2.eval()
model16.eval()
with torch.no_grad():
    mu2, _ = model2.encode(Xte)
mu2 = mu2.numpy()

fig, ax = plt.subplots(figsize=(8, 7))
cmap = plt.get_cmap("tab10")
for k in range(10):
    m = (yte == k).numpy()
    ax.scatter(mu2[m, 0], mu2[m, 1], s=4, alpha=0.6, color=cmap(k), label=str(k))
ax.set(title="Figura 4: μ das 10 000 imagens de teste (d = 2)",
       xlabel="μ₁ (dimensão latente 1)", ylabel="μ₂ (dimensão latente 2)")
ax.legend(title="Dígito", markerscale=4, loc="center left", bbox_to_anchor=(1.01, 0.5))
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig4.png", dpi=150)
plt.close(fig)

# C. Figura 5: plano decodificado em grade 15x15 pelos quantis de N(0, 1)
n = 15
q = np.linspace(0.05, 0.95, n)
zs = norm.ppf(q)  # z = Phi^{-1}(q)
grid = np.array([[z1, z2] for z2 in zs[::-1] for z1 in zs], dtype=np.float32)
with torch.no_grad():
    imgs = model2.decode(torch.from_numpy(grid))

fig, ax = plt.subplots(figsize=(8.5, 8.5))
ax.imshow(tile(imgs, n, n), cmap="gray")
ticks = (np.arange(n) + 0.5) * 28
ax.set_xticks(ticks[::2])
ax.set_xticklabels([f"{v:.2f}" for v in zs[::2]], fontsize=8)
ax.set_yticks(ticks[::2])
ax.set_yticklabels([f"{v:.2f}" for v in zs[::-1][::2]], fontsize=8)
ax.set(title="Figura 5: plano latente decodificado, z = Φ⁻¹(q), q ∈ [0,05; 0,95]",
       xlabel="z₁", ylabel="z₂")
fig.tight_layout()
fig.savefig(FIG_DIR / "fig5.png", dpi=150)
plt.close(fig)

idx_a = (yte == A_DIGIT).nonzero(as_tuple=True)[0][0].item()
idx_b = (yte == B_DIGIT).nonzero(as_tuple=True)[0][0].item()
xa, xb = Xte[idx_a], Xte[idx_b]
print(f"\nInterpolação: índice A = {idx_a} (dígito {yte[idx_a].item()}), "
      f"índice B = {idx_b} (dígito {yte[idx_b].item()})")

ts = torch.linspace(0, 1, 10)
pix_row = torch.stack([(1 - t) * xa + t * xb for t in ts])
with torch.no_grad():
    mu_a, _ = model16.encode(xa.unsqueeze(0))
    mu_b, _ = model16.encode(xb.unsqueeze(0))
    lat_row = model16.decode(torch.stack([(1 - t) * mu_a[0] + t * mu_b[0] for t in ts]))

fig, axes = plt.subplots(2, 10, figsize=(13, 3.4))
for j, t in enumerate(ts):
    for r, img in enumerate([pix_row[j], lat_row[j]]):
        axes[r, j].imshow(img.view(28, 28), cmap="gray")
        axes[r, j].set_xticks([])
        axes[r, j].set_yticks([])
    axes[0, j].set_title(f"t = {t:.2f}", fontsize=8)
axes[0, 0].set_ylabel("Pixels")
axes[1, 0].set_ylabel("Latente")
fig.suptitle(f"Figura 6: interpolação entre teste[{idx_a}] (dígito {A_DIGIT}) "
             f"e teste[{idx_b}] (dígito {B_DIGIT})")
fig.supxlabel(f"t — peso de teste[{idx_b}] (dígito {B_DIGIT})")
fig.tight_layout()
fig.savefig(FIG_DIR / "fig6.png", dpi=150)
plt.close(fig)

with open(FIG_DIR / "results_ex2.json", "w") as f:
    json.dump({"d16": {"rec": rec16, "kl": kl16}, "d2": {"rec": rec2, "kl": kl2},
               "idx_a": idx_a, "idx_b": idx_b}, f, indent=2)

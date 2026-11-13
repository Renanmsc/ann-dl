"""Componentes compartilhados do VAE: dados, modelo, perda e laço de treino.

Tudo que é específico de VAE (reparametrização, BCE, KL em forma fechada)
está escrito aqui à mão. Nada de pythae, diffusers ou
torch.distributions.kl_divergence.
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import datasets, transforms

HERE = Path(__file__).resolve().parent
FIG_DIR = HERE.parent / "figures"
DATA_DIR = HERE / "data"
FIG_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cpu")  # CPU: reprodutível e suficiente (~30 s por modelo)


# ----------------------------------------------------------------------------
# A. Dados
# ----------------------------------------------------------------------------
def load_data():
    """MNIST em [0, 1], achatado em 784. 50k treino, 10k validação, 10k teste."""
    train_full = datasets.MNIST(DATA_DIR, train=True, download=True,
                                transform=transforms.ToTensor())
    test = datasets.MNIST(DATA_DIR, train=False, download=True,
                          transform=transforms.ToTensor())
    train, val = random_split(train_full, [50_000, 10_000],
                              generator=torch.Generator().manual_seed(42))

    # Materializa em tensores (mesmo valor do ToTensor: uint8 / 255).
    def flat(u8):
        return u8.float().div(255.0).view(len(u8), -1)

    return {
        "train": (flat(train_full.data[train.indices]), train_full.targets[train.indices]),
        "val": (flat(train_full.data[val.indices]), train_full.targets[val.indices]),
        "test": (flat(test.data), test.targets),
    }


# ----------------------------------------------------------------------------
# B. Modelo e perda
# ----------------------------------------------------------------------------
class VAE(nn.Module):
    def __init__(self, d=16):
        super().__init__()
        self.d = d
        self.encoder = nn.Sequential(
            nn.Linear(784, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
        )
        self.fc_mu = nn.Linear(256, d)
        self.fc_logvar = nn.Linear(256, d)
        self.decoder = nn.Sequential(
            nn.Linear(d, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, 784), nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    @staticmethod
    def reparameterize(mu, logvar, eps=None):
        """z = mu + sigma * eps, eps ~ N(0, I). Mantém o gradiente em mu e logvar."""
        if eps is None:
            eps = torch.randn_like(mu)
        return mu + torch.exp(0.5 * logvar) * eps

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def loss_terms(x, x_hat, mu, logvar):
    """Os dois termos da perda, em nats por imagem.

    recon = BCE somada sobre os 784 pixels, média sobre o lote
    kl    = 1/2 * sum_i (mu_i^2 + sigma_i^2 - 1 - log sigma_i^2), média sobre o lote
    """
    batch = x.size(0)
    recon = F.binary_cross_entropy(x_hat, x, reduction="sum") / batch
    kl = 0.5 * torch.sum(mu.pow(2) + logvar.exp() - 1.0 - logvar) / batch
    return recon, kl


# ----------------------------------------------------------------------------
# Avaliação e treino
# ----------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, X, batch_size=1000, seed=0):
    """Reconstrução e KL médias (nats/imagem). O ruído eps usa um gerador próprio
    com semente fixa, então o número é reprodutível e não mexe no RNG global."""
    model.eval()
    g = torch.Generator().manual_seed(seed)
    rec_sum = kl_sum = 0.0
    for i in range(0, len(X), batch_size):
        x = X[i:i + batch_size]
        mu, logvar = model.encode(x)
        eps = torch.randn(mu.shape, generator=g)
        x_hat = model.decode(model.reparameterize(mu, logvar, eps))
        recon, kl = loss_terms(x, x_hat, mu, logvar)
        rec_sum += recon.item() * len(x)
        kl_sum += kl.item() * len(x)
    return rec_sum / len(X), kl_sum / len(X)


def train_vae(d, data, epochs=20, batch_size=128, lr=1e-3, beta=1.0):
    """Treina um VAE com latente de dimensão d. Decisões olham só a validação."""
    torch.manual_seed(42)  # antes de construir o modelo
    model = VAE(d).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xtr = data["train"][0]
    loader = DataLoader(TensorDataset(Xtr), batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(42))

    hist = {"train_rec": [], "train_kl": [], "val_rec": [], "val_kl": []}
    for epoch in range(1, epochs + 1):
        model.train()
        rec_sum = kl_sum = 0.0
        for (x,) in loader:
            x_hat, mu, logvar = model(x)
            recon, kl = loss_terms(x, x_hat, mu, logvar)
            loss = recon + beta * kl
            opt.zero_grad()
            loss.backward()
            opt.step()
            rec_sum += recon.item() * len(x)
            kl_sum += kl.item() * len(x)

        val_rec, val_kl = evaluate(model, data["val"][0])
        hist["train_rec"].append(rec_sum / len(Xtr))
        hist["train_kl"].append(kl_sum / len(Xtr))
        hist["val_rec"].append(val_rec)
        hist["val_kl"].append(val_kl)
        print(f"[d={d}] época {epoch:2d} | treino rec {hist['train_rec'][-1]:7.2f} "
              f"KL {hist['train_kl'][-1]:6.2f} | val rec {val_rec:7.2f} KL {val_kl:6.2f}")
    return model, hist


# ----------------------------------------------------------------------------
# Utilidade de figura
# ----------------------------------------------------------------------------
def tile(imgs, nrows, ncols):
    """Arruma (nrows*ncols, 784) em uma grade única de imagens 28x28."""
    a = imgs.detach().cpu().numpy().reshape(nrows, ncols, 28, 28)
    return a.transpose(0, 2, 1, 3).reshape(nrows * 28, ncols * 28)

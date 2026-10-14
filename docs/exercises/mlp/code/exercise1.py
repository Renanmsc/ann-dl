"""Exercício 1: um passo de retropropagação, na mão.

Verificação numérica da resolução feita no relatório. Usa apenas a biblioteca
padrão (math): nenhuma biblioteca de redes neurais ou de álgebra linear.
Execute com:  python exercise1.py
"""

import math

# ---------------------------------------------------------------------------
# Dados do enunciado
# ---------------------------------------------------------------------------
x = [0.5, -0.2]
y = 1.0
W1 = [[0.3, -0.1], [0.2, 0.4]]  # linha i = neurônio oculto i
b1 = [0.1, -0.2]
W2 = [0.5, -0.3]
b2 = 0.2
ETA = 0.3

tanh = math.tanh


def forward(x, W1, b1, W2, b2):
    """Passagem direta. Retorna (z1, h1, u2, y_hat, L)."""
    z1 = [W1[i][0] * x[0] + W1[i][1] * x[1] + b1[i] for i in range(2)]
    h1 = [tanh(z) for z in z1]
    u2 = W2[0] * h1[0] + W2[1] * h1[1] + b2
    y_hat = tanh(u2)
    L = (y - y_hat) ** 2
    return z1, h1, u2, y_hat, L


def backward(x, h1, y_hat, W2):
    """Passagem reversa, derivada à mão (regra da cadeia)."""
    dL_dyhat = -2.0 * (y - y_hat)                       # dL/dy_hat
    dL_du2 = dL_dyhat * (1.0 - y_hat ** 2)              # tanh'(u2) = 1 - y_hat^2
    dL_dW2 = [dL_du2 * h1[0], dL_du2 * h1[1]]
    dL_db2 = dL_du2
    dL_dh1 = [W2[0] * dL_du2, W2[1] * dL_du2]
    dL_dz1 = [dL_dh1[i] * (1.0 - h1[i] ** 2) for i in range(2)]
    dL_dW1 = [[dL_dz1[i] * x[j] for j in range(2)] for i in range(2)]
    dL_db1 = list(dL_dz1)
    return dL_dyhat, dL_du2, dL_dW2, dL_db2, dL_dh1, dL_dz1, dL_dW1, dL_db1


def fmt(v):
    if isinstance(v, list):
        return "[" + ", ".join(fmt(e) for e in v) + "]"
    return f"{v:+.6f}"


def numerical_check(eps=1e-6):
    """Confere cada gradiente por diferenças finitas centrais."""
    params = {
        "W1": W1, "b1": b1, "W2": W2, "b2": b2,
    }

    def loss_with(W1_, b1_, W2_, b2_):
        return forward(x, W1_, b1_, W2_, b2_)[4]

    max_err = 0.0
    z1, h1, u2, y_hat, L = forward(x, W1, b1, W2, b2)
    g = backward(x, h1, y_hat, W2)
    dW2, db2_, dW1, db1_ = g[2], g[3], g[6], g[7]

    def central(setter):
        return (setter(+eps) - setter(-eps)) / (2 * eps)

    for i in range(2):
        for j in range(2):
            def f(d, i=i, j=j):
                W = [r[:] for r in W1]
                W[i][j] += d
                return loss_with(W, b1, W2, b2)
            max_err = max(max_err, abs(central(f) - dW1[i][j]))
        def fb(d, i=i):
            b = b1[:]
            b[i] += d
            return loss_with(W1, b, W2, b2)
        max_err = max(max_err, abs(central(fb) - db1_[i]))
        def fw2(d, i=i):
            W = W2[:]
            W[i] += d
            return loss_with(W1, b1, W, b2)
        max_err = max(max_err, abs(central(fw2) - dW2[i]))
    def fb2(d):
        return loss_with(W1, b1, W2, b2 + d)
    max_err = max(max_err, abs(central(fb2) - db2_))
    return max_err


def main() -> None:
    # A: passagem direta
    z1, h1, u2, y_hat, L = forward(x, W1, b1, W2, b2)
    print("== A: passagem direta ==")
    print("z1 =", fmt(z1))
    print("h1 =", fmt(h1))
    print("u2 =", fmt(u2))
    print("y_hat =", fmt(y_hat))
    print("L =", fmt(L))

    # B: passagem reversa
    dyhat, du2, dW2, db2_, dh1, dz1, dW1, db1_ = backward(x, h1, y_hat, W2)
    print("\n== B: passagem reversa ==")
    print("dL/dy_hat =", fmt(dyhat))
    print("tanh'(u2) = 1 - y_hat^2 =", fmt(1 - y_hat ** 2))
    print("dL/du2 =", fmt(du2))
    print("dL/dW2 =", fmt(dW2))
    print("dL/db2 =", fmt(db2_))
    print("dL/dh1 =", fmt(dh1))
    print("tanh'(z1) = 1 - h1^2 =", fmt([1 - h ** 2 for h in h1]))
    print("dL/dz1 =", fmt(dz1))
    print("dL/dW1 =", fmt(dW1))
    print("dL/db1 =", fmt(db1_))
    print("erro máximo vs. diferenças finitas:", f"{numerical_check():.2e}")

    # C: atualização
    W2n = [W2[i] - ETA * dW2[i] for i in range(2)]
    b2n = b2 - ETA * db2_
    W1n = [[W1[i][j] - ETA * dW1[i][j] for j in range(2)] for i in range(2)]
    b1n = [b1[i] - ETA * db1_[i] for i in range(2)]
    print("\n== C: parâmetros atualizados (eta = 0.3) ==")
    print("W2 =", fmt(W2n))
    print("b2 =", fmt(b2n))
    print("W1 =", fmt(W1n))
    print("b1 =", fmt(b1n))
    _, h1n, u2n, y_hat_n, L_n = forward(x, W1n, b1n, W2n, b2n)
    print("segunda passagem direta: y_hat =", fmt(y_hat_n), " L =", fmt(L_n))
    print("variação da perda:", fmt(L_n - L))


if __name__ == "__main__":
    main()

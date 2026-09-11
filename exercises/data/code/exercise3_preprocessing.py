"""Exercise 3 — Preparando dados do mundo real para uma rede neural.

Pré-processa o dataset Spaceship Titanic (Kaggle) para uma rede que usa
tanh nas camadas ocultas: split estratificado antes de qualquer estatística,
imputação por tipo de coluna, one-hot para categóricas, feature de gasto
total, log1p para as colunas de cauda pesada e escalonamento para [-1, 1].

Uso (a partir da raiz do repositório):

    python docs/exercises/data/code/exercise3_preprocessing.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

DATA_DIR = Path(__file__).resolve().parents[1] / "raw"
FIGURES = Path(__file__).resolve().parents[1] / "figures"
TRAIN_PATH = DATA_DIR / "train.csv"
RANDOM_STATE = 42

SPEND_COLS = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
NUMERIC_COLS = ["Age", *SPEND_COLS]
CATEGORICAL_COLS = ["HomePlanet", "CryoSleep", "Destination", "VIP"]
DROP_COLS = ["Cabin", "Name", "PassengerId"]
TARGET = "Transported"


def describe_dataset(df: pd.DataFrame) -> None:
    n = len(df)
    balance = df[TARGET].value_counts()
    print(f"Dataset: {n} passageiros, {df.shape[1]} colunas")
    print(f"Transported representa se o passageiro foi transportado para outra "
          f"dimensão (o alvo binário a prever).")
    print("\nBalanceamento do alvo:")
    for cls, count in balance.items():
        print(f"  {cls}: {count} ({count / n:.1%})")

    print(f"\nFeatures numéricas: {NUMERIC_COLS}")
    print(f"Features categóricas: {CATEGORICAL_COLS}")
    print(f"Descartadas: {DROP_COLS}")

    print("\nValores faltantes por coluna:")
    missing = df.isna().sum()
    missing_pct = df.isna().mean() * 100
    table = pd.DataFrame({"faltantes": missing, "percentual": missing_pct.round(2)})
    print(table[table["faltantes"] > 0].sort_values("faltantes", ascending=False))

    print("\nEstatísticas das colunas de gasto (valores brutos, com NaN ignorado):")
    stats = df[SPEND_COLS].agg(["mean", "median", "max"]).T
    print(stats.round(2))
    print(
        "\nMédia >> mediana em todas as colunas (mediana = 0): a maioria dos "
        "passageiros gasta zero (ex.: quem está em CryoSleep) e uma minoria "
        "gasta muito, puxando a média para cima -> distribuição fortemente "
        "assimétrica à direita (cauda pesada)."
    )



def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df[TARGET], random_state=RANDOM_STATE,
    )
    print(f"\nSplit 80/20 estratificado: treino={len(train_df)}, teste={len(test_df)}")
    print(f"  Balanceamento treino: {train_df[TARGET].mean():.3f}")
    print(f"  Balanceamento teste:  {test_df[TARGET].mean():.3f}")
    print(
        "  Motivo de separar antes de imputar/escalar: qualquer estatística "
        "(média, mediana, categorias vistas, min/max) calculada no dataset "
        "inteiro 'vaza' informação do teste para o treino. O modelo passaria "
        "a se beneficiar, na hora de treinar, de estatísticas que só deveriam "
        "existir depois de ver dados novos -- o que infla artificialmente o "
        "desempenho reportado e não reflete o cenário real de produção."
    )
    return train_df, test_df



def impute(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Imputa numéricas com a mediana (robusta à assimetria) e categóricas com a moda.

    Os dois imputadores são ajustados (`fit`) apenas no treino e depois
    aplicados (`transform`) em treino e teste, evitando vazamento de dados.
    """
    train_df = train_df.copy()
    test_df = test_df.copy()

    num_imputer = SimpleImputer(strategy="median")
    train_df[NUMERIC_COLS] = num_imputer.fit_transform(train_df[NUMERIC_COLS])
    test_df[NUMERIC_COLS] = num_imputer.transform(test_df[NUMERIC_COLS])

    cat_imputer = SimpleImputer(strategy="most_frequent")
    train_df[CATEGORICAL_COLS] = cat_imputer.fit_transform(train_df[CATEGORICAL_COLS])
    test_df[CATEGORICAL_COLS] = cat_imputer.transform(test_df[CATEGORICAL_COLS])

    return train_df, test_df


def encode_categorical(
    train_df: pd.DataFrame, test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One-hot para as 4 colunas categóricas, ajustado só no treino.

    ``handle_unknown="ignore"`` faz com que uma categoria vista no teste mas
    não no treino vire um vetor de zeros (nenhuma das colunas dummy ativa),
    em vez de o encoder falhar.
    """
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoder.fit(train_df[CATEGORICAL_COLS])

    train_ohe = pd.DataFrame(
        encoder.transform(train_df[CATEGORICAL_COLS]),
        columns=encoder.get_feature_names_out(CATEGORICAL_COLS),
        index=train_df.index,
    )
    test_ohe = pd.DataFrame(
        encoder.transform(test_df[CATEGORICAL_COLS]),
        columns=encoder.get_feature_names_out(CATEGORICAL_COLS),
        index=test_df.index,
    )
    return train_ohe, test_ohe


def add_total_spend(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["TotalSpend"] = df[SPEND_COLS].sum(axis=1)
    return df


def log_transform(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    df[cols] = np.log1p(df[cols])
    return df


def scale_numeric(
    train_df: pd.DataFrame, test_df: pd.DataFrame, cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, MinMaxScaler]:
    """Normalização para [-1, 1], ajustada só no treino (compatível com tanh)."""
    scaler = MinMaxScaler(feature_range=(-1, 1))
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df[cols] = scaler.fit_transform(train_df[cols])
    test_df[cols] = scaler.transform(test_df[cols])
    return train_df, test_df, scaler


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(TRAIN_PATH)

    print("=" * 70, "\nA — Conheça os dados\n", "=" * 70, sep="")
    describe_dataset(df)

    print("\n", "=" * 70, "\nB — Separe antes de transformar\n", "=" * 70, sep="")
    train_df, test_df = split_data(df)

    print("\n", "=" * 70, "\nC — Pré-processe\n", "=" * 70, sep="")

    train_df, test_df = impute(train_df, test_df)
    print("Imputação: numéricas -> mediana do treino (robusta à cauda pesada); "
          "categóricas -> moda do treino (preserva a categoria dominante sem "
          "inventar uma classe nova).")

    train_df = add_total_spend(train_df)
    test_df = add_total_spend(test_df)
    numeric_cols = [*NUMERIC_COLS, "TotalSpend"]
    print(f"Feature criada: TotalSpend (soma de {SPEND_COLS}). "
          f"Colunas descartadas: {DROP_COLS}.")

    raw_foodcourt = train_df["FoodCourt"].copy()

    log_cols = [*SPEND_COLS, "TotalSpend"]
    train_df = log_transform(train_df, log_cols)
    test_df = log_transform(test_df, log_cols)
    print(f"log1p aplicado em: {log_cols}. Isso comprime a cauda longa de "
          f"valores altos, aproximando a distribuição de uma forma mais "
          f"simétrica -- essencial para tanh, que satura (derivada ~ 0) "
          f"para entradas com magnitude grande, dificultando o aprendizado.")

    train_ohe, test_ohe = encode_categorical(train_df, test_df)
    print(f"One-hot em {CATEGORICAL_COLS} -> {train_ohe.shape[1]} colunas dummy. "
          f"Categoria do teste ausente no treino vira vetor de zeros "
          f"(handle_unknown='ignore').")

    train_df, test_df, scaler = scale_numeric(train_df, test_df, numeric_cols)
    print(f"Escalonamento: Normalização Min-Max para [-1, 1] em {numeric_cols} "
          f"(mesmo intervalo de saída do tanh). "
          f"min={train_df[numeric_cols].min().min():.3f}  "
          f"max={train_df[numeric_cols].max().max():.3f}")

    X_train = pd.concat([train_df[numeric_cols].reset_index(drop=True), train_ohe.reset_index(drop=True)], axis=1)
    X_test = pd.concat([test_df[numeric_cols].reset_index(drop=True), test_ohe.reset_index(drop=True)], axis=1)
    y_train = train_df[TARGET].reset_index(drop=True)
    y_test = test_df[TARGET].reset_index(drop=True)

    print("\n", "=" * 70, "\nD — Verifique e visualize\n", "=" * 70, sep="")

    # Figura 6 - FoodCourt antes (bruto, pós-imputação) e depois (log1p + escalonado)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].hist(raw_foodcourt, bins=40, color="tab:blue")
    axes[0].set_title("FoodCourt — bruto (pós-imputação)")
    axes[0].set_xlabel("FoodCourt")
    axes[0].set_ylabel("contagem")

    axes[1].hist(X_train["FoodCourt"], bins=40, color="tab:orange")
    axes[1].set_title("FoodCourt — log1p + escalonado [-1, 1]")
    axes[1].set_xlabel("FoodCourt (processado)")
    axes[1].set_ylabel("contagem")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig06-foodcourt-before-after.png", dpi=150)
    plt.close(fig)

    n_nan_train = int(X_train.isna().sum().sum())
    n_nan_test = int(X_test.isna().sum().sum())
    print(f"NaN remanescente: treino={n_nan_train}, teste={n_nan_test}")
    print(f"Shape final: X_train={X_train.shape}, X_test={X_test.shape}")
    print(f"Intervalo de valores: X_train min={X_train.values.min():.3f}, "
          f"max={X_train.values.max():.3f}  "
          f"(compatível com tanh, que produz saídas em [-1, 1])")

    X_train.to_csv(DATA_DIR / "X_train_processed.csv", index=False)
    X_test.to_csv(DATA_DIR / "X_test_processed.csv", index=False)
    y_train.to_csv(DATA_DIR / "y_train.csv", index=False)
    y_test.to_csv(DATA_DIR / "y_test.csv", index=False)


if __name__ == "__main__":
    main()
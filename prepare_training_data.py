from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from settings import PCA_VARIANCE, RANDOM_STATE, TEST_RATIO

MPLCONFIGDIR = Path(".matplotlib_cache")
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR.resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(MPLCONFIGDIR.resolve()))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def prepare_training_data(
    df_training: pd.DataFrame,
    output_dir: str | Path = "model_outputs",
) -> dict:
    """Delar upp, skalar och PCA-transformerar data inför modellträning.

    df_training ska innehålla målkolumnen i df_training.attrs["target_column"].
    Funktionen returnerar både originalfeatures för XGBoost och PCA-features
    för logistisk regression. Skalare och PCA tränas bara på träningsdelen för
    att testdatan inte ska påverka modellen.
    """
    print("\n-- PCA och träningsklara matriser ------------------------------")

    target_column = df_training.attrs.get("target_column", "type_1_encoded")
    if target_column not in df_training.columns:
        raise ValueError(f"Träningsdata saknar målkolumnen: {target_column}")

    feature_columns = df_training.attrs.get("feature_columns")
    if feature_columns is None:
        feature_columns = [
            column for column in df_training.columns
            if column != target_column
        ]

    X = df_training[feature_columns].astype(float)
    y = df_training[target_column].astype(int)

    X_train_original, X_test_original, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_RATIO,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_original)
    X_test_scaled = scaler.transform(X_test_original)

    pca = PCA(n_components=PCA_VARIANCE, random_state=RANDOM_STATE)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    pca_columns = [
        f"pca_component_{index + 1}"
        for index in range(X_train_pca.shape[1])
    ]
    df_train_pca = pd.DataFrame(
        X_train_pca,
        columns=pca_columns,
        index=X_train_original.index,
    )
    df_test_pca = pd.DataFrame(
        X_test_pca,
        columns=pca_columns,
        index=X_test_original.index,
    )
    df_train_pca[target_column] = y_train
    df_test_pca[target_column] = y_test

    explained_variance = pca.explained_variance_ratio_.sum()
    explained_variance_table = _build_explained_variance_table(pca_columns, pca)
    pca_loadings = _build_pca_loadings(
        pca,
        pca_columns,
        feature_columns,
    )
    top_pca_loadings = _build_top_pca_loadings(pca_loadings)
    target_map = df_training.attrs.get("target_map", {})
    target_labels = {
        int(encoded): label
        for label, encoded in target_map.items()
    }

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    explained_variance_table.to_csv(
        output_path / "pca_explained_variance.csv",
        index=False,
    )
    top_pca_loadings.to_csv(output_path / "pca_top_loadings.csv", index=False)
    _save_explained_variance_plot(explained_variance_table, output_path)

    print(f"Rader träning/test:  {X_train_original.shape[0]} / {X_test_original.shape[0]}")
    print(f"Features före PCA:   {X_train_original.shape[1]}")
    print(f"Features efter PCA:  {X_train_pca.shape[1]}")
    print(f"Bevarad varians:     {explained_variance:.2%}")
    print(f"Målkolumn:           {target_column}")
    print(f"PCA-rapport sparad:  {output_path}")

    return {
        "X_train_original": X_train_original,
        "X_test_original": X_test_original,
        "X_train_pca": X_train_pca,
        "X_test_pca": X_test_pca,
        "y_train": y_train,
        "y_test": y_test,
        "df_train_pca": df_train_pca,
        "df_test_pca": df_test_pca,
        "explained_variance_table": explained_variance_table,
        "pca_loadings": pca_loadings,
        "top_pca_loadings": top_pca_loadings,
        "pca_columns": pca_columns,
        "feature_columns": feature_columns,
        "target_column": target_column,
        "target_labels": target_labels,
        "scaler": scaler,
        "pca": pca,
        "explained_variance": explained_variance,
        "train_index": X_train_original.index,
        "test_index": X_test_original.index,
    }


def _build_explained_variance_table(
    pca_columns: list[str],
    pca: PCA,
) -> pd.DataFrame:
    """Skapar en tabell med förklarad varians per PCA-komponent."""
    explained_variance = pd.DataFrame(
        {
            "component": pca_columns,
            "explained_variance_ratio": pca.explained_variance_ratio_,
        }
    )
    explained_variance["cumulative_explained_variance"] = (
        explained_variance["explained_variance_ratio"].cumsum()
    )
    return explained_variance


def _build_pca_loadings(
    pca: PCA,
    pca_columns: list[str],
    feature_columns: list[str],
) -> pd.DataFrame:
    """Beräknar hur originalfeatures laddar på varje PCA-komponent."""
    return pd.DataFrame(
        pca.components_.T,
        index=feature_columns,
        columns=pca_columns,
    )


def _build_top_pca_loadings(
    pca_loadings: pd.DataFrame,
    top_n: int = 10,
) -> pd.DataFrame:
    """Tar fram de starkaste absoluta laddningarna för de första komponenterna."""
    rows = []
    selected_components = pca_loadings.columns[:10]
    for component in selected_components:
        top_features = (
            pca_loadings[component]
            .abs()
            .sort_values(ascending=False)
            .head(top_n)
            .index
        )
        for feature in top_features:
            rows.append(
                {
                    "component": component,
                    "feature": feature,
                    "loading": pca_loadings.loc[feature, component],
                    "absolute_loading": abs(pca_loadings.loc[feature, component]),
                }
            )
    return pd.DataFrame(rows)


def _save_explained_variance_plot(
    explained_variance: pd.DataFrame,
    output_path: Path,
) -> None:
    """Sparar en figur som visar kumulativ förklarad varians."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        explained_variance.index + 1,
        explained_variance["cumulative_explained_variance"],
        linewidth=2,
    )
    ax.axhline(PCA_VARIANCE, color="red", linestyle="--", linewidth=1)
    ax.set_title("Kumulativ förklarad varians efter PCA")
    ax.set_xlabel("Antal PCA-komponenter")
    ax.set_ylabel("Kumulativ förklarad varians")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path / "pca_explained_variance.png", dpi=150)
    plt.close(fig)

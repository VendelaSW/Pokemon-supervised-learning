"""
Träningsförberedelse — Split, skalning och PCA
==============================================
Tar emot df_training och skapar de matriser som används av modellerna.

Funktionen gör en stratifierad train/test-split och fit:ar all skalning
och PCA enbart på träningsdelen. Tabellfeatures skalas inför PCA, medan
bildpixlar redan är normaliserade till 0-1 och skickas direkt till en
separat PCA som centrerar datan internt. Image-only-vyer skapas från
image_pca för att kunna testa om sprite-bilderna räcker utan tabellfeatures.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from settings import IMAGE_PCA_VARIANCE, PCA_VARIANCE, RANDOM_STATE, TEST_RATIO

MPLCONFIGDIR = Path(".matplotlib_cache")
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR.resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(MPLCONFIGDIR.resolve()))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def prepare_training_data(
    df_training: pd.DataFrame,
    image_matrix: np.ndarray | None = None,
    image_valid_mask: np.ndarray | None = None,
    output_dir: str | Path = "model_outputs",
) -> dict:
    """Delar upp, skalar och PCA-transformerar data inför modellträning.

    df_training ska innehålla målkolumnen i df_training.attrs["target_column"].
    Funktionen returnerar både originalfeatures för XGBoost och PCA-features
    för logistisk regression. Om bildfeatures finns byggs även kombinerade
    matriser med tabellfeatures och image_pca-kolumner. Tabellscalers och PCA
    tränas bara på träningsdelen för att testdatan inte ska påverka modellen.
    Bild-PCA fit:as på 0-1-normaliserade träningspixlar utan StandardScaler.
    Image-only-logreg får därefter en separat scaler på image_pca-features.
    """
    print("\n-- PCA och träningsklara matriser ------------------------------")

    df_training, image_matrix, has_image_features, image_valid_count = (
        _prepare_image_inputs(df_training, image_matrix, image_valid_mask)
    )

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

    if has_image_features:
        (
            X_train_original,
            X_test_original,
            image_train_raw,
            image_test_raw,
            y_train,
            y_test,
        ) = train_test_split(
            X,
            image_matrix,
            y,
            test_size=TEST_RATIO,
            stratify=y,
            random_state=RANDOM_STATE,
        )
    else:
        X_train_original, X_test_original, y_train, y_test = train_test_split(
            X,
            y,
            test_size=TEST_RATIO,
            stratify=y,
            random_state=RANDOM_STATE,
        )
        image_train_raw = None
        image_test_raw = None

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
    _save_explained_variance_plot(explained_variance_table, output_path)

    image_training_data = _build_empty_image_training_data()
    if has_image_features:
        image_training_data = _build_image_training_data(
            image_train_raw=image_train_raw,
            image_test_raw=image_test_raw,
            X_train_original=X_train_original,
            X_test_original=X_test_original,
            feature_columns=feature_columns,
        )

    print(f"Rader träning/test:  {X_train_original.shape[0]} / {X_test_original.shape[0]}")
    print(f"Features före PCA:   {X_train_original.shape[1]}")
    print(f"Features efter PCA:  {X_train_pca.shape[1]}")
    print(f"Bevarad varians:     {explained_variance:.2%}")
    print(f"Målkolumn:           {target_column}")
    if has_image_features:
        print(f"Rader med bild:      {image_valid_count}")
        print(f"Råa bildfeatures:    {image_matrix.shape[1]}")
        print(f"Image PCA-features:  {len(image_training_data['image_pca_columns'])}")
        print(
            "Features med bild:   "
            f"{image_training_data['X_train_with_images'].shape[1]}"
        )
        print(
            "PCA med bild:        "
            f"{image_training_data['X_train_with_images_pca'].shape[1]}"
        )
    print(f"PCA-PNG sparad i:    {output_path}")

    training_data = {
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
    training_data.update(image_training_data)
    return training_data


def _prepare_image_inputs(
    df_training: pd.DataFrame,
    image_matrix: np.ndarray | None,
    image_valid_mask: np.ndarray | None,
) -> tuple[pd.DataFrame, np.ndarray | None, bool, int]:
    """Validerar bildfeatures och filtrerar bort rader utan giltig bild."""
    if image_matrix is None and image_valid_mask is None:
        return df_training, None, False, 0
    if image_matrix is None or image_valid_mask is None:
        raise ValueError("image_matrix och image_valid_mask måste skickas in tillsammans.")

    image_matrix = np.asarray(image_matrix, dtype=np.float32)
    image_valid_mask = np.asarray(image_valid_mask, dtype=bool)

    if image_matrix.ndim != 2:
        raise ValueError("image_matrix måste vara tvådimensionell.")
    if image_valid_mask.ndim != 1:
        raise ValueError("image_valid_mask måste vara endimensionell.")
    if image_matrix.shape[0] != len(df_training):
        raise ValueError("image_matrix måste ha samma radantal som df_training.")
    if image_valid_mask.shape[0] != len(df_training):
        raise ValueError("image_valid_mask måste ha samma radantal som df_training.")
    if not image_valid_mask.any():
        raise ValueError("Inga giltiga bildfeatures hittades.")

    filtered_df = df_training.loc[image_valid_mask].copy()
    filtered_df.attrs = df_training.attrs.copy()
    return (
        filtered_df,
        image_matrix[image_valid_mask],
        True,
        int(image_valid_mask.sum()),
    )


def _build_empty_image_training_data() -> dict:
    """Skapar tomma image-keys när pipeline körs utan bildfeatures."""
    return {
        "X_train_with_images": None,
        "X_test_with_images": None,
        "X_train_with_images_pca": None,
        "X_test_with_images_pca": None,
        "with_images_feature_columns": [],
        "with_images_pca_columns": [],
        "image_only_feature_columns": [],
        "image_pca_columns": [],
        "image_pca": None,
        "image_scaler": None,
        "image_only_scaler": None,
        "X_train_image_only": None,
        "X_test_image_only": None,
        "X_train_image_only_scaled": None,
        "X_test_image_only_scaled": None,
        "with_images_scaler": None,
        "with_images_pca": None,
        "has_image_features": False,
        "image_valid_count": 0,
        "image_explained_variance_table": None,
        "with_images_explained_variance_table": None,
    }


def _build_image_training_data(
    *,
    image_train_raw: np.ndarray,
    image_test_raw: np.ndarray,
    X_train_original: pd.DataFrame,
    X_test_original: pd.DataFrame,
    feature_columns: list[str],
) -> dict:
    """Bygger image_pca direkt från 0-1-normaliserade pixlar."""
    image_pca = PCA(
        n_components=IMAGE_PCA_VARIANCE,
        random_state=RANDOM_STATE,
    )
    image_train_pca = image_pca.fit_transform(image_train_raw)
    image_test_pca = image_pca.transform(image_test_raw)
    image_pca_columns = [
        f"image_pca_{index + 1:03d}"
        for index in range(image_train_pca.shape[1])
    ]
    X_train_image_pca = pd.DataFrame(
        image_train_pca,
        columns=image_pca_columns,
        index=X_train_original.index,
    )
    X_test_image_pca = pd.DataFrame(
        image_test_pca,
        columns=image_pca_columns,
        index=X_test_original.index,
    )
    X_train_image_only = X_train_image_pca
    X_test_image_only = X_test_image_pca
    image_only_feature_columns = image_pca_columns.copy()

    image_only_scaler = StandardScaler()
    X_train_image_only_scaled = pd.DataFrame(
        image_only_scaler.fit_transform(X_train_image_only),
        columns=image_only_feature_columns,
        index=X_train_image_only.index,
    )
    X_test_image_only_scaled = pd.DataFrame(
        image_only_scaler.transform(X_test_image_only),
        columns=image_only_feature_columns,
        index=X_test_image_only.index,
    )

    X_train_with_images = pd.concat([X_train_original, X_train_image_pca], axis=1)
    X_test_with_images = pd.concat([X_test_original, X_test_image_pca], axis=1)
    with_images_feature_columns = feature_columns + image_pca_columns

    with_images_scaler = StandardScaler()
    X_train_with_images_scaled = with_images_scaler.fit_transform(X_train_with_images)
    X_test_with_images_scaled = with_images_scaler.transform(X_test_with_images)

    with_images_pca = PCA(n_components=PCA_VARIANCE, random_state=RANDOM_STATE)
    X_train_with_images_pca = with_images_pca.fit_transform(X_train_with_images_scaled)
    X_test_with_images_pca = with_images_pca.transform(X_test_with_images_scaled)
    with_images_pca_columns = [
        f"with_images_pca_component_{index + 1}"
        for index in range(X_train_with_images_pca.shape[1])
    ]

    return {
        "X_train_with_images": X_train_with_images,
        "X_test_with_images": X_test_with_images,
        "X_train_with_images_pca": X_train_with_images_pca,
        "X_test_with_images_pca": X_test_with_images_pca,
        "with_images_feature_columns": with_images_feature_columns,
        "with_images_pca_columns": with_images_pca_columns,
        "image_only_feature_columns": image_only_feature_columns,
        "image_pca_columns": image_pca_columns,
        "image_pca": image_pca,
        "image_scaler": None,
        "image_only_scaler": image_only_scaler,
        "X_train_image_only": X_train_image_only,
        "X_test_image_only": X_test_image_only,
        "X_train_image_only_scaled": X_train_image_only_scaled,
        "X_test_image_only_scaled": X_test_image_only_scaled,
        "with_images_scaler": with_images_scaler,
        "with_images_pca": with_images_pca,
        "has_image_features": True,
        "image_valid_count": len(X_train_original) + len(X_test_original),
        "image_explained_variance_table": _build_explained_variance_table(
            image_pca_columns,
            image_pca,
        ),
        "with_images_explained_variance_table": _build_explained_variance_table(
            with_images_pca_columns,
            with_images_pca,
        ),
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

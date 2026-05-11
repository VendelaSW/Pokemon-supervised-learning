"""
Steg 5 — Modellträning och utvärdering
=======================================
Tränar Logistic Regression och Random Forest Classifier på tre
olika feature-uppsättningar:

  1. Tabell     — enbart numeriska stats och kodade kategorier
  2. Bild (PCA) — pixeldata dimensionsreducerad med PCA
  3. Kombinerat — tabell + bild-PCA ihopslaget

PCA reducerar bildfeatures från tusentals pixelvärden till ett
mindre antal komponenter som bevarar en vald andel av variansen.
StandardScaler används på båda featuretyper innan modellering.

Varje kombination av modell × features utvärderas med accuracy
och classification_report.  Slutligen skrivs en sammanfattande
jämförelsetabell ut.
"""

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from settings import (
    PCA_VARIANCE,
    TEST_RATIO,
    RANDOM_STATE,
    LR_MAX_ITER,
    RF_ESTIMATORS,
)


def train_and_evaluate(data: dict, label_maps: dict) -> pd.DataFrame:
    """Tränar modeller på tre feature-uppsättningar och skriver ut resultat.

    Steg:
      1. Stratifierad train/test-split (gemensam för alla uppsättningar)
      2. StandardScaler på tabellfeatures
      3. StandardScaler + PCA på bildfeatures
      4. Träning och utvärdering av varje modell × feature-kombination

    Returnerar
    ----------
    results_df : pd.DataFrame — accuracy per modell och feature-uppsättning
    """
    print("\n── Modellträning ──────────────────────────────────")

    y         = data["y"]
    X_tabular = data["X_tabular"]
    X_image   = data["X_image_raw"]

    # Inverterad kodning för utskrift av typnamn
    inv_type1 = {v: k for k, v in label_maps["type_1"].items()}
    type_names = [inv_type1[i] for i in sorted(inv_type1.keys())]

    # ── Gemensam train/test-split ────────────────────────────
    idx = np.arange(len(y))
    idx_train, idx_test, y_train, y_test = train_test_split(
        idx, y,
        test_size=TEST_RATIO,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"  Träningsdata: {len(idx_train)} rader")
    print(f"  Testdata:     {len(idx_test)} rader")

    # ── Skala tabellfeatures ─────────────────────────────────
    tab_scaler = StandardScaler()
    X_tab_train = tab_scaler.fit_transform(X_tabular[idx_train])
    X_tab_test = tab_scaler.transform(X_tabular[idx_test])

    # ── PCA på bildfeatures ──────────────────────────────────
    # Standardisera först, sedan reducera med PCA.
    # PCA_VARIANCE (t.ex. 0.95) anger att vi behåller så många
    # komponenter att 95 % av variansen bevaras.
    img_scaler = StandardScaler()
    X_img_train = img_scaler.fit_transform(X_image[idx_train])
    X_img_test = img_scaler.transform(X_image[idx_test])

    pca = PCA(n_components=PCA_VARIANCE, random_state=RANDOM_STATE)
    X_img_train_pca = pca.fit_transform(X_img_train)
    X_img_test_pca = pca.transform(X_img_test)

    explained = pca.explained_variance_ratio_.sum() * 100
    print(f"  PCA: {X_image.shape[1]} bildfeatures → {pca.n_components_} komponenter ({explained:.1f} % varians bevarad)")

    # ── Kombinerade features ─────────────────────────────────
    X_comb_train = np.hstack([X_tab_train, X_img_train_pca])
    X_comb_test = np.hstack([X_tab_test, X_img_test_pca])

    # ── Feature-uppsättningar ────────────────────────────────
    feature_sets = {
        "tabular":  (X_tab_train, X_tab_test),
        "image_pca": (X_img_train_pca, X_img_test_pca),
        "combined": (X_comb_train, X_comb_test),
    }

    # ── Modeller ─────────────────────────────────────────────
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=LR_MAX_ITER,
            solver="lbfgs",
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=RF_ESTIMATORS,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "XGBClassifier": XGBClassifier(
            random_state=RANDOM_STATE
        )
    }

    # ── Träna och utvärdera alla kombinationer ────────────────
    results = []

    for feat_name, (X_train, X_test) in feature_sets.items():
        for model_name, model_template in models.items():
            model = clone(model_template)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)

            results.append({
                "model": model_name,
                "features": feat_name,
                "accuracy": acc,
            })

            print(f"\n{'=' * 65}")
            print(f"  {model_name}  |  Features: {feat_name}")
            print(f"  Accuracy: {acc:.4f}  ({acc * 100:.1f} %)")
            print(f"{'=' * 65}")
            print(classification_report(
                y_test, y_pred,
                target_names=type_names,
                zero_division=0,
            ))

    # ── Sammanfattning ───────────────────────────────────────
    print("\n" + "=" * 65)
    print("  SAMMANFATTNING")
    print("=" * 65)

    results_df = pd.DataFrame(results)
    pivot = results_df.pivot(index="features", columns="model", values="accuracy")
    pivot = pivot.reindex(["tabular", "image_pca", "combined"])
    pivot = (pivot * 100).round(1)
    print(pivot.to_string())

    best = results_df.loc[results_df["accuracy"].idxmax()]
    n_classes = len(type_names)
    print(f"\nBäst: {best['model']} med {best['features']} → {best['accuracy'] * 100:.1f} %")
    print(f"(Slumpbaslinje för {n_classes} klasser: ~{100 / n_classes:.1f} %)")

    return results_df

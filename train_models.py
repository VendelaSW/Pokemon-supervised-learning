"""
Modellträning — Logreg, XGBoost och utvärdering
===============================================
Tränar upp till elva modellflöden på den förberedda träningsdatan:
logistisk regression på PCA-features, XGBoost på fulla training
features och XGBoost på stripped features. Om image features finns
körs motsvarande tre flöden även med PCA-komprimerade sprite-features.
RandomForest körs som en fast baseline med och utan image features.
Sist körs tre image-only-flöden som bara använder image_pca-features.

Modulen sparar tränade modeller som pkl och skapar fokuserade PNG-
figurer för jämförelse, confusion matrix och feature importance.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.utils.class_weight import compute_sample_weight
from tqdm.auto import tqdm
from xgboost import XGBClassifier

from settings import (
    GRID_SEARCH_CV,
    LR_MAX_ITER,
    MODEL_OUTPUT_DIR,
    PERMUTATION_REPEATS,
    RANDOM_STATE,
    RF_ESTIMATORS,
    RF_MAX_DEPTH,
    RF_MIN_SAMPLES_LEAF,
    XGB_PARAM_GRID,
)

MPLCONFIGDIR = Path(".matplotlib_cache")
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR.resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(MPLCONFIGDIR.resolve()))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns


DUMMY_SOURCE_COLUMNS = [
    "color",
    "shape",
    "habitat",
    "growth_rate",
]

STRIPPED_FEATURE_PREFIXES = tuple(
    f"{column}_"
    for column in DUMMY_SOURCE_COLUMNS
)
STRIPPED_NUMERIC_FEATURES = [
    "sp_attack",
]


def train_models(
    training_data: dict,
    output_dir: str | Path = MODEL_OUTPUT_DIR,
    run_label: str = "pokemon_type_training",
) -> dict:
    """Kör modellflödena och sparar modeller samt PNG-figurer."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    has_image_features = training_data.get("has_image_features", False)
    total_steps = 11 if has_image_features else 4

    print(f"\n-- Körning: {run_label} ----------------------------------------")
    result_items = []
    with tqdm(total=total_steps, desc="Modellträning", unit="modell") as progress:
        logreg_results = train_multinomial_regression(
            training_data,
            output_path,
            run_label,
            total_steps,
        )
        progress.update(1)
        xgboost_results = train_xgboost_grid_search(
            training_data,
            output_path,
            run_label,
            total_steps,
        )
        progress.update(1)
        xgboost_stripped_results = train_xgboost_stripped_features(
            training_data,
            output_path,
            run_label,
            total_steps,
        )
        progress.update(1)
        result_items.extend(
            [
                ("logreg", logreg_results),
                ("xgboost_grid_search", xgboost_results),
                ("xgboost_stripped_features", xgboost_stripped_results),
            ]
        )

        if has_image_features:
            logreg_with_images_results = train_multinomial_regression_with_images(
                training_data,
                output_path,
                run_label,
                total_steps,
            )
            progress.update(1)
            xgboost_with_images_results = train_xgboost_grid_search_with_images(
                training_data,
                output_path,
                run_label,
                total_steps,
            )
            progress.update(1)
            xgboost_stripped_with_images_results = (
                train_xgboost_stripped_features_with_images(
                    training_data,
                    output_path,
                    run_label,
                    total_steps,
                )
            )
            progress.update(1)
            result_items.extend(
                [
                    ("logreg_pca_with_images", logreg_with_images_results),
                    ("xgboost_grid_search_with_images", xgboost_with_images_results),
                    (
                        "xgboost_stripped_features_with_images",
                        xgboost_stripped_with_images_results,
                    ),
                ]
            )

        random_forest_step = 7 if has_image_features else 4
        random_forest_results = train_random_forest(
            training_data,
            output_path,
            run_label,
            total_steps,
            random_forest_step,
        )
        progress.update(1)
        result_items.append(("random_forest", random_forest_results))

        if has_image_features:
            random_forest_with_images_results = train_random_forest_with_images(
                training_data,
                output_path,
                run_label,
                total_steps,
            )
            progress.update(1)
            result_items.append(
                ("random_forest_with_images", random_forest_with_images_results)
            )

            logreg_image_only_results = train_logreg_image_only(
                training_data,
                output_path,
                run_label,
                total_steps,
            )
            progress.update(1)
            xgboost_image_only_results = train_xgboost_image_only(
                training_data,
                output_path,
                run_label,
                total_steps,
            )
            progress.update(1)
            random_forest_image_only_results = train_random_forest_image_only(
                training_data,
                output_path,
                run_label,
                total_steps,
            )
            progress.update(1)
            result_items.extend(
                [
                    ("logreg_image_only", logreg_image_only_results),
                    ("xgboost_image_only", xgboost_image_only_results),
                    ("random_forest_image_only", random_forest_image_only_results),
                ]
            )

    comparison = pd.DataFrame(
        [results["metrics"] for _, results in result_items]
    )
    _save_model_comparison_plot(comparison, output_path, run_label)

    print("\n-- Modelljämförelse --------------------------------------------")
    print(f"Körning: {run_label}")
    print(comparison.round(3).to_string(index=False))
    print(f"\nViktiga modellfiler och PNG-figurer sparade i: {output_path}")

    results = {name: result for name, result in result_items}
    results["model_comparison"] = comparison
    return results


def train_multinomial_regression(
    training_data: dict,
    output_dir: Path,
    run_label: str,
    total_steps: int = 3,
) -> dict:
    """Tränar multinomial logistisk regression på PCA-transformerad data."""
    print(f"\n-- Träning 1/{total_steps}: Multinomial logistisk regression ---------------")
    _print_feature_columns(
        "logreg_pca: originalfeatures före scaler/PCA",
        training_data["feature_columns"],
    )
    _print_feature_columns(
        "logreg_pca: PCA-komponenter som modellen tränas på",
        training_data["pca_columns"],
    )

    model = LogisticRegression(
        solver="lbfgs",
        max_iter=LR_MAX_ITER,
        random_state=RANDOM_STATE,
    )
    model.fit(training_data["X_train_pca"], training_data["y_train"])
    y_pred = model.predict(training_data["X_test_pca"])

    results = _evaluate_model(
        model_name="logreg_pca",
        y_test=training_data["y_test"],
        y_pred=y_pred,
        target_labels=training_data.get("target_labels", {}),
        output_dir=output_dir,
        run_label=run_label,
    )
    joblib.dump(
        {
            "model": model,
            "scaler": training_data["scaler"],
            "pca": training_data["pca"],
            "pca_columns": training_data["pca_columns"],
            "target_labels": training_data.get("target_labels", {}),
        },
        output_dir / f"{run_label}_logreg_pca_model.pkl",
    )
    results["model"] = model
    return results


def train_multinomial_regression_with_images(
    training_data: dict,
    output_dir: Path,
    run_label: str,
    total_steps: int,
) -> dict:
    """Tränar logistisk regression på PCA av tabellfeatures och image_pca."""
    print(
        f"\n-- Träning 4/{total_steps}: "
        "Multinomial logistisk regression med bildfeatures --"
    )
    _print_feature_columns(
        "logreg_pca_with_images: features före scaler/PCA",
        training_data["with_images_feature_columns"],
    )
    _print_feature_columns(
        "logreg_pca_with_images: PCA-komponenter som modellen tränas på",
        training_data["with_images_pca_columns"],
    )

    model = LogisticRegression(
        solver="lbfgs",
        max_iter=LR_MAX_ITER,
        random_state=RANDOM_STATE,
    )
    model.fit(training_data["X_train_with_images_pca"], training_data["y_train"])
    y_pred = model.predict(training_data["X_test_with_images_pca"])

    results = _evaluate_model(
        model_name="logreg_pca_with_images",
        y_test=training_data["y_test"],
        y_pred=y_pred,
        target_labels=training_data.get("target_labels", {}),
        output_dir=output_dir,
        run_label=run_label,
    )
    joblib.dump(
        {
            "model": model,
            "image_scaler": training_data["image_scaler"],
            "image_pca": training_data["image_pca"],
            "image_pca_columns": training_data["image_pca_columns"],
            "with_images_scaler": training_data["with_images_scaler"],
            "with_images_pca": training_data["with_images_pca"],
            "with_images_pca_columns": training_data["with_images_pca_columns"],
            "with_images_feature_columns": training_data["with_images_feature_columns"],
            "target_labels": training_data.get("target_labels", {}),
        },
        output_dir / f"{run_label}_logreg_pca_with_images_model.pkl",
    )
    results["model"] = model
    return results


def train_xgboost_stripped_features(
    training_data: dict,
    output_dir: Path,
    run_label: str,
    total_steps: int = 3,
) -> dict:
    """Tränar XGBoost på dummy-kategorier plus utvalda numeriska features."""
    print(
        f"\n-- Träning 3/{total_steps}: "
        "XGBoost stripped features med GridSearchCV -----"
    )

    X_train, X_test, stripped_columns = _strip_xgboost_features(training_data)
    y_train = training_data["y_train"]
    y_test = training_data["y_test"]
    sample_weight = _balanced_sample_weight(y_train)
    _print_feature_columns(
        "xgboost_stripped_features: features som modellen tränas på",
        stripped_columns,
    )

    base_model = XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        num_class=len(np.unique(y_train)),
        tree_method="hist",
        n_jobs=1,
        random_state=RANDOM_STATE,
    )
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=XGB_PARAM_GRID,
        scoring="balanced_accuracy",
        cv=GRID_SEARCH_CV,
        n_jobs=-1,
        refit=True,
        verbose=0,
    )
    _fit_grid_search_with_progress(
        grid_search,
        X_train,
        y_train,
        sample_weight=sample_weight,
        description="xgboost_stripped_features",
    )

    model = grid_search.best_estimator_
    y_pred = model.predict(X_test)
    results = _evaluate_model(
        model_name="xgboost_stripped_features",
        y_test=y_test,
        y_pred=y_pred,
        target_labels=training_data.get("target_labels", {}),
        output_dir=output_dir,
        run_label=run_label,
    )
    joblib.dump(
        {
            "model": model,
            "feature_columns": stripped_columns,
            "target_labels": training_data.get("target_labels", {}),
            "best_params": grid_search.best_params_,
            "best_cv_score": grid_search.best_score_,
            "sample_weight": "balanced",
        },
        output_dir / f"{run_label}_xgboost_stripped_features_model.pkl",
    )

    importance = _save_feature_importance(
        model,
        stripped_columns,
        output_dir,
        run_label,
        model_name="xgboost_stripped_features",
    )
    permutation = _save_permutation_importance(
        model,
        X_test,
        y_test,
        output_dir,
        run_label,
        model_name="xgboost_stripped_features",
    )

    print(f"Bästa stripped XGBoost-parametrar: {grid_search.best_params_}")
    print(f"Bästa stripped CV balanced accuracy: {grid_search.best_score_:.3f}")

    results["model"] = model
    results["best_params"] = grid_search.best_params_
    results["best_cv_score"] = grid_search.best_score_
    results["feature_importance"] = importance
    results["permutation_importance"] = permutation
    return results


def train_xgboost_grid_search(
    training_data: dict,
    output_dir: Path,
    run_label: str,
    total_steps: int = 3,
) -> dict:
    """Tränar XGBoost på alla originalfeatures från df_training."""
    print(f"\n-- Träning 2/{total_steps}: XGBoost med GridSearchCV -----------------------")

    X_train = training_data["X_train_original"]
    X_test = training_data["X_test_original"]
    y_train = training_data["y_train"]
    y_test = training_data["y_test"]
    sample_weight = _balanced_sample_weight(y_train)
    _print_feature_columns(
        "xgboost_grid_search: features som modellen tränas på",
        training_data["feature_columns"],
    )

    base_model = XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        num_class=len(np.unique(y_train)),
        tree_method="hist",
        n_jobs=1,
        random_state=RANDOM_STATE,
    )
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=XGB_PARAM_GRID,
        scoring="balanced_accuracy",
        cv=GRID_SEARCH_CV,
        n_jobs=-1,
        refit=True,
        verbose=0,
    )
    _fit_grid_search_with_progress(
        grid_search,
        X_train,
        y_train,
        sample_weight=sample_weight,
        description="xgboost_grid_search",
    )

    model = grid_search.best_estimator_
    y_pred = model.predict(X_test)
    results = _evaluate_model(
        model_name="xgboost_grid_search",
        y_test=y_test,
        y_pred=y_pred,
        target_labels=training_data.get("target_labels", {}),
        output_dir=output_dir,
        run_label=run_label,
    )
    joblib.dump(
        {
            "model": model,
            "feature_columns": training_data["feature_columns"],
            "target_labels": training_data.get("target_labels", {}),
            "best_params": grid_search.best_params_,
            "best_cv_score": grid_search.best_score_,
            "sample_weight": "balanced",
        },
        output_dir / f"{run_label}_xgboost_grid_search_model.pkl",
    )

    importance = _save_feature_importance(
        model,
        training_data["feature_columns"],
        output_dir,
        run_label,
        model_name="xgboost_grid_search",
    )
    permutation = _save_permutation_importance(
        model,
        X_test,
        y_test,
        output_dir,
        run_label,
        model_name="xgboost_grid_search",
    )

    print(f"Bästa XGBoost-parametrar: {grid_search.best_params_}")
    print(f"Bästa CV balanced accuracy: {grid_search.best_score_:.3f}")

    results["model"] = model
    results["best_params"] = grid_search.best_params_
    results["best_cv_score"] = grid_search.best_score_
    results["feature_importance"] = importance
    results["permutation_importance"] = permutation
    return results


def train_xgboost_grid_search_with_images(
    training_data: dict,
    output_dir: Path,
    run_label: str,
    total_steps: int,
) -> dict:
    """Tränar XGBoost på tabellfeatures och PCA-komprimerade bildfeatures."""
    print(
        f"\n-- Träning 5/{total_steps}: "
        "XGBoost med image features och GridSearchCV ------"
    )

    X_train = training_data["X_train_with_images"]
    X_test = training_data["X_test_with_images"]
    y_train = training_data["y_train"]
    y_test = training_data["y_test"]
    sample_weight = _balanced_sample_weight(y_train)
    feature_columns = training_data["with_images_feature_columns"]
    _print_feature_columns(
        "xgboost_grid_search_with_images: features som modellen tränas på",
        feature_columns,
    )

    base_model = XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        num_class=len(np.unique(y_train)),
        tree_method="hist",
        n_jobs=1,
        random_state=RANDOM_STATE,
    )
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=XGB_PARAM_GRID,
        scoring="balanced_accuracy",
        cv=GRID_SEARCH_CV,
        n_jobs=-1,
        refit=True,
        verbose=0,
    )
    _fit_grid_search_with_progress(
        grid_search,
        X_train,
        y_train,
        sample_weight=sample_weight,
        description="xgboost_grid_search_with_images",
    )

    model = grid_search.best_estimator_
    y_pred = model.predict(X_test)
    results = _evaluate_model(
        model_name="xgboost_grid_search_with_images",
        y_test=y_test,
        y_pred=y_pred,
        target_labels=training_data.get("target_labels", {}),
        output_dir=output_dir,
        run_label=run_label,
    )
    joblib.dump(
        {
            "model": model,
            "feature_columns": feature_columns,
            "image_scaler": training_data["image_scaler"],
            "image_pca": training_data["image_pca"],
            "image_pca_columns": training_data["image_pca_columns"],
            "target_labels": training_data.get("target_labels", {}),
            "best_params": grid_search.best_params_,
            "best_cv_score": grid_search.best_score_,
            "sample_weight": "balanced",
        },
        output_dir / f"{run_label}_xgboost_grid_search_with_images_model.pkl",
    )

    importance = _save_feature_importance(
        model,
        feature_columns,
        output_dir,
        run_label,
        model_name="xgboost_grid_search_with_images",
    )
    permutation = _save_permutation_importance(
        model,
        X_test,
        y_test,
        output_dir,
        run_label,
        model_name="xgboost_grid_search_with_images",
    )

    print(f"Bästa XGBoost-parametrar med bild: {grid_search.best_params_}")
    print(f"Bästa CV balanced accuracy med bild: {grid_search.best_score_:.3f}")

    results["model"] = model
    results["best_params"] = grid_search.best_params_
    results["best_cv_score"] = grid_search.best_score_
    results["feature_importance"] = importance
    results["permutation_importance"] = permutation
    return results


def train_xgboost_stripped_features_with_images(
    training_data: dict,
    output_dir: Path,
    run_label: str,
    total_steps: int,
) -> dict:
    """Tränar stripped XGBoost på tabellfeatures och image_pca."""
    print(
        f"\n-- Träning 6/{total_steps}: "
        "XGBoost stripped features med image features --------"
    )

    X_train, X_test, stripped_columns = _strip_xgboost_features(
        training_data,
        include_image_features=True,
    )
    y_train = training_data["y_train"]
    y_test = training_data["y_test"]
    sample_weight = _balanced_sample_weight(y_train)
    _print_feature_columns(
        "xgboost_stripped_features_with_images: features som modellen tränas på",
        stripped_columns,
    )

    base_model = XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        num_class=len(np.unique(y_train)),
        tree_method="hist",
        n_jobs=1,
        random_state=RANDOM_STATE,
    )
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=XGB_PARAM_GRID,
        scoring="balanced_accuracy",
        cv=GRID_SEARCH_CV,
        n_jobs=-1,
        refit=True,
        verbose=0,
    )
    _fit_grid_search_with_progress(
        grid_search,
        X_train,
        y_train,
        sample_weight=sample_weight,
        description="xgboost_stripped_features_with_images",
    )

    model = grid_search.best_estimator_
    y_pred = model.predict(X_test)
    results = _evaluate_model(
        model_name="xgboost_stripped_features_with_images",
        y_test=y_test,
        y_pred=y_pred,
        target_labels=training_data.get("target_labels", {}),
        output_dir=output_dir,
        run_label=run_label,
    )
    joblib.dump(
        {
            "model": model,
            "feature_columns": stripped_columns,
            "image_scaler": training_data["image_scaler"],
            "image_pca": training_data["image_pca"],
            "image_pca_columns": training_data["image_pca_columns"],
            "target_labels": training_data.get("target_labels", {}),
            "best_params": grid_search.best_params_,
            "best_cv_score": grid_search.best_score_,
            "sample_weight": "balanced",
        },
        output_dir / f"{run_label}_xgboost_stripped_features_with_images_model.pkl",
    )

    importance = _save_feature_importance(
        model,
        stripped_columns,
        output_dir,
        run_label,
        model_name="xgboost_stripped_features_with_images",
    )
    permutation = _save_permutation_importance(
        model,
        X_test,
        y_test,
        output_dir,
        run_label,
        model_name="xgboost_stripped_features_with_images",
    )

    print(f"Bästa stripped XGBoost-parametrar med bild: {grid_search.best_params_}")
    print(f"Bästa stripped CV balanced accuracy med bild: {grid_search.best_score_:.3f}")

    results["model"] = model
    results["best_params"] = grid_search.best_params_
    results["best_cv_score"] = grid_search.best_score_
    results["feature_importance"] = importance
    results["permutation_importance"] = permutation
    return results


def train_random_forest(
    training_data: dict,
    output_dir: Path,
    run_label: str,
    total_steps: int,
    step_number: int,
) -> dict:
    """Tränar RandomForest på fulla tabellfeatures."""
    print(
        f"\n-- Träning {step_number}/{total_steps}: "
        "RandomForest på training features -------------------"
    )

    X_train = training_data["X_train_original"]
    X_test = training_data["X_test_original"]
    y_train = training_data["y_train"]
    y_test = training_data["y_test"]
    feature_columns = training_data["feature_columns"]
    _print_feature_columns(
        "random_forest: features som modellen tränas på",
        feature_columns,
    )

    model = _make_random_forest()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results = _evaluate_model(
        model_name="random_forest",
        y_test=y_test,
        y_pred=y_pred,
        target_labels=training_data.get("target_labels", {}),
        output_dir=output_dir,
        run_label=run_label,
    )
    joblib.dump(
        {
            "model": model,
            "feature_columns": feature_columns,
            "target_labels": training_data.get("target_labels", {}),
            "class_weight": "balanced",
        },
        output_dir / f"{run_label}_random_forest_model.pkl",
    )

    importance = _save_feature_importance(
        model,
        feature_columns,
        output_dir,
        run_label,
        model_name="random_forest",
    )
    permutation = _save_permutation_importance(
        model,
        X_test,
        y_test,
        output_dir,
        run_label,
        model_name="random_forest",
    )

    results["model"] = model
    results["feature_importance"] = importance
    results["permutation_importance"] = permutation
    return results


def train_random_forest_with_images(
    training_data: dict,
    output_dir: Path,
    run_label: str,
    total_steps: int,
) -> dict:
    """Tränar RandomForest på tabellfeatures och image_pca."""
    print(
        f"\n-- Träning 8/{total_steps}: "
        "RandomForest med image features --------------------"
    )

    X_train = training_data["X_train_with_images"]
    X_test = training_data["X_test_with_images"]
    y_train = training_data["y_train"]
    y_test = training_data["y_test"]
    feature_columns = training_data["with_images_feature_columns"]
    _print_feature_columns(
        "random_forest_with_images: features som modellen tränas på",
        feature_columns,
    )

    model = _make_random_forest()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results = _evaluate_model(
        model_name="random_forest_with_images",
        y_test=y_test,
        y_pred=y_pred,
        target_labels=training_data.get("target_labels", {}),
        output_dir=output_dir,
        run_label=run_label,
    )
    joblib.dump(
        {
            "model": model,
            "feature_columns": feature_columns,
            "image_scaler": training_data["image_scaler"],
            "image_pca": training_data["image_pca"],
            "image_pca_columns": training_data["image_pca_columns"],
            "target_labels": training_data.get("target_labels", {}),
            "class_weight": "balanced",
        },
        output_dir / f"{run_label}_random_forest_with_images_model.pkl",
    )

    importance = _save_feature_importance(
        model,
        feature_columns,
        output_dir,
        run_label,
        model_name="random_forest_with_images",
    )
    permutation = _save_permutation_importance(
        model,
        X_test,
        y_test,
        output_dir,
        run_label,
        model_name="random_forest_with_images",
    )

    results["model"] = model
    results["feature_importance"] = importance
    results["permutation_importance"] = permutation
    return results


def train_logreg_image_only(
    training_data: dict,
    output_dir: Path,
    run_label: str,
    total_steps: int,
) -> dict:
    """Tränar logistisk regression endast på skalade image_pca-features."""
    print(
        f"\n-- Träning 9/{total_steps}: "
        "Multinomial logistisk regression image-only ------"
    )

    feature_columns = training_data["image_only_feature_columns"]
    _print_feature_columns(
        "logreg_image_only: image_pca-features som modellen tränas på",
        feature_columns,
    )

    model = LogisticRegression(
        solver="lbfgs",
        max_iter=LR_MAX_ITER,
        random_state=RANDOM_STATE,
    )
    model.fit(training_data["X_train_image_only_scaled"], training_data["y_train"])
    y_pred = model.predict(training_data["X_test_image_only_scaled"])

    results = _evaluate_model(
        model_name="logreg_image_only",
        y_test=training_data["y_test"],
        y_pred=y_pred,
        target_labels=training_data.get("target_labels", {}),
        output_dir=output_dir,
        run_label=run_label,
    )
    joblib.dump(
        {
            "model": model,
            "image_only_scaler": training_data["image_only_scaler"],
            "image_pca": training_data["image_pca"],
            "image_pca_columns": training_data["image_pca_columns"],
            "image_only_feature_columns": feature_columns,
            "target_labels": training_data.get("target_labels", {}),
        },
        output_dir / f"{run_label}_logreg_image_only_model.pkl",
    )
    results["model"] = model
    return results


def train_xgboost_image_only(
    training_data: dict,
    output_dir: Path,
    run_label: str,
    total_steps: int,
) -> dict:
    """Tränar XGBoost endast på image_pca-features."""
    print(
        f"\n-- Träning 10/{total_steps}: "
        "XGBoost image-only med GridSearchCV -------------"
    )

    X_train = training_data["X_train_image_only"]
    X_test = training_data["X_test_image_only"]
    y_train = training_data["y_train"]
    y_test = training_data["y_test"]
    feature_columns = training_data["image_only_feature_columns"]
    sample_weight = _balanced_sample_weight(y_train)
    _print_feature_columns(
        "xgboost_image_only: image_pca-features som modellen tränas på",
        feature_columns,
    )

    base_model = XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        num_class=len(np.unique(y_train)),
        tree_method="hist",
        n_jobs=1,
        random_state=RANDOM_STATE,
    )
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=XGB_PARAM_GRID,
        scoring="balanced_accuracy",
        cv=GRID_SEARCH_CV,
        n_jobs=-1,
        refit=True,
        verbose=0,
    )
    _fit_grid_search_with_progress(
        grid_search,
        X_train,
        y_train,
        sample_weight=sample_weight,
        description="xgboost_image_only",
    )

    model = grid_search.best_estimator_
    y_pred = model.predict(X_test)
    results = _evaluate_model(
        model_name="xgboost_image_only",
        y_test=y_test,
        y_pred=y_pred,
        target_labels=training_data.get("target_labels", {}),
        output_dir=output_dir,
        run_label=run_label,
    )
    joblib.dump(
        {
            "model": model,
            "feature_columns": feature_columns,
            "image_pca": training_data["image_pca"],
            "image_pca_columns": training_data["image_pca_columns"],
            "target_labels": training_data.get("target_labels", {}),
            "best_params": grid_search.best_params_,
            "best_cv_score": grid_search.best_score_,
            "sample_weight": "balanced",
        },
        output_dir / f"{run_label}_xgboost_image_only_model.pkl",
    )

    importance = _save_feature_importance(
        model,
        feature_columns,
        output_dir,
        run_label,
        model_name="xgboost_image_only",
    )
    permutation = _save_permutation_importance(
        model,
        X_test,
        y_test,
        output_dir,
        run_label,
        model_name="xgboost_image_only",
    )

    print(f"Bästa image-only XGBoost-parametrar: {grid_search.best_params_}")
    print(f"Bästa image-only CV balanced accuracy: {grid_search.best_score_:.3f}")

    results["model"] = model
    results["best_params"] = grid_search.best_params_
    results["best_cv_score"] = grid_search.best_score_
    results["feature_importance"] = importance
    results["permutation_importance"] = permutation
    return results


def train_random_forest_image_only(
    training_data: dict,
    output_dir: Path,
    run_label: str,
    total_steps: int,
) -> dict:
    """Tränar RandomForest endast på image_pca-features."""
    print(
        f"\n-- Träning 11/{total_steps}: "
        "RandomForest image-only ---------------------------"
    )

    X_train = training_data["X_train_image_only"]
    X_test = training_data["X_test_image_only"]
    y_train = training_data["y_train"]
    y_test = training_data["y_test"]
    feature_columns = training_data["image_only_feature_columns"]
    _print_feature_columns(
        "random_forest_image_only: image_pca-features som modellen tränas på",
        feature_columns,
    )

    model = _make_random_forest()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results = _evaluate_model(
        model_name="random_forest_image_only",
        y_test=y_test,
        y_pred=y_pred,
        target_labels=training_data.get("target_labels", {}),
        output_dir=output_dir,
        run_label=run_label,
    )
    joblib.dump(
        {
            "model": model,
            "feature_columns": feature_columns,
            "image_pca": training_data["image_pca"],
            "image_pca_columns": training_data["image_pca_columns"],
            "target_labels": training_data.get("target_labels", {}),
            "class_weight": "balanced",
        },
        output_dir / f"{run_label}_random_forest_image_only_model.pkl",
    )

    importance = _save_feature_importance(
        model,
        feature_columns,
        output_dir,
        run_label,
        model_name="random_forest_image_only",
    )
    permutation = _save_permutation_importance(
        model,
        X_test,
        y_test,
        output_dir,
        run_label,
        model_name="random_forest_image_only",
    )

    results["model"] = model
    results["feature_importance"] = importance
    results["permutation_importance"] = permutation
    return results


def _make_random_forest() -> RandomForestClassifier:
    """Skapar RandomForest-baseline med gemensamma settings."""
    return RandomForestClassifier(
        n_estimators=RF_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        min_samples_leaf=RF_MIN_SAMPLES_LEAF,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )


def _balanced_sample_weight(y_train: pd.Series) -> np.ndarray:
    """Skapar sample weights så ovanliga klasser väger mer vid XGBoost-träning."""
    return compute_sample_weight(class_weight="balanced", y=y_train)


@contextmanager
def _tqdm_joblib(progress_bar: tqdm):
    """Kopplar joblib-parallellism till en tqdm-progressbar."""
    old_callback = joblib.parallel.BatchCompletionCallBack

    class TqdmBatchCompletionCallback(old_callback):
        def __call__(self, *args, **kwargs):
            progress_bar.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield progress_bar
    finally:
        joblib.parallel.BatchCompletionCallBack = old_callback
        progress_bar.close()


def _fit_grid_search_with_progress(
    grid_search: GridSearchCV,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    sample_weight: np.ndarray,
    description: str,
) -> None:
    """Fit:ar GridSearchCV och visar progress för varje CV-fit."""
    total_fits = len(ParameterGrid(grid_search.param_grid)) * grid_search.cv
    progress_bar = tqdm(
        total=total_fits,
        desc=description,
        unit="fit",
        leave=False,
    )
    with _tqdm_joblib(progress_bar):
        grid_search.fit(X_train, y_train, sample_weight=sample_weight)


def _print_feature_columns(title: str, columns: list[str]) -> None:
    """Skriver ut hela feature-listan utan trunkering."""
    print(f"\n{title}")
    print(f"Antal features: {len(columns)}")
    for index, column in enumerate(columns, start=1):
        print(f"{index:03d}. {column}")


def _evaluate_model(
    *,
    model_name: str,
    y_test: pd.Series,
    y_pred: np.ndarray,
    target_labels: dict[int, str],
    output_dir: Path,
    run_label: str,
) -> dict:
    """Beräknar kärn-metrics och sparar confusion matrix som PNG."""
    class_labels = np.array(sorted(y_test.unique()))
    class_names = _label_names(class_labels, target_labels)
    confusion = pd.DataFrame(
        confusion_matrix(y_test, y_pred, labels=class_labels),
        index=class_names,
        columns=class_names,
    )
    _save_confusion_matrix_plot(confusion, model_name, output_dir, run_label)

    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "macro_f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
    }
    print(
        f"{model_name}: accuracy={metrics['accuracy']:.3f}, "
        f"balanced_accuracy={metrics['balanced_accuracy']:.3f}, "
        f"macro_f1={metrics['macro_f1']:.3f}"
    )
    print(f"\nClassification report för {model_name} ({run_label}):")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=class_labels,
            target_names=class_names,
            zero_division=0,
        )
    )
    return {
        "metrics": metrics,
        "confusion_matrix": confusion,
    }


def _strip_xgboost_features(
    training_data: dict,
    include_image_features: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Väljer stripped tabellfeatures och eventuella image_pca-features."""
    if include_image_features:
        all_feature_columns = training_data["with_images_feature_columns"]
        X_train = training_data["X_train_with_images"]
        X_test = training_data["X_test_with_images"]
    else:
        all_feature_columns = training_data["feature_columns"]
        X_train = training_data["X_train_original"]
        X_test = training_data["X_test_original"]

    feature_columns = [
        column
        for column in all_feature_columns
        if column in STRIPPED_NUMERIC_FEATURES
        or column.startswith(STRIPPED_FEATURE_PREFIXES)
        or (include_image_features and column.startswith("image_pca_"))
    ]
    if not feature_columns:
        raise ValueError("Hittade inga stripped features för XGBoost.")
    return (
        X_train[feature_columns],
        X_test[feature_columns],
        feature_columns,
    )


def _label_names(labels: np.ndarray, target_labels: dict[int, str]) -> list[str]:
    """Hämtar läsbara klassnamn för figurer om mappningen finns."""
    return [
        target_labels.get(int(label), str(label))
        for label in labels
    ]


def _save_model_comparison_plot(
    comparison: pd.DataFrame,
    output_dir: Path,
    run_label: str,
) -> None:
    """Sparar en kompakt PNG som jämför modellernas viktigaste metrics."""
    plot_data = comparison.set_index("model")[
        ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"]
    ]
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_data.T.plot(kind="bar", ax=ax)
    ax.set_title(f"Modelljämförelse: {run_label}")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=30)
    ax.legend(title="Modell")
    fig.tight_layout()
    fig.savefig(output_dir / f"{run_label}_model_comparison.png", dpi=150)
    plt.close(fig)


def _save_confusion_matrix_plot(
    confusion: pd.DataFrame,
    model_name: str,
    output_dir: Path,
    run_label: str,
) -> None:
    """Sparar confusion matrix som värmekarta."""
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title(f"Confusion matrix: {model_name} ({run_label})")
    ax.set_xlabel("Predikterad klass")
    ax.set_ylabel("Faktisk klass")
    fig.tight_layout()
    fig.savefig(output_dir / f"{run_label}_{model_name}_confusion_matrix.png", dpi=150)
    plt.close(fig)


def _save_feature_importance(
    model: XGBClassifier | RandomForestClassifier,
    feature_columns: list[str],
    output_dir: Path,
    run_label: str,
    model_name: str,
) -> dict:
    """Sparar modellens feature importance som PNG."""
    importance = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": model.feature_importances_,
        }
    )
    importance["feature_group"] = importance["feature"].map(_feature_group)
    importance = importance.sort_values("importance", ascending=False)
    grouped_importance = _group_importance(
        importance,
        value_column="importance",
        grouped_value_column="total_importance",
    )
    _save_importance_plot(
        importance,
        value_column="importance",
        title=f"Feature importance: {model_name} ({run_label})",
        output_path=output_dir / f"{run_label}_{model_name}_feature_importance.png",
    )
    _save_importance_plot(
        grouped_importance,
        feature_column="feature_group",
        value_column="total_importance",
        title=f"Grupperad feature importance: {model_name} ({run_label})",
        output_path=output_dir / f"{run_label}_{model_name}_grouped_feature_importance.png",
    )
    return {
        "feature_importance": importance,
        "grouped_feature_importance": grouped_importance,
    }


def _save_permutation_importance(
    model: XGBClassifier | RandomForestClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: Path,
    run_label: str,
    model_name: str,
) -> dict:
    """Beräknar permutation importance och sparar den som PNG."""
    print("Beräknar permutation importance...")
    permutation = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=PERMUTATION_REPEATS,
        random_state=RANDOM_STATE,
        scoring="balanced_accuracy",
        n_jobs=-1,
    )
    importance = pd.DataFrame(
        {
            "feature": X_test.columns,
            "importance_mean": permutation.importances_mean,
            "importance_std": permutation.importances_std,
        }
    )
    importance["feature_group"] = importance["feature"].map(_feature_group)
    importance = importance.sort_values("importance_mean", ascending=False)
    grouped_importance = _group_importance(
        importance,
        value_column="importance_mean",
        grouped_value_column="total_importance_mean",
    )
    _save_importance_plot(
        importance,
        value_column="importance_mean",
        title=f"Permutation importance: {model_name} ({run_label})",
        output_path=output_dir / f"{run_label}_{model_name}_permutation_importance.png",
    )
    _save_importance_plot(
        grouped_importance,
        feature_column="feature_group",
        value_column="total_importance_mean",
        title=f"Grupperad permutation importance: {model_name} ({run_label})",
        output_path=output_dir / f"{run_label}_{model_name}_grouped_permutation_importance.png",
    )
    return {
        "permutation_importance": importance,
        "grouped_permutation_importance": grouped_importance,
    }


def _feature_group(feature: str) -> str:
    """Grupperar dummy-features tillbaka till sin ursprungliga kolumn."""
    if feature.startswith("image_pca_"):
        return "image_pca"
    for source_column in DUMMY_SOURCE_COLUMNS:
        if feature.startswith(f"{source_column}_"):
            return source_column
    return feature


def _group_importance(
    importance: pd.DataFrame,
    *,
    value_column: str,
    grouped_value_column: str,
) -> pd.DataFrame:
    """Summerar importance per feature-grupp."""
    return (
        importance.groupby("feature_group", as_index=False)
        .agg(
            **{
                grouped_value_column: (value_column, "sum"),
                "feature_count": ("feature", "count"),
            }
        )
        .sort_values(grouped_value_column, ascending=False)
    )


def _save_importance_plot(
    importance: pd.DataFrame,
    *,
    value_column: str,
    title: str,
    output_path: Path,
    feature_column: str = "feature",
    top_n: int = 20,
) -> None:
    """Sparar ett horisontellt stapeldiagram för de viktigaste features."""
    plot_data = importance.head(top_n).sort_values(value_column, ascending=True)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(plot_data[feature_column], plot_data[value_column])
    ax.set_title(title)
    ax.set_xlabel(value_column)
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

from __future__ import annotations

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from settings import (
    GRID_SEARCH_CV,
    LR_MAX_ITER,
    MODEL_OUTPUT_DIR,
    PERMUTATION_REPEATS,
    RANDOM_STATE,
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
    """Kör de tre modellflödena och sparar modeller samt PNG-figurer."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n-- Körning: {run_label} ----------------------------------------")
    logreg_results = train_multinomial_regression(training_data, output_path, run_label)
    xgboost_results = train_xgboost_grid_search(training_data, output_path, run_label)
    xgboost_stripped_results = train_xgboost_stripped_features(
        training_data,
        output_path,
        run_label,
    )
    comparison = pd.DataFrame(
        [
            logreg_results["metrics"],
            xgboost_results["metrics"],
            xgboost_stripped_results["metrics"],
        ]
    )
    _save_model_comparison_plot(comparison, output_path, run_label)

    print("\n-- Modelljämförelse --------------------------------------------")
    print(f"Körning: {run_label}")
    print(comparison.round(3).to_string(index=False))
    print(f"\nViktiga modellfiler och PNG-figurer sparade i: {output_path}")

    return {
        "logreg": logreg_results,
        "xgboost_grid_search": xgboost_results,
        "xgboost_stripped_features": xgboost_stripped_results,
        "model_comparison": comparison,
    }


def train_multinomial_regression(
    training_data: dict,
    output_dir: Path,
    run_label: str,
) -> dict:
    """Tränar multinomial logistisk regression på PCA-transformerad data."""
    print("\n-- Träning 1/3: Multinomial logistisk regression ---------------")
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


def train_xgboost_stripped_features(
    training_data: dict,
    output_dir: Path,
    run_label: str,
) -> dict:
    """Tränar XGBoost på dummy-kategorier plus utvalda numeriska features."""
    print("\n-- Träning 3/3: XGBoost stripped features med GridSearchCV -----")

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
        verbose=1,
    )
    grid_search.fit(X_train, y_train, sample_weight=sample_weight)

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

    importance = _save_xgboost_feature_importance(
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
) -> dict:
    """Tränar XGBoost på alla originalfeatures från df_training."""
    print("\n-- Träning 2/3: XGBoost med GridSearchCV -----------------------")

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
        verbose=1,
    )
    grid_search.fit(X_train, y_train, sample_weight=sample_weight)

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

    importance = _save_xgboost_feature_importance(
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


def _balanced_sample_weight(y_train: pd.Series) -> np.ndarray:
    """Skapar sample weights så ovanliga klasser väger mer vid XGBoost-träning."""
    return compute_sample_weight(class_weight="balanced", y=y_train)


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


def _strip_xgboost_features(training_data: dict) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Väljer dummy-kategorier och sp_attack för stripped XGBoost."""
    feature_columns = [
        column
        for column in training_data["feature_columns"]
        if column in STRIPPED_NUMERIC_FEATURES
        or column.startswith(STRIPPED_FEATURE_PREFIXES)
    ]
    if not feature_columns:
        raise ValueError("Hittade inga stripped features för XGBoost.")
    return (
        training_data["X_train_original"][feature_columns],
        training_data["X_test_original"][feature_columns],
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


def _save_xgboost_feature_importance(
    model: XGBClassifier,
    feature_columns: list[str],
    output_dir: Path,
    run_label: str,
    model_name: str,
) -> dict:
    """Sparar XGBoost feature importance som PNG."""
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
    model: XGBClassifier,
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

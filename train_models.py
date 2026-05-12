from __future__ import annotations

import json
import os
from pathlib import Path

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
    "ability_1",
    "ability_2",
    "ability_3",
    "color",
    "shape",
    "habitat",
    "growth_rate",
    "egg_group_1",
    "egg_group_2",
]


def train_models(
    training_data: dict,
    output_dir: str | Path = MODEL_OUTPUT_DIR,
) -> dict:
    """Tränar logreg och XGBoost samt sparar gemensam utvärdering."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logreg_results = train_multinomial_regression(training_data, output_path)
    xgboost_results = train_xgboost_grid_search(training_data, output_path)

    comparison = pd.DataFrame(
        [
            logreg_results["metrics"],
            xgboost_results["metrics"],
        ]
    )
    comparison.to_csv(output_path / "model_comparison.csv", index=False)

    print("\n-- Modelljämförelse --------------------------------------------")
    print(comparison.to_string(index=False))
    print(f"\nModellresultat sparade i: {output_path}")

    return {
        "logreg": logreg_results,
        "xgboost": xgboost_results,
        "xgboost_best_params": xgboost_results["best_params"],
        "model_comparison": comparison,
        "predictions": {
            "logreg": logreg_results["predictions"],
            "xgboost": xgboost_results["predictions"],
        },
    }


def train_multinomial_regression(
    training_data: dict,
    output_dir: str | Path = MODEL_OUTPUT_DIR,
) -> dict:
    """Tränar multinomial logistisk regression på PCA-transformerad data."""
    print("\n-- Multinomial logistisk regression ----------------------------")

    X_train = training_data["X_train_pca"]
    X_test = training_data["X_test_pca"]
    y_train = training_data["y_train"]
    y_test = training_data["y_test"]

    model = LogisticRegression(
        solver="lbfgs",
        max_iter=LR_MAX_ITER,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return _evaluate_and_save_model(
        model_name="logreg_pca",
        model=model,
        y_test=y_test,
        y_pred=y_pred,
        target_labels=training_data.get("target_labels", {}),
        output_dir=Path(output_dir),
    )


def train_xgboost_grid_search(
    training_data: dict,
    output_dir: str | Path = MODEL_OUTPUT_DIR,
) -> dict:
    """Tränar XGBoost på originalfeatures med GridSearchCV."""
    print("\n-- XGBoost med GridSearchCV ------------------------------------")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    X_train = training_data["X_train_original"]
    X_test = training_data["X_test_original"]
    y_train = training_data["y_train"]
    y_test = training_data["y_test"]

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
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    best_params_path = output_path / "xgboost_best_params.json"
    best_params_path.write_text(
        json.dumps(grid_search.best_params_, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    cv_results = pd.DataFrame(grid_search.cv_results_)
    cv_results.to_csv(output_path / "xgboost_grid_search_results.csv", index=False)

    print(f"Bästa XGBoost-parametrar: {grid_search.best_params_}")
    print(f"Bästa CV balanced accuracy: {grid_search.best_score_:.3f}")

    results = _evaluate_and_save_model(
        model_name="xgboost",
        model=best_model,
        y_test=y_test,
        y_pred=y_pred,
        target_labels=training_data.get("target_labels", {}),
        output_dir=output_path,
    )
    results["grid_search"] = grid_search
    results["best_params"] = grid_search.best_params_
    results["best_cv_score"] = grid_search.best_score_

    importance = _save_xgboost_feature_importance(
        best_model,
        training_data["feature_columns"],
        output_path,
    )
    permutation = _save_permutation_importance(
        best_model,
        X_test,
        y_test,
        output_path,
    )
    results["feature_importance"] = importance
    results["permutation_importance"] = permutation

    return results


def _evaluate_and_save_model(
    *,
    model_name: str,
    model,
    y_test: pd.Series,
    y_pred: np.ndarray,
    target_labels: dict[int, str],
    output_dir: Path,
) -> dict:
    """Beräknar metrics och sparar rapporter för en modell."""
    output_dir.mkdir(parents=True, exist_ok=True)

    class_labels = np.array(sorted(y_test.unique()))
    class_names = _label_names(class_labels, target_labels)
    report_text = classification_report(
        y_test,
        y_pred,
        labels=class_labels,
        target_names=class_names,
        zero_division=0,
    )
    report_dict = classification_report(
        y_test,
        y_pred,
        labels=class_labels,
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )
    confusion = pd.DataFrame(
        confusion_matrix(y_test, y_pred, labels=class_labels),
        index=class_names,
        columns=class_names,
    )
    predictions = _make_prediction_table(y_test, y_pred, target_labels)

    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "macro_f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
    }

    report_path = output_dir / f"{model_name}_classification_report.txt"
    report_path.write_text(report_text, encoding="utf-8")
    pd.DataFrame(report_dict).T.to_csv(
        output_dir / f"{model_name}_classification_report.csv"
    )
    confusion.to_csv(output_dir / f"{model_name}_confusion_matrix.csv")
    predictions.to_csv(output_dir / f"{model_name}_predictions.csv")
    _save_confusion_matrix_plot(confusion, model_name, output_dir)

    print(f"\n{model_name}")
    print(f"Accuracy:            {metrics['accuracy']:.3f}")
    print(f"Balanced accuracy:   {metrics['balanced_accuracy']:.3f}")
    print(f"Macro F1:            {metrics['macro_f1']:.3f}")
    print(f"Weighted F1:         {metrics['weighted_f1']:.3f}")
    print("\nKlassificeringsrapport:")
    print(report_text)

    return {
        "model": model,
        "predictions": predictions,
        "classification_report": report_text,
        "classification_report_dict": report_dict,
        "confusion_matrix": confusion,
        "metrics": metrics,
    }


def _label_names(labels: np.ndarray, target_labels: dict[int, str]) -> list[str]:
    """Hämtar läsbara klassnamn för rapporter om mappningen finns."""
    return [
        target_labels.get(int(label), str(label))
        for label in labels
    ]


def _make_prediction_table(
    y_test: pd.Series,
    y_pred: np.ndarray,
    target_labels: dict[int, str],
) -> pd.DataFrame:
    """Skapar en resultat-tabell som kan kopplas tillbaka till df_reference."""
    predictions = pd.DataFrame(index=y_test.index)
    predictions["actual_encoded"] = y_test.astype(int)
    predictions["predicted_encoded"] = y_pred.astype(int)
    predictions["actual_type_1"] = predictions["actual_encoded"].map(target_labels)
    predictions["predicted_type_1"] = predictions["predicted_encoded"].map(target_labels)
    predictions["is_correct"] = (
        predictions["actual_encoded"] == predictions["predicted_encoded"]
    )
    return predictions


def _save_confusion_matrix_plot(
    confusion: pd.DataFrame,
    model_name: str,
    output_dir: Path,
) -> None:
    """Sparar confusion matrix som värmekarta."""
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title(f"Confusion matrix: {model_name}")
    ax.set_xlabel("Predikterad klass")
    ax.set_ylabel("Faktisk klass")
    fig.tight_layout()
    fig.savefig(output_dir / f"{model_name}_confusion_matrix.png", dpi=150)
    plt.close(fig)


def _save_xgboost_feature_importance(
    model: XGBClassifier,
    feature_columns: list[str],
    output_dir: Path,
) -> dict:
    """Sparar feature importance från XGBoost på originalfeatures."""
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

    importance.to_csv(output_dir / "xgboost_feature_importance.csv", index=False)
    grouped_importance.to_csv(
        output_dir / "xgboost_grouped_feature_importance.csv",
        index=False,
    )
    _save_importance_plot(
        importance,
        value_column="importance",
        title="XGBoost feature importance",
        output_path=output_dir / "xgboost_feature_importance.png",
    )
    _save_importance_plot(
        grouped_importance,
        feature_column="feature_group",
        value_column="total_importance",
        title="XGBoost grupperad feature importance",
        output_path=output_dir / "xgboost_grouped_feature_importance.png",
    )

    print("\nTop 10 XGBoost feature importance:")
    print(importance[["feature", "importance"]].head(10).to_string(index=False))

    return {
        "feature_importance": importance,
        "grouped_feature_importance": grouped_importance,
    }


def _save_permutation_importance(
    model: XGBClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: Path,
) -> dict:
    """Beräknar och sparar permutation importance på testdata."""
    print("\nBeräknar permutation importance...")
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

    importance.to_csv(output_dir / "xgboost_permutation_importance.csv", index=False)
    grouped_importance.to_csv(
        output_dir / "xgboost_grouped_permutation_importance.csv",
        index=False,
    )
    _save_importance_plot(
        importance,
        value_column="importance_mean",
        title="XGBoost permutation importance",
        output_path=output_dir / "xgboost_permutation_importance.png",
    )
    _save_importance_plot(
        grouped_importance,
        feature_column="feature_group",
        value_column="total_importance_mean",
        title="XGBoost grupperad permutation importance",
        output_path=output_dir / "xgboost_grouped_permutation_importance.png",
    )

    print("\nTop 10 permutation importance:")
    print(importance[["feature", "importance_mean"]].head(10).to_string(index=False))

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
    grouped = (
        importance.groupby("feature_group", as_index=False)
        .agg(
            **{
                grouped_value_column: (value_column, "sum"),
                "mean_importance": (value_column, "mean"),
                "max_importance": (value_column, "max"),
                "feature_count": ("feature", "count"),
            }
        )
        .sort_values(grouped_value_column, ascending=False)
    )
    return grouped


def _save_importance_plot(
    importance: pd.DataFrame,
    *,
    value_column: str,
    title: str,
    output_path: Path,
    feature_column: str = "feature",
    top_n: int = 30,
) -> None:
    """Sparar ett horisontellt stapeldiagram för de viktigaste features."""
    plot_data = importance.head(top_n).sort_values(value_column, ascending=True)
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.barh(plot_data[feature_column], plot_data[value_column])
    ax.set_title(title)
    ax.set_xlabel(value_column)
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

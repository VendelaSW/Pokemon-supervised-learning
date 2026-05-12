from __future__ import annotations

"""
Exploratory Data Analysis för Pokémon-datasetet.

Modulen används före och efter datarensning för att visa hur datasetet
förändras. Den sparar främst PNG-figurer eftersom de är enklast att använda
i presentationen av datan.
"""

import os
import re
import unicodedata
from pathlib import Path

import pandas as pd

MPLCONFIGDIR = Path(".matplotlib_cache")
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR.resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(MPLCONFIGDIR.resolve()))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns


BATTLE_STAT_COLUMNS = [
    "hp",
    "attack",
    "defense",
    "sp_attack",
    "sp_defense",
    "speed",
]

CORE_NUMERIC_COLUMNS = [
    *BATTLE_STAT_COLUMNS,
    "height_m",
    "weight_kg",
    "capture_rate",
    "base_happiness",
]

SELECTED_CATEGORICAL_COLUMNS = [
    "type_1",
    "color",
    "shape",
    "habitat",
    "growth_rate",
    "egg_groups",
    "generation",
]

def print_dataframe_overview(df: pd.DataFrame, title: str) -> None:
    """Skriver ut en kort översikt utan att fylla terminalen med hela tabeller."""
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)
    print(f"Rader: {df.shape[0]}")
    print(f"Kolumner: {df.shape[1]}")
    print(f"Saknade värden: {int(df.isna().sum().sum())}")
    print(f"Dubbletter: {int(df.duplicated().sum())}")
    preview = df.columns.to_list()[:12]
    suffix = " ..." if df.shape[1] > len(preview) else ""
    print(f"Kolumnexempel: {preview}{suffix}")


def run_eda(
    df: pd.DataFrame,
    dataset_name: str,
    output_dir: str | Path = "eda_outputs",
) -> dict:
    """Kör EDA på en DataFrame och sparar förklarande PNG-figurer.

    Funktionen är avsedd för både rådata och rensad referensdata.
    Träningsmatrisen med one-hot-kolumner analyseras inte här eftersom
    den främst är en modellrepresentation.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    slug = _slugify(dataset_name)
    plots_dir = output_path / f"{slug}_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    summary = _build_summary(df, dataset_name)
    plot_paths = _save_plots(df, dataset_name, slug, plots_dir)
    summary["plot_paths"] = plot_paths

    print(f"\nEDA klar för {dataset_name}.")
    print(f"PNG-figurer: {plots_dir}")

    return summary


def compare_eda_results(
    raw_summary: dict,
    cleaned_summary: dict,
    output_dir: str | Path = "eda_outputs",
) -> Path:
    """Sparar en PNG-jämförelse mellan EDA före och efter rensning."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    path = output_path / "raw_vs_cleaned_comparison.png"
    comparison = pd.DataFrame(
        {
            "dataset": ["Rådata", "Rensad data"],
            "kolumner": [raw_summary["shape"][1], cleaned_summary["shape"][1]],
            "saknade_värden": [
                raw_summary["missing_total"],
                cleaned_summary["missing_total"],
            ],
            "dubbletter": [
                raw_summary["duplicate_rows"],
                cleaned_summary["duplicate_rows"],
            ],
        }
    ).set_index("dataset")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, column in zip(axes, comparison.columns):
        comparison[column].plot(kind="bar", ax=ax, color=["#4C78A8", "#54A24B"])
        ax.set_title(column.replace("_", " ").title())
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=0)
    fig.suptitle("Jämförelse före och efter rensning")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\nEDA-jämförelse sparad som PNG: {path}")
    return path


def _build_summary(df: pd.DataFrame, dataset_name: str) -> dict:
    missing_count = df.isna().sum()
    return {
        "dataset_name": dataset_name,
        "shape": df.shape,
        "missing_total": int(missing_count.sum()),
        "duplicate_rows": int(df.duplicated().sum()),
    }


def _save_plots(
    df: pd.DataFrame,
    dataset_name: str,
    slug: str,
    plots_dir: Path,
) -> list[Path]:
    plot_paths = []
    sns.set_theme(style="whitegrid")

    plot_paths.append(_plot_target_distribution(df, dataset_name, slug, plots_dir))
    plot_paths.append(_plot_missing_values(df, dataset_name, slug, plots_dir))
    plot_paths.append(_plot_correlation(df, dataset_name, slug, plots_dir))
    plot_paths.append(_plot_numeric_histograms(df, dataset_name, slug, plots_dir))
    plot_paths.append(_plot_stat_boxplots(df, dataset_name, slug, plots_dir))
    plot_paths.extend(_plot_categorical_top_values(df, dataset_name, slug, plots_dir))
    plot_paths.append(_plot_scatter_matrix(df, dataset_name, slug, plots_dir))

    return [path for path in plot_paths if path is not None]


def _plot_target_distribution(df: pd.DataFrame, dataset_name: str, slug: str, plots_dir: Path) -> Path:
    path = plots_dir / f"{slug}_type_1_distribution.png"
    fig, ax = plt.subplots(figsize=(10, 7))
    if "type_1" in df.columns:
        df["type_1"].value_counts().sort_values().plot(kind="barh", ax=ax, color="#4C78A8")
        ax.set_title(f"Fördelning av type_1 ({dataset_name})")
        ax.set_xlabel("Antal")
        ax.set_ylabel("type_1")
    else:
        _write_empty_plot(ax, "Kolumnen type_1 saknas")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_missing_values(df: pd.DataFrame, dataset_name: str, slug: str, plots_dir: Path) -> Path:
    path = plots_dir / f"{slug}_missing_values.png"
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values()
    fig, ax = plt.subplots(figsize=(10, max(4, len(missing) * 0.35)))
    if not missing.empty:
        missing.plot(kind="barh", ax=ax, color="#F58518")
        ax.set_title(f"Saknade värden ({dataset_name})")
        ax.set_xlabel("Antal saknade värden")
        ax.set_ylabel("Kolumn")
    else:
        _write_empty_plot(ax, "Inga saknade värden")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_correlation(df: pd.DataFrame, dataset_name: str, slug: str, plots_dir: Path) -> Path:
    path = plots_dir / f"{slug}_correlation_heatmap.png"
    numeric_columns = _existing_columns(df, CORE_NUMERIC_COLUMNS)
    if len(numeric_columns) < 2:
        numeric_columns = df.select_dtypes(include="number").columns.to_list()[:12]

    fig, ax = plt.subplots(figsize=(10, 8))
    if len(numeric_columns) >= 2:
        corr = df[numeric_columns].corr()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
        ax.set_title(f"Korrelationer mellan numeriska kolumner ({dataset_name})")
    else:
        _write_empty_plot(ax, "För få numeriska kolumner för korrelation")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_numeric_histograms(df: pd.DataFrame, dataset_name: str, slug: str, plots_dir: Path) -> Path:
    path = plots_dir / f"{slug}_numeric_histograms.png"
    numeric_columns = _existing_columns(df, CORE_NUMERIC_COLUMNS)
    if not numeric_columns:
        numeric_columns = df.select_dtypes(include="number").columns.to_list()[:10]

    if numeric_columns:
        axes = df[numeric_columns].hist(figsize=(14, 10), bins=25, color="#54A24B")
        fig = axes.flatten()[0].figure
        fig.suptitle(f"Histogram för numeriska kolumner ({dataset_name})", y=1.02)
    else:
        fig, ax = plt.subplots(figsize=(8, 4))
        _write_empty_plot(ax, "Inga numeriska kolumner")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_stat_boxplots(df: pd.DataFrame, dataset_name: str, slug: str, plots_dir: Path) -> Path:
    path = plots_dir / f"{slug}_battle_stat_boxplots.png"
    stat_columns = _existing_columns(df, BATTLE_STAT_COLUMNS)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    if "type_1" in df.columns and stat_columns:
        for ax, column in zip(axes, stat_columns):
            sns.boxplot(data=df, x="type_1", y=column, ax=ax)
            ax.set_title(column)
            ax.tick_params(axis="x", rotation=60)
            ax.set_xlabel("")
        for ax in axes[len(stat_columns):]:
            ax.axis("off")
        fig.suptitle(f"Battle stats per type_1 ({dataset_name})")
    else:
        for ax in axes[1:]:
            ax.axis("off")
        _write_empty_plot(axes[0], "type_1 eller battle stats saknas")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_categorical_top_values(
    df: pd.DataFrame,
    dataset_name: str,
    slug: str,
    plots_dir: Path,
) -> list[Path]:
    paths = []
    for column in SELECTED_CATEGORICAL_COLUMNS:
        if column not in df.columns:
            continue

        path = plots_dir / f"{slug}_{column}_top_values.png"
        values = df[column].value_counts(dropna=False).head(15).sort_values()
        fig, ax = plt.subplots(figsize=(10, max(4, len(values) * 0.35)))
        values.plot(kind="barh", ax=ax, color="#B279A2")
        ax.set_title(f"Toppvärden i {column} ({dataset_name})")
        ax.set_xlabel("Antal")
        ax.set_ylabel(column)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)
    return paths


def _plot_scatter_matrix(df: pd.DataFrame, dataset_name: str, slug: str, plots_dir: Path) -> Path:
    path = plots_dir / f"{slug}_scatter_matrix.png"
    scatter_columns = _existing_columns(df, ["hp", "attack", "defense", "speed"])
    if len(scatter_columns) >= 2:
        axes = pd.plotting.scatter_matrix(
            df[scatter_columns],
            figsize=(10, 10),
            diagonal="hist",
            alpha=0.55,
        )
        fig = axes.flatten()[0].figure
        fig.suptitle(f"Scatter matrix för kärnstatistik ({dataset_name})", y=1.02)
    else:
        fig, ax = plt.subplots(figsize=(8, 4))
        _write_empty_plot(ax, "För få kolumner för scatter matrix")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _existing_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [column for column in columns if column in df.columns]


def _write_empty_plot(ax: plt.Axes, text: str) -> None:
    ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=14)
    ax.set_axis_off()


def _slugify(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", normalized.strip().lower()).strip("_")
    return slug or "dataset"

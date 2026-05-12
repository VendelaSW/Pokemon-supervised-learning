from __future__ import annotations

"""
Exploratory Data Analysis för Pokémon-datasetet.

Modulen används före och efter datarensning för att visa hur datasetet
förändras. Den sparar både textbaserade rapporter och figurer, men den
analyserar inte den breda träningsmatrisen med dummy-kolumner eftersom
den främst är skapad för modellträning.
"""

import os
import re
import unicodedata
from io import StringIO
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

CROSSTAB_COLUMNS = [
    "color",
    "shape",
    "habitat",
    "growth_rate",
    "egg_groups",
]


def print_dataframe_overview(df: pd.DataFrame, title: str) -> None:
    """Skriver ut en tydlig översikt av en DataFrame i terminalen."""
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)
    print(f"Rader: {df.shape[0]}")
    print(f"Kolumner: {df.shape[1]}")
    print("\nKolumnnamn:")
    print(df.columns.to_list())
    print("\nFörsta fem raderna:")
    print(df.head())
    print("\nDataformat:")
    info_buffer = StringIO()
    df.info(buf=info_buffer)
    print(info_buffer.getvalue())


def run_eda(
    df: pd.DataFrame,
    dataset_name: str,
    output_dir: str | Path = "eda_outputs",
) -> dict:
    """Kör EDA på en DataFrame och sparar rapport samt figurer.

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
    report_path = output_path / f"{slug}_eda_report.md"
    report_path.write_text(_format_report(summary), encoding="utf-8")

    plot_paths = _save_plots(df, dataset_name, slug, plots_dir)
    summary["report_path"] = report_path
    summary["plot_paths"] = plot_paths

    print(f"\nEDA klar för {dataset_name}.")
    print(f"Rapport: {report_path}")
    print(f"Figurer: {plots_dir}")

    return summary


def compare_eda_results(
    raw_summary: dict,
    cleaned_summary: dict,
    output_dir: str | Path = "eda_outputs",
) -> Path:
    """Sparar en jämförelserapport mellan EDA före och efter rensning."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    report_path = output_path / "raw_vs_cleaned_comparison.md"

    raw_columns = set(raw_summary["columns"])
    cleaned_columns = set(cleaned_summary["columns"])
    added_columns = sorted(cleaned_columns - raw_columns)
    removed_columns = sorted(raw_columns - cleaned_columns)
    shared_columns = sorted(raw_columns & cleaned_columns)
    dtype_changes = [
        (column, raw_summary["dtypes"][column], cleaned_summary["dtypes"][column])
        for column in shared_columns
        if raw_summary["dtypes"][column] != cleaned_summary["dtypes"][column]
    ]

    lines = [
        "# Jämförelse: rådata vs rensad data",
        "",
        "## Storlek",
        "",
        f"- Rådata: {raw_summary['shape'][0]} rader, {raw_summary['shape'][1]} kolumner",
        f"- Rensad data: {cleaned_summary['shape'][0]} rader, {cleaned_summary['shape'][1]} kolumner",
        "",
        "## Saknade värden",
        "",
        f"- Rådata: {raw_summary['missing_total']} saknade värden totalt",
        f"- Rensad data: {cleaned_summary['missing_total']} saknade värden totalt",
        "",
        "## Dubbletter",
        "",
        f"- Rådata: {raw_summary['duplicate_rows']} dubbla rader",
        f"- Rensad data: {cleaned_summary['duplicate_rows']} dubbla rader",
        "",
        "## Kolumnförändringar",
        "",
        _format_list("Tillagda kolumner", added_columns),
        _format_list("Borttagna kolumner", removed_columns),
        "",
        "## Datatyper som ändrats",
        "",
        _format_dtype_changes(dtype_changes),
        "",
    ]

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nJämförelserapport sparad: {report_path}")
    return report_path


def _build_summary(df: pd.DataFrame, dataset_name: str) -> dict:
    missing_count = df.isna().sum()
    missing_percent = (missing_count / len(df) * 100).round(2) if len(df) else missing_count
    missing_table = (
        pd.DataFrame(
            {
                "missing_count": missing_count,
                "missing_percent": missing_percent,
            }
        )
        .query("missing_count > 0")
        .sort_values(["missing_count", "missing_percent"], ascending=False)
    )

    numeric_columns = df.select_dtypes(include="number").columns.to_list()
    categorical_columns = df.select_dtypes(
        include=["object", "string", "category", "bool"]
    ).columns.to_list()

    categorical_cardinality = (
        df[categorical_columns].nunique(dropna=False).sort_values(ascending=False)
        if categorical_columns
        else pd.Series(dtype="int64")
    )

    top_values = {
        column: df[column].value_counts(dropna=False).head(10)
        for column in categorical_columns
    }

    outlier_summary = _iqr_outlier_summary(df, numeric_columns)
    crosstabs = _build_crosstabs(df)

    return {
        "dataset_name": dataset_name,
        "shape": df.shape,
        "columns": df.columns.to_list(),
        "dtypes": {column: str(dtype) for column, dtype in df.dtypes.items()},
        "head": df.head(),
        "info": _info_to_string(df),
        "missing_table": missing_table,
        "missing_total": int(missing_count.sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "numeric_describe": df[numeric_columns].describe().T if numeric_columns else pd.DataFrame(),
        "categorical_cardinality": categorical_cardinality,
        "top_values": top_values,
        "target_distribution": (
            df["type_1"].value_counts(dropna=False) if "type_1" in df.columns else pd.Series(dtype="int64")
        ),
        "correlation": df[numeric_columns].corr() if len(numeric_columns) >= 2 else pd.DataFrame(),
        "outlier_summary": outlier_summary,
        "crosstabs": crosstabs,
    }


def _format_report(summary: dict) -> str:
    lines = [
        f"# EDA: {summary['dataset_name']}",
        "",
        "## Grundstruktur",
        "",
        f"- Rader: {summary['shape'][0]}",
        f"- Kolumner: {summary['shape'][1]}",
        f"- Dubbletter: {summary['duplicate_rows']}",
        f"- Saknade värden totalt: {summary['missing_total']}",
        "",
        "## Kolumner och datatyper",
        "",
        _format_code_block(summary["info"]),
        "",
        "## Första fem raderna",
        "",
        _format_dataframe(summary["head"]),
        "",
        "## Saknade värden",
        "",
        _format_dataframe(summary["missing_table"], empty_text="Inga saknade värden."),
        "",
        "## Numeriska summeringar",
        "",
        _format_dataframe(summary["numeric_describe"], empty_text="Inga numeriska kolumner."),
        "",
        "## Kategorisk kardinalitet",
        "",
        _format_series(summary["categorical_cardinality"], empty_text="Inga kategoriska kolumner."),
        "",
        "## Fördelning av målvariabeln type_1",
        "",
        _format_series(summary["target_distribution"], empty_text="Kolumnen type_1 saknas."),
        "",
        "## Korrelationer",
        "",
        _format_dataframe(summary["correlation"], empty_text="För få numeriska kolumner för korrelation."),
        "",
        "## IQR-baserade avvikelser",
        "",
        _format_dataframe(summary["outlier_summary"], empty_text="Inga numeriska kolumner att analysera."),
        "",
        "## Toppvärden per kategorisk kolumn",
        "",
    ]

    for column, values in summary["top_values"].items():
        lines.extend([f"### {column}", "", _format_series(values), ""])

    lines.extend(["## Kontingenstabeller mot type_1", ""])
    if summary["crosstabs"]:
        for column, table in summary["crosstabs"].items():
            lines.extend([f"### type_1 x {column}", "", _format_dataframe(table), ""])
    else:
        lines.append("Inga kontingenstabeller skapades.")

    return "\n".join(lines)


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


def _iqr_outlier_summary(df: pd.DataFrame, numeric_columns: list[str]) -> pd.DataFrame:
    rows = []
    for column in numeric_columns:
        values = df[column].dropna()
        if values.empty:
            continue

        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = ((values < lower) | (values > upper)).sum()
        rows.append(
            {
                "column": column,
                "lower_bound": round(lower, 3),
                "upper_bound": round(upper, 3),
                "outlier_count": int(outliers),
                "outlier_percent": round(outliers / len(values) * 100, 2),
            }
        )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("outlier_count", ascending=False)


def _build_crosstabs(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if "type_1" not in df.columns:
        return {}

    crosstabs = {}
    for column in CROSSTAB_COLUMNS:
        if column in df.columns:
            crosstabs[column] = pd.crosstab(df["type_1"], df[column], dropna=False)
    return crosstabs


def _info_to_string(df: pd.DataFrame) -> str:
    buffer = StringIO()
    df.info(buf=buffer)
    return buffer.getvalue()


def _format_dataframe(df: pd.DataFrame, empty_text: str = "Inga data.") -> str:
    if df.empty:
        return empty_text
    return _format_code_block(df.to_string())


def _format_series(series: pd.Series, empty_text: str = "Inga data.") -> str:
    if series.empty:
        return empty_text
    return _format_code_block(series.to_string())


def _format_code_block(text: str) -> str:
    return f"```text\n{text}\n```"


def _format_list(title: str, values: list[str]) -> str:
    if not values:
        return f"- {title}: inga"
    return f"- {title}: {', '.join(values)}"


def _format_dtype_changes(changes: list[tuple[str, str, str]]) -> str:
    if not changes:
        return "Inga datatyper ändrades för gemensamma kolumner."
    rows = [f"- {column}: {before} -> {after}" for column, before, after in changes]
    return "\n".join(rows)


def _existing_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [column for column in columns if column in df.columns]


def _write_empty_plot(ax: plt.Axes, text: str) -> None:
    ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=14)
    ax.set_axis_off()


def _slugify(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", normalized.strip().lower()).strip("_")
    return slug or "dataset"

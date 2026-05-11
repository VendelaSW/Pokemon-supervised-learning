"""
Steg 4 — Förbered features och målvariabel
===========================================
Tar den förbehandlade DataFrame:n och bildmatrisen och separerar
dem i:

  - X_tabular     : numeriska tabellfeatures (stats, kodade kategorier)
  - X_image_raw   : plattade pixelvektorer
  - y             : målvariabel (type_1_encoded)

Kolumner som läcker information om målet (type_1_encoded, type_2_encoded)
och identitetskolumner (namn, pokedex_number) tas bort ur X.
Endast rader med giltiga bilder behålls.
"""

import numpy as np
import pandas as pd


def build_features(
    df: pd.DataFrame,
    image_matrix: np.ndarray,
    valid_mask: np.ndarray,
) -> dict:
    """Bygger featurematriser för tabell- och bilddata.

    Returnerar
    ----------
    dict med nycklar:
        X_tabular       : np.ndarray — tabellfeatures
        X_image_raw     : np.ndarray — plattade pixelvektorer
        y               : np.ndarray — målvariabel (type_1_encoded)
        tabular_columns : list[str]  — kolumnnamn för tabellfeatures
        df              : pd.DataFrame — filtrerat dataset
    """
    print("\n── Bygger featurematriser ──────────────────────────")

    # Filtrera till rader med giltiga bilder
    df_valid = df[valid_mask].reset_index(drop=True)
    X_image_valid = image_matrix[valid_mask]

    # Målvariabel
    y = df_valid["type_1_encoded"].values

    # Tabellfeatures — ta bort identitet, mål och råa strängkolumner
    exclude = {
        "pokedex_number", "name", "image",
        "type_1_encoded", "type_2_encoded",
        "sprite_url", "flavor_text", "genus", "egg_groups",
        "abilities", "hidden_ability",
        "type_1", "type_2", "color", "shape", "habitat", "growth_rate",
    }

    tabular_columns = [
        col for col in df_valid.columns
        if col not in exclude
        and df_valid[col].dtype in [np.int64, np.float64, np.int32, np.float32]
    ]
    # Ta bort eventuella dubbletter men behåll ordning
    tabular_columns = list(dict.fromkeys(tabular_columns))

    X_tabular = df_valid[tabular_columns].values.astype(np.float32)

    print(f"  Tabellfeatures ({X_tabular.shape[1]}): {tabular_columns}")
    print(f"  Bildfeatures: {X_image_valid.shape[1]} pixelvärden per rad")
    print(f"  Rader med giltiga bilder: {len(y)}")

    return {
        "X_tabular": X_tabular,
        "X_image_raw": X_image_valid,
        "y": y,
        "tabular_columns": tabular_columns,
        "df": df_valid,
    }

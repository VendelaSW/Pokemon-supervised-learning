"""
Steg 2 — Förbehandling
======================
Tar den råa DataFrame:n från load_data och gör den redo för analys:

  1. Fyll i saknade värden (type_2, habitat, abilities m.fl.)
  2. Konvertera generation-strängar till heltal
  3. Omvandla booleska kolumner till 0/1
  4. Skapa härledda features (antal abilities, har dold ability)
  5. Label-koda kategoriska kolumner

Inga bilder hanteras här — det sker i load_images.
"""

import numpy as np
import pandas as pd


def preprocess_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Rensar och berikar datasetet, returnerar DataFrame + etikettkodningar.

    Returnerar
    ----------
    df          : pd.DataFrame — rensat dataset med nya kolumner
    label_maps  : dict         — mappningar för kategoriska kolumner
                                 (t.ex. {"type_1": {"Bug": 0, ...}})
    """
    print("\n── Förbehandling ──────────────────────────────────")

    # ── Saknade värden ────────────────────────────────────────
    # type_2 saknas för Pokémon med bara en typ — vi sätter "None"
    df["type_2"] = df["type_2"].fillna("None")
    df["habitat"] = df["habitat"].fillna("unknown")
    df["hidden_ability"] = df["hidden_ability"].fillna("none")
    df["abilities"] = df["abilities"].fillna("unknown")
    # base_experience saknas för ett fåtal — fyll med medianen
    df["base_experience"] = df["base_experience"].fillna(df["base_experience"].median())

    missing = df.isna().sum()
    if missing.sum() == 0:
        print("Inga saknade värden kvar efter ifyllnad.")
    else:
        print("Kvarvarande saknade värden:")
        print(missing[missing > 0])

    # ── Generation: sträng → heltal ──────────────────────────
    gen_map = {
        "gen-i": 1, "gen-ii": 2, "gen-iii": 3, "gen-iv": 4,
        "gen-v": 5, "gen-vi": 6, "gen-vii": 7, "gen-viii": 8, "gen-ix": 9,
    }
    df["generation"] = df["generation"].map(gen_map)

    # ── Booleska kolumner → 0/1 ──────────────────────────────
    for col in ["is_legendary", "is_mythical", "is_baby"]:
        df[col] = df[col].astype(int)

    # ── Härledda features ────────────────────────────────────
    # Antal abilities ger en numerisk representation av förmågorna
    df["num_abilities"] = df["abilities"].apply(lambda x: len(str(x).split("|")))
    df["has_hidden_ability"] = (df["hidden_ability"] != "none").astype(int)

    # ── Label-kodning av kategoriska kolumner ─────────────────
    # Vi skapar heltalskodade versioner och sparar mappningarna
    # så att vi kan avkoda resultaten efteråt.
    cat_columns = ["type_1", "type_2", "color", "shape", "habitat", "growth_rate"]
    label_maps = {}

    for col in cat_columns:
        categories = sorted(df[col].unique())
        mapping = {cat: i for i, cat in enumerate(categories)}
        label_maps[col] = mapping
        df[col + "_encoded"] = df[col].map(mapping)

    print(f"Förbehandling klar — {len(df)} rader, {len(df.columns)} kolumner")
    print(f"Kodade kategoriska kolumner: {cat_columns}")

    return df, label_maps

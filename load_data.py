"""
Datainsamling — Rå CSV
======================
Läser in Pokémon-datasetet från en semikolonseparerad CSV-fil och
validerar att filen existerar.

Datasetet innehåller statistik, typ, färg, habitat och sprite-URL.
Modulen gör ingen rensning utan returnerar en rå DataFrame från disk.
"""

from pathlib import Path
import pandas as pd


def load_data(data_path: Path) -> pd.DataFrame:
    """Läser in CSV-filen och returnerar en rå DataFrame.

    Om filen saknas listas CSV-filer i dataset-mappen som felsökningsstöd.

    Returnerar
    ----------
    df : pd.DataFrame — obearbetad data rakt från disk
    """
    if not data_path.exists():
        csv_files = list(Path("dataset").glob("*.csv"))
        print("CSV-filer som hittades i dataset-mappen:")
        print(csv_files)
        raise FileNotFoundError(f"Hittar inte filen: {data_path}")

    # Datasetet är semikolonseparerat.
    df = pd.read_csv(data_path, sep=";")

    print("Datasetet har laddats in korrekt.")
    print(f"Fil: {data_path}")
    print(f"Antal rader: {df.shape[0]}")
    print(f"Antal kolumner: {df.shape[1]}")

    return df

"""
Steg 1 — Datainsamling
======================
Läser in Pokémon-datasetet från en semikolonseparerad CSV-fil och
validerar att filen existerar.

Datasetet innehåller statistik, typ, färg, habitat och sprite-URL
för ca 1 000 Pokémon (efter borttagning av megaevolveringar).
Inget avancerat händer här — vi laddar filen, skriver ut en kort
sammanfattning och returnerar en rå DataFrame.
"""

from pathlib import Path
import pandas as pd


def load_data(data_path: Path) -> pd.DataFrame:
    """Läser in CSV-filen och returnerar en rå DataFrame.

    Om filen inte hittas listar vi alla CSV-filer i dataset-mappen
    för att hjälpa användaren hitta rätt sökväg.

    Returnerar
    ----------
    df : pd.DataFrame — obearbetad data rakt från disk
    """
    if not data_path.exists():
        csv_files = list(Path("dataset").glob("*.csv"))
        print("CSV-filer som hittades i dataset-mappen:")
        print(csv_files)
        raise FileNotFoundError(f"Hittar inte filen: {data_path}")

    # semikolonseparerad fil — vanligt i europeiska dataset
    df = pd.read_csv(data_path, sep=";")

    print("Datasetet har laddats in korrekt.")
    print(f"Fil: {data_path}")
    print(f"Antal rader: {df.shape[0]}")
    print(f"Antal kolumner: {df.shape[1]}")

    return df

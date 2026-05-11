"""
Pokémon-typklassificering — ML-pipeline
========================================
Startpunkt som knyter ihop alla steg:

    load_data  →  preprocess_data  →  load_images  →  build_features
              →  modellering (Logistic Regression + Random Forest)

Kör med:  python app.py

Pipelinen läser in Pokémon-data från en CSV-fil, rensar och kodar
features, laddar in sprite-bilder som pixelmatriser, reducerar
bilddimensioner med PCA och tränar klassificeringsmodeller för
att förutsäga primärtyp (type_1).

Förväntad struktur på disk:
    dataset/
        pokemon_complete.csv
    images/
        <pokedex_number>/
            <pokedex_number>.png
"""

import json
from settings import DATA_PATH, IMAGE_DIR

from load_data import load_data
from preprocess_data import preprocess_data
from load_images import load_images
from build_features import build_features
from train_models import train_and_evaluate


# ── 1. Ladda rådata ──────────────────────────────────────────
# Läser in CSV-filen och skriver ut grundläggande info om datasetet.

df_raw = load_data(DATA_PATH)


# ── 2. Förbehandla ───────────────────────────────────────────
# Fyller i saknade värden, konverterar datatyper, skapar härledda
# features och label-kodar kategoriska kolumner.

df, label_maps = preprocess_data(df_raw)


# ── 3. Ladda bilder ─────────────────────────────────────────
# Läser sprites från disk, ersätter transparens med svart bakgrund,
# skalar till 32×32 RGB och normaliserar pixelvärden till 0–1.

image_matrix, valid_mask = load_images(df, IMAGE_DIR)


# ── 4. Bygg featurematriser ─────────────────────────────────
# Separerar tabelldata och bilddata.  Filtrerar bort rader utan
# giltig bild.  Identifierar vilka kolumner som är numeriska
# features respektive målvariabel (type_1_encoded).

data = build_features(df, image_matrix, valid_mask)

print(f"\n  Slutgiltigt dataset: {len(data['y'])} rader, {len(label_maps['type_1'])} typklasser")


# ── 5. Modellträning och utvärdering ────────────────────────
# Tränar Logistic Regression och Random Forest på tre feature-
# uppsättningar: tabell, bild-PCA och kombinerat.  PCA reducerar
# bildfeatures till de komponenter som bevarar 95 % av variansen.
# Utvärderar med accuracy och classification_report per typ.

results = train_and_evaluate(data, label_maps)


# ── Spara etikettkodningar för referens ──────────────────────
maps_path = DATA_PATH.parent / "label_maps.json"
with open(maps_path, "w") as f:
    json.dump(label_maps, f, indent=2, ensure_ascii=False)
print(f"\nEtikettkodningar sparade till {maps_path}")

print("\nPipelinen är klar.")

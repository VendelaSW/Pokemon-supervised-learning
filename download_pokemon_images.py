"""
Bildnedladdning — Pokémon-sprites
==================================
Läser sprite-URL:er från pokemon_complete.csv och laddar ned
varje bild till en mapp namngiven efter pokédex-numret:

    images/<pokedex_number>/<pokedex_number>.png

Rader utan URL hoppas över och loggas.  Redan nedladdade bilder
hoppas över så att skriptet kan köras igen utan att ladda ned allt
på nytt.

Kör med:  python download_pokemon_images.py
"""

import csv
import time
import urllib.request
import urllib.error
from pathlib import Path

from settings import DATA_PATH, IMAGE_DIR

# ── Nedladdningsinställningar ────────────────────────────────
ANTAL_FÖRSÖK  = 3      # antal försök per bild vid nätverksfel
FÖRSÖKSPAUS   = 2      # sekunder mellan omförsök
ARTIGHETS_PAUS = 0.1   # sekunder mellan nedladdningar


def ladda_ned_bild(url: str, dest: Path) -> bool:
    """Laddar ned en bild från url till dest.  Returnerar True vid lyckat försök."""
    for försök in range(1, ANTAL_FÖRSÖK + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Pokemon-Sprite-Downloader/1.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                dest.write_bytes(resp.read())
            return True
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as e:
            if försök < ANTAL_FÖRSÖK:
                print(f"    Försök {försök}/{ANTAL_FÖRSÖK} för {url} ({e})")
                time.sleep(FÖRSÖKSPAUS)
            else:
                print(f"    MISSLYCKADES efter {ANTAL_FÖRSÖK} försök: {e}")
                return False


def main():
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    hoppade_över = []
    misslyckade = []
    nedladdade = 0

    with open(DATA_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")

        for rad in reader:
            pokedex_nr = rad["pokedex_number"].strip()
            namn = rad["name"].strip()
            url = rad.get("sprite_url", "").strip()

            # Hoppa över rader utan URL
            if not url:
                hoppade_över.append((pokedex_nr, namn))
                continue

            # Skapa mapp för denna Pokémon
            pokemon_mapp = IMAGE_DIR / pokedex_nr
            pokemon_mapp.mkdir(parents=True, exist_ok=True)

            # Bestäm filnamn från URL
            filnamn = url.split("/")[-1]
            if not filnamn:
                filnamn = f"{pokedex_nr}.png"
            dest = pokemon_mapp / filnamn

            # Hoppa över redan nedladdade bilder
            if dest.exists() and dest.stat().st_size > 0:
                nedladdade += 1
                continue

            print(f"[{nedladdade + 1}] Laddar ned {namn} (#{pokedex_nr})...")
            ok = ladda_ned_bild(url, dest)

            if ok:
                nedladdade += 1
            else:
                misslyckade.append((pokedex_nr, namn, url))

            time.sleep(ARTIGHETS_PAUS)

    # ── Sammanfattning ───────────────────────────────────────
    print("\n" + "=" * 50)
    print(f"Nedladdade:         {nedladdade}")
    print(f"Hoppade över (ingen URL): {len(hoppade_över)}")
    print(f"Misslyckade:        {len(misslyckade)}")

    if hoppade_över:
        print(f"\nPokémon utan bild-URL:")
        for pid, pnamn in hoppade_över:
            print(f"  #{pid} {pnamn}")

    if misslyckade:
        print(f"\nMisslyckade nedladdningar:")
        for pid, pnamn, purl in misslyckade:
            print(f"  #{pid} {pnamn} — {purl}")


if __name__ == "__main__":
    main()

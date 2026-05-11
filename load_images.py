"""
Steg 3 — Bildladdning
=====================
Läser in sprite-bilder från disk, konverterar till RGB med svart
bakgrund (alfa ersätts), skalar till IMG_SIZE × IMG_SIZE och
returnerar en pixelmatris + en mask som anger vilka rader som
har giltiga bilder.

Bilderna förväntas ligga i  images/<pokedex_number>/*.png
(skapade av download_pokemon_images.py).
"""

import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

from settings import IMG_SIZE, CHANNELS


def load_images(df: pd.DataFrame, image_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Laddar sprites, skalar och plattar ut till en pixelmatris.

    Varje bild konverteras till RGBA, sedan klistras den på en svart
    RGB-bakgrund — det ger en konsekvent bakgrund utan transparens.
    Pixelvärden normaliseras till intervallet 0–1.

    Returnerar
    ----------
    image_matrix : np.ndarray — shape (n_rader, IMG_SIZE*IMG_SIZE*CHANNELS)
    valid_mask   : np.ndarray — bool-array, True för rader med giltig bild
    """
    print(f"\n── Bildladdning ({IMG_SIZE}×{IMG_SIZE} RGB, alfa → svart) ──")

    images = []
    valid = []

    for pid in df["pokedex_number"]:
        folder = image_dir / str(pid)
        img_path = None

        if folder.exists():
            png_files = list(folder.glob("*.png"))
            if png_files:
                img_path = png_files[0]

        if img_path and img_path.exists():
            try:
                # Öppna som RGBA oavsett ursprungligt format
                img = Image.open(img_path).convert("RGBA")
                img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)

                # Ersätt transparens med svart bakgrund och konvertera till RGB
                background = Image.new("RGB", img.size, (0, 0, 0))
                background.paste(img, mask=img.split()[3])

                arr = np.array(background, dtype=np.float32) / 255.0
                images.append(arr.flatten())
                valid.append(True)
            except Exception as e:
                print(f"  Hoppar över #{pid}: {e}")
                images.append(np.zeros(IMG_SIZE * IMG_SIZE * CHANNELS, dtype=np.float32))
                valid.append(False)
        else:
            images.append(np.zeros(IMG_SIZE * IMG_SIZE * CHANNELS, dtype=np.float32))
            valid.append(False)

    valid_mask = np.array(valid)
    image_matrix = np.stack(images)
    n_valid = valid_mask.sum()
    print(f"  Giltiga bilder: {n_valid}/{len(df)} ({len(df) - n_valid} hoppades över)")

    return image_matrix, valid_mask

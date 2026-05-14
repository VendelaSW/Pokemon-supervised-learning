"""
Bildladdning — Sprite-features
==============================
Läser in sprite-bilder från disk och gör dem till bildfeatures:

  sprite PNG → RGBA → RGB med svart bakgrund → IMG_SIZE × IMG_SIZE
             → float32-pixlar i intervallet 0-1 → platt pixelvektor

Funktionen returnerar även en mask som anger vilka rader som har
giltiga bilder.  Den masken används senare så att tabellfeatures,
bildfeatures och målvariabeln får exakt samma rader.

Bilderna förväntas ligga i  images/<pokedex_number>/*.png
(skapade av download_pokemon_images.py).
"""

import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image, ImageFile

from settings import IMG_SIZE, CHANNELS

# Vissa sprites har trasig PNG-metadata trots läsbar bilddata.
ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_images(df: pd.DataFrame, image_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Laddar sprites, skalar och plattar ut till en pixelmatris.

    Varje bild konverteras till RGBA, sedan klistras den på en svart
    RGB-bakgrund — det ger en konsekvent bakgrund utan transparens.
    Pixelvärden normaliseras till intervallet 0-1 och plattas ut till
    en featurevektor med längden IMG_SIZE * IMG_SIZE * CHANNELS.

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
                # Öppna som RGBA oavsett ursprungligt format så att
                # alfakanalen kan användas för bakgrundskonverteringen.
                img = Image.open(img_path).convert("RGBA")
                img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)

                # Ersätt transparens med svart bakgrund och konvertera
                # bildens kanaler till vanliga RGB-features.
                background = Image.new("RGB", img.size, (0, 0, 0))
                background.paste(img, mask=img.split()[3])

                # RGB-bilden blir en float32-matris i 0-1 och plattas
                # sedan ut till en rad i X_image_raw.
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

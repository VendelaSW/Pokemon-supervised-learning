"""
Settings-paket — Re-export av gemensamma konstanter
===================================================
Gör projektets konstanter tillgängliga via korta importer som
from settings import TEST_RATIO, RANDOM_STATE.
"""

from .settings import (
    DATA_PATH,
    MODEL_OUTPUT_DIR,
    IMAGE_DIR,
    IMG_SIZE,
    CHANNELS,
    PCA_VARIANCE,
    TEST_RATIO,
    RANDOM_STATE,
    LR_MAX_ITER,
    GRID_SEARCH_CV,
    PERMUTATION_REPEATS,
    XGB_PARAM_GRID,
)

__all__ = [
    "DATA_PATH",
    "MODEL_OUTPUT_DIR",
    "IMAGE_DIR",
    "IMG_SIZE",
    "CHANNELS",
    "PCA_VARIANCE",
    "TEST_RATIO",
    "RANDOM_STATE",
    "LR_MAX_ITER",
    "GRID_SEARCH_CV",
    "PERMUTATION_REPEATS",
    "XGB_PARAM_GRID",
]

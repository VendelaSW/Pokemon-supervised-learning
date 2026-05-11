# Återexporterar alla konstanter så att övriga moduler kan skriva:
#     from settings import TEST_RATIO, RANDOM_STATE, ...

from .settings import (
    DATA_PATH,
    IMAGE_DIR,
    IMG_SIZE,
    CHANNELS,
    PCA_VARIANCE,
    TEST_RATIO,
    RANDOM_STATE,
    LR_MAX_ITER,
    RF_ESTIMATORS,
)

__all__ = [
    "DATA_PATH",
    "IMAGE_DIR",
    "IMG_SIZE",
    "CHANNELS",
    "PCA_VARIANCE",
    "TEST_RATIO",
    "RANDOM_STATE",
    "LR_MAX_ITER",
    "RF_ESTIMATORS",
]

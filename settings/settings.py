"""
Konfiguration — Projektets gemensamma inställningar
===================================================
Samlar sökvägar, reproducerbarhetsparametrar och modellparametrar
som används av pipeline-modulerna.

Övriga moduler importerar normalt dessa värden via settings-paketet.
"""

from pathlib import Path

# ── KONFIGURATION ─────────────────────────────────────────────
# Alla justerbara parametrar samlade på ett ställe.
# Övriga moduler importerar härifrån via settings-paketet.

DATA_PATH = Path("dataset") / "pokemon_complete.csv"   # sökväg till CSV-filen
IMAGE_DIR = Path("images")                              # mapp med nedladdade sprites
MODEL_OUTPUT_DIR = Path("model_outputs")                # mapp för modellrapporter

# ── Bildbehandling ───────────────────────────────────────────
IMG_SIZE   = 96      # bilder skalas till IMG_SIZE × IMG_SIZE pixlar
CHANNELS   = 3       # RGB (alfa ersätts med svart bakgrund)
IMAGE_PCA_VARIANCE = 0.90  # behåll 90 % av variansen i råa sprite-pixlar

# ── PCA ──────────────────────────────────────────────────────
PCA_VARIANCE = 0.95  # behåll 95 % av variansen i träningsfeatures

# ── Uppdelning & reproducerbarhet ────────────────────────────
TEST_RATIO   = 0.2   # 80/20 stratifierad train/test-split
RANDOM_STATE = 42    # fast seed för reproducerbara resultat

# ── Modeller ─────────────────────────────────────────────────
LR_MAX_ITER       = 2000           # max iterationer för logistisk regression
GRID_SEARCH_CV    = 3              # antal folds i GridSearchCV
PERMUTATION_REPEATS = 5           # antal upprepningar för permutation importance
RF_ESTIMATORS = 300               # antal träd i RandomForest-baseline
RF_MAX_DEPTH = None               # None låter träden växa tills stoppvillkor nås
RF_MIN_SAMPLES_LEAF = 1           # minsta antal rader i ett löv

XGB_PARAM_GRID = {
    "n_estimators": [250, 255, 275],
    "max_depth": [2],
    "learning_rate": [0.1, 0.2],
    "subsample": [0.5, 0.7],
    "gamma": [0.0, 0.01],
    "reg_lambda": [1, 5],
    "reg_alpha": [0, 0.1, 0.2],
}

# Bästa fulla XGBoost-resultat hittills:
# gamma=0.0, learning_rate=0.1, max_depth=2, n_estimators=250,
# reg_alpha=0.1, reg_lambda=5, subsample=0.5
#
# Bästa stripped XGBoost-resultat hittills:
# gamma=0.0, learning_rate=0.1, max_depth=2, n_estimators=250,
# reg_alpha=0.1, reg_lambda=1, subsample=0.5
#
# Aktuell grid söker nära bästa resultat för att finjustera modellen.
#
# Tidigare testade värden med sämre resultat:
# n_estimators = 50, 100, 200, 300, 350
# learning_rate = 0.05
# max_depth = 2, 3, 4, 5, 10, 20, 30, 40
# subsample = 0.8, 0.9
# gamma = 0.1, 1

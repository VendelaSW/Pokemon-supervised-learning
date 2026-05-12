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

# ── PCA ──────────────────────────────────────────────────────
PCA_VARIANCE = 0.95  # behåll 95 % av variansen i träningsfeatures

# ── Uppdelning & reproducerbarhet ────────────────────────────
TEST_RATIO   = 0.2   # 80/20 stratifierad train/test-split
RANDOM_STATE = 42    # fast seed för reproducerbara resultat

# ── Modeller ─────────────────────────────────────────────────
LR_MAX_ITER       = 2000           # max iterationer för logistisk regression
RF_ESTIMATORS     = 100            # antal träd i Random Forest
GRID_SEARCH_CV    = 5              # antal folds i GridSearchCV
PERMUTATION_REPEATS = 10           # antal upprepningar för permutation importance

XGB_PARAM_GRID = {
    "n_estimators": [50, 100],
    "max_depth": [5, 10],
    "learning_rate": [0.1],
    "subsample": [0.8, 1.0]
}

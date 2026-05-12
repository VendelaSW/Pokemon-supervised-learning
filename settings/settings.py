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
GRID_SEARCH_CV    = 3              # antal folds i GridSearchCV
PERMUTATION_REPEATS = 5           # antal upprepningar för permutation importance

XGB_PARAM_GRID = {
    "n_estimators": [250, 255, 275],
    "max_depth": [2],
    "learning_rate": [0.1, 0.2],
    "subsample": [0.5, 0.7],
    "gamma": [0.0, 0.01],
    "reg_lambda": [1, 5],
    "reg_alpha": [0, 0.1, 0.2],
}

# Vi har testat, som gav sämre resultat:
# n_estimators = 50, 100, 200, 300, 350
# learning_rate = 0.05
# max_depth = 2, 3, 4, 5, 10, 20, 30, 40
# subsample = 0.8, 0.9
# gamma = 1
from pathlib import Path

# ── KONFIGURATION ─────────────────────────────────────────────
# Alla justerbara parametrar samlade på ett ställe.
# Övriga moduler importerar härifrån via settings-paketet.

DATA_PATH = Path("dataset") / "pokemon_complete.csv"   # sökväg till CSV-filen
IMAGE_DIR = Path("images")                              # mapp med nedladdade sprites

# ── Bildbehandling ───────────────────────────────────────────
IMG_SIZE   = 96      # bilder skalas till IMG_SIZE × IMG_SIZE pixlar
CHANNELS   = 3       # RGB (alfa ersätts med svart bakgrund)

# ── PCA ──────────────────────────────────────────────────────
PCA_VARIANCE = 0.95  # behåll 95 % av variansen i bilddata

# ── Uppdelning & reproducerbarhet ────────────────────────────
TEST_RATIO   = 0.2   # 80/20 stratifierad train/test-split
RANDOM_STATE = 42    # fast seed för reproducerbara resultat

# ── Modeller ─────────────────────────────────────────────────
LR_MAX_ITER       = 2000           # max iterationer för Logistic Regression
RF_ESTIMATORS     = 100            # antal träd i Random Forest

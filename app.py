"""
Huvudflöde — Pokémon supervised learning
========================================
Kör projektets pipeline från rå CSV till rensad referensdata,
EDA, bildfeatures, träningsmatris, PCA och modellträning.

Flödet håller isär df_raw, df_reference och df_training så att
modellfeatures kan ändras utan att referensdata förloras.
"""

from settings import DATA_PATH, IMAGE_DIR

from data_clean import build_training_dataframe, clean_data
from eda import compare_eda_results, print_dataframe_overview, run_eda
from load_data import load_data
from load_images import load_images
from prepare_training_data import prepare_training_data
from train_models import train_models


RUN_LABEL = "pokemon_type_training"


# ── 1. Ladda och analysera rådata ─────────────────────────────

df_raw = load_data(DATA_PATH)
print_dataframe_overview(df_raw, "Data före rensning")
raw_eda_summary = run_eda(df_raw, "Rådata före rensning")

# ── 2. Rensa data och analysera referensdatan ─────────────────

df_reference = clean_data(df_raw)
print_dataframe_overview(df_reference, "Data efter rensning")
cleaned_eda_summary = run_eda(df_reference, "Rensad referensdata")
compare_eda_results(raw_eda_summary, cleaned_eda_summary)

# ── 3. Beskriv modellflödet ───────────────────────────────────

print("\n-- Modellflöde -------------------------------------------------")
print(f"Körning: {RUN_LABEL}")
print("1. training features -> PCA -> logreg")
print("2. training features -> XGBoost med GridSearchCV")
print(
    "3. training features -> stripped features "
    "(color, shape, habitat, growth_rate, sp_attack) -> XGBoost med GridSearchCV"
)
print("4. training features + image_pca -> PCA -> logreg")
print("5. training features + image_pca -> XGBoost med GridSearchCV")
print(
    "6. stripped training features + image_pca "
    "-> XGBoost med GridSearchCV"
)
print("7. training features -> RandomForest")
print("8. training features + image_pca -> RandomForest")
print("9. image_pca -> logreg")
print("10. image_pca -> XGBoost med GridSearchCV")
print("11. image_pca -> RandomForest")

# ── 4. Ladda bildfeatures, kör PCA och träna modeller ─────────

image_matrix, image_valid_mask = load_images(df_reference, IMAGE_DIR)
df_training = build_training_dataframe(df_reference)
training_data = prepare_training_data(
    df_training,
    image_matrix=image_matrix,
    image_valid_mask=image_valid_mask,
)
model_results = train_models(training_data, run_label=RUN_LABEL)

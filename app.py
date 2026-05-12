from settings import DATA_PATH

from data_clean import build_training_dataframe, clean_data
from eda import compare_eda_results, print_dataframe_overview, run_eda
from load_data import load_data
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
print("3. training features -> stripped features -> XGBoost med GridSearchCV")

# ── 4. Bygg träningsdata, kör PCA och träna modell ────────────

df_training = build_training_dataframe(df_reference)
training_data = prepare_training_data(df_training)
model_results = train_models(training_data, run_label=RUN_LABEL)

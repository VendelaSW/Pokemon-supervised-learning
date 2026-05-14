# Pokemon-supervised-learning

Projektet bygger en supervised learning-pipeline för att förutsäga en Pokémons primärtyp (`type_1`) från tabellfeatures. Flödet är skrivet så att rådata, rensad referensdata och träningsdata hålls isär.

## Flöde

1. `load_data.py` läser in `dataset/pokemon_complete.csv`.
2. `eda.py` kör EDA på rådata och sparar förklarande PNG-figurer i `eda_outputs/`.
3. `data_clean.py` rensar rådata till `df_reference`.
4. `eda.py` kör samma EDA på `df_reference` och skapar en jämförelse mot rådata.
5. `data_clean.py` bygger `df_training`, där referenskolumner tas bort och kategorier omvandlas med `pd.get_dummies()`.
6. `prepare_training_data.py` delar upp data i train/test, skalar features och kör PCA på träningsdelen.
7. `train_models.py` tränar logreg på PCA-data, XGBoost på fulla training features och en extra XGBoost på stripped features.
8. `train_models.py` sparar tränade modeller som `.pkl` och viktiga utvärderingsfigurer som PNG.

## Data

Projektet förväntar sig följande lokala struktur:

```text
dataset/
  pokemon_complete.csv
images/
  <pokedex_number>/
    <sprite>.png
```

`dataset/`, `images/`, `eda_outputs/`, `model_outputs/` och `.matplotlib_cache/` är ignorerade i Git eftersom de är lokala data- eller output-mappar.

## Träningsdata

`df_reference` behåller rensad data för EDA, uppslag och tolkning av resultat. `df_training` är den modellklara tabellen med numeriska training features och dummy-kodade kategorier.

Referenskolumner som `name`, `type_2`, abilities, egg groups och metadata exkluderas från träning. Den fulla träningsdatan använder numeriska stats som `hp`, `attack`, `defense`, `sp_attack`, `sp_defense`, `speed`, `height_m`, `weight_kg` och `capture_rate`, plus dummy-kolumner från kategorierna nedan.

Kategorier som används i träningen kodas med dummy-kolumner:

```text
color
shape
habitat
growth_rate
```

Den tredje XGBoost-körningen strippar sedan ner fulla training features till dummy-kolumner från `color`, `shape`, `habitat`, `growth_rate` samt den numeriska kolumnen `sp_attack`.

Målvariabeln är `type_1_encoded`, skapad från `type_1`.

## PCA och modeller

PCA körs efter train/test-split för att testdatan inte ska påverka skalning eller komponenter. `prepare_training_data.py` returnerar både originalfeatures och PCA-transformerade features, så logreg kan använda PCA medan XGBoost kan använda tolkningsbara originalfeatures.

Eftersom `type_1` har fler än två klasser används multinomial logistisk regression i stället för binär `Logit`.

XGBoost tränas med `GridSearchCV`, flera hyperparametrar och balanced `sample_weight` så ovanliga typer väger mer vid träning. Resultaten jämförs med accuracy, balanced accuracy, macro F1 och weighted F1.

## Modelloutput

Modelloutput hålls kort och fokuserad. Tränade modeller sparas som `.pkl`, och utvärdering/importance sparas som PNG:

```text
pokemon_type_training_logreg_pca_model.pkl
pokemon_type_training_xgboost_grid_search_model.pkl
pokemon_type_training_xgboost_stripped_features_model.pkl
pokemon_type_training_model_comparison.png
pokemon_type_training_logreg_pca_confusion_matrix.png
pokemon_type_training_xgboost_grid_search_confusion_matrix.png
pokemon_type_training_xgboost_stripped_features_confusion_matrix.png
pca_explained_variance.png
pokemon_type_training_xgboost_grid_search_feature_importance.png
pokemon_type_training_xgboost_grid_search_permutation_importance.png
pokemon_type_training_xgboost_grid_search_grouped_feature_importance.png
pokemon_type_training_xgboost_grid_search_grouped_permutation_importance.png
pokemon_type_training_xgboost_stripped_features_feature_importance.png
pokemon_type_training_xgboost_stripped_features_permutation_importance.png
pokemon_type_training_xgboost_stripped_features_grouped_feature_importance.png
pokemon_type_training_xgboost_stripped_features_grouped_permutation_importance.png
```

Feature importance och permutation importance sparas som PNG både per enskild feature och grupperat tillbaka till ursprungliga kategorikolumner som `color`, `shape`, `habitat` och `growth_rate`.

## Körning

Installera beroenden och kör sedan:

```bash
python app.py
```

I den lokala Anaconda-miljön som använts under utvecklingen:

```bash
/opt/anaconda3/bin/conda run -n Statistics_AI python app.py
```

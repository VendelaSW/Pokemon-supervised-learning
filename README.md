# Pokemon-supervised-learning

Projektet bygger en supervised learning-pipeline för att förutsäga en Pokémons primärtyp (`type_1`) från tabellfeatures och sprite-bilder. Flödet är skrivet så att rådata, rensad referensdata och träningsdata hålls isär.

## Flöde

1. `load_data.py` läser in `dataset/pokemon_complete.csv`.
2. `eda.py` kör EDA på rådata och sparar förklarande PNG-figurer i `eda_outputs/`.
3. `data_clean.py` rensar rådata till `df_reference`.
4. `eda.py` kör samma EDA på `df_reference` och skapar en jämförelse mot rådata.
5. `data_clean.py` bygger `df_training`, där referenskolumner tas bort och kategorier omvandlas med `pd.get_dummies()`.
6. `load_images.py` laddar sprite-bilder och plattar ut dem till råa pixelfeatures.
7. `prepare_training_data.py` delar upp data i train/test, skalar features och kör PCA på träningsdelen. Råa pixelfeatures komprimeras separat till `image_pca_*` med 90 % bevarad varians.
8. `train_models.py` tränar fyra tabellbaserade modeller och fyra motsvarande modeller med image features.
9. `train_models.py` sparar tränade modeller som `.pkl` och viktiga utvärderingsfigurer som PNG.

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

När sprite-bilder används görs råa pixlar först om till `image_pca_*`-features med 90 % bevarad varians. Steg 4-6 och 8 använder samma train/test-split som steg 1-3 och 7, men lägger till dessa image features.

Målvariabeln är `type_1_encoded`, skapad från `type_1`.

## PCA och modeller

PCA körs efter train/test-split för att testdatan inte ska påverka skalning eller komponenter. `prepare_training_data.py` returnerar både originalfeatures och PCA-transformerade features, så logreg kan använda PCA medan XGBoost kan använda tolkningsbara originalfeatures. Bildfeatures PCA-komprimeras separat innan de kombineras med tabellfeatures.

Eftersom `type_1` har fler än två klasser används multinomial logistisk regression i stället för binär `Logit`.

XGBoost tränas med `GridSearchCV`, flera hyperparametrar och balanced `sample_weight` så ovanliga typer väger mer vid träning. RandomForest tränas som en fast baseline med `class_weight="balanced"`. Resultaten jämförs med accuracy, balanced accuracy, macro F1 och weighted F1.

Modellflödena är:

```text
1. training features -> PCA -> logreg
2. training features -> XGBoost med GridSearchCV
3. training features -> stripped features -> XGBoost med GridSearchCV
4. training features + image_pca -> PCA -> logreg
5. training features + image_pca -> XGBoost med GridSearchCV
6. stripped training features + image_pca -> XGBoost med GridSearchCV
7. training features -> RandomForest
8. training features + image_pca -> RandomForest
```

## Modelloutput

Modelloutput hålls kort och fokuserad. Tränade modeller sparas som `.pkl`, och utvärdering/importance sparas som PNG:

```text
pokemon_type_training_logreg_pca_model.pkl
pokemon_type_training_xgboost_grid_search_model.pkl
pokemon_type_training_xgboost_stripped_features_model.pkl
pokemon_type_training_logreg_pca_with_images_model.pkl
pokemon_type_training_xgboost_grid_search_with_images_model.pkl
pokemon_type_training_xgboost_stripped_features_with_images_model.pkl
pokemon_type_training_random_forest_model.pkl
pokemon_type_training_random_forest_with_images_model.pkl
pokemon_type_training_model_comparison.png
pokemon_type_training_logreg_pca_confusion_matrix.png
pokemon_type_training_xgboost_grid_search_confusion_matrix.png
pokemon_type_training_xgboost_stripped_features_confusion_matrix.png
pokemon_type_training_logreg_pca_with_images_confusion_matrix.png
pokemon_type_training_xgboost_grid_search_with_images_confusion_matrix.png
pokemon_type_training_xgboost_stripped_features_with_images_confusion_matrix.png
pokemon_type_training_random_forest_confusion_matrix.png
pokemon_type_training_random_forest_with_images_confusion_matrix.png
pca_explained_variance.png
pokemon_type_training_xgboost_grid_search_feature_importance.png
pokemon_type_training_xgboost_grid_search_permutation_importance.png
pokemon_type_training_xgboost_grid_search_grouped_feature_importance.png
pokemon_type_training_xgboost_grid_search_grouped_permutation_importance.png
pokemon_type_training_xgboost_stripped_features_feature_importance.png
pokemon_type_training_xgboost_stripped_features_permutation_importance.png
pokemon_type_training_xgboost_stripped_features_grouped_feature_importance.png
pokemon_type_training_xgboost_stripped_features_grouped_permutation_importance.png
pokemon_type_training_xgboost_grid_search_with_images_feature_importance.png
pokemon_type_training_xgboost_grid_search_with_images_permutation_importance.png
pokemon_type_training_xgboost_grid_search_with_images_grouped_feature_importance.png
pokemon_type_training_xgboost_grid_search_with_images_grouped_permutation_importance.png
pokemon_type_training_xgboost_stripped_features_with_images_feature_importance.png
pokemon_type_training_xgboost_stripped_features_with_images_permutation_importance.png
pokemon_type_training_xgboost_stripped_features_with_images_grouped_feature_importance.png
pokemon_type_training_xgboost_stripped_features_with_images_grouped_permutation_importance.png
pokemon_type_training_random_forest_feature_importance.png
pokemon_type_training_random_forest_permutation_importance.png
pokemon_type_training_random_forest_grouped_feature_importance.png
pokemon_type_training_random_forest_grouped_permutation_importance.png
pokemon_type_training_random_forest_with_images_feature_importance.png
pokemon_type_training_random_forest_with_images_permutation_importance.png
pokemon_type_training_random_forest_with_images_grouped_feature_importance.png
pokemon_type_training_random_forest_with_images_grouped_permutation_importance.png
```

Feature importance och permutation importance sparas som PNG både per enskild feature och grupperat tillbaka till ursprungliga kategorikolumner som `color`, `shape`, `habitat`, `growth_rate` och `image_pca`.

## Körning

Installera beroenden och kör sedan:

```bash
python app.py
```

I den lokala Anaconda-miljön som använts under utvecklingen:

```bash
/opt/anaconda3/bin/conda run -n Statistics_AI python app.py
```

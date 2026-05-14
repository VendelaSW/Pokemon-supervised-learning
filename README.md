# Pokemon-supervised-learning

Projektet bygger en supervised learning-pipeline för att förutsäga en Pokémons primärtyp (`type_1`) från tabellfeatures och sprite-bilder. Flödet är skrivet så att rådata, rensad referensdata och träningsdata hålls isär.

## Projektbakgrund

Neintindo har anlitat oss för att hjälpa till med utvecklingen av nästa generations Mockemon-spel. De vill undersöka om Mockemon-karaktärernas egenskaper tydligt speglar vilken typ de tillhör.

Uppdraget är därför att analysera om karaktärernas statistik och visuella design är tillräckligt distinkta för att deras typ ska kunna förutsägas med hjälp av supervised learning. Modellen används som en indikation på om det finns tydliga mönster som skiljer de olika Mockemon-typerna från varandra.

Till vår hjälp har vi fått tillgång till detaljerad data om varje Mockemon. Datan innehåller bland annat information om deras egenskaper och fysiska attribut, såsom HP, Attack, Defense, Special Attack, Special Defense, Speed, Height, Weight och Shape, samt sprite-bilder av karaktärerna.

Genom att kombinera tabellbaserade egenskaper med visuella bildfeatures undersöker vi om Mockemon-karaktärernas statistik och visuella kännetecken har ett tydligt samband med deras respektive typer.

## Flöde

1. `load_data.py` läser in `dataset/pokemon_complete.csv`.
2. `eda.py` kör EDA på rådata och sparar förklarande PNG-figurer i `eda_outputs/`.
3. `data_clean.py` rensar rådata till `df_reference`.
4. `eda.py` kör samma EDA på `df_reference` och skapar en jämförelse mot rådata.
5. `data_clean.py` bygger `df_training`, där referenskolumner tas bort och kategorier omvandlas med `pd.get_dummies()`.
6. `load_images.py` laddar sprite-bilder och plattar ut dem till råa pixelfeatures.
7. `prepare_training_data.py` delar upp data i train/test, skalar tabellfeatures och kör PCA på träningsdelen. Bildpixlar är redan normaliserade till 0-1 och komprimeras separat till `image_pca_*` med 90 % bevarad varians.
8. `train_models.py` tränar fyra tabellbaserade modeller, fyra modeller med tabellfeatures och image features samt tre image-only-modeller.
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

När sprite-bilder används normaliseras pixlarna först till 0-1 i `load_images.py` och görs sedan om till `image_pca_*`-features med 90 % bevarad varians. Bild-PCA använder inte `StandardScaler`; PCA centrerar pixeldatan internt. Steg 4-6 och 8 använder samma train/test-split som steg 1-3 och 7, men lägger till dessa image features. Steg 9-11 använder endast `image_pca_*` för att testa om bilderna ensamma kan förutsäga `type_1`.

Målvariabeln är `type_1_encoded`, skapad från `type_1`.

## PCA och modeller

PCA körs efter train/test-split för att testdatan inte ska påverka skalning eller komponenter. `prepare_training_data.py` returnerar både originalfeatures och PCA-transformerade features, så logreg kan använda PCA medan XGBoost kan använda tolkningsbara originalfeatures. Bildfeatures PCA-komprimeras separat från 0-1-normaliserade pixlar innan de kombineras med tabellfeatures.

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
9. image_pca -> logreg
10. image_pca -> XGBoost med GridSearchCV
11. image_pca -> RandomForest
```

## Tolkning av modellmått

Projektet är en multiclass classification där modellen predikterar `type_1`. Pokémon-typerna är obalanserade, vilket betyder att vissa typer förekommer oftare än andra. Därför ska `accuracy` inte tolkas ensam.

| Mått | Tolkning i projektet |
| --- | --- |
| `accuracy` | Andel rätt predikterade Pokémon totalt. Måttet är lätt att förstå, men kan bli missvisande om modellen främst lyckas på vanliga typer. |
| `balanced_accuracy` / `bal_acc` | Genomsnittlig recall över alla `type_1`-klasser. Varje typ får lika stor vikt, även ovanliga typer. |
| `macro_f1` | Genomsnittligt F1-score där varje typ väger lika. Bra för att se om modellen fungerar brett över alla typer, inte bara de vanligaste. |
| `weighted_f1` | F1-score viktat efter hur många exempel varje typ har. Bra som helhetsmått, men kan dölja svaga resultat på ovanliga typer. |

Vid modelljämförelse bör `balanced_accuracy` och `macro_f1` prioriteras när målet är att jämföra modeller rättvist över alla Pokémon-typer. `accuracy` och `weighted_f1` används som sekundära helhetsmått. Om `accuracy` är tydligt högre än `balanced_accuracy` tyder det ofta på att modellen är starkare på vanliga typer än på ovanliga. Om `macro_f1` är låg tyder det på problem med precision eller recall för flera typer.

Classification report används för att se vilka specifika typer som fungerar bra eller dåligt. Confusion matrix används för att se vilka typer modellen blandar ihop.

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
pokemon_type_training_logreg_image_only_model.pkl
pokemon_type_training_xgboost_image_only_model.pkl
pokemon_type_training_random_forest_image_only_model.pkl
pokemon_type_training_model_comparison.png
pokemon_type_training_logreg_pca_confusion_matrix.png
pokemon_type_training_xgboost_grid_search_confusion_matrix.png
pokemon_type_training_xgboost_stripped_features_confusion_matrix.png
pokemon_type_training_logreg_pca_with_images_confusion_matrix.png
pokemon_type_training_xgboost_grid_search_with_images_confusion_matrix.png
pokemon_type_training_xgboost_stripped_features_with_images_confusion_matrix.png
pokemon_type_training_random_forest_confusion_matrix.png
pokemon_type_training_random_forest_with_images_confusion_matrix.png
pokemon_type_training_logreg_image_only_confusion_matrix.png
pokemon_type_training_xgboost_image_only_confusion_matrix.png
pokemon_type_training_random_forest_image_only_confusion_matrix.png
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
pokemon_type_training_xgboost_image_only_feature_importance.png
pokemon_type_training_xgboost_image_only_permutation_importance.png
pokemon_type_training_xgboost_image_only_grouped_feature_importance.png
pokemon_type_training_xgboost_image_only_grouped_permutation_importance.png
pokemon_type_training_random_forest_image_only_feature_importance.png
pokemon_type_training_random_forest_image_only_permutation_importance.png
pokemon_type_training_random_forest_image_only_grouped_feature_importance.png
pokemon_type_training_random_forest_image_only_grouped_permutation_importance.png
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

# Pokemon-supervised-learning

Projektet bygger en supervised learning-pipeline för att förutsäga en Pokémons primärtyp (`type_1`) från tabellfeatures. Flödet är skrivet så att rådata, rensad referensdata och träningsdata hålls isär.

## Flöde

1. `load_data.py` läser in `dataset/pokemon_complete.csv`.
2. `eda.py` kör EDA på rådata och sparar rapporter/figurer i `eda_outputs/`.
3. `data_clean.py` rensar rådata till `df_reference`.
4. `eda.py` kör samma EDA på `df_reference` och skapar en jämförelse mot rådata.
5. `data_clean.py` bygger `df_training`, där referenskolumner tas bort och kategorier omvandlas med `pd.get_dummies()`.
6. `prepare_training_data.py` delar upp data i train/test, skalar features och kör PCA på träningsdelen.
7. `train_models.py` tränar multinomial logistisk regression på PCA-data och XGBoost på originalfeatures.
8. `train_models.py` utvärderar modellerna och sparar feature importance samt permutation importance.

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

`df_reference` behåller rensad data för EDA, uppslag och tolkning av resultat. `df_training` är den modellklara tabellen.

Exkluderade träningskolumner inkluderar bland annat `name`, `type_2`, `genus`, `hidden_ability`, `generation`, `base_stat_total`, `base_experience`, `base_happiness`, legendary/mythical/baby-flaggor, antal abilities/egg groups, `evolution_chain_id` och `sprite_url`.

Kategorier som används i träningen kodas med dummy-kolumner:

```text
color
shape
habitat
growth_rate
```

Målvariabeln är `type_1_encoded`, skapad från `type_1`.

## PCA och modeller

PCA körs efter train/test-split för att testdatan inte ska påverka skalning eller komponenter. `prepare_training_data.py` returnerar både originalfeatures och PCA-transformerade features, så logreg kan använda PCA medan XGBoost kan använda tolkningsbara originalfeatures.

Eftersom `type_1` har fler än två klasser används multinomial logistisk regression i stället för binär `Logit`.

XGBoost tränas med `GridSearchCV` och flera hyperparametrar. Resultaten jämförs med accuracy, balanced accuracy, macro F1 och weighted F1.

## Modelloutput

Modellrapporter sparas i `model_outputs/`:

```text
model_comparison.csv
xgboost_best_params.json
classification reports
confusion matrices
pca_explained_variance.csv/png
xgboost_feature_importance.csv/png
xgboost_permutation_importance.csv/png
```

Feature importance och permutation importance sparas både per enskild dummy-feature och grupperat tillbaka till ursprungliga kategorikolumner som `color`, `shape` och `habitat`.

## Körning

Installera beroenden och kör sedan:

```bash
python app.py
```

I den lokala Anaconda-miljön som använts under utvecklingen:

```bash
/opt/anaconda3/bin/conda run -n Statistics_AI python app.py
```

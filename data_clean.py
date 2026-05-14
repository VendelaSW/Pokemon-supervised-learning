from __future__ import annotations

import pandas as pd


EXPECTED_COLUMNS = [
    "pokedex_number",
    "name",
    "type_1",
    "type_2",
    "hp",
    "attack",
    "defense",
    "sp_attack",
    "sp_defense",
    "speed",
    "base_stat_total",
    "height_m",
    "weight_kg",
    "base_experience",
    "abilities",
    "hidden_ability",
    "generation",
    "is_legendary",
    "is_mythical",
    "is_baby",
    "color",
    "shape",
    "egg_groups",
    "habitat",
    "growth_rate",
    "capture_rate",
    "base_happiness",
    "genus",
    "evolution_chain_id",
    "sprite_url",
]

GENERATION_MAP = {
    "gen-i": 1,
    "gen-ii": 2,
    "gen-iii": 3,
    "gen-iv": 4,
    "gen-v": 5,
    "gen-vi": 6,
    "gen-vii": 7,
    "gen-viii": 8,
    "gen-ix": 9,
}

DUMMY_COLUMNS = [
    "color",
    "shape",
    "habitat",
    "growth_rate",
]

TARGET_COLUMN = "type_1_encoded"
TARGET_SOURCE_COLUMN = "type_1"

REFERENCE_ONLY_COLUMNS = [
    # Identitetskolumner som bara används för uppslag och tolkning.
    "pokedex_number",
    "name",
    # Rå målvariabel och sekundär typ sparas i df_reference, men träningsdatan
    # använder type_1_encoded som y och exkluderar type_2.
    "type_1",
    "type_2",
    # Abilities och egg groups extraheras för EDA/referens, men används inte
    # som features i nuvarande modellflöde.
    "abilities",
    "ability_1",
    "ability_2",
    "ability_3",
    "hidden_ability",
    "egg_groups",
    "egg_group_1",
    "egg_group_2",
    # Dessa råkategorier ersätts av dummy-kolumner från DUMMY_COLUMNS.
    "color",
    "shape",
    "habitat",
    "growth_rate",
    "genus",
    # Kolumner som medvetet inte ska användas vid modellträning.
    "base_stat_total",
    "base_experience",
    "base_happiness",
    "generation",
    "is_legendary",
    "is_mythical",
    "is_baby",
    "has_hidden_ability",
    "num_abilities",
    "num_egg_groups",
    "evolution_chain_id",
    "sprite_url",
]


def _require_columns(df: pd.DataFrame) -> None:
    missing = [column for column in EXPECTED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Datasetet saknar kolumner: {missing}")


def _clean_text(series: pd.Series, fill_value: str | None = None) -> pd.Series:
    cleaned = series.astype("string").str.strip()
    cleaned = cleaned.replace("", pd.NA)
    if fill_value is not None:
        cleaned = cleaned.fillna(fill_value)
    return cleaned


def _clean_pipe_text(series: pd.Series, fill_value: str) -> pd.Series:
    cleaned = _clean_text(series, fill_value).str.lower()
    return cleaned.str.replace(r"\s*\|\s*", "|", regex=True)


def _clean_number(
    series: pd.Series,
    *,
    dtype: str,
    fill_value: float | int | None = None,
) -> pd.Series:
    cleaned = pd.to_numeric(series, errors="coerce")
    if fill_value is None:
        fill_value = cleaned.median()
    cleaned = cleaned.fillna(fill_value)
    if dtype == "int":
        return cleaned.round().astype(int)
    if dtype == "float":
        return cleaned.astype(float)
    raise ValueError(f"Okand numerisk dtype: {dtype}")


def _clean_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.astype(int)

    normalized = series.astype("string").str.strip().str.lower()
    return normalized.map(
        {
            "true": 1,
            "false": 0,
            "1": 1,
            "0": 0,
            "yes": 1,
            "no": 0,
        }
    ).fillna(0).astype(int)


def _split_pipe_values(series: pd.Series) -> pd.Series:
    return series.fillna("").str.split("|")


def _pipe_value(parts: pd.Series, index: int, fill_value: str = "none") -> pd.Series:
    value = parts.str[index].fillna(fill_value)
    return value.replace("", fill_value)


def _pipe_count(parts: pd.Series, ignored_values: set[str]) -> pd.Series:
    def count_values(values: list[str]) -> int:
        return sum(1 for value in values if value and value not in ignored_values)

    return parts.apply(count_values).astype(int)


def _add_target_encoding(df: pd.DataFrame, column: str) -> dict[str, int]:
    categories = sorted(df[column].dropna().unique())
    mapping = {category: index for index, category in enumerate(categories)}
    df[TARGET_COLUMN] = df[column].map(mapping).astype(int)
    return mapping


def _make_dummy_columns(df: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, list[str]]:
    dummies = pd.get_dummies(
        df[columns],
        columns=columns,
        prefix=columns,
        prefix_sep="_",
        dtype=int,
    )
    return dummies, dummies.columns.to_list()


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Rensar rå Pokémon-data genom att extrahera, bearbeta och skriva tillbaka kolumner.

    Det viktiga mönstret är:
      1. Kopiera rådata till en ny DataFrame.
      2. Extrahera varje kolumn till en namngiven Series.
      3. Bearbeta dessa Series separat.
      4. Skriv tillbaka de rensade värdena till df.
    """
    print("\n-- Datarensning ------------------------------------------------")

    _require_columns(df)
    df = df.copy()

    # Extrahera varje råkolumn som en egen Series så att varje värde
    # kan rensas eller omvandlas innan det skrivs tillbaka till DataFrame:n.
    pokedex_number = df["pokedex_number"]
    name = df["name"]
    type_1 = df["type_1"]
    type_2 = df["type_2"]
    hp = df["hp"]
    attack = df["attack"]
    defense = df["defense"]
    sp_attack = df["sp_attack"]
    sp_defense = df["sp_defense"]
    speed = df["speed"]
    height_m = df["height_m"]
    weight_kg = df["weight_kg"]
    base_experience = df["base_experience"]
    abilities = df["abilities"]
    hidden_ability = df["hidden_ability"]
    generation = df["generation"]
    is_legendary = df["is_legendary"]
    is_mythical = df["is_mythical"]
    is_baby = df["is_baby"]
    color = df["color"]
    shape = df["shape"]
    egg_groups = df["egg_groups"]
    habitat = df["habitat"]
    growth_rate = df["growth_rate"]
    capture_rate = df["capture_rate"]
    base_happiness = df["base_happiness"]
    genus = df["genus"]
    evolution_chain_id = df["evolution_chain_id"]
    sprite_url = df["sprite_url"]

    # Bearbeta de extraherade värdena.
    pokedex_number = _clean_number(pokedex_number, dtype="int")
    name = _clean_text(name)
    type_1 = _clean_text(type_1)
    type_2 = _clean_text(type_2, "None")

    hp = _clean_number(hp, dtype="int")
    attack = _clean_number(attack, dtype="int")
    defense = _clean_number(defense, dtype="int")
    sp_attack = _clean_number(sp_attack, dtype="int")
    sp_defense = _clean_number(sp_defense, dtype="int")
    speed = _clean_number(speed, dtype="int")
    base_stat_total = hp + attack + defense + sp_attack + sp_defense + speed

    height_m = _clean_number(height_m, dtype="float")
    weight_kg = _clean_number(weight_kg, dtype="float")
    base_experience = _clean_number(base_experience, dtype="int")

    abilities = _clean_pipe_text(abilities, "unknown")
    ability_parts = _split_pipe_values(abilities)
    ability_1 = _pipe_value(ability_parts, 0, "unknown")
    ability_2 = _pipe_value(ability_parts, 1, "none")
    ability_3 = _pipe_value(ability_parts, 2, "none")
    num_abilities = _pipe_count(ability_parts, {"", "unknown", "none"})

    hidden_ability = _clean_text(hidden_ability, "none").str.lower()
    has_hidden_ability = (hidden_ability != "none").astype(int)

    generation = _clean_text(generation).str.lower().map(GENERATION_MAP)
    generation = generation.fillna(generation.median()).astype(int)

    is_legendary = _clean_bool(is_legendary)
    is_mythical = _clean_bool(is_mythical)
    is_baby = _clean_bool(is_baby)

    color = _clean_text(color, "unknown").str.lower()
    shape = _clean_text(shape, "unknown").str.lower()

    egg_groups = _clean_pipe_text(egg_groups, "unknown")
    egg_group_parts = _split_pipe_values(egg_groups)
    egg_group_1 = _pipe_value(egg_group_parts, 0, "unknown")
    egg_group_2 = _pipe_value(egg_group_parts, 1, "none")
    num_egg_groups = _pipe_count(egg_group_parts, {"", "unknown", "none"})

    habitat = _clean_text(habitat, "unknown").str.lower()
    growth_rate = _clean_text(growth_rate, "unknown").str.lower()
    capture_rate = _clean_number(capture_rate, dtype="int")
    base_happiness = _clean_number(base_happiness, dtype="int")
    genus = _clean_text(genus, "unknown")
    evolution_chain_id = _clean_number(evolution_chain_id, dtype="int")
    sprite_url = _clean_text(sprite_url, "")

    # Skriv tillbaka de bearbetade värdena till DataFrame:n.
    df["pokedex_number"] = pokedex_number
    df["name"] = name
    df["type_1"] = type_1
    df["type_2"] = type_2
    df["hp"] = hp
    df["attack"] = attack
    df["defense"] = defense
    df["sp_attack"] = sp_attack
    df["sp_defense"] = sp_defense
    df["speed"] = speed
    df["base_stat_total"] = base_stat_total
    df["height_m"] = height_m
    df["weight_kg"] = weight_kg
    df["base_experience"] = base_experience
    df["abilities"] = abilities
    df["ability_1"] = ability_1
    df["ability_2"] = ability_2
    df["ability_3"] = ability_3
    df["num_abilities"] = num_abilities
    df["hidden_ability"] = hidden_ability
    df["has_hidden_ability"] = has_hidden_ability
    df["generation"] = generation
    df["is_legendary"] = is_legendary
    df["is_mythical"] = is_mythical
    df["is_baby"] = is_baby
    df["color"] = color
    df["shape"] = shape
    df["egg_groups"] = egg_groups
    df["egg_group_1"] = egg_group_1
    df["egg_group_2"] = egg_group_2
    df["num_egg_groups"] = num_egg_groups
    df["habitat"] = habitat
    df["growth_rate"] = growth_rate
    df["capture_rate"] = capture_rate
    df["base_happiness"] = base_happiness
    df["genus"] = genus
    df["evolution_chain_id"] = evolution_chain_id
    df["sprite_url"] = sprite_url

    target_map = _add_target_encoding(df, TARGET_SOURCE_COLUMN)
    df.attrs["target_map"] = target_map

    print(f"Rensad data: {df.shape[0]} rader, {df.shape[1]} kolumner")
    print(f"Target-kodad kolumn: {TARGET_SOURCE_COLUMN} -> {TARGET_COLUMN}")
    print("Extraherade referenskolumner: abilities och egg_groups har delats upp")

    return df


def build_training_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Skapar en tränings-DataFrame utan att ändra referensdatan.

    Den returnerade DataFrame:n behåller TARGET_COLUMN som y, numeriska
    träningsfeatures och dummy-kodade kategorier från DUMMY_COLUMNS. Råkolumner
    som behövs för tolkning ligger kvar i df_reference men tas bort här.
    """
    required_columns = REFERENCE_ONLY_COLUMNS + DUMMY_COLUMNS
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Kan inte skapa träningsdata. Saknade kolumner: {missing}")

    dummies, dummy_columns = _make_dummy_columns(df, DUMMY_COLUMNS)
    df_training = pd.concat(
        [df.drop(columns=REFERENCE_ONLY_COLUMNS), dummies],
        axis=1,
    ).copy()
    feature_columns = [
        column for column in df_training.columns
        if column != TARGET_COLUMN
    ]

    df_training.attrs["target_column"] = TARGET_COLUMN
    df_training.attrs["feature_columns"] = feature_columns
    df_training.attrs["dummy_columns"] = dummy_columns
    df_training.attrs["reference_only_columns"] = REFERENCE_ONLY_COLUMNS
    df_training.attrs["target_map"] = df.attrs.get("target_map", {})
    df.attrs["training_columns"] = df_training.columns.to_list()

    print("\n-- Träningsdata ------------------------------------------------")
    print(f"Referensdata behålls: {df.shape[0]} rader, {df.shape[1]} kolumner")
    print(f"Träningsdata:         {df_training.shape[0]} rader, {df_training.shape[1]} kolumner")
    print(f"Målkolumn:            {TARGET_COLUMN}")
    print(f"Dummy-featurekällor:  {DUMMY_COLUMNS}")
    print(f"Dummy-kolumner:       {len(dummy_columns)}")
    print(f"Feature-kolumner:    {len(feature_columns)} totalt")

    return df_training

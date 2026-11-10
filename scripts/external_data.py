"""Laden, Bereinigen und Matchen der externen Datenquellen.

Verwendete Quellen:
- IGDB-Spielemetadaten (Kaggle: emirshn/igdb-dataset-for-data-mining-projects)
- Metacritic-Spielebewertungen (Kaggle: stadynromn/metacritic-video-games-reviews-up-to-sep-2025)
- Steam-Preishistorie (Kaggle: andrii222/steam-price-history-nov-2023-oct-2025)

Die Quellen werden ueber einen normalisierten Spielnamen und das
Veroeffentlichungsjahr (IGDB, Metacritic) bzw. die Steam-AppID (Preishistorie)
an den FronkonGames-Datensatz angebunden.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import numpy as np
import pandas as pd

from common import DATA, normalize_name, save_table

IGDB_SLUG = "emirshn/igdb-dataset-for-data-mining-projects"
IGDB_VERSION = 1
MC_SLUG = "stadynromn/metacritic-video-games-reviews-up-to-sep-2025"
MC_VERSION = 1
PRICE_SLUG = "andrii222/steam-price-history-nov-2023-oct-2025"
PRICE_VERSION = 1

IGDB_GENRES = [
    "Adventure", "Arcade", "Card & Board Game", "Fighting",
    "Hack and slash/Beat 'em up", "Indie", "MOBA", "Music", "Platform",
    "Point-and-click", "Puzzle", "Quiz/Trivia", "Racing",
    "Real Time Strategy (RTS)", "Role-playing (RPG)", "Shooter", "Simulator",
    "Sport", "Strategy", "Tactical", "Turn-based strategy (TBS)", "Visual Novel",
]


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _dataset_file(slug: str, filename: str, version: int) -> Path:
    import kagglehub

    root = Path(kagglehub.dataset_download(f"{slug}/versions/{version}"))
    candidates = list(root.rglob(filename))
    if not candidates:
        raise FileNotFoundError(f"{filename} im Kaggle-Download {slug} nicht gefunden")
    return candidates[0]


def parse_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if not isinstance(value, str) or not value.strip() or value.strip().lower() == "nan":
        return []
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(parsed, (list, tuple, set)):
        return [str(v).strip() for v in parsed if str(v).strip()]
    return [str(parsed).strip()]


def _normalise_name(series: pd.Series) -> pd.Series:
    norm = series.apply(normalize_name)
    norm = norm.where(norm.str.len() >= 3, "")
    return norm


def load_igdb() -> pd.DataFrame:
    path = _dataset_file(IGDB_SLUG, "game_dataset_cleaned.csv", IGDB_VERSION)
    cols = [
        "id", "name", "category", "release_date", "rating", "aggregated_rating",
        "genres", "themes", "franchise", "main_developers", "publishers",
        "platforms", "game_modes", "player_perspectives",
    ]
    df = pd.read_csv(path, usecols=cols)

    df["igdb_release_year"] = pd.to_datetime(df["release_date"], errors="coerce").dt.year
    df["_norm"] = _normalise_name(df["name"])

    for col in ["genres", "themes", "platforms", "game_modes", "player_perspectives"]:
        df[col] = df[col].apply(parse_list)
    for col in ["franchise", "main_developers", "publishers"]:
        df[col] = df[col].apply(parse_list)

    df["_quality"] = (
        df["rating"].notna().astype(int) * 4
        + df["aggregated_rating"].notna().astype(int) * 3
        + df["genres"].apply(len).clip(upper=3)
        + df["game_modes"].apply(len).clip(upper=2)
        + df["platforms"].apply(len).clip(upper=2)
    )
    df = df[df["_norm"] != ""].copy()
    df = df.sort_values(
        ["_norm", "igdb_release_year", "_quality", "rating", "id"],
        ascending=[True, True, False, False, True],
    )
    df = df.drop_duplicates(["_norm", "igdb_release_year"])

    for genre in IGDB_GENRES:
        df["igdb_genre_" + _slug(genre)] = df["genres"].apply(lambda xs, g=genre: g in xs)

    df["igdb_singleplayer"] = df["game_modes"].apply(lambda xs: "Single player" in xs)
    df["igdb_multiplayer"] = df["game_modes"].apply(lambda xs: "Multiplayer" in xs)
    df["igdb_coop"] = df["game_modes"].apply(lambda xs: "Co-operative" in xs)
    df["igdb_mmo"] = df["game_modes"].apply(lambda xs: "MMO" in xs)
    df["igdb_pc"] = df["platforms"].apply(
        lambda xs: any("PC (Microsoft Windows)" in x for x in xs)
    )
    df["igdb_n_genres"] = df["genres"].apply(lambda xs: len(set(xs)))
    df["igdb_n_platforms"] = df["platforms"].apply(lambda xs: len(set(xs)))

    for col in ["genres", "themes", "platforms", "game_modes", "player_perspectives",
                "franchise", "main_developers", "publishers"]:
        df[col] = df[col].apply(lambda xs: "; ".join(xs))

    rename = {
        "id": "igdb_id",
        "name": "igdb_name",
        "category": "igdb_category",
        "release_date": "igdb_release_date",
        "rating": "igdb_rating",
        "aggregated_rating": "igdb_aggregated_rating",
        "genres": "igdb_genres",
        "themes": "igdb_themes",
        "franchise": "igdb_franchise",
        "main_developers": "igdb_main_developers",
        "publishers": "igdb_publishers",
        "platforms": "igdb_platforms",
        "game_modes": "igdb_game_modes",
        "player_perspectives": "igdb_player_perspectives",
    }
    df = df.rename(columns=rename)
    keep = ["_norm", "igdb_release_year"] + list(rename.values()) + [
        "igdb_singleplayer", "igdb_multiplayer", "igdb_coop", "igdb_mmo",
        "igdb_pc", "igdb_n_genres", "igdb_n_platforms",
    ] + ["igdb_genre_" + _slug(g) for g in IGDB_GENRES]
    return df[keep].copy()


def load_metacritic() -> pd.DataFrame:
    path = _dataset_file(MC_SLUG, "metacritic_dataset_clean.csv", MC_VERSION)
    df = pd.read_csv(path)

    df = df[df["platform"].astype(str).str.fullmatch("PC", case=False, na=False)].copy()
    df["mc_release_year"] = pd.to_datetime(df["release_date"], errors="coerce").dt.year
    df["_norm"] = _normalise_name(df["name"])
    df = df[df["_norm"] != ""].copy()
    df = df.sort_values(
        ["_norm", "mc_release_year", "metascore", "user_score"],
        ascending=[True, True, False, False],
    )
    df = df.drop_duplicates(["_norm", "mc_release_year"])

    rename = {
        "name": "mc_name",
        "platform": "mc_platform",
        "metascore": "mc_metascore",
        "user_score": "mc_user_score",
        "developer": "mc_developer",
        "publisher": "mc_publisher",
        "genre": "mc_genre",
        "release_date": "mc_release_date",
    }
    df = df.rename(columns=rename)
    return df[
        ["_norm", "mc_release_year", "mc_name", "mc_platform", "mc_metascore",
         "mc_user_score", "mc_developer", "mc_publisher", "mc_genre", "mc_release_date"]
    ].copy()


def load_price_history() -> pd.DataFrame:
    path = _dataset_file(PRICE_SLUG, "steam_price_history.csv", PRICE_VERSION)
    df = pd.read_csv(path)

    for col in ["app_id", "base_price", "lowest_price", "price_changes", "days_on_sale"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["month"] = pd.to_datetime(df["month"], errors="coerce")
    df = df.dropna(subset=["app_id", "month"]).copy()
    df["app_id"] = df["app_id"].astype(int)

    df["discount_pct"] = np.where(
        df["base_price"] > 0,
        (df["base_price"] - df["lowest_price"]) / df["base_price"] * 100.0,
        np.nan,
    )
    df["discount_pct"] = df["discount_pct"].clip(lower=0, upper=100)

    monthly = (
        df.groupby(["app_id", "month"], as_index=False)
        .agg(
            base_price=("base_price", "median"),
            lowest_price=("lowest_price", "min"),
            price_changes=("price_changes", "sum"),
            days_on_sale=("days_on_sale", "max"),
        )
    )
    monthly["discount_pct"] = np.where(
        monthly["base_price"] > 0,
        (monthly["base_price"] - monthly["lowest_price"]) / monthly["base_price"] * 100.0,
        np.nan,
    )
    monthly["discount_pct"] = monthly["discount_pct"].clip(lower=0, upper=100)

    agg = (
        monthly.groupby("app_id", as_index=False)
        .agg(
            ph_months=("month", "nunique"),
            ph_first_month=("month", "min"),
            ph_last_month=("month", "max"),
            ph_base_price=("base_price", "median"),
            ph_lowest_price=("lowest_price", "min"),
            ph_max_discount_pct=("discount_pct", "max"),
            ph_mean_discount_pct=("discount_pct", "mean"),
            ph_price_changes_total=("price_changes", "sum"),
            ph_days_on_sale_total=("days_on_sale", "sum"),
            ph_months_on_sale=("days_on_sale", lambda s: int((s > 0).sum())),
        )
    )
    agg["ph_mean_price_changes"] = agg["ph_price_changes_total"] / agg["ph_months"].clip(lower=1)
    agg["ph_share_months_on_sale"] = agg["ph_months_on_sale"] / agg["ph_months"].clip(lower=1)
    agg["ph_mean_days_on_sale"] = agg["ph_days_on_sale_total"] / agg["ph_months"].clip(lower=1)
    agg["ph_has_sale"] = agg["ph_months_on_sale"] > 0
    return agg.rename(columns={"app_id": "AppID_int"})


def attach_external_sources(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Matcht alle externen Quellen an den Basis-Datensatz.

    Erwartet die Spalten AppID, Name, release_year und analytic_sample.
    Gibt den erweiterten Datensatz sowie Coverage- und Overlap-Tabellen zurueck.
    """
    base = df[["AppID", "Name", "release_year", "analytic_sample"]].copy()
    base["_norm"] = _normalise_name(base["Name"])
    base["AppID_int"] = pd.to_numeric(base["AppID"], errors="coerce")

    igdb = load_igdb()
    mc = load_metacritic()
    price = load_price_history()

    base = base.merge(
        igdb,
        how="left",
        left_on=["_norm", "release_year"],
        right_on=["_norm", "igdb_release_year"],
        validate="many_to_one",
    )
    base = base.drop(columns=["igdb_release_year"], errors="ignore")

    base = base.merge(price, how="left", on="AppID_int", validate="many_to_one")
    base = base.drop(columns=["AppID_int"], errors="ignore")

    base = base.merge(
        mc,
        how="left",
        left_on=["_norm", "release_year"],
        right_on=["_norm", "mc_release_year"],
        validate="many_to_one",
    )
    base = base.drop(columns=["_norm", "mc_release_year"], errors="ignore")

    base["has_igdb"] = base["igdb_id"].notna()
    base["has_mc_external"] = base["mc_metascore"].notna()
    base["has_price_history"] = base["ph_months"].notna()
    base["n_sources"] = (
        1 + base["has_igdb"].astype(int) + base["has_mc_external"].astype(int)
        + base["has_price_history"].astype(int)
    )

    external_cols = [c for c in base.columns if c.startswith(("igdb_", "mc_", "ph_"))]
    feature_cols = ["AppID"] + external_cols + ["has_igdb", "has_mc_external",
                                                "has_price_history", "n_sources"]
    base[feature_cols].to_parquet(DATA / "external_features.parquet", index=False)

    out = df.merge(
        base.drop(columns=["Name", "release_year", "analytic_sample"]),
        how="left",
        on="AppID",
        validate="one_to_one",
    )

    n_analytic = int(df["analytic_sample"].sum())
    coverage = pd.DataFrame(
        [
            {
                "Quelle": "FronkonGames (Steam)",
                "Datensatz": "steam-games-dataset, Version 57",
                "N_Quelle": len(df),
                "N_gematcht_gesamt": len(df),
                "Match_Rate_gesamt": 1.0,
                "N_gematcht_analytisch": n_analytic,
                "Match_Rate_analytisch": 1.0,
                "Schluessel": "AppID",
                "Hinweis": "Basisdatensatz",
            },
            {
                "Quelle": "IGDB",
                "Datensatz": "emirshn/igdb-dataset-for-data-mining-projects, Version 1",
                "N_Quelle": len(igdb),
                "N_gematcht_gesamt": int(base["has_igdb"].sum()),
                "Match_Rate_gesamt": base["has_igdb"].mean(),
                "N_gematcht_analytisch": int((base["has_igdb"] & base["analytic_sample"]).sum()),
                "Match_Rate_analytisch": (base["has_igdb"] & base["analytic_sample"]).sum() / n_analytic,
                "Schluessel": "normalisierter Name + Jahr",
                "Hinweis": "ein IGDB-Treffer pro Name/Jahr",
            },
            {
                "Quelle": "Metacritic extern",
                "Datensatz": "stadynromn/metacritic-video-games-reviews-up-to-sep-2025, Version 1",
                "N_Quelle": len(mc),
                "N_gematcht_gesamt": int(base["has_mc_external"].sum()),
                "Match_Rate_gesamt": base["has_mc_external"].mean(),
                "N_gematcht_analytisch": int((base["has_mc_external"] & base["analytic_sample"]).sum()),
                "Match_Rate_analytisch": (base["has_mc_external"] & base["analytic_sample"]).sum() / n_analytic,
                "Schluessel": "normalisierter Name + Jahr (nur PC)",
                "Hinweis": "nur PC-Versionen des Metacritic-Datensatzes",
            },
            {
                "Quelle": "Preishistorie",
                "Datensatz": "andrii222/steam-price-history-nov-2023-oct-2025, Version 1",
                "N_Quelle": len(price),
                "N_gematcht_gesamt": int(base["has_price_history"].sum()),
                "Match_Rate_gesamt": base["has_price_history"].mean(),
                "N_gematcht_analytisch": int((base["has_price_history"] & base["analytic_sample"]).sum()),
                "Match_Rate_analytisch": (base["has_price_history"] & base["analytic_sample"]).sum() / n_analytic,
                "Schluessel": "Steam-AppID",
                "Hinweis": "monatliche Preishistorie 2023-11 bis 2025-11",
            },
        ]
    )
    save_table(coverage, "source_coverage")

    overlap = (
        out.loc[out["analytic_sample"], "n_sources"]
        .value_counts()
        .sort_index()
        .rename_axis("Anzahl_Datenquellen")
        .reset_index(name="N")
    )
    overlap["Anteil"] = overlap["N"] / max(int(out["analytic_sample"].sum()), 1)
    save_table(overlap, "source_overlap")

    return out, coverage, overlap

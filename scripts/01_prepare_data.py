"""Datenaufbereitung, Bereinigung und Feature Engineering.

Input : FronkonGames Steam Games Dataset (games.csv)
Output: data/steam_prepared.parquet, data/analytic_sample.parquet,
        outputs/tables/sample_flow.csv, missing_values.csv, variable_overview.csv
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from common import (
    DATA,
    MIN_REVIEWS,
    MIN_REVIEWS_ROBUST,
    NON_GAME_PATTERN,
    REFERENCE_YEAR,
    SOFTWARE_GENRES,
    STANDARD_GENRES,
    load_raw,
    normalize_name,
    save_table,
    split_list,
    wilson_lower_bound,
)
from external_data import attach_external_sources

OWNER_CLASSES = {
    0: (1, "0 - 20.000"),
    20000: (2, "20.000 - 50.000"),
    50000: (3, "50.000 - 100.000"),
    100000: (4, "100.000 - 200.000"),
    200000: (5, "200.000 - 500.000"),
    500000: (6, "500.000 - 1 Mio."),
    1000000: (7, "1 - 2 Mio."),
    2000000: (8, "2 - 5 Mio."),
    5000000: (9, "5 - 10 Mio."),
    10000000: (10, "10 - 20 Mio."),
    20000000: (11, "20 - 50 Mio."),
    50000000: (12, "50 - 100 Mio."),
    100000000: (13, "100 - 200 Mio."),
}

MULTI_TOKENS = {
    "Multi-player", "MMO", "Online PvP", "Online Co-op", "PvP",
    "Shared/Split Screen PvP", "Shared/Split Screen Co-op",
    "Cross-Platform Multiplayer", "LAN Co-op", "LAN PvP",
    "Remote Play Together",
}
COOP_TOKENS = {"Co-op", "Online Co-op", "Shared/Split Screen Co-op", "LAN Co-op"}
PVP_TOKENS = {"PvP", "Online PvP", "Shared/Split Screen PvP", "LAN PvP"}


def to_number(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def parse_owners(series: pd.Series) -> pd.DataFrame:
    extracted = series.str.extract(r"^\s*(\d+)\s*-\s*(\d+)\s*$")
    low = pd.to_numeric(extracted[0], errors="coerce")
    high = pd.to_numeric(extracted[1], errors="coerce")
    valid = high.fillna(0) > 0
    mid = (low + high) / 2.0

    classes = low.map(lambda x: OWNER_CLASSES.get(int(x), (np.nan, np.nan))[0] if pd.notna(x) else np.nan)
    labels = low.map(lambda x: OWNER_CLASSES.get(int(x), (np.nan, np.nan))[1] if pd.notna(x) else np.nan)

    return pd.DataFrame(
        {
            "owners_low": low.where(valid),
            "owners_high": high.where(valid),
            "owners_mid": mid.where(valid),
            "owners_class": classes.where(valid),
            "owners_label": labels.where(valid),
        }
    )


def build_price_class(price: pd.Series) -> pd.Series:
    result = pd.Series(pd.NA, index=price.index, dtype="object")
    result[price == 0] = "Free"
    paid = price > 0
    result[paid] = pd.cut(
        price[paid],
        bins=[0, 5, 10, 20, 40, np.inf],
        labels=["0,01 - 4,99", "5 - 9,99", "10 - 19,99", "20 - 39,99", "40+"],
        right=False,
    ).astype(str)
    return result


def main() -> None:
    df = load_raw()
    n_raw = len(df)

    numeric_cols = [
        "Peak CCU", "Price", "Discount", "DLC count", "Metacritic score",
        "User score", "Positive", "Negative", "Recommendations", "Achievements",
    ]
    for col in numeric_cols:
        df[col] = to_number(df[col])
    for col in ["Positive", "Negative", "Peak CCU", "Discount", "DLC count"]:
        df[col] = df[col].fillna(0).clip(lower=0)

    df["release_date"] = pd.to_datetime(df["Release date"], format="%b %d, %Y", errors="coerce")
    df["release_year"] = df["release_date"].dt.year

    df = pd.concat([df, parse_owners(df["Estimated owners"])], axis=1)
    df["log_owners"] = np.log10(df["owners_mid"])
    df["peak_ccu"] = df["Peak CCU"]
    df["log_peak_ccu"] = np.log10(1.0 + df["peak_ccu"])

    df["price"] = df["Price"].clip(lower=0)
    df["is_free"] = df["price"] == 0
    df["price_paid"] = df["price"].where(df["price"] > 0)
    df["price_class"] = build_price_class(df["price"])
    df["discount"] = df["Discount"].clip(0, 100)
    df["has_discount"] = df["discount"] > 0
    df["dlc_count"] = df["DLC count"].astype(int)

    for col in ["Windows", "Mac", "Linux"]:
        df[col.lower()] = df[col].map({"True": True, "False": False}).fillna(False).astype(bool)
    df["n_platforms"] = df[["windows", "mac", "linux"]].sum(axis=1)
    df["platform_combo"] = (
        df["windows"].map({True: "W", False: ""})
        + df["mac"].map({True: "+M", False: ""})
        + df["linux"].map({True: "+L", False: ""})
    ).str.lstrip("+")

    df["metacritic"] = df["Metacritic score"].where(df["Metacritic score"] > 0)
    df["has_metacritic"] = df["metacritic"].notna()

    df["review_count"] = (df["Positive"] + df["Negative"]).astype(int)
    df["positive_ratio"] = np.where(
        df["review_count"] > 0, df["Positive"] / df["review_count"], np.nan
    )
    df["wilson"] = [
        wilson_lower_bound(pos, neg)
        for pos, neg in zip(df["Positive"].to_numpy(), df["Negative"].to_numpy())
    ]

    df["genres"] = df["Genres"].apply(split_list)
    df["categories"] = df["Categories"].apply(split_list)
    df["n_genres"] = df["genres"].apply(lambda g: len(set(g) & set(STANDARD_GENRES)))
    df["has_genre"] = df["n_genres"] > 0
    df["free_to_play_genre"] = df["genres"].apply(lambda g: "Free To Play" in g)
    df["early_access"] = df["genres"].apply(lambda g: "Early Access" in g)

    for genre in STANDARD_GENRES:
        col = "genre_" + re.sub(r"[^a-z0-9]+", "_", genre.lower()).strip("_")
        df[col] = df["genres"].apply(lambda g, genre=genre: genre in g)

    df["is_software"] = df.apply(
        lambda row: bool(set(row["genres"]) & SOFTWARE_GENRES)
        and not (set(row["genres"]) & set(STANDARD_GENRES))
        and "Free To Play" not in row["genres"]
        and "Early Access" not in row["genres"],
        axis=1,
    )

    df["single_player"] = df["categories"].apply(lambda c: "Single-player" in c)
    df["multi_player"] = df["categories"].apply(lambda c: bool(set(c) & MULTI_TOKENS))
    df["coop"] = df["categories"].apply(lambda c: bool(set(c) & COOP_TOKENS))
    df["pvp"] = df["categories"].apply(lambda c: bool(set(c) & PVP_TOKENS))
    df["mmo"] = df["categories"].apply(lambda c: "MMO" in c)

    def mode(row: pd.Series) -> str:
        if row["single_player"] and row["multi_player"]:
            return "Single + Multiplayer"
        if row["single_player"]:
            return "Nur Singleplayer"
        if row["multi_player"]:
            return "Nur Multiplayer"
        return "Keine Angabe"

    df["mode"] = df.apply(mode, axis=1)

    df["non_game_name"] = df["Name"].fillna("").str.contains(NON_GAME_PATTERN)
    df["valid_owners"] = df["owners_mid"].notna()
    df["released"] = df["release_year"].between(1997, REFERENCE_YEAR)
    df["age_years"] = (REFERENCE_YEAR - df["release_year"]).clip(lower=0)

    # Duplikate / Editionsvarianten: normalisierter Name + Erscheinungsjahr.
    candidates = df[df["valid_owners"] & df["released"] & ~df["non_game_name"] & ~df["is_software"]].copy()
    candidates["_norm_name"] = candidates["Name"].apply(normalize_name)
    candidates.loc[candidates["_norm_name"].str.len() < 3, "_norm_name"] = ""
    candidates["_dedupe_key"] = np.where(
        candidates["_norm_name"] != "",
        candidates["_norm_name"] + "|" + candidates["release_year"].astype("Int64").astype(str),
        "__appid_" + candidates["AppID"].astype(str),
    )
    candidates = candidates.sort_values(
        ["_dedupe_key", "owners_mid", "review_count", "release_date"],
        ascending=[True, False, False, True],
    )
    keep_ids = set(candidates.drop_duplicates("_dedupe_key")["AppID"])
    df["is_duplicate"] = False
    df.loc[df["AppID"].isin(candidates["AppID"]), "is_duplicate"] = ~df.loc[
        df["AppID"].isin(candidates["AppID"]), "AppID"
    ].isin(keep_ids)

    df["analytic_sample"] = (
        df["valid_owners"] & df["released"] & ~df["non_game_name"]
        & ~df["is_software"] & ~df["is_duplicate"]
    )
    df["rating_sample"] = df["analytic_sample"] & (df["review_count"] >= MIN_REVIEWS)
    df["rating_sample_50"] = df["analytic_sample"] & (df["review_count"] >= MIN_REVIEWS_ROBUST)
    df["meta_sample"] = df["analytic_sample"] & df["has_metacritic"]
    df["peak_sample"] = df["analytic_sample"] & (df["peak_ccu"] > 0)

    # Externe Quellen anbinden (IGDB, Metacritic, Preishistorie).
    df, coverage, overlap = attach_external_sources(df)
    df["igdb_sample"] = df["analytic_sample"] & df["has_igdb"]
    df["mc_external_sample"] = df["analytic_sample"] & df["has_mc_external"]
    df["price_history_sample"] = df["analytic_sample"] & df["has_price_history"]
    df["multi_source_sample"] = (
        df["analytic_sample"] & df["has_igdb"] & df["has_mc_external"]
        & df["has_price_history"]
    )

    analytic = df[df["analytic_sample"]].copy()
    q75 = analytic["owners_low"].quantile(0.75)
    q90 = analytic["owners_low"].quantile(0.90)
    peak_q75 = analytic.loc[analytic["peak_ccu"] > 0, "peak_ccu"].quantile(0.75)

    df["success_top25"] = df["owners_low"] >= q75
    df["success_top10"] = df["owners_low"] >= q90
    df["success_100k"] = df["owners_low"] >= 100000
    df["success_peak_top25"] = df["peak_ccu"] >= peak_q75
    df.loc[~df["peak_sample"], "success_peak_top25"] = False
    analytic = df[df["analytic_sample"]].copy()

    # Sample-Flow dokumentieren: sequenzielle Bereinigung bis zur Analysestichprobe,
    # danach Teilstichproben fuer einzelne Fragestellungen.
    sequential = [
        ("Rohdatensatz", np.ones(n_raw, dtype=bool)),
        ("Besitzerangabe vorhanden (nicht '0 - 0')", df["valid_owners"].to_numpy()),
        ("Veroeffentlichung 1997-2025", (df["valid_owners"] & df["released"]).to_numpy()),
        (
            "ohne Demo/Playtest/Soundtrack/DLC usw.",
            (df["valid_owners"] & df["released"] & ~df["non_game_name"]).to_numpy(),
        ),
        (
            "ohne Software/Utilities",
            (df["valid_owners"] & df["released"] & ~df["non_game_name"] & ~df["is_software"]).to_numpy(),
        ),
        ("Analysestichprobe (ohne Namens-/Editionsduplikate)", df["analytic_sample"].to_numpy()),
    ]
    flow_rows = []
    previous = n_raw
    for label, mask in sequential:
        n = int(mask.sum())
        flow_rows.append(
            {
                "Schritt": label,
                "Basis": "sequenzielle Bereinigung",
                "N": n,
                "Anteil_am_Rohdatensatz": n / n_raw,
                "Entfernt_gegenueber_Vorzeile": previous - n,
            }
        )
        previous = n
    subsets = [
        ("Teilstichprobe: mind. 10 Reviews (F3, Bewertungsanalysen)", df["rating_sample"].to_numpy()),
        ("Teilstichprobe: mind. 50 Reviews (Robustheit F3)", df["rating_sample_50"].to_numpy()),
        ("Teilstichprobe: FronkonGames-Metacritic vorhanden (nur Vergleich)", df["meta_sample"].to_numpy()),
        ("Teilstichprobe: Peak CCU > 0 (ergaenzende Zielgroesse)", df["peak_sample"].to_numpy()),
        ("Teilstichprobe: IGDB-Match vorhanden", df["igdb_sample"].to_numpy()),
        ("Teilstichprobe: externe Metacritic-Daten vorhanden", df["mc_external_sample"].to_numpy()),
        ("Teilstichprobe: Preishistorie vorhanden", df["price_history_sample"].to_numpy()),
        ("Teilstichprobe: alle vier Datenquellen vorhanden", df["multi_source_sample"].to_numpy()),
    ]
    for label, mask in subsets:
        n = int(mask.sum())
        flow_rows.append(
            {
                "Schritt": label,
                "Basis": "Teilmenge der Analysestichprobe",
                "N": n,
                "Anteil_am_Rohdatensatz": n / n_raw,
                "Entfernt_gegenueber_Vorzeile": np.nan,
            }
        )
    flow = pd.DataFrame(flow_rows)
    save_table(flow, "sample_flow")

    quality = pd.DataFrame(
        [
            {"Pruefung": "Zeilen im Rohdatensatz", "Wert": n_raw, "Anteil": 1.0},
            {"Pruefung": "Eindeutige AppIDs", "Wert": int(df["AppID"].nunique()), "Anteil": df["AppID"].nunique() / n_raw},
            {"Pruefung": "Doppelte AppIDs", "Wert": int(df["AppID"].duplicated().sum()), "Anteil": df["AppID"].duplicated().mean()},
            {"Pruefung": "Fehlender Name", "Wert": int(df["Name"].isna().sum()), "Anteil": df["Name"].isna().mean()},
            {"Pruefung": "Fehlendes Release-Datum", "Wert": int(df["release_date"].isna().sum()), "Anteil": df["release_date"].isna().mean()},
            {"Pruefung": "Release-Jahr nach 2025", "Wert": int((df["release_year"] > REFERENCE_YEAR).sum()), "Anteil": (df["release_year"] > REFERENCE_YEAR).mean()},
            {"Pruefung": "Besitzer '0 - 0' (fehlend)", "Wert": int((~df["valid_owners"]).sum()), "Anteil": (~df["valid_owners"]).mean()},
            {"Pruefung": "Preis = 0", "Wert": int((df["price"] == 0).sum()), "Anteil": (df["price"] == 0).mean()},
            {"Pruefung": "Preis fehlend/negativ", "Wert": int((df["price"].isna() | (df["price"] < 0)).sum()), "Anteil": (df["price"].isna() | (df["price"] < 0)).mean()},
            {"Pruefung": "Keine Reviews", "Wert": int((df["review_count"] == 0).sum()), "Anteil": (df["review_count"] == 0).mean()},
            {"Pruefung": "Kein Metacritic-Score", "Wert": int((~df["has_metacritic"]).sum()), "Anteil": (~df["has_metacritic"]).mean()},
            {"Pruefung": "Peak CCU = 0", "Wert": int((df["peak_ccu"] == 0).sum()), "Anteil": (df["peak_ccu"] == 0).mean()},
            {"Pruefung": "Keine Genre-Angabe", "Wert": int((~df["has_genre"]).sum()), "Anteil": (~df["has_genre"]).mean()},
            {"Pruefung": "Keine Kategorie-Angabe", "Wert": int((df["categories"].apply(len) == 0).sum()), "Anteil": (df["categories"].apply(len) == 0).mean()},
            {"Pruefung": "Software/Utilities", "Wert": int(df["is_software"].sum()), "Anteil": df["is_software"].mean()},
            {"Pruefung": "Nicht-Spiel-Name (Demo/Playtest usw.)", "Wert": int(df["non_game_name"].sum()), "Anteil": df["non_game_name"].mean()},
            {"Pruefung": "Namens-/Editionsduplikate (entfernt)", "Wert": int(df["is_duplicate"].sum()), "Anteil": df["is_duplicate"].mean()},
            {"Pruefung": "Analysestichprobe", "Wert": int(df["analytic_sample"].sum()), "Anteil": df["analytic_sample"].mean()},
            {"Pruefung": "IGDB-Match in Analysestichprobe", "Wert": int(df["igdb_sample"].sum()), "Anteil": df["igdb_sample"].mean()},
            {"Pruefung": "Externer Metacritic-Match in Analysestichprobe", "Wert": int(df["mc_external_sample"].sum()), "Anteil": df["mc_external_sample"].mean()},
            {"Pruefung": "Preishistorie-Match in Analysestichprobe", "Wert": int(df["price_history_sample"].sum()), "Anteil": df["price_history_sample"].mean()},
            {"Pruefung": "Mindestens zwei Datenquellen", "Wert": int((df["analytic_sample"] & (df["n_sources"] >= 2)).sum()), "Anteil": (df["analytic_sample"] & (df["n_sources"] >= 2)).mean()},
            {"Pruefung": "Alle vier Datenquellen", "Wert": int(df["multi_source_sample"].sum()), "Anteil": df["multi_source_sample"].mean()},
        ]
    )
    save_table(quality, "data_quality_summary")

    thresholds = pd.DataFrame(
        [
            {"Kriterium": "Top 25 % (oberes Quartil)", "Schwelle_owners_low": q75,
             "Anteil_der_analysestichprobe": float(df.loc[df["analytic_sample"], "success_top25"].mean())},
            {"Kriterium": "Top 10 % (Perzentilschwelle)", "Schwelle_owners_low": q90,
             "Anteil_der_analysestichprobe": float(df.loc[df["analytic_sample"], "success_top10"].mean())},
            {"Kriterium": "Mindestens 100.000 Besitzer", "Schwelle_owners_low": 100000,
             "Anteil_der_analysestichprobe": float(df.loc[df["analytic_sample"], "success_100k"].mean())},
            {"Kriterium": "Peak CCU oberes Quartil", "Schwelle_owners_low": np.nan,
             "Anteil_der_analysestichprobe": float(df.loc[df["peak_sample"], "success_peak_top25"].mean())},
        ]
    )
    save_table(thresholds, "success_thresholds")

    missing = (
        df.loc[df["analytic_sample"]]
        .isna()
        .mean()
        .sort_values(ascending=False)
        .rename("Fehlanteil")
        .reset_index()
        .rename(columns={"index": "Variable"})
    )
    save_table(missing, "missing_values")

    overview = pd.DataFrame(
        [
            ("owners_low", "Untere Grenze der Besitzer-Spanne", "metrisch", "Zielgroesse (Schwellen)"),
            ("owners_mid", "Mittelpunkt der Besitzer-Spanne", "metrisch", "Zielgroesse (log-transformiert)"),
            ("log_owners", "log10(owners_mid)", "metrisch", "Zielgroesse in Korrelationen/Tests"),
            ("peak_ccu", "Peak Concurrent Players", "metrisch", "ergaenzende Zielgroesse"),
            ("release_year", "Veroeffentlichungsjahr", "metrisch", "Kontrollvariable / F1"),
            ("n_platforms", "Anzahl unterstuetzter Plattformen", "metrisch", "F1"),
            ("genre_*", "Genre-Indikatoren", "binaer", "F1"),
            ("mode", "Single-/Multiplayer-Kategorie", "kategorial", "F1"),
            ("price", "Aktueller Preis in USD", "metrisch", "F2"),
            ("is_free", "Free-to-Play-Indikator", "binaer", "F2"),
            ("discount", "Aktueller Rabatt in Prozent", "metrisch", "F2"),
            ("dlc_count", "Anzahl DLCs", "metrisch", "F2"),
            ("wilson", "Wilson-Untergrenze der positiven Bewertungen", "metrisch", "F3"),
            ("positive_ratio", "Anteil positiver Bewertungen", "metrisch", "F3 (Robustheit)"),
            ("metacritic", "Metacritic-Score aus FronkonGames", "metrisch", "Vergleichswert"),
            ("mc_metascore", "Externer Metacritic-Kritikerscore", "metrisch", "F3 (externe Quelle)"),
            ("mc_user_score", "Externer Metacritic-Nutzerscore", "metrisch", "F3 (externe Quelle)"),
            ("igdb_rating", "IGDB-Nutzerbewertung", "metrisch", "externe Validierung"),
            ("igdb_aggregated_rating", "IGDB-Kritikerbewertung", "metrisch", "externe Validierung"),
            ("igdb_multiplayer", "IGDB: Multiplayer-Modus", "binaer", "externe Validierung"),
            ("igdb_singleplayer", "IGDB: Singleplayer-Modus", "binaer", "externe Validierung"),
            ("igdb_n_genres", "Anzahl IGDB-Genres", "metrisch", "externe Validierung"),
            ("ph_max_discount_pct", "Maximaler Preisnachlass in der Preishistorie", "metrisch", "F2 (externe Quelle)"),
            ("ph_share_months_on_sale", "Anteil Monate mit Rabatt", "metrisch", "F2 (externe Quelle)"),
            ("ph_price_changes_total", "Preisaenderungen in der Preishistorie", "metrisch", "F2 (externe Quelle)"),
            ("n_sources", "Anzahl verfuegbarer Datenquellen", "metrisch", "Datenabdeckung"),
            ("review_count", "Anzahl Reviews", "metrisch", "nur Stichprobenrestriktion, kein Praediktor"),
            ("success_top25", "Erfolg: oberes Quartil der Besitzer", "binaer", "Erfolgsziel (deskriptiv)"),
            ("success_top10", "Erfolg: obere 10-%-Schwelle", "binaer", "Robustheitsziel"),
            ("success_100k", "Erfolg: mind. 100.000 Besitzer", "binaer", "Robustheitsziel"),
        ],
        columns=["Variable", "Beschreibung", "Typ", "Rolle"],
    )
    save_table(overview, "variable_overview")

    save_cols = [
        "AppID", "Name", "release_date", "release_year", "age_years",
        "owners_low", "owners_high", "owners_mid", "owners_class", "owners_label",
        "log_owners", "peak_ccu", "log_peak_ccu", "price", "price_paid",
        "price_class", "is_free", "discount", "has_discount", "dlc_count",
        "windows", "mac", "linux", "n_platforms", "platform_combo",
        "metacritic", "has_metacritic", "Positive", "Negative", "review_count",
        "positive_ratio", "wilson", "mode", "single_player", "multi_player",
        "coop", "pvp", "mmo", "n_genres", "has_genre", "free_to_play_genre",
        "early_access", "is_software", "non_game_name", "is_duplicate",
        "valid_owners", "released", "analytic_sample", "rating_sample",
        "rating_sample_50", "meta_sample", "peak_sample", "success_top25",
        "success_top10", "success_100k", "success_peak_top25",
        "igdb_sample", "mc_external_sample", "price_history_sample", "multi_source_sample",
        "has_igdb", "has_mc_external", "has_price_history", "n_sources",
        "Developers", "Publishers", "Genres", "Categories",
    ] + ["genre_" + re.sub(r"[^a-z0-9]+", "_", g.lower()).strip("_") for g in STANDARD_GENRES]
    external_cols = [c for c in df.columns if c.startswith(("igdb_", "mc_", "ph_")) and c not in save_cols]
    save_cols = save_cols + external_cols
    df[save_cols].to_parquet(DATA / "steam_prepared.parquet", index=False)
    analytic[save_cols].to_parquet(DATA / "analytic_sample.parquet", index=False)
    analytic[analytic["rating_sample"]][save_cols].to_parquet(DATA / "rating_sample.parquet", index=False)
    analytic[analytic["rating_sample_50"]][save_cols].to_parquet(DATA / "rating_sample_50.parquet", index=False)
    analytic[analytic["meta_sample"]][save_cols].to_parquet(DATA / "meta_sample.parquet", index=False)
    analytic[analytic["igdb_sample"]][save_cols].to_parquet(DATA / "igdb_sample.parquet", index=False)
    analytic[analytic["mc_external_sample"]][save_cols].to_parquet(DATA / "mc_external_sample.parquet", index=False)
    analytic[analytic["price_history_sample"]][save_cols].to_parquet(DATA / "price_history_sample.parquet", index=False)
    analytic[analytic["multi_source_sample"]][save_cols].to_parquet(DATA / "multi_source_sample.parquet", index=False)

    print(f"Rohdatensatz: {n_raw:,}")
    print(f"Analysestichprobe: {int(df['analytic_sample'].sum()):,}")
    print(f"Bewertungsstichprobe (>=10 Reviews): {int(df['rating_sample'].sum()):,}")
    print(f"Metacritic-Teilstichprobe (FronkonGames): {int(df['meta_sample'].sum()):,}")
    print(f"IGDB-Teilstichprobe: {int(df['igdb_sample'].sum()):,}")
    print(f"Externe Metacritic-Teilstichprobe: {int(df['mc_external_sample'].sum()):,}")
    print(f"Preishistorie-Teilstichprobe: {int(df['price_history_sample'].sum()):,}")
    print(f"Alle vier Datenquellen: {int(df['multi_source_sample'].sum()):,}")
    print(f"Mindestens zwei Datenquellen: {int((df['analytic_sample'] & (df['n_sources'] >= 2)).sum()):,}")
    print("\nQuellenabdeckung:")
    print(coverage[["Quelle", "N_gematcht_analytisch", "Match_Rate_analytisch"]].to_string(index=False))
    print(f"Erfolgsschwelle Top 25 % (owners_low): {q75:,.0f}")
    print(f"Erfolgsschwelle Top 10 % (owners_low): {q90:,.0f}")
    print(flow.to_string(index=False))


if __name__ == "__main__":
    main()

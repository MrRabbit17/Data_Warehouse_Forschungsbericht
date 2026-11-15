"""Datenaufbereitung, Bereinigung und Feature Engineering.

Input : FronkonGames Steam Games Dataset (games.csv, Version 58, hashgeprueft)
        sowie die externen Quellen aus external_data.py
Output: data/steam_prepared.parquet, data/analytic_sample.parquet und Teilstichproben,
        outputs/tables/sample_flow.csv, data_quality_summary.csv,
        success_thresholds.csv, missing_values.csv, variable_overview.csv,
        steamspy_coverage_by_month.csv, exclusion_samples.csv, data_provenance.csv
"""

from __future__ import annotations

import re
import sys
import unicodedata

import numpy as np
import pandas as pd

from common import (
    CLUSTER_FROM_YEAR,
    CLUSTER_PRICE,
    DATA,
    MIN_REVIEWS,
    MIN_REVIEWS_ROBUST,
    RANDOM_STATE,
    SOFTWARE_GENRES,
    STANDARD_GENRES,
    STEAMSPY_CUTOFF,
    load_raw,
    non_game_name,
    normalize_name,
    save_table,
    split_list,
    wilson_lower_bound,
    write_provenance,
)
from external_data import attach_external_sources

# Preis im Label-Text mit Dezimalkomma wie im uebrigen Bericht.
CLUSTER_PRICE_DE = f"{CLUSTER_PRICE:.2f}".replace(".", ",")

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

# Steam-Kategorien, die mehrere Spieler voraussetzen. "Remote Play Together"
# zaehlt mit, weil die Funktion einen lokalen Mehrspielermodus voraussetzt.
MULTI_TOKENS = {
    "Multi-player", "MMO", "Online PvP", "Online Co-op", "PvP", "Co-op",
    "Shared/Split Screen", "Shared/Split Screen PvP", "Shared/Split Screen Co-op",
    "Cross-Platform Multiplayer", "LAN Co-op", "LAN PvP",
    "Remote Play Together",
}
COOP_TOKENS = {"Co-op", "Online Co-op", "Shared/Split Screen Co-op", "LAN Co-op"}
PVP_TOKENS = {"PvP", "Online PvP", "Shared/Split Screen PvP", "LAN PvP"}
F2P_GENRE = "Free To Play"

# Erfolgsschwellen auf der unteren Grenze der SteamSpy-Besitzerklasse. Wegen der
# groben Klassen faellt das 75-%-Quantil auf die Klassengrenze 20.000 (geprueft
# in main); der tatsaechliche Anteil wird in success_thresholds.csv berichtet.
SUCCESS_TOP25_OWNERS = 20_000
SUCCESS_50K_OWNERS = 50_000
SUCCESS_100K_OWNERS = 100_000
PRICE_OUTLIER_USD = 100.0
# Schaetzung des Markteintritts (Early-Access-Absolventen): Steam-AppIDs werden
# fortlaufend vergeben. Liegt die AppID eines Spiels unter dem Median der AppIDs
# des Jahrgangs, der APPID_PROXY_YEARS Jahre vor seinem Store-Jahr erschien, war
# die Store-Seite sehr wahrscheinlich Jahre vor dem Store-Datum angelegt. Als
# Referenz dienen nur Jahrgaenge der Analysestichprobe mit genuegend Spielen.
APPID_PROXY_YEARS = 3
APPID_REF_MIN_N = 200
# Schluesselwoerter fuer die Gegenprobe des Namensfilters (exclusion_samples.csv):
# behaltene Titel mit diesen Woertern werden stichprobenartig ausgegeben.
KEEP_CHECK_PATTERN = re.compile(
    r"\b(?:soundtrack|ost|dlc|demo|expansion|server|playtest|beta|test|trial|edition|benchmark|"
    r"artbook|bundle|pack|tool|editor|wallpaper)\b",
    re.IGNORECASE,
)
PRICE_CLASS_LABELS = ["0,01 - 4,99", "5 - 9,99", "10 - 19,99", "20 - 39,99", "40+"]

# Platzhalter in Entwickler-/Publisherfeldern (werden seit keep_default_na=False
# als Text eingelesen und fuer den Duplikatschluessel als fehlend behandelt).
COMPANY_PLACEHOLDERS = {"", "none", "n/a", "na", "null", "-", "unknown", "tbd", "tba"}
LEGAL_FORMS = {
    "ltd", "limited", "inc", "incorporated", "llc", "gmbh", "co", "corp",
    "corporation", "sa", "srl", "oy", "ab", "kk", "plc", "pty", "bv", "sas", "ug", "kg", "ag",
}


def to_number(series: pd.Series, name: str) -> pd.Series:
    """Text in Zahlen umwandeln; nicht leere, aber unlesbare Werte sind ein Fehler."""
    numbers = pd.to_numeric(series, errors="coerce")
    failed = series.notna() & numbers.isna()
    if failed.any():
        examples = series[failed].unique()[:5].tolist()
        raise ValueError(f"Spalte {name}: {int(failed.sum())} nicht numerische Werte, z. B. {examples}")
    return numbers


def parse_owners(series: pd.Series) -> pd.DataFrame:
    extracted = series.str.extract(r"^\s*(\d+)\s*-\s*(\d+)\s*$")
    low = pd.to_numeric(extracted[0], errors="coerce")
    high = pd.to_numeric(extracted[1], errors="coerce")
    classes = low.map(lambda x: OWNER_CLASSES.get(int(x), (np.nan, np.nan))[0] if pd.notna(x) else np.nan)
    labels = low.map(lambda x: OWNER_CLASSES.get(int(x), (np.nan, np.nan))[1] if pd.notna(x) else np.nan)
    return pd.DataFrame(
        {
            "owners_low": low,
            "owners_high": high,
            "owners_mid": (low + high) / 2.0,
            "owners_class": classes,
            "owners_label": labels,
        }
    )


def _distance_to_price_points(values: np.ndarray, cents: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Abstand zum naechsten positiven Preispunkt mit den angegebenen Cent-Endungen."""
    floor = np.floor(values)
    candidates = np.stack([base + c / 100.0 for base in (floor - 1.0, floor, floor + 1.0) for c in cents], axis=1)
    # Ein Listenpreis von 0 waere "kostenlos"; ein rabattierter Tagespreis > 0 darf
    # daher nie auf 0 gesetzt werden.
    candidates = np.where(candidates > 0, candidates, np.nan)
    distance = np.abs(candidates - values[:, None])
    distance = np.where(np.isnan(distance), np.inf, distance)
    best_idx = np.argmin(distance, axis=1)
    rows = np.arange(len(values))
    return candidates[rows, best_idx], distance[rows, best_idx]


def reconstruct_list_price(price: pd.Series, discount: pd.Series) -> tuple[pd.Series, pd.Series, dict]:
    """Listenpreis aus Tagespreis und Rabatt: price / (1 - discount/100).

    Der Tagespreis ist auf Cent gerundet; der Rundungsfehler des rekonstruierten
    Werts betraegt daher bis zu 0,01 / (1 - discount/100). Liegt der Wert
    innerhalb dieser Toleranz (mindestens 0,02) neben einem ueblichen Preispunkt
    (x,99 oder ganzer Dollar, immer > 0), wird er auf diesen gesetzt. Ohne Rabatt
    ist der Tagespreis bereits der Listenpreis. Nicht bestimmbar (NaN) ist der
    Listenpreis bei 100 % Rabatt und bei Tagespreis 0 trotz Rabatt zwischen 1 und
    99 %: Die Formel ergaebe dort 0, obwohl ein rabattiertes Spiel nicht kostenlos
    ist (z. B. Call of Duty mit -45 %, Tagespreis 0 in der Quelle).

    Widerspruechliche Quellzeilen: Liegt der rekonstruierte Wert auch mit Toleranz
    neben keinem Preispunkt (x,99, x,49, x,50, ganzer Dollar), der Tagespreis aber
    exakt auf x,99 bzw. einem ganzen Dollar, passt die Rabattangabe nicht zum
    Tagespreis (z. B. 39,99 USD bei 85 % ergaebe 266,60 USD). Der Tagespreis ist
    dann bereits der Listenpreis (gleich, ob Rabatt oder Preis veraltet ist) und
    wird uebernommen; diese Zeilen werden markiert und gezaehlt. Weil der Rabatt in
    der Quelle teils fehlt, ist der Listenpreis sonst eine Untergrenze.
    """
    zero_with_discount = (price == 0) & (discount > 0) & (discount < 100)
    defined = (discount < 100) & ~zero_with_discount
    raw = (price / (1.0 - discount / 100.0)).where(defined).round(2)
    tolerance = np.maximum(0.02, 0.01 / (1.0 - discount.where(defined) / 100.0)).to_numpy()

    values = raw.to_numpy(dtype=float)
    discounted = np.isfinite(values) & (discount.to_numpy() > 0) & (values > 0)
    best, dist = _distance_to_price_points(np.where(discounted, values, 1.0), (99, 0))
    snap = discounted & (dist <= tolerance + 1e-9) & (np.round(best, 2) != values)
    on_point = discounted & (dist <= tolerance + 1e-9)
    _, dist_wide = _distance_to_price_points(np.where(discounted, values, 1.0), (99, 49, 50, 0))
    price_cents = np.round(price.to_numpy(dtype=float) * 100).astype("int64") % 100
    conflict = discounted & (dist_wide > tolerance + 1e-9) & np.isin(price_cents, (99, 0))
    result = raw.copy()
    result[snap] = np.round(best[snap], 2)
    result[conflict] = price[conflict]
    conflict_series = pd.Series(conflict, index=price.index)
    stats = {
        "rabattiert": int(((discount > 0) & defined).sum()),
        "auf_preispunkt_gesetzt": int(snap.sum()),
        "nicht_auf_preispunkt": int((discounted & ~on_point).sum()),
        "widerspruch_tagespreis_uebernommen": int(conflict.sum()),
        "nicht_bestimmbar_rabatt_100": int((discount >= 100).sum()),
        "nicht_bestimmbar_tagespreis_0_mit_rabatt": int(zero_with_discount.sum()),
    }
    return result, conflict_series, stats


def build_price_class(price: pd.Series) -> pd.Series:
    result = pd.Series(pd.NA, index=price.index, dtype="object")
    result[price == 0] = "Free"
    paid = price > 0
    result[paid] = pd.cut(
        price[paid],
        bins=[0, 5, 10, 20, 40, np.inf],
        labels=PRICE_CLASS_LABELS,
        right=False,
    ).astype(str)
    return result


def normalize_company(value: object) -> str:
    """Schluessel fuer den ersten Entwickler/Publisher (ohne Rechtsform, Platzhalter leer)."""
    if not isinstance(value, str):
        return ""
    first = value.split(",")[0]
    text = unicodedata.normalize("NFKC", first).casefold().strip()
    if text in COMPANY_PLACEHOLDERS:
        return ""
    tokens = re.findall(r"[^\W_]+", text)
    while len(tokens) > 1 and tokens[-1] in LEGAL_FORMS:
        tokens.pop()
    return "".join(tokens)


def english_flag(value: object) -> bool:
    return isinstance(value, str) and re.search(r"\benglish\b", value, re.IGNORECASE) is not None


def steamspy_coverage(df: pd.DataFrame) -> pd.DataFrame:
    """SteamSpy-Abdeckung je Release-Monat (Rohdaten, ab 2024) zur Pruefung des Stichtags.

    Reviews, Besitzer und Peak CCU stammen aus SteamSpy, Recommendations aus dem
    Store. Bricht nur die SteamSpy-Abdeckung am Stichtag ein, waehrend die
    Store-Abdeckung gleich bleibt, ist der SteamSpy-Stand bestaetigt.
    """
    d = df[df["release_date"] >= pd.Timestamp("2024-01-01")].copy()
    d["Release_Monat"] = d["release_date"].dt.to_period("M").astype(str)
    reviews = d["Positive"] + d["Negative"]
    table = (
        d.assign(
            _rev=reviews > 0,
            _peak=d["Peak CCU"] > 0,
            _own20=d["owners_low_raw"] >= 20000,
            _zero=d["owners_zero"],
            _rec=d["Recommendations"] > 0,
        )
        .groupby("Release_Monat")
        .agg(
            N_Spiele=("AppID", "size"),
            Anteil_Reviews_gt0=("_rev", "mean"),
            Anteil_PeakCCU_gt0=("_peak", "mean"),
            Anteil_Besitzer_ab_20000=("_own20", "mean"),
            Anteil_Besitzer_0_0=("_zero", "mean"),
            Anteil_Recommendations_gt0=("_rec", "mean"),
        )
        .reset_index()
    )
    table["Vor_SteamSpy_Stichtag"] = pd.PeriodIndex(table["Release_Monat"], freq="M").end_time < STEAMSPY_CUTOFF
    table.insert(1, "Basis", "Rohdatensatz (alle Einträge mit Release-Datum ab 2024)")
    return table


def check_steamspy_cutoff(coverage: pd.DataFrame, df: pd.DataFrame) -> str:
    """Den Bruch der SteamSpy-Abdeckung am Stichtag pruefen und beschreiben."""
    before = (STEAMSPY_CUTOFF - pd.DateOffset(months=1)).strftime("%Y-%m")
    after = STEAMSPY_CUTOFF.strftime("%Y-%m")
    rows = coverage.set_index("Release_Monat")
    if before not in rows.index or after not in rows.index:
        raise RuntimeError("SteamSpy-Stichtag nicht pruefbar: Monate um den Stichtag fehlen.")
    share_before = rows.loc[before, "Anteil_Reviews_gt0"]
    share_after = rows.loc[after, "Anteil_Reviews_gt0"]
    rec_before = rows.loc[before, "Anteil_Recommendations_gt0"]
    rec_after = rows.loc[after, "Anteil_Recommendations_gt0"]
    if not (share_before >= 3 * share_after and share_after <= 0.15):
        raise RuntimeError(
            f"Kein Bruch der SteamSpy-Abdeckung am Stichtag {STEAMSPY_CUTOFF.date()}: "
            f"Reviews > 0 {share_before:.1%} ({before}) vs. {share_after:.1%} ({after})."
        )
    # Naeherung fuer den Abruftag: erster Release-Tag einer Serie von fuenf Tagen
    # (je mind. 20 Neuerscheinungen), an denen kein Spiel einen Peak-CCU-Wert hat.
    # Spaeter vereinzelt auftretende Werte stammen von Early-Access-Absolventen.
    window = df[(df["release_date"] >= STEAMSPY_CUTOFF - pd.Timedelta(days=30))
                & (df["release_date"] < STEAMSPY_CUTOFF + pd.Timedelta(days=60))]
    daily = window.groupby("release_date")["Peak CCU"].agg(n="size", share=lambda s: (s > 0).mean())
    daily = daily[daily["n"] >= 20]
    shares = daily["share"].to_numpy()
    first_zero = next(
        (daily.index[i] for i in range(len(shares) - 4) if (shares[i:i + 5] < 0.01).all()), None
    )
    zero_text = first_zero.strftime("%d.%m.%Y") if first_zero is not None else "nicht bestimmbar"
    return (
        f"SteamSpy-Stichtag bestaetigt: Reviews > 0 bei {share_before:.1%} der Releases {before}, "
        f"aber nur {share_after:.1%} im {after}; das Store-Feld Recommendations > 0 bleibt stabil "
        f"({rec_before:.1%} vs. {rec_after:.1%}). Peak CCU > 0 endet bei Releases ab {zero_text} "
        "(SteamSpy-Abruf etwa an diesem Tag)."
    )


def mark_duplicates(df: pd.DataFrame, candidates: pd.Series) -> tuple[pd.Series, pd.Series, dict]:
    """Duplikate ueber normalisierten Namen + Release-Jahr + ersten Entwickler finden.

    Fehlt der Entwickler, wird der erste Publisher verwendet; fehlen beide oder
    ist der normalisierte Name leer, wird nicht zusammengelegt. Je Gruppe bleibt
    der Eintrag mit gueltiger und hoechster Besitzerangabe, dann den meisten
    Reviews und dem fruehesten Datum (danach kleinste AppID).
    """
    cand = df.loc[candidates, [
        "AppID", "Name", "release_year", "release_date", "Developers", "Publishers",
        "valid_owners", "owners_mid", "review_count",
    ]].copy()
    # Vollstaendiger Schluessel in allen Schriften: Die Reduktion gemischtschriftlicher
    # Namen auf den lateinischen Teil (nur fuer das Matching gedacht) wuerde
    # verschiedene Titel wie "<chinesischer Titel>Online" zu "online" zusammenlegen.
    cand["_norm_name"] = cand["Name"].apply(lambda v: normalize_name(v, latin_part=False))
    company = cand["Developers"].apply(normalize_company)
    publisher = cand["Publishers"].apply(normalize_company)
    cand["_company"] = company.where(company != "", publisher)
    usable = (cand["_norm_name"] != "") & (cand["_company"] != "") & cand["release_year"].notna()
    cand["_dedupe_key"] = np.where(
        usable,
        cand["_norm_name"] + "|" + cand["release_year"].astype("Int64").astype(str) + "|" + cand["_company"],
        "__appid_" + cand["AppID"].astype(str),
    )
    cand["_appid_num"] = pd.to_numeric(cand["AppID"], errors="coerce")
    cand = cand.sort_values(
        ["_dedupe_key", "valid_owners", "owners_mid", "review_count", "release_date", "_appid_num"],
        ascending=[True, False, False, False, True, True],
        na_position="last",
        kind="mergesort",
    )
    kept = cand.drop_duplicates("_dedupe_key")
    kept_by_key = kept.set_index("_dedupe_key")["AppID"]
    cand["_kept_appid"] = cand["_dedupe_key"].map(kept_by_key)
    dup = cand[cand["AppID"] != cand["_kept_appid"]]

    is_duplicate = pd.Series(False, index=df.index)
    is_duplicate.loc[dup.index] = True
    duplicate_of = pd.Series(pd.NA, index=df.index, dtype="string")
    duplicate_of.loc[dup.index] = dup["_kept_appid"].astype("string")

    kept_names = df.set_index("AppID")["Name"]
    same_name = (
        dup["Name"].fillna("").str.casefold()
        == dup["_kept_appid"].map(kept_names).fillna("").str.casefold()
    )
    stats = {
        "kandidaten": int(len(cand)),
        "ohne_schluessel": int((~usable).sum()),
        "gruppen_mit_duplikaten": int(dup["_dedupe_key"].nunique()),
        "entfernt": int(len(dup)),
        "entfernt_gleicher_name": int(same_name.sum()),
        "entfernt_editionsvariante": int((~same_name).sum()),
    }
    return is_duplicate, duplicate_of, stats


def prepare_steam(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Merkmale bilden und Stichprobenfilter setzen (alles vor dem Anbinden externer Quellen)."""
    info: dict = {}
    numeric_cols = [
        "Peak CCU", "Price", "Discount", "DLC count", "Metacritic score",
        "User score", "Positive", "Negative", "Recommendations", "Achievements",
    ]
    for col in numeric_cols:
        df[col] = to_number(df[col], col)
    for col in ["Positive", "Negative", "Peak CCU", "Price", "Discount", "DLC count", "Recommendations"]:
        n_missing = int(df[col].isna().sum())
        n_negative = int((df[col] < 0).sum())
        if n_missing or n_negative:
            raise ValueError(f"Spalte {col}: {n_missing} fehlende und {n_negative} negative Werte")
    if (df["Discount"] > 100).any():
        raise ValueError("Spalte Discount: Werte ueber 100 %")

    df["release_date"] = pd.to_datetime(df["Release date"], format="%b %d, %Y", errors="coerce")
    df["release_year"] = df["release_date"].dt.year

    # SteamSpy-Felder: "0 - 0" ist keine Besitzerangabe; eine SteamSpy-Luecke liegt
    # vor, wenn Steam selbst Reviews meldet (Recommendations), SteamSpy aber keine.
    owners = parse_owners(df["Estimated owners"])
    if owners["owners_low"].isna().any():
        raise ValueError(f"{int(owners['owners_low'].isna().sum())} unlesbare Besitzerangaben")
    df["owners_low_raw"] = owners["owners_low"]
    df["owners_zero"] = owners["owners_high"] == 0
    raw_reviews = df["Positive"] + df["Negative"]
    df["steamspy_missing"] = (raw_reviews == 0) & (df["Recommendations"] > 0)
    df["steamspy_signal"] = (raw_reviews > 0) | (df["Peak CCU"] > 0) | (owners["owners_low"] > 0)
    df["valid_owners"] = ~df["owners_zero"] & ~df["steamspy_missing"]
    for col in owners.columns:
        df[col] = owners[col].where(df["valid_owners"])
    df["log_owners"] = np.log10(df["owners_mid"])

    ss_missing = df["steamspy_missing"]
    df["Positive"] = df["Positive"].where(~ss_missing)
    df["Negative"] = df["Negative"].where(~ss_missing)
    df["peak_ccu"] = df["Peak CCU"].where(~ss_missing)
    df["log_peak_ccu"] = np.log10(1.0 + df["peak_ccu"])
    df["review_count"] = df["Positive"] + df["Negative"]
    df["positive_ratio"] = (df["Positive"] / df["review_count"]).where(df["review_count"] > 0)
    df["wilson"] = [
        wilson_lower_bound(pos, neg)
        for pos, neg in zip(df["Positive"].to_numpy(), df["Negative"].to_numpy())
    ]

    # Preise: Price ist der Tagespreis zum Store-Snapshot (ggf. rabattiert).
    df["price_current"] = df["Price"]
    df["discount"] = df["Discount"]
    df["has_discount"] = df["discount"] > 0
    df["price_zero_with_discount"] = (df["price_current"] == 0) & df["has_discount"] & (df["discount"] < 100)
    df["list_price"], df["list_price_conflict"], info["listenpreis"] = reconstruct_list_price(
        df["price_current"], df["discount"])
    # Kostenlos nur bei bestimmbarem Listenpreis 0 (NaN == 0 ist False).
    df["is_free"] = df["list_price"] == 0
    df["price_class"] = build_price_class(df["list_price"])
    df["price_class_current"] = build_price_class(df["price_current"])
    df["price_outlier"] = df["list_price"] > PRICE_OUTLIER_USD
    df["price_cluster"] = (~df["price_outlier"] & df["list_price"].round(2).eq(CLUSTER_PRICE)
                           & (df["release_year"] >= CLUSTER_FROM_YEAR))
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

    df["genres"] = df["Genres"].apply(split_list)
    df["categories"] = df["Categories"].apply(split_list)
    df["n_genres"] = df["genres"].apply(lambda g: len(set(g) & set(STANDARD_GENRES)))
    df["has_genre"] = df["n_genres"] > 0
    df["genres_empty"] = df["genres"].apply(len) == 0
    df["f2p_tag"] = df["genres"].apply(lambda g: F2P_GENRE in g)
    df["early_access"] = df["genres"].apply(lambda g: "Early Access" in g)
    for genre in STANDARD_GENRES:
        col = "genre_" + re.sub(r"[^a-z0-9]+", "_", genre.lower()).strip("_")
        df[col] = df["genres"].apply(lambda g, genre=genre: genre in g)

    # Reine Software: Software-Genre ohne eines der Steam-Standardgenres
    # (unabhaengig von Free-To-Play- oder Early-Access-Markierung).
    df["is_software"] = df["genres"].apply(
        lambda g: bool(set(g) & SOFTWARE_GENRES) and not (set(g) & set(STANDARD_GENRES))
    )

    df["single_player"] = df["categories"].apply(lambda c: "Single-player" in c)
    df["multi_player"] = df["categories"].apply(lambda c: bool(set(c) & MULTI_TOKENS))
    df["coop"] = df["categories"].apply(lambda c: bool(set(c) & COOP_TOKENS))
    df["pvp"] = df["categories"].apply(lambda c: bool(set(c) & PVP_TOKENS))
    df["mmo"] = df["categories"].apply(lambda c: "MMO" in c)
    df["mode"] = np.select(
        [df["single_player"] & df["multi_player"], df["single_player"], df["multi_player"]],
        ["Single + Multiplayer", "Nur Singleplayer", "Nur Multiplayer"],
        default="Keine Angabe",
    )

    languages = df["Supported languages"].fillna("")
    df["languages_listed"] = ~languages.str.strip().isin(["", "[]"])
    df["english_supported"] = languages.apply(english_flag)

    df["non_game_name"] = non_game_name(df["Name"])
    df["released"] = (df["release_year"] >= 1997) & (df["release_date"] < STEAMSPY_CUTOFF)
    df["released_after_cutoff"] = df["release_date"] >= STEAMSPY_CUTOFF
    # Nach dem Stichtag datiert, aber mit SteamSpy-Daten: fast nur Spiele, deren
    # Store-Datum spaeter auf die Vollversion gesetzt wurde (Early-Access-Absolventen).
    df["dated_after_cutoff_with_steamspy"] = df["released_after_cutoff"] & df["steamspy_signal"]
    df["age_years"] = ((STEAMSPY_CUTOFF - df["release_date"]).dt.days / 365.25).round(3)

    dedupe_candidates = (
        df["released"] & ~df["steamspy_missing"] & ~df["non_game_name"] & ~df["is_software"]
        & (df["valid_owners"] | df["owners_zero"])
    )
    df["is_duplicate"], df["duplicate_of"], info["duplikate"] = mark_duplicates(df, dedupe_candidates)

    base = df["released"] & ~df["non_game_name"] & ~df["is_software"] & ~df["is_duplicate"]
    df["analytic_sample"] = base & df["valid_owners"]
    # Sensitivitaet: "0 - 0" als unterste Klasse bzw. Nicht-Erfolg einbeziehen.
    df["sample_with_owners_zero"] = base & ~df["steamspy_missing"] & (df["valid_owners"] | df["owners_zero"])
    df["rating_sample"] = df["analytic_sample"] & (df["review_count"] >= MIN_REVIEWS)
    df["rating_sample_50"] = df["analytic_sample"] & (df["review_count"] >= MIN_REVIEWS_ROBUST)
    df["meta_sample"] = df["analytic_sample"] & df["has_metacritic"]
    df["peak_sample"] = df["analytic_sample"] & (df["peak_ccu"] > 0)
    return df, info


def add_release_alternatives(df: pd.DataFrame) -> None:
    """Alternatives Release-Jahr aus der IGDB-Erstveroeffentlichung (Sensitivitaet).

    Das Steam-Datum ist das aktuelle Store-Datum, bei Early-Access-Absolventen also
    das Datum der Vollversion. release_year_alt nimmt das fruehere von Steam- und
    IGDB-Jahr; release_shifted markiert Spiele, deren IGDB-Erstveroeffentlichung
    mehr als ein Jahr vor dem Steam-Jahr liegt. Ein solcher Abstand ist nur ueber
    den Namens-Lookup in external_data.py moeglich (das Matching erlaubt hoechstens
    ein Jahr). Betroffen sind ueberwiegend Neuauflagen und Portierungen aelterer
    Spiele, seltener fruehere Veroeffentlichungen ausserhalb von Steam;
    release_year_alt ist fuer Portierungen daher eine obere Schranke des Alters auf
    Steam. Early-Access-Absolventen erfasst release_shifted nur teilweise: IGDB
    datiert sie teils auf den Early-Access-Start, teils auf die Vollversion, und
    erfasst werden nur Phasen ab zwei Jahren bei Spielen ohne IGDB-Treffer (dafuer
    zusaetzlich add_market_entry_proxy).
    """
    if "igdb_first_release_year" not in df.columns:
        raise RuntimeError("Spalte igdb_first_release_year fehlt; external_data.py liefert sie nicht.")
    igdb_year = pd.to_numeric(df["igdb_first_release_year"], errors="coerce")
    df["release_year_alt"] = np.fmin(df["release_year"], igdb_year)
    df["release_shifted"] = (igdb_year < df["release_year"] - 1).fillna(False).astype(bool)
    if igdb_year.notna().any() and not df["release_shifted"].any():
        print("WARNUNG: release_shifted ist ueberall False; der Namens-Lookup fuer "
              "igdb_first_release_year liefert keine frueheren Jahre.")


def add_market_entry_proxy(df: pd.DataFrame) -> dict:
    """Geschaetzter Markteintritt fuer Spiele, deren Store-Datum spaeter liegt (Sensitivitaet).

    Bei Early-Access-Absolventen ist das Store-Datum das Datum der Vollversion. Das
    IGDB-Erstveroeffentlichungsjahr entspricht bei vielen Proxy-Faellen ebenfalls
    diesem Jahr, und release_shifted greift nur ohne IGDB-Treffer und ab zwei Jahren
    Abstand; es erfasst diese Faelle daher nur teilweise. Zwei Hinweise auf einen frueheren
    Markteintritt (nur fuer Spiele ohne aktuelle Early-Access-Markierung, deren
    Store-Datum bereits der Eintritt ist):

    - AppID: AppID unter dem Median des Jahrgangs APPID_PROXY_YEARS Jahre vor dem
      Store-Jahr. Geschaetztes Eintrittsjahr = erster Referenzjahrgang, dessen
      Median-AppID ueber der AppID liegt (fruehestens der erste Referenzjahrgang).
      Der Proxy mischt Early-Access-Absolventen mit lange vorab angekuendigten
      Titeln; fuer Letztere ueberschaetzt er das Alter auf dem Markt.
    - Preishistorie: erster Preismonat mehr als einen Monat vor dem Monat des
      Store-Datums; Eintrittsjahr = Jahr des ersten Preismonats (wegen des Fensters
      ab Nov. 2023 eine obere Schranke).

    market_entry_year = Minimum aus Store-Jahr und den geschaetzten Jahren.
    """
    appid = pd.to_numeric(df["AppID"], errors="coerce")
    analytic = df["analytic_sample"]
    ref = (pd.DataFrame({"year": df.loc[analytic, "release_year"], "appid": appid[analytic]})
           .groupby("year")["appid"].agg(["size", "median"]))
    ref = ref[ref["size"] >= APPID_REF_MIN_N]["median"]
    if not ref.is_monotonic_increasing:
        raise RuntimeError("Median-AppID der Referenzjahrgaenge steigt nicht monoton; AppID-Proxy ungeeignet.")
    years, medians = ref.index.to_numpy(), ref.to_numpy(dtype=float)
    candidate = ~df["early_access"] & appid.notna()

    threshold = (df["release_year"] - APPID_PROXY_YEARS).map(ref)
    proxy_appid = candidate & threshold.notna() & (appid < threshold)
    idx = np.searchsorted(medians, appid.fillna(np.inf).to_numpy(), side="right")
    appid_year = pd.Series(np.where(idx < len(years), years[np.minimum(idx, len(years) - 1)], np.nan),
                           index=df.index)

    first_month = pd.to_datetime(df["ph_first_month"], errors="coerce")
    release = df["release_date"]
    months_before = ((release.dt.year - first_month.dt.year) * 12 + release.dt.month - first_month.dt.month)
    proxy_ph = candidate & df["has_price_history"] & (months_before > 1).fillna(False)

    entry = df["release_year"].astype(float)
    entry = entry.where(~proxy_appid, np.fmin(entry, appid_year))
    entry = entry.where(~proxy_ph, np.fmin(entry, first_month.dt.year.astype(float)))
    df["entry_proxy_appid"] = proxy_appid.astype(bool)
    df["entry_proxy_ph"] = proxy_ph.astype(bool)
    df["ea_graduate_proxy"] = (proxy_appid | proxy_ph).astype(bool)
    df["market_entry_year"] = entry
    return {
        "referenzjahre": f"{int(years.min())}-{int(years.max())}",
        "appid": int((proxy_appid & analytic).sum()),
        "preishistorie": int((proxy_ph & analytic).sum()),
        "beide": int((proxy_appid & proxy_ph & analytic).sum()),
        "gesamt": int((df["ea_graduate_proxy"] & analytic).sum()),
    }


def flow_table(df: pd.DataFrame, n_raw: int) -> pd.DataFrame:
    cutoff = STEAMSPY_CUTOFF.strftime("%d.%m.%Y")
    m_owners = ~df["owners_zero"]
    m_released = m_owners & df["released"]
    m_steamspy = m_released & ~df["steamspy_missing"]
    m_name = m_steamspy & ~df["non_game_name"]
    m_software = m_name & ~df["is_software"]
    m_final = df["analytic_sample"]
    sequential = [
        ("raw", "Rohdatensatz", pd.Series(True, index=df.index)),
        ("owners_present", "Besitzerangabe vorhanden (nicht '0 - 0')", m_owners),
        ("released_before_cutoff", f"Veröffentlichung 1997 bis vor dem SteamSpy-Stichtag ({cutoff})", m_released),
        ("steamspy_present", "SteamSpy-Daten vorhanden (ohne SteamSpy-Lücke: 0 Reviews trotz Steam-Recommendations)", m_steamspy),
        ("no_non_game_name", "ohne Demo/Playtest/Soundtrack/DLC/Testserver usw.", m_name),
        ("no_software", "ohne Software (Software-Genre ohne Steam-Standardgenre)", m_software),
        ("analytic_sample", "Analysestichprobe (ohne Duplikate: Name + Jahr + Entwickler)", m_final),
    ]
    rows = []
    previous = n_raw
    for key, label, mask in sequential:
        n = int(mask.sum())
        rows.append({"Kennung": key, "Schritt": label, "Basis": "sequenzielle Bereinigung", "N": n,
                     "Anteil_am_Rohdatensatz": n / n_raw, "Entfernt_gegenueber_Vorzeile": previous - n})
        previous = n
    after_cutoff = m_owners & df["released_after_cutoff"]
    info_rows = [
        ("removed_after_cutoff", f"darunter entfernt: veröffentlicht ab dem SteamSpy-Stichtag ({cutoff})", after_cutoff),
        ("removed_after_cutoff_with_steamspy",
         "davon trotzdem mit SteamSpy-Daten (v. a. Early-Access-Absolventen mit späterem Store-Datum)",
         after_cutoff & df["steamspy_signal"]),
        ("removed_before_1997_or_no_date", "darunter entfernt: vor 1997 oder ohne Release-Datum",
         m_owners & ~df["released"] & ~df["released_after_cutoff"]),
    ]
    for key, label, mask in info_rows:
        n = int(mask.sum())
        rows.append({"Kennung": key, "Schritt": label, "Basis": "Information zur Zeile 'Veröffentlichung'",
                     "N": n, "Anteil_am_Rohdatensatz": n / n_raw, "Entfernt_gegenueber_Vorzeile": np.nan})
    subsets = [
        ("rating_sample", "Teilstichprobe: mind. 10 Reviews (Bewertungsanalysen)", "rating_sample"),
        ("rating_sample_50", "Teilstichprobe: mind. 50 Reviews (Robustheit)", "rating_sample_50"),
        ("meta_sample", "Teilstichprobe: FronkonGames-Metacritic vorhanden (nur Vergleich)", "meta_sample"),
        ("peak_sample", "Teilstichprobe: Peak CCU > 0 (ergänzende Zielgröße)", "peak_sample"),
        ("igdb_sample", "Teilstichprobe: IGDB-Match vorhanden", "igdb_sample"),
        ("mc_external_sample", "Teilstichprobe: externe Metacritic-Daten vorhanden", "mc_external_sample"),
        ("price_history_sample", "Teilstichprobe: Preishistorie vorhanden", "price_history_sample"),
        ("multi_source_sample", "Teilstichprobe: alle vier Datenquellen vorhanden", "multi_source_sample"),
    ]
    n_analytic = int(m_final.sum())
    for key, label, col in subsets:
        n = int(df[col].sum())
        rows.append({"Kennung": key, "Schritt": label, "Basis": "Teilmenge der Analysestichprobe", "N": n,
                     "Anteil_am_Rohdatensatz": n / n_raw, "Entfernt_gegenueber_Vorzeile": np.nan,
                     "Anteil_an_Analysestichprobe": n / n_analytic})
    n = int(df["sample_with_owners_zero"].sum())
    rows.append({"Kennung": "sample_with_owners_zero",
                 "Schritt": "Sensitivitätsstichprobe: Analysestichprobe plus Einträge mit '0 - 0'",
                 "Basis": "Erweiterung der Analysestichprobe", "N": n, "Anteil_am_Rohdatensatz": n / n_raw,
                 "Entfernt_gegenueber_Vorzeile": np.nan, "Anteil_an_Analysestichprobe": n / n_analytic})
    return pd.DataFrame(rows)


def quality_table(df: pd.DataFrame, n_raw: int) -> pd.DataFrame:
    analytic = df[df["analytic_sample"]]
    n_an = len(analytic)
    cutoff = STEAMSPY_CUTOFF.strftime("%d.%m.%Y")
    rows = []

    def add(key: str, label: str, basis: str, count: int, note: str = "", n_basis: int | None = None) -> None:
        if n_basis is None:
            n_basis = n_raw if basis == "Rohdatensatz" else n_an
        rows.append({"Kennung": key, "Pruefung": label, "Basis": basis, "N_Basis": n_basis,
                     "Wert": int(count), "Anteil": count / n_basis if n_basis else np.nan, "Hinweis": note})

    raw = "Rohdatensatz"
    ana = "Analysestichprobe"
    add("raw_rows", "Zeilen im Rohdatensatz", raw, n_raw)
    add("unique_appids", "Eindeutige AppIDs", raw, df["AppID"].nunique())
    add("duplicate_appids", "Doppelte AppIDs", raw, df["AppID"].duplicated().sum())
    add("missing_name", "Fehlender Name", raw, df["Name"].isna().sum())
    add("missing_release", "Fehlendes Release-Datum", raw, df["release_date"].isna().sum())
    add("release_after_cutoff", f"Release ab dem SteamSpy-Stichtag ({cutoff})", raw, df["released_after_cutoff"].sum(),
        "keine SteamSpy-Ergebnisdaten; aus der Analysestichprobe ausgeschlossen")
    add("release_after_2025", "Release-Jahr nach 2025", raw, (df["release_year"] > 2025).sum())
    add("owners_zero", "Besitzer '0 - 0' (fehlend)", raw, df["owners_zero"].sum(),
        "immer 0 Reviews und Peak CCU 0; Anteil steigt stark mit dem Release-Jahr")
    add("steamspy_missing", "SteamSpy-Lücke (0 Reviews, aber Steam-Recommendations > 0)", raw,
        df["steamspy_missing"].sum(), "Besitzer, Reviews und Peak CCU auf NaN gesetzt und ausgeschlossen")
    add("price_current_zero", "Tagespreis = 0", raw, (df["price_current"] == 0).sum())
    add("list_price_zero", "Listenpreis = 0 (kostenlos)", raw, df["is_free"].sum())
    add("list_price_undefined", "Listenpreis nicht bestimmbar", raw, df["list_price"].isna().sum(),
        "Rabatt 100 % oder Tagespreis 0 trotz Rabatt 1–99 %; weder kostenlos noch kostenpflichtig gezählt")
    add("price_zero_with_discount", "Tagespreis 0 trotz Rabatt 1–99 % (Datenfehler)", raw,
        df["price_zero_with_discount"].sum(), "Listenpreis fehlend statt 0; nicht als kostenlos gewertet")
    add("price_outlier", f"Listenpreis > {PRICE_OUTLIER_USD:.0f} USD (Preisausreißer)", raw, df["price_outlier"].sum())
    # Pruefung der Listenpreis-Rekonstruktion bei rabattierten Spielen.
    def off_point(frame: pd.DataFrame) -> pd.Series:
        lp = frame["list_price"]
        cents = (np.round(lp * 100) % 100).fillna(-1)
        return frame["has_discount"] & (lp > 0) & ~cents.isin([99, 0])

    raw_disc = df[df["has_discount"] & (df["list_price"] > 0)]
    add("list_price_conflict", "Rabattangabe widerspricht dem Tagespreis: Tagespreis als Listenpreis übernommen",
        "Rohdatensatz, rabattiert mit Listenpreis > 0", df["list_price_conflict"].sum(),
        "Tagespreis exakt x,99/ganzer Dollar, Tagespreis/(1 − Rabatt) neben keinem Preispunkt "
        "(z. B. 39,99 bei 85 % → 266,60); Rabattgruppe folgt weiter der Quelle", n_basis=len(raw_disc))
    add("list_price_off_point", "Rabattiert: Listenpreis nicht auf x,99/ganzem Dollar (beibehalten)",
        "Rohdatensatz, rabattiert mit Listenpreis > 0", off_point(raw_disc).sum(),
        "ungewöhnliche, aber mit Tagespreis und Rabatt vereinbare Preise (z. B. 0,93 bei 50 % → 1,86)",
        n_basis=len(raw_disc))
    add("no_metacritic", "Kein Metacritic-Score (FronkonGames)", raw, (~df["has_metacritic"]).sum())
    add("no_standard_genre", "Kein Steam-Standardgenre", raw, (~df["has_genre"]).sum(),
        "umfasst auch Einträge nur mit Free To Play, Early Access oder Software-Genres")
    add("genres_empty", "Genre-Feld leer", raw, df["genres_empty"].sum())
    add("no_categories", "Keine Kategorie-Angabe", raw, (df["categories"].apply(len) == 0).sum())
    add("no_languages", "Keine Sprachangabe", raw, (~df["languages_listed"]).sum())
    add("software_flag", "Software (Software-Genre ohne Standardgenre), Merkmal", raw, df["is_software"].sum(),
        "Merkmal über alle Zeilen; tatsächlich entfernt (nach den vorherigen Schritten) siehe sample_flow")
    add("non_game_flag", "Nicht-Spiel-Name (Demo/Playtest/Soundtrack usw.), Merkmal", raw, df["non_game_name"].sum(),
        "Merkmal über alle Zeilen; die meisten Treffer haben bereits '0 - 0' (siehe sample_flow)")
    add("duplicates_removed", "Duplikate (Name + Jahr + Entwickler), entfernt", raw,
        (df["is_duplicate"] & df["valid_owners"]).sum(),
        "nur Einträge mit gültiger Besitzerangabe; zusammen mit '0 - 0'-Einträgen (Sensitivitätsstichprobe) "
        f"{int(df['is_duplicate'].sum())}")
    add("analytic_sample", "Analysestichprobe", raw, n_an)
    add("an_no_reviews", "Keine Reviews", ana, (analytic["review_count"] == 0).sum())
    add("an_lt10_reviews", "Weniger als 10 Reviews", ana, (analytic["review_count"] < MIN_REVIEWS).sum())
    add("an_peak_zero", "Peak CCU = 0", ana, (analytic["peak_ccu"] == 0).sum(),
        "Peak CCU = gleichzeitige Spieler am Vortag des SteamSpy-Abrufs")
    add("an_list_price_zero", "Listenpreis = 0 (kostenlos)", ana, analytic["is_free"].sum())
    add("an_list_price_undefined", "Listenpreis nicht bestimmbar", ana, analytic["list_price"].isna().sum(),
        "Rabatt 100 % oder Tagespreis 0 trotz Rabatt 1–99 %")
    add("an_price_zero_with_discount", "Tagespreis 0 trotz Rabatt 1–99 % (Datenfehler)", ana,
        analytic["price_zero_with_discount"].sum(), "nicht als kostenlos gewertet (Listenpreis fehlend)")
    add("an_f2p_tag", "Genre 'Free To Play'", ana, analytic["f2p_tag"].sum())
    add("an_price_outlier", f"Listenpreis > {PRICE_OUTLIER_USD:.0f} USD", ana, analytic["price_outlier"].sum())
    add("an_price_cluster", f"Listenpreis exakt {CLUSTER_PRICE_DE} USD, Release ab {CLUSTER_FROM_YEAR} (Preis-Cluster)",
        ana, analytic["price_cluster"].sum(), "Serien von Kleinstspielen; in der Preisübersicht getrennt ausgewiesen")
    an_disc = analytic[analytic["has_discount"] & (analytic["list_price"] > 0)]
    add("an_list_price_conflict", "Rabattangabe widerspricht dem Tagespreis: Tagespreis als Listenpreis übernommen",
        "Analysestichprobe, rabattiert mit Listenpreis > 0", analytic["list_price_conflict"].sum(),
        n_basis=len(an_disc))
    add("an_list_price_off_point", "Rabattiert: Listenpreis nicht auf x,99/ganzem Dollar (beibehalten)",
        "Analysestichprobe, rabattiert mit Listenpreis > 0", off_point(an_disc).sum(), n_basis=len(an_disc))
    add("an_discount", "Rabatt > 0 zum Store-Snapshot", ana, analytic["has_discount"].sum())
    add("an_no_english", "Englisch nicht unterstützt (bei vorhandener Sprachangabe)", ana,
        (analytic["languages_listed"] & ~analytic["english_supported"]).sum())
    add("an_early_access", "Early Access (Genre-Markierung)", ana, analytic["early_access"].sum())
    for key, label, col in [
        ("an_igdb", "IGDB-Match", "igdb_sample"),
        ("an_mc_external", "Externer Metacritic-Match", "mc_external_sample"),
        ("an_price_history", "Preishistorie-Match", "price_history_sample"),
    ]:
        add(key, label, ana, analytic[col].sum())
    add("an_two_sources", "Mindestens zwei Datenquellen", ana, (analytic["n_sources"] >= 2).sum())
    add("an_all_sources", "Alle vier Datenquellen", ana, analytic["multi_source_sample"].sum())
    add("an_release_shifted", "IGDB-Erstveröffentlichung > 1 Jahr vor dem Steam-Jahr (release_shifted)", ana,
        analytic["release_shifted"].sum(),
        "überwiegend Neuauflagen/Portierungen, teils Early-Access-Absolventen; Sensitivitätsanalyse")
    shifted = analytic[analytic["release_shifted"]]
    # Das Matching erlaubt hoechstens ein Jahr Abstand; release_shifted kann daher nur
    # aus dem Namens-Lookup (Spiele ohne IGDB-Treffer) stammen.
    add("an_release_shifted_lookup", "davon IGDB-Jahr aus dem Namens-Lookup (ohne IGDB-Treffer)",
        "Analysestichprobe, release_shifted", shifted["igdb_year_from_lookup"].eq(True).sum(),
        "Treffer der Stufen 1/2 haben höchstens 1 Jahr Abstand, Stufe 3 kein Jahr", n_basis=len(shifted))
    gap = shifted["release_year"] - shifted["igdb_first_release_year"]
    add("an_release_shifted_gap23", "davon Abstand 2 bis 3 Jahre", "Analysestichprobe, release_shifted",
        gap.between(2, 3).sum(), "Größenordnung typischer Early-Access-Phasen, aber auch Portierungen",
        n_basis=len(shifted))
    add("an_ea_graduate_proxy", "Hinweis auf früheren Markteintritt (EA-Absolventen-Proxy, ohne aktuelle EA-Markierung)",
        ana, analytic["ea_graduate_proxy"].sum(),
        f"AppID unter dem Median des Jahrgangs {APPID_PROXY_YEARS} Jahre vor dem Store-Jahr oder erster Preismonat "
        "> 1 Monat vor dem Store-Datum; mischt EA-Absolventen und lange angekündigte Titel")
    add("an_entry_proxy_appid", "davon über die AppID", ana, analytic["entry_proxy_appid"].sum())
    add("an_entry_proxy_ph", "davon über die Preishistorie", ana, analytic["entry_proxy_ph"].sum())
    add("an_entry_proxy_shifted", "EA-Absolventen-Proxy mit release_shifted",
        "Analysestichprobe, EA-Absolventen-Proxy", (analytic["ea_graduate_proxy"] & analytic["release_shifted"]).sum(),
        n_basis=int(analytic["ea_graduate_proxy"].sum()))
    # Wie datiert IGDB die Proxy-Faelle? Gleiches Jahr wie Steam spricht fuer eine
    # Datierung auf die Vollversion (oder einen Fehlgriff des Proxys).
    ea_igdb = analytic[analytic["ea_graduate_proxy"] & analytic["igdb_first_release_year"].notna()]
    add("an_entry_proxy_igdb_year", "EA-Absolventen-Proxy mit IGDB-Erstveröffentlichungsjahr",
        "Analysestichprobe, EA-Absolventen-Proxy", len(ea_igdb), n_basis=int(analytic["ea_graduate_proxy"].sum()))
    add("an_entry_proxy_igdb_same", "davon IGDB-Jahr = Store-Jahr", "EA-Absolventen-Proxy mit IGDB-Jahr",
        (ea_igdb["igdb_first_release_year"] == ea_igdb["release_year"]).sum(),
        "IGDB datiert wie Steam (Vollversion) oder der Proxy trifft keinen früheren Markteintritt",
        n_basis=len(ea_igdb))
    kept_kw = analytic[analytic["Name"].fillna("").str.contains(KEEP_CHECK_PATTERN)]
    add("an_kept_keyword", "Gegenprobe Namensfilter: behalten trotz Schlüsselwort im Namen", ana, len(kept_kw),
        "Schlüsselwörter wie Edition, Test, Pack, Beta, Demo; 25 Beispiele in exclusion_samples.csv")
    add("an_kept_keyword_genre", "Gegenprobe: davon mit Steam-Standardgenre", "Gegenprobe Namensfilter",
        kept_kw["has_genre"].sum(), "ohne Standardgenre: u. a. Software ohne Genreangabe, die kein Filter erkennt",
        n_basis=len(kept_kw))

    # Verfuegbarkeit der IGDB-Merkmale innerhalb der IGDB-Teilstichprobe: Modus-
    # und Genre-Anteile werden auf Spiele mit Angabe bezogen, nicht auf alle Treffer.
    igdb = analytic[analytic["igdb_sample"]]
    n_igdb = len(igdb)
    ig = "IGDB-Teilstichprobe"
    for stage in (1, 2, 3):
        add(f"igdb_stage{stage}", f"IGDB-Treffer in Stufe {stage}", ig,
            igdb["igdb_match_stage"].eq(stage).sum(), n_basis=n_igdb)
    add("igdb_has_modes", "IGDB-Spielmodus angegeben", ig, igdb["igdb_has_modes"].eq(True).sum(),
        "Basis der Modus-Anteile; ohne Angabe sind die Modus-Merkmale fehlend", n_basis=n_igdb)
    add("igdb_has_genres", "IGDB-Genre angegeben", ig, igdb["igdb_has_genres"].eq(True).sum(),
        "Basis der Genre-Anteile", n_basis=n_igdb)
    add("igdb_rating", "IGDB-Nutzerbewertung vorhanden", ig, igdb["igdb_rating"].notna().sum(), n_basis=n_igdb)
    add("igdb_rating_integer", "IGDB-Nutzerbewertung ganzzahlig", "IGDB-Treffer mit Nutzerbewertung",
        igdb["igdb_rating_integer"].eq(True).sum(),
        "ganzzahlige Werte deuten auf sehr wenige Stimmen (IGDB-Datei ohne Stimmenzahl)",
        n_basis=int(igdb["igdb_rating"].notna().sum()))
    add("igdb_aggregated_rating", "IGDB-Kritikerbewertung vorhanden", ig,
        igdb["igdb_aggregated_rating"].notna().sum(), n_basis=n_igdb)
    stage3 = igdb[igdb["igdb_match_stage"].eq(3)]
    add("igdb_stage3_has_modes", "Stufe-3-Treffer mit Spielmodus-Angabe", "IGDB-Treffer in Stufe 3",
        stage3["igdb_has_modes"].eq(True).sum(), "undatierte IGDB-Einträge sind meist spärlich dokumentiert",
        n_basis=len(stage3))
    mc = analytic[analytic["mc_external_sample"]]
    add("mc_user_score", "Externer Metacritic-Nutzerscore vorhanden", "Externe Metacritic-Teilstichprobe",
        mc["mc_user_score"].notna().sum(), n_basis=len(mc))
    ph = analytic[analytic["price_history_sample"]]
    add("ph_full_window", "Preishistorie mit vollem Fenster (Beginn Nov. oder Dez. 2023 bis Okt. 2025)", "Preishistorie-Teilstichprobe",
        ph["ph_full_window"].eq(True).sum(), n_basis=len(ph))
    add("ph_lowest_above_base", "Preishistorie: Tiefstpreis über statischem Basispreis in mind. einem Monat",
        "Preishistorie-Teilstichprobe", ph["ph_lowest_above_base"].eq(True).sum(), n_basis=len(ph))
    # Spiele mit Preisdaten vor dem Store-Datum (v. a. Early-Access-Absolventen):
    # Wie oft erfasst IGDB bzw. release_shifted den frueheren Markteintritt?
    ph_early = ph[ph["entry_proxy_ph"]]
    add("ph_before_release", "Preishistorie beginnt > 1 Monat vor dem Store-Datum", "Preishistorie-Teilstichprobe",
        len(ph_early), "Hinweis auf früheren Markteintritt (v. a. Early-Access-Absolventen)", n_basis=len(ph))
    add("ph_before_release_shifted", "davon mit release_shifted", "Preishistorie beginnt vor dem Store-Datum",
        ph_early["release_shifted"].sum(), n_basis=len(ph_early))
    early_igdb = ph_early[ph_early["igdb_first_release_year"].notna()]
    add("ph_before_release_igdb_same", "davon IGDB-Erstveröffentlichungsjahr = Store-Jahr",
        "Preishistorie beginnt vor dem Store-Datum, IGDB-Jahr vorhanden",
        (early_igdb["igdb_first_release_year"] == early_igdb["release_year"]).sum(),
        "IGDB datiert diese Spiele wie Steam auf die Vollversion (Preisfenster beginnt erst Nov. 2023)", n_basis=len(early_igdb))
    add("ph_before_release_appid", "davon auch über die AppID erkannt", "Preishistorie beginnt vor dem Store-Datum",
        ph_early["entry_proxy_appid"].sum(), n_basis=len(ph_early))
    # Plausibilisierung des Snapshot-Rabatts: Kostet ein Spiel ohne erfassten Rabatt
    # hoechstens 90 % des (statischen) Basispreises der Preishistorie, fehlt der
    # Rabatt sehr wahrscheinlich. Grenzen: Basispreis aus Nov. 2023 bis Okt. 2025,
    # Snapshot ca. Maerz 2026; dauerhafte Preissenkungen wirken genauso.
    ph_paid0 = ph[(ph["list_price"] > 0) & (ph["discount"] == 0) & ph["ph_base_price"].notna()]
    basis_ph0 = "Preishistorie-Teilstichprobe, kostenpflichtig, Rabatt 0 zum Snapshot"
    add("ph_disc0_le90", "Rabatt 0, aber Tagespreis ≤ 90 % des Basispreises der Preishistorie", basis_ph0,
        (ph_paid0["price_current"] <= 0.9 * ph_paid0["ph_base_price"]).sum(),
        "Hinweis auf nicht erfassten Rabatt (oder dauerhafte Preissenkung nach Okt. 2025)", n_basis=len(ph_paid0))
    add("ph_disc0_below", "Rabatt 0, aber Tagespreis unter dem Basispreis der Preishistorie", basis_ph0,
        (ph_paid0["price_current"] < ph_paid0["ph_base_price"] - 0.005).sum(), n_basis=len(ph_paid0))
    add("ph_disc0_above", "Rabatt 0, Tagespreis über dem Basispreis der Preishistorie", basis_ph0,
        (ph_paid0["price_current"] > ph_paid0["ph_base_price"] + 0.005).sum(), n_basis=len(ph_paid0))
    ph_disc = ph[(ph["list_price"] > 0) & (ph["discount"] > 0) & ph["ph_base_price"].notna()]
    add("ph_disc_list_eq_base", "Rabatt > 0: rekonstruierter Listenpreis = Basispreis der Preishistorie (± 0,02)",
        "Preishistorie-Teilstichprobe, kostenpflichtig, Rabatt > 0 zum Snapshot",
        ((ph_disc["list_price"] - ph_disc["ph_base_price"]).abs() <= 0.02).sum(),
        "Prüfung der Listenpreis-Rekonstruktion", n_basis=len(ph_disc))
    return pd.DataFrame(rows)


def de_int(value: float) -> str:
    return f"{value:,.0f}".replace(",", ".")


def success_definitions(df: pd.DataFrame) -> pd.DataFrame:
    analytic = df["analytic_sample"]
    q75 = float(df.loc[analytic, "owners_low"].quantile(0.75))
    q90 = float(df.loc[analytic, "owners_low"].quantile(0.90))
    if q75 != SUCCESS_TOP25_OWNERS:
        print(f"WARNUNG: 75-%-Quantil von owners_low ist {q75:,.0f}, nicht {SUCCESS_TOP25_OWNERS:,}.")

    df["success_top25"] = df["owners_low"] >= SUCCESS_TOP25_OWNERS
    df["success_50k"] = df["owners_low"] >= SUCCESS_50K_OWNERS
    df["success_100k"] = df["owners_low"] >= SUCCESS_100K_OWNERS
    peak_threshold = float(df.loc[df["peak_sample"], "peak_ccu"].quantile(0.75))
    peak_success = pd.Series(pd.NA, index=df.index, dtype="boolean")
    peak_success[df["peak_sample"]] = df.loc[df["peak_sample"], "peak_ccu"] >= peak_threshold
    df["success_peak_top25"] = peak_success

    n_an = int(analytic.sum())
    n_peak = int(df["peak_sample"].sum())
    rows = []
    for key, label, var, threshold, note in [
        ("success_top25", "Oberes Quartil: mind. 20.000 Besitzer (Hauptdefinition)", "owners_low",
         SUCCESS_TOP25_OWNERS,
         f"75-%-Quantil von owners_low = {de_int(q75)}; wegen der Besitzerklassen weicht der Anteil von 25 % ab"),
        ("success_50k", "Mindestens 50.000 Besitzer (Robustheit)", "owners_low", SUCCESS_50K_OWNERS,
         f"90-%-Quantil von owners_low = {de_int(q90)}; keine 'Top 10 %', sondern der tatsächliche Anteil"),
        ("success_100k", "Mindestens 100.000 Besitzer (Robustheit)", "owners_low", SUCCESS_100K_OWNERS, ""),
    ]:
        k = int(df.loc[analytic, key].sum())
        rows.append({"Kennung": key, "Kriterium": label, "Variable": var, "Schwelle": threshold,
                     "Basis": "Analysestichprobe", "N_Basis": n_an, "N_erfolgreich": k,
                     "Anteil": k / n_an, "Anteil_an_Analysestichprobe": k / n_an, "Hinweis": note})
    k = int(df.loc[df["peak_sample"], "success_peak_top25"].sum())
    rows.append({
        "Kennung": "success_peak_top25",
        "Kriterium": "Peak CCU oberes Quartil (nur Spiele mit Peak CCU > 0; nur dokumentiert, nicht ausgewertet)",
        "Variable": "peak_ccu", "Schwelle": peak_threshold,
        "Basis": "Analysestichprobe mit Peak CCU > 0", "N_Basis": n_peak, "N_erfolgreich": k,
        "Anteil": k / n_peak if n_peak else np.nan, "Anteil_an_Analysestichprobe": k / n_an,
        "Hinweis": "in keiner Analyse verwendet; Peak CCU wird zweiteilig ausgewertet (Peak > 0 und Intensität). "
                   "Peak CCU = gleichzeitige Spieler am Vortag des SteamSpy-Abrufs (ca. Anfang Juni 2025), "
                   "kein Allzeit-Peak; außerhalb von Peak CCU > 0 nicht definiert",
    })
    return pd.DataFrame(rows)


def variable_overview() -> pd.DataFrame:
    rows = [
        ("owners_low", "Untere Grenze der SteamSpy-Besitzerklasse (Stand ca. Anfang Juni 2025)", "ordinal (Klassengrenze)", "Zielgröße (Erfolgsschwellen)"),
        ("owners_mid", "Mittelpunkt der Besitzerklasse", "ordinal (13 Klassenmitten)", "Zielgröße in Mann-Whitney/Cliff's δ und Kruskal-Wallis (rangbasiert)"),
        ("log_owners", "log10(owners_mid)", "ordinal (13 Werte)", "Zielgröße in Spearman-Korrelationen; für Rangverfahren gleichwertig zu owners_mid (monotone Transformation)"),
        ("owners_zero", "Besitzerangabe '0 - 0' (fehlend, nicht zufällig verteilt)", "binär", "Ausschluss; Sensitivitätsanalyse"),
        ("steamspy_missing", "SteamSpy-Lücke: 0 Reviews trotz Steam-Recommendations > 0", "binär", "Ausschluss (Besitzer/Reviews/Peak auf NaN)"),
        ("peak_ccu", "Höchste Zahl gleichzeitiger Spieler am Vortag des SteamSpy-Abrufs (ca. Anfang Juni 2025), kein Allzeit-Peak", "metrisch", "ergänzende Zielgröße (Momentaufnahme, nur Peak > 0)"),
        ("release_year", "Jahr des aktuellen Steam-Store-Datums (bei Early-Access-Absolventen Datum der Vollversion)", "metrisch", "Kontrollvariable (Alter/Kohorte)"),
        ("release_year_alt", "Früheres von Steam-Jahr und IGDB-Erstveröffentlichungsjahr (IGDB-Treffer oder Namens-Lookup); bei Portierungen obere Schranke des Alters auf Steam", "metrisch", "Sensitivität der Alterskontrolle"),
        ("release_shifted", "IGDB-Erstveröffentlichung mehr als ein Jahr vor dem Steam-Jahr (überwiegend Neuauflagen/Portierungen, seltener frühere Veröffentlichung außerhalb von Steam; Early-Access-Absolventen nur teilweise, da IGDB sie teils auf die Vollversion datiert); stammt nur aus dem Namens-Lookup, weil IGDB-Treffer höchstens 1 Jahr Abstand haben", "binär", "Sensitivität (Ausschluss)"),
        ("market_entry_year", f"Geschätztes Jahr des Markteintritts: Minimum aus Store-Jahr, AppID-Jahr (falls AppID unter dem Median des Jahrgangs {APPID_PROXY_YEARS} Jahre früher) und Jahr des ersten Preismonats (falls > 1 Monat vor dem Store-Datum); nur ohne aktuelle Early-Access-Markierung", "metrisch", "Sensitivität der Alterskontrolle (Early Access)"),
        ("ea_graduate_proxy", "Hinweis auf früheren Markteintritt (entry_proxy_appid oder entry_proxy_ph); mischt Early-Access-Absolventen und lange vorab angekündigte Titel", "binär", "Sensitivität (Ausschluss)"),
        ("age_years", "Jahre zwischen Store-Datum und SteamSpy-Stichtag (01.06.2025)", "metrisch", "Kontrollvariable (deskriptiv)"),
        ("n_platforms", "Anzahl unterstützter Plattformen", "metrisch", "F1"),
        ("platform_combo", "Plattformkombination (W, W+M, W+L, W+M+L)", "kategorial", "F1"),
        ("genre_*", "Steam-Standardgenres", "binär", "F1"),
        ("mode", "Single-/Multiplayer aus Steam-Kategorien (Multiplayer: Multi-player, PvP, Co-op, Shared/Split Screen, MMO, LAN, Cross-Platform, Remote Play Together)", "kategorial", "F1"),
        ("price_current", "Tagespreis zum Store-Snapshot (ca. Anfang März 2026) in USD, ggf. rabattiert", "metrisch", "nur Sensitivität (Tagespreis)"),
        ("list_price", "Rekonstruierter Listenpreis = Tagespreis / (1 - Rabatt/100), auf Preispunkte x,99/ganze Dollar gesetzt; Tagespreis, wenn die Rabattangabe dem Tagespreis widerspricht (list_price_conflict); Untergrenze, da der Rabatt teils fehlt; fehlend bei 100 % Rabatt und bei Tagespreis 0 trotz Rabatt 1–99 %", "metrisch", "F2 (Preisniveau)"),
        ("list_price_conflict", "Rabattangabe widerspricht dem Tagespreis (Tagespreis exakt x,99/ganzer Dollar, rekonstruierter Wert neben keinem Preispunkt); Tagespreis als Listenpreis übernommen", "binär", "Datenqualität"),
        ("price_class", "Preisklasse nach Listenpreis ('Free' = Listenpreis 0; fehlend ohne bestimmbaren Listenpreis)", "ordinal", "F2"),
        ("price_class_current", "Preisklasse nach Tagespreis", "ordinal", "nur Sensitivität"),
        ("is_free", "kostenlos (Listenpreis 0); nicht: Tagespreis 0 durch 100 % Rabatt oder Tagespreis 0 trotz Rabatt 1–99 %", "binär", "F2 (Tests nur auf Spielen mit bestimmbarem Listenpreis)"),
        ("price_zero_with_discount", "Tagespreis 0 trotz Rabatt 1–99 % (Datenfehler der Quelle); Listenpreis dort fehlend", "binär", "Datenqualität"),
        ("f2p_tag", "Steam-Genre 'Free To Play'", "binär", "F2 (Sensitivität)"),
        ("price_outlier", f"Listenpreis > {PRICE_OUTLIER_USD:.0f} USD", "binär", "in Preisniveau-Analysen ausgeschlossen"),
        ("price_cluster", f"Listenpreis exakt {CLUSTER_PRICE_DE} USD bei Release ab {CLUSTER_FROM_YEAR} (Serien von Kleinstspielen)", "binär", "in der Preisübersicht getrennt ausgewiesen; in den Tests enthalten (Sensitivität ohne)"),
        ("discount", "Rabatt zum Store-Snapshot in Prozent (teils fehlend, daher Untergrenze)", "metrisch", "F2"),
        ("has_discount", "Rabatt > 0 zum Store-Snapshot", "binär", "F2"),
        ("dlc_count", "Anzahl DLCs", "metrisch", "F2"),
        ("positive_ratio", "Anteil positiver Steam-Reviews (SteamSpy-Stand ca. Juni 2025)", "metrisch", "F3 (primäres Maß der Nutzerbewertung)"),
        ("wilson", "Wilson-Untergrenze des Anteils positiver Reviews", "metrisch", "nur Diagnose (stark vom Review-Volumen abhängig)"),
        ("review_count", "Anzahl Reviews (Positive + Negative)", "metrisch", "Stichprobenrestriktion (mind. 10) und Diagnose der Volumenabhängigkeit; kein inhaltlicher Prädiktor"),
        ("metacritic", "Metacritic-Score aus FronkonGames", "metrisch", "Vergleichswert"),
        ("mc_metascore", "Externer Metacritic-Kritikerscore", "metrisch", "F3 (Kritikerbewertung)"),
        ("mc_user_score", "Externer Metacritic-Nutzerscore", "metrisch", "F3 (nur wo vorhanden)"),
        ("mc_match_stage", "Match-Stufe Metacritic (1 = Name + Jahr, 2 = eindeutiger Name, Steam-Jahr >= MC-Jahr - 1)", "kategorial", "Matching-Diagnose"),
        ("igdb_match_stage", "Match-Stufe IGDB (1 = Name + Jahr, 2 = eindeutiger Name +/- 1 Jahr, 3 = eindeutiger Name, IGDB undatiert)", "kategorial", "Matching-Diagnose, Selektionstests"),
        ("igdb_rating", "IGDB-Nutzerbewertung (0 = Platzhalter -> fehlend)", "metrisch", "H4 (externe Konsistenzprüfung)"),
        ("igdb_rating_integer", "IGDB-Nutzerbewertung ganzzahlig (Hinweis auf sehr wenige Stimmen; IGDB-Datei ohne Stimmenzahl)", "binär (fehlend ohne Nutzerbewertung)", "Stratifizierung der IGDB-Nutzerbewertung"),
        ("igdb_aggregated_rating", "IGDB-Kritikerbewertung (0 = Platzhalter -> fehlend)", "metrisch", "H4 (externe Konsistenzprüfung)"),
        ("igdb_has_modes", "IGDB-Treffer mit Spielmodus-Angabe", "binär", "Basis der IGDB-Modus-Anteile und -Tests"),
        ("igdb_multiplayer", "IGDB: Multiplayer, Co-operative, Split screen, MMO oder Battle Royale (analog Steam)", "binär (fehlend ohne Modusangabe)", "H4 (Konsistenz mit Steam, keine unabhängige Bestätigung)"),
        ("igdb_singleplayer", "IGDB: Singleplayer-Modus", "binär (fehlend ohne Modusangabe)", "H4 (Konsistenz mit Steam)"),
        ("igdb_coop", "IGDB: Co-operative oder Split screen", "binär (fehlend ohne Modusangabe)", "H4 (Konsistenz mit Steam)"),
        ("igdb_mmo", "IGDB: Massively Multiplayer Online (MMO)", "binär (fehlend ohne Modusangabe)", "H4 (Konsistenz mit Steam)"),
        ("igdb_genre_*", "IGDB-Genres (exakte IGDB-Bezeichnungen, inkl. Pinball)", "binär (fehlend ohne Genreangabe)", "H4 (Konsistenz mit Steam)"),
        ("igdb_n_genres", "Anzahl IGDB-Genres (fehlend ohne Genreangabe)", "metrisch", "H4 (Dokumentationstiefe)"),
        ("igdb_pc_like", "IGDB-Eintrag mit PC-naher Plattform (Windows, Mac, Linux, SteamVR); fehlend ohne Plattformangabe", "binär", "Matching-Diagnose"),
        ("igdb_first_release_year", "IGDB-Erstveröffentlichungsjahr: Jahr des datierten Treffers, sonst Namens-Lookup (igdb_year_from_lookup)", "metrisch", "Basis für release_year_alt/release_shifted"),
        ("ph_price_changes_per_month", "Preisänderungen je Beobachtungsmonat (Preishistorie)", "metrisch", "H4 (externe Quelle)"),
        ("ph_days_on_sale_per_month", "Rabatttage je Beobachtungsmonat (bezogen auf den statischen Basispreis der Quelle)", "metrisch", "H4 (externe Quelle)"),
        ("ph_share_months_on_sale", "Anteil der Beobachtungsmonate mit Rabatt", "metrisch", "H4 (externe Quelle)"),
        ("ph_max_discount_pct", "Maximaler Preisnachlass gegenüber dem Basispreis (nur bei vollem Beobachtungsfenster interpretierbar)", "metrisch", "H4 (externe Quelle)"),
        ("ph_mean_monthly_max_discount_pct", "Mittel der monatlichen Höchstrabatte (aus dem monatlichen Tiefstpreis; Monate ohne Rabatt mit 0); kein mittlerer Rabatt je Verkaufstag", "metrisch", "H4 (externe Quelle)"),
        ("ph_months", "Beobachtete Monate in der Preishistorie (Nov. 2023 bis Okt. 2025, 2025-11 verworfen)", "metrisch", "Kontrollvariable (Fensterlänge)"),
        ("ph_full_window", "Volles Beobachtungsfenster (Beginn Nov. oder Dez. 2023 bis Okt. 2025, 24 bzw. 23 Monate)", "binär", "Teilmenge für Preishistorie-Tests"),
        ("ph_lowest_above_base", "Tiefstpreis eines Monats über dem statischen Basispreis (Rabatt dort nicht bestimmbar)", "binär", "Sensitivität (Ausschluss)"),
        ("n_sources", "Anzahl verfügbarer Datenquellen", "metrisch", "Datenabdeckung"),
        ("english_supported", "Englisch in 'Supported languages' (False auch bei fehlender Sprachangabe, siehe languages_listed)", "binär", "Selektionstests"),
        ("languages_listed", "Sprachangabe vorhanden", "binär", "Selektionstests (Basis für english_supported)"),
        ("success_top25", "Erfolg: mind. 20.000 Besitzer (oberes Quartil)", "binär", "Hauptzielgröße"),
        ("success_50k", "Erfolg: mind. 50.000 Besitzer", "binär", "Robustheitsziel"),
        ("success_100k", "Erfolg: mind. 100.000 Besitzer", "binär", "Robustheitsziel"),
        ("success_peak_top25", "Peak CCU im oberen Quartil der Spiele mit Peak CCU > 0 (Schwelle siehe success_thresholds.csv)", "binär (fehlend außerhalb Peak > 0)", "nicht ausgewertet (nur dokumentiert)"),
    ]
    return pd.DataFrame(rows, columns=["Variable", "Beschreibung", "Typ", "Rolle"])


def exclusion_samples(df: pd.DataFrame) -> pd.DataFrame:
    """Zufaellige Beispiele je Ausschlussschritt (Seed 42) zur Plausibilitaetspruefung."""
    m_released = ~df["owners_zero"] & df["released"]
    m_steamspy = m_released & ~df["steamspy_missing"]
    m_name = m_steamspy & ~df["non_game_name"]
    m_software = m_name & ~df["is_software"]
    # Gegenprobe zum Namensfilter: behaltene Spiele, deren Name ein Schluesselwort
    # enthaelt (sollten ueberwiegend echte Spiele sein).
    kept_keyword = df["analytic_sample"] & df["Name"].fillna("").str.contains(KEEP_CHECK_PATTERN)
    reasons = [
        ("Nach dem SteamSpy-Stichtag datiert, mit SteamSpy-Daten", ~df["owners_zero"] & df["dated_after_cutoff_with_steamspy"]),
        ("SteamSpy-Lücke", m_released & df["steamspy_missing"]),
        ("Nicht-Spiel-Name", m_steamspy & df["non_game_name"]),
        ("Software", m_name & df["is_software"]),
        ("Duplikat", m_software & df["is_duplicate"]),
        ("Gegenprobe: behalten trotz Schlüsselwort im Namen", kept_keyword),
    ]
    names = df.set_index("AppID")["Name"]
    frames = []
    for label, mask in reasons:
        part = df[mask]
        part = part.sample(min(25, len(part)), random_state=RANDOM_STATE) if len(part) else part
        frame = pd.DataFrame({
            "Ausschlussgrund": label,
            "N_gesamt": int(mask.sum()),
            "AppID": part["AppID"],
            "Name": part["Name"],
            "Release": part["release_date"].dt.strftime("%Y-%m-%d"),
            "Besitzer_roh": part["Estimated owners"],
            "Recommendations": part["Recommendations"],
            "Genres": part["Genres"],
            "Entwickler": part["Developers"],
            "Behalten_statt_dessen": part["duplicate_of"].map(
                lambda x: f"{x}: {names.get(x, '')}" if pd.notna(x) else ""
            ),
        })
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    df = load_raw()
    n_raw = len(df)
    df, info = prepare_steam(df)

    coverage_by_month = steamspy_coverage(df)
    save_table(coverage_by_month, "steamspy_coverage_by_month")
    cutoff_message = check_steamspy_cutoff(coverage_by_month, df)

    # Externe Quellen anbinden (IGDB, Metacritic, Preishistorie).
    df, coverage, overlap = attach_external_sources(df)
    add_release_alternatives(df)
    info["markteintritt"] = add_market_entry_proxy(df)
    df["igdb_sample"] = df["analytic_sample"] & df["has_igdb"]
    df["mc_external_sample"] = df["analytic_sample"] & df["has_mc_external"]
    df["price_history_sample"] = df["analytic_sample"] & df["has_price_history"]
    df["multi_source_sample"] = (
        df["analytic_sample"] & df["has_igdb"] & df["has_mc_external"]
        & df["has_price_history"]
    )

    thresholds = success_definitions(df)
    save_table(thresholds, "success_thresholds")
    analytic = df[df["analytic_sample"]].copy()

    flow = flow_table(df, n_raw)
    save_table(flow, "sample_flow")
    save_table(quality_table(df, n_raw), "data_quality_summary")
    save_table(exclusion_samples(df), "exclusion_samples")

    missing = (
        analytic.drop(columns=["genres", "categories", "duplicate_of"])
        .isna()
        .mean()
        .sort_values(ascending=False, kind="mergesort")
        .rename("Fehlanteil")
        .reset_index()
        .rename(columns={"index": "Variable"})
    )
    save_table(missing, "missing_values")
    save_table(variable_overview(), "variable_overview")

    save_cols = [
        "AppID", "Name", "release_date", "release_year", "release_year_alt", "release_shifted",
        "market_entry_year", "ea_graduate_proxy", "entry_proxy_appid", "entry_proxy_ph",
        "age_years", "owners_low", "owners_high", "owners_mid", "owners_class", "owners_label",
        "log_owners", "owners_zero", "steamspy_missing", "steamspy_signal",
        "peak_ccu", "log_peak_ccu", "price_current", "list_price", "price_class",
        "price_class_current", "is_free", "f2p_tag", "price_outlier", "price_cluster", "discount", "has_discount",
        "price_zero_with_discount", "list_price_conflict",
        "dlc_count", "windows", "mac", "linux", "n_platforms", "platform_combo",
        "metacritic", "has_metacritic", "Positive", "Negative", "Recommendations", "review_count",
        "positive_ratio", "wilson", "mode", "single_player", "multi_player",
        "coop", "pvp", "mmo", "n_genres", "has_genre", "genres_empty",
        "early_access", "is_software", "non_game_name", "is_duplicate", "duplicate_of",
        "valid_owners", "released", "released_after_cutoff", "dated_after_cutoff_with_steamspy",
        "languages_listed", "english_supported",
        "analytic_sample", "sample_with_owners_zero", "rating_sample",
        "rating_sample_50", "meta_sample", "peak_sample", "success_top25",
        "success_50k", "success_100k", "success_peak_top25",
        "igdb_sample", "mc_external_sample", "price_history_sample", "multi_source_sample",
        "has_igdb", "has_mc_external", "has_price_history", "n_sources",
        "Developers", "Publishers", "Genres", "Categories",
    ] + ["genre_" + re.sub(r"[^a-z0-9]+", "_", g.lower()).strip("_") for g in STANDARD_GENRES]
    external_cols = [c for c in df.columns if c.startswith(("igdb_", "mc_", "ph_")) and c not in save_cols]
    save_cols = save_cols + external_cols
    df[save_cols].to_parquet(DATA / "steam_prepared.parquet", index=False)
    analytic[save_cols].to_parquet(DATA / "analytic_sample.parquet", index=False)
    for name in [
        "rating_sample", "rating_sample_50", "meta_sample", "igdb_sample",
        "mc_external_sample", "price_history_sample", "multi_source_sample",
    ]:
        analytic[analytic[name]][save_cols].to_parquet(DATA / f"{name}.parquet", index=False)

    provenance_path = write_provenance()

    dup = info["duplikate"]
    lp = info["listenpreis"]
    print(cutoff_message)
    print(f"Rohdatensatz: {n_raw:,} (Provenienz: {provenance_path.name})")
    print(f"Analysestichprobe: {int(df['analytic_sample'].sum()):,}")
    print(f"Bewertungsstichprobe (>=10 Reviews): {int(df['rating_sample'].sum()):,}")
    print(f"Peak-Teilstichprobe (Peak CCU > 0): {int(df['peak_sample'].sum()):,}")
    print(f"Metacritic-Teilstichprobe (FronkonGames): {int(df['meta_sample'].sum()):,}")
    print(f"IGDB-Teilstichprobe: {int(df['igdb_sample'].sum()):,}")
    print(f"Externe Metacritic-Teilstichprobe: {int(df['mc_external_sample'].sum()):,}")
    print(f"Preishistorie-Teilstichprobe: {int(df['price_history_sample'].sum()):,}")
    print(f"Alle vier Datenquellen: {int(df['multi_source_sample'].sum()):,}")
    print(f"Mindestens zwei Datenquellen: {int((df['analytic_sample'] & (df['n_sources'] >= 2)).sum()):,}")
    print(
        f"Duplikate: {dup['entfernt']:,} Eintraege in {dup['gruppen_mit_duplikaten']:,} Gruppen entfernt "
        f"({dup['entfernt_gleicher_name']:,} mit gleichem Namen, {dup['entfernt_editionsvariante']:,} "
        f"Editions-/Schreibvarianten); {dup['ohne_schluessel']:,} von {dup['kandidaten']:,} Kandidaten ohne "
        "verwertbaren Schluessel (Name oder Entwickler/Publisher leer) nie zusammengelegt."
    )
    print(
        f"Listenpreis: {lp['rabattiert']:,} rabattierte Eintraege rekonstruiert, davon "
        f"{lp['auf_preispunkt_gesetzt']:,} auf einen Preispunkt gesetzt, {lp['nicht_auf_preispunkt']:,} neben "
        f"jedem Preispunkt (davon {lp['widerspruch_tagespreis_uebernommen']:,} mit Rabattangabe im Widerspruch "
        "zum Tagespreis: Tagespreis uebernommen); "
        f"{lp['nicht_bestimmbar_rabatt_100']:,} mit 100 % Rabatt und "
        f"{lp['nicht_bestimmbar_tagespreis_0_mit_rabatt']:,} mit Tagespreis 0 trotz Rabatt 1-99 % ohne Listenpreis "
        f"(nicht als kostenlos gezaehlt; Analysestichprobe: {int(analytic['list_price'].isna().sum()):,})."
    )
    kept = int((analytic["Name"].fillna("").str.contains(KEEP_CHECK_PATTERN)).sum())
    print(f"Gegenprobe Namensfilter: {kept:,} behaltene Spiele mit Schluesselwort im Namen "
          "(Stichprobe in exclusion_samples.csv)")
    shifted = analytic["release_shifted"]
    print(f"Release-Jahr verschoben (IGDB > 1 Jahr frueher), Analysestichprobe: {int(shifted.sum()):,} "
          f"(Early-Access-Markierung {analytic.loc[shifted, 'early_access'].mean():.1%}, "
          f"Erfolgsquote {analytic.loc[shifted, 'success_top25'].mean():.1%} vs. "
          f"{analytic['success_top25'].mean():.1%} insgesamt)")
    me = info["markteintritt"]
    print(f"Frueherer Markteintritt (EA-Absolventen-Proxy), Analysestichprobe: {me['gesamt']:,} "
          f"(AppID {me['appid']:,}, Preishistorie {me['preishistorie']:,}, beide {me['beide']:,}; "
          f"Referenzjahrgaenge {me['referenzjahre']}); davon release_shifted: "
          f"{int((analytic['ea_graduate_proxy'] & analytic['release_shifted']).sum()):,}")
    print("\nQuellenabdeckung:")
    coverage_cols = [c for c in ["Quelle", "N_gematcht_analytisch", "Match_Rate_analytisch"] if c in coverage.columns]
    print((coverage[coverage_cols] if coverage_cols else coverage).to_string(index=False))
    print("\nErfolgsdefinitionen:")
    print(thresholds[["Kennung", "Schwelle", "Basis", "N_Basis", "Anteil"]].to_string(index=False))
    print()
    print(flow[["Schritt", "N", "Entfernt_gegenueber_Vorzeile"]].to_string(index=False))


if __name__ == "__main__":
    main()

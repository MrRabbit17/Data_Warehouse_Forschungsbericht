"""Laden, Bereinigen und Matchen der externen Datenquellen.

Verwendete Quellen (Kaggle-Handle, Version, Datei, Bytes und SHA-256 stehen
zentral in ``common.SOURCES``; ``common.fetch_source`` laedt und prueft sie):

- IGDB-Spielemetadaten (emirshn/igdb-dataset-for-data-mining-projects,
  game_dataset_cleaned.csv): Momentaufnahme Juli 2025 (Kaggle-Version 1 vom
  26.07.2025). Spaeter erschienene Spiele fehlen weitgehend.
- Metacritic-Spielebewertungen (stadynromn/metacritic-video-games-reviews-up-to-sep-2025,
  metacritic_dataset_raw.csv): Bewertungen bis Anfang Oktober 2025. Die
  clean-Datei der Quelle ist ein Complete-Case-Auszug, der auch einen
  Nutzerscore verlangt; sie wird deshalb nicht verwendet.
- Steam-Preishistorie (andrii222/steam-price-history-nov-2023-oct-2025,
  steam_price_history.csv): rollierende Fenster von bis zu 24 Monaten zwischen
  Nov. 2023 und Okt. 2025, kuerzer bei Release im Zeitraum. Die Quelle enthaelt
  keine kostenlosen Titel.

IGDB und Metacritic werden ueber den normalisierten Spielnamen in Stufen
angebunden (Spalten ``igdb_match_stage`` und ``mc_match_stage``):

1. Name und Jahr exakt.
2. Nur fuer noch ungematchte Steam-Spiele und unbenutzte Quelleintraege, deren
   Name auf beiden Seiten eindeutig ist: IGDB mit |Steam-Jahr - IGDB-Jahr| <= 1,
   Metacritic mit Steam-Jahr >= MC-Jahr - 1. Metacritic fuehrt je Spiel nur das
   Erstveroeffentlichungsdatum ueber alle Plattformen, spaete PC-Portierungen
   haben daher ein spaeteres Steam-Jahr.
3. Nur IGDB: undatierte IGDB-Eintraege mit beidseitig eindeutigem Namen; geben
   beide Seiten Entwickler/Publisher an, muessen sich diese ueberschneiden.

Jeder Quelleintrag wird hoechstens einer AppID zugeordnet. Die Eindeutigkeit des
Namens wird auf Steam-Seite ueber alle uebergebenen Spiele geprueft, nicht nur
ueber die Analysestichprobe. Die Preishistorie wird exakt ueber die Steam-AppID
verknuepft.

``igdb_first_release_year`` ist das Jahr des datierten IGDB-Treffers; fuer Spiele
ohne datierten Treffer wird es zusaetzlich per Namens-Lookup ermittelt (Name
beidseitig eindeutig, Firmen nicht widerspruechlich, Spalte
``igdb_year_from_lookup``). Nur so kann ein IGDB-Jahr deutlich vor dem
Steam-Jahr liegen (Neuauflagen, Portierungen, fruehere Veroeffentlichung
ausserhalb von Steam oder im Early Access).

Alle Zeilen, die verworfen, zusammengelegt oder auf NaN gesetzt werden, werden
gezaehlt und als ``outputs/tables/source_counts.csv`` ausgegeben.
"""

from __future__ import annotations

import ast
import re
import unicodedata
from collections.abc import Iterable

import numpy as np
import pandas as pd

from common import (
    DATA,
    PROVENANCE,
    SOURCES,
    fetch_source,
    has_non_latin_letters,
    normalize_name,
    record_rows,
    save_table,
)

# Exakte Bezeichnungen aus der IGDB-Rohdatei. load_igdb prueft, dass jede
# Bezeichnung dort vorkommt, damit ein Tippfehler nicht still ein Nullmerkmal
# erzeugt.
IGDB_GENRES = [
    "Adventure", "Arcade", "Card & Board Game", "Fighting",
    "Hack and slash/Beat 'em up", "Indie", "MOBA", "Music", "Pinball", "Platform",
    "Point-and-click", "Puzzle", "Quiz/Trivia", "Racing",
    "Real Time Strategy (RTS)", "Role-playing (RPG)", "Shooter", "Simulator",
    "Sport", "Strategy", "Tactical", "Turn-based strategy (TBS)", "Visual Novel",
]
MODE_SINGLE = "Single player"
MODE_MULTI = "Multiplayer"
MODE_COOP = "Co-operative"
MODE_SPLIT = "Split screen"
MODE_MMO = "Massively Multiplayer Online (MMO)"
MODE_BR = "Battle Royale"
IGDB_MODES = [MODE_SINGLE, MODE_MULTI, MODE_COOP, MODE_SPLIT, MODE_MMO, MODE_BR]
# Mehrspieler wie bei Steam: jede Form gemeinsamen Spielens zaehlt.
IGDB_MULTIPLAYER_MODES = {MODE_MULTI, MODE_COOP, MODE_SPLIT, MODE_MMO, MODE_BR}
IGDB_COOP_MODES = {MODE_COOP, MODE_SPLIT}
# Plattformen, unter denen ein Steam-Spiel laufen kann.
PC_LIKE_PLATFORMS = {"PC (Microsoft Windows)", "Mac", "Linux", "SteamVR"}

# Upload-Datum der IGDB-Momentaufnahme (Datenstand-Text in common.SOURCES).
IGDB_SNAPSHOT = pd.Timestamp("2025-07-26")

# Namen kuerzer als diese Grenzen (nach Normalisierung) sind zu unspezifisch.
MIN_NAME_LEN_EXACT = 2
MIN_NAME_LEN_FUZZY = 4
# Zulaessige Differenz Steam-Jahr minus Quelljahr in Stufe 2 (None = offen).
IGDB_YEAR_WINDOW = (-1, 1)
MC_YEAR_WINDOW = (-1, None)
# Ab diesem Abstand (Steam-Jahr minus MC-Jahr) wird ein Stufe-2-Treffer nur mit
# gemeinsamem Entwickler/Publisher oder identischem Steam-Metacritic-Score
# akzeptiert (Stichprobenpruefung: Fehlzuordnungen fast nur bei grossem Abstand).
MC_CHECK_GAP = 2

# Preishistorie: der Monat 2025-11 deckt nur die ersten Novembertage ab
# (mittlere Rabatttage 0,9 statt etwa 6) und wird verworfen. 2023-11 ist
# vollstaendig; dass dort weniger Apps vorkommen, folgt aus den versetzten
# 24-Monats-Fenstern.
PH_FIRST_MONTH = pd.Timestamp("2023-11-01")
PH_LAST_COMPLETE_MONTH = pd.Timestamp("2025-10-01")
# Volles Fenster: Beobachtung ab spaetestens Dez. 2023 bis Okt. 2025. Die
# Quelle hat zwei um einen Monat versetzte Fenster (Nov. 2023-Okt. 2025 und
# Dez. 2023-Nov. 2025); nach Entfernen von 2025-11 sind das 24 bzw. 23 Monate.
PH_FULL_WINDOW_START = pd.Timestamp("2023-12-01")
PH_EXPECTED_APPS = 651
PH_PRICE_FIELDS = ["base_price", "lowest_price", "price_changes", "days_on_sale"]

_COUNTS: list[dict] = []
# Name/Jahr-Schluessel der IGDB-Eintraege, die nur Nicht-PC-Plattformen haben
# (fuer die Zaehlung der dadurch verlorenen Treffer).
_IGDB_NON_PC_KEYS: set[tuple[str, float]] = set()

# Woerter, die in Firmennamen nichts ueber die Identitaet aussagen.
_COMPANY_STOP = {
    "a", "s", "ab", "and", "as", "bv", "co", "company", "corp", "corporation", "development",
    "developer", "digital", "entertainment", "game", "games", "global", "gmbh", "group", "inc",
    "incorporated", "interactive", "international", "kk", "lab", "labs", "limited", "llc",
    "ltd", "media", "of", "online", "oy", "plc", "production", "productions", "pty",
    "publishing", "sa", "sas", "sl", "software", "srl", "studio", "studios", "systems", "team",
    "technologies", "the", "works", "america", "europe", "japan", "usa", "uk",
}


# --------------------------------------------------------------------------
# Hilfsfunktionen
# --------------------------------------------------------------------------

def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _count(quelle: str, schritt: str, n: int, n_analytic: int | None = None,
           hinweis: str = "") -> None:
    """Protokolliert einen Bereinigungs- oder Matching-Schritt."""
    row = {
        "Quelle": quelle,
        "Schritt": schritt,
        "N": int(n),
        "N_analytisch": np.nan if n_analytic is None else int(n_analytic),
        "Hinweis": hinweis,
    }
    _COUNTS.append(row)
    extra = "" if n_analytic is None else f" (Analysestichprobe: {int(n_analytic):,})"
    print(f"[{quelle}] {schritt}: {int(n):,}{extra}")


def _version(key: str) -> object:
    """Tatsaechlich geladene Version laut Provenienzprotokoll (sonst Sollwert)."""
    for row in PROVENANCE:
        if row.get("Quelle") == key and row.get("Version") is not None:
            return row["Version"]
    return SOURCES[key]["version"]


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
    return series.map(normalize_name).fillna("")


def _company_keys(text: object) -> frozenset[str]:
    """Vergleichsschluessel fuer Entwickler/Publisher aus ';'/','-getrenntem Text.

    Je Firma der zusammengezogene Name ohne Rechtsform- und Fuellwoerter sowie
    dessen markante Einzelwoerter, damit z. B. "Bend Studio" und "Sony Bend"
    oder "BANDAI NAMCO Entertainment" und "Namco" uebereinstimmen.
    """
    keys = set()
    for part in re.split(r"[;,]", text if isinstance(text, str) else ""):
        tokens = re.findall(r"[^\W_]+", unicodedata.normalize("NFKC", part).casefold())
        core = [t for t in tokens if t not in _COMPANY_STOP]
        joined = "".join(core)
        if len(joined) >= 3:
            keys.add(joined)
        keys.update(t for t in core if len(t) >= 4)
    return frozenset(keys)


def _company_overlap(left: frozenset[str], right: frozenset[str]) -> bool:
    for a in left:
        for b in right:
            if a == b or a.startswith(b) or b.startswith(a):
                return True
    return False


def _flag(lists: pd.Series, labels: set[str]) -> pd.Series:
    """Indikator mit NA, wenn die Liste leer ist (unbekannt statt nein)."""
    values = [pd.NA if not xs else bool(labels.intersection(xs)) for xs in lists]
    return pd.Series(values, index=lists.index, dtype="boolean")


def _check_labels(lists: pd.Series, labels: Iterable[str], what: str) -> None:
    present = set()
    for xs in lists:
        present.update(xs)
    missing = sorted(set(labels) - present)
    if missing:
        raise AssertionError(f"IGDB-{what} nicht in den Daten gefunden: {missing}")


def _dedupe_name_year(df: pd.DataFrame, year_col: str, order: list[str],
                      ascending: list[bool]) -> tuple[pd.DataFrame, int]:
    """Ein Eintrag je (Name, Jahr). Undatierte Eintraege bleiben einzeln erhalten,
    damit die Eindeutigkeitspruefung des Namens sie mitzaehlt."""
    dated = df[df[year_col].notna()].sort_values(["_norm", year_col] + order,
                                                 ascending=[True, True] + ascending)
    kept = dated.drop_duplicates(["_norm", year_col])
    undated = df[df[year_col].isna()]
    out = pd.concat([kept, undated], ignore_index=True)
    return out, len(dated) - len(kept)


# --------------------------------------------------------------------------
# IGDB
# --------------------------------------------------------------------------

def load_igdb() -> pd.DataFrame:
    """IGDB laden, bereinigen und je normalisiertem Namen/Jahr deduplizieren."""
    path = fetch_source("igdb")
    cols = [
        "id", "name", "category", "release_date", "rating", "aggregated_rating",
        "genres", "themes", "franchise", "main_developers", "publishers",
        "platforms", "game_modes", "player_perspectives",
    ]
    df = pd.read_csv(path, usecols=cols, keep_default_na=False, na_values=[""])
    record_rows("igdb", len(df))
    _count("IGDB", "Rohzeilen", len(df))

    for col in ["rating", "aggregated_rating"]:
        raw = df[col]
        df[col] = pd.to_numeric(raw, errors="coerce")
        bad = int((raw.notna() & df[col].isna()).sum())
        if bad:
            _count("IGDB", f"{col}: nicht numerisch -> NaN", bad)
    dates = pd.to_datetime(df["release_date"], errors="coerce", format="%Y-%m-%d")
    bad_dates = int((df["release_date"].notna() & dates.isna()).sum())
    if bad_dates:
        _count("IGDB", "release_date nicht lesbar -> undatiert", bad_dates)
    df["igdb_release_year"] = dates.dt.year.astype(float)

    for col in ["genres", "themes", "platforms", "game_modes", "player_perspectives",
                "franchise", "main_developers", "publishers"]:
        df[col] = df[col].apply(parse_list)
    _check_labels(df["genres"], IGDB_GENRES, "Genres")
    _check_labels(df["game_modes"], IGDB_MODES, "Spielmodi")
    _check_labels(df["platforms"], PC_LIKE_PLATFORMS, "Plattformen")

    # Kritiker- und Nutzerwert 0 sind Platzhalter (naechstkleinerer echter Wert 8 bzw. 10).
    for col, label in [("aggregated_rating", "Kritikerbewertung"), ("rating", "Nutzerbewertung")]:
        zero = df[col] == 0
        _count("IGDB", f"{label} = 0 (Platzhalter) -> NaN", int(zero.sum()))
        df.loc[zero, col] = np.nan

    df["_norm"] = _normalise_name(df["name"])
    short = df["_norm"].str.len() < MIN_NAME_LEN_EXACT
    _count("IGDB", f"ohne verwertbaren Namen (< {MIN_NAME_LEN_EXACT} Zeichen nach Normalisierung) entfernt",
           int(short.sum()))
    df = df[~short].copy()

    df["igdb_pc_like"] = _flag(df["platforms"], PC_LIKE_PLATFORMS)
    non_pc = df["igdb_pc_like"].eq(False).fillna(False).astype(bool)
    global _IGDB_NON_PC_KEYS
    _IGDB_NON_PC_KEYS = set(
        zip(df.loc[non_pc & df["igdb_release_year"].notna(), "_norm"],
            df.loc[non_pc & df["igdb_release_year"].notna(), "igdb_release_year"])
    )
    _count("IGDB", "Plattformangabe ohne PC-nahe Plattform (Windows/Mac/Linux/SteamVR) entfernt",
           int(non_pc.sum()))
    df = df[~non_pc].copy()

    df["_main"] = (df["category"] == "main_game").astype(int)
    df["_pc"] = df["igdb_pc_like"].fillna(False).astype(int)
    df["_quality"] = (
        df["rating"].notna().astype(int) * 4
        + df["aggregated_rating"].notna().astype(int) * 3
        + df["genres"].apply(len).clip(upper=3)
        + df["game_modes"].apply(len).clip(upper=2)
        + df["platforms"].apply(len).clip(upper=2)
    )
    df, n_dupes = _dedupe_name_year(
        df, "igdb_release_year",
        order=["_main", "_pc", "_quality", "rating", "id"],
        ascending=[False, False, False, False, True],
    )
    _count("IGDB", "Name/Jahr-Dubletten entfernt (Hauptspiel und PC-Plattform bevorzugt)", n_dupes)
    _count("IGDB", "Einträge für das Matching", len(df))
    _count("IGDB", "davon ohne Veröffentlichungsdatum (nur Stufe 3 möglich)",
           int(df["igdb_release_year"].isna().sum()))
    _count("IGDB", "davon Kritikerbewertung vorhanden", int(df["aggregated_rating"].notna().sum()))
    _count("IGDB", "davon Nutzerbewertung vorhanden", int(df["rating"].notna().sum()))

    for genre in IGDB_GENRES:
        df["igdb_genre_" + _slug(genre)] = _flag(df["genres"], {genre})
    df["igdb_has_genres"] = df["genres"].apply(len) > 0
    df["igdb_has_modes"] = df["game_modes"].apply(len) > 0
    df["igdb_singleplayer"] = _flag(df["game_modes"], {MODE_SINGLE})
    df["igdb_multiplayer"] = _flag(df["game_modes"], IGDB_MULTIPLAYER_MODES)
    df["igdb_coop"] = _flag(df["game_modes"], IGDB_COOP_MODES)
    df["igdb_mmo"] = _flag(df["game_modes"], {MODE_MMO})
    df["igdb_n_genres"] = df["genres"].apply(lambda xs: len(set(xs)) if xs else np.nan)
    df["igdb_n_platforms"] = df["platforms"].apply(lambda xs: len(set(xs)) if xs else np.nan)
    # IGDB liefert keine Stimmenzahl. Ganzzahlige Nutzerwerte (v. a. 50/60/70/80)
    # entstehen typischerweise aus sehr wenigen Stimmen.
    df["igdb_rating_integer"] = pd.Series(
        np.where(df["rating"].notna(), df["rating"].mod(1).eq(0), pd.NA),
        index=df.index, dtype="boolean",
    )
    df["igdb_first_release_year"] = df["igdb_release_year"]
    df["_company_text"] = [
        "; ".join(list(d) + list(p)) for d, p in zip(df["main_developers"], df["publishers"])
    ]

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
    keep = ["_norm", "igdb_release_year", "_company_text"] + list(rename.values()) + [
        "igdb_first_release_year", "igdb_has_modes", "igdb_singleplayer", "igdb_multiplayer",
        "igdb_coop", "igdb_mmo", "igdb_pc_like", "igdb_has_genres", "igdb_n_genres",
        "igdb_n_platforms", "igdb_rating_integer",
    ] + ["igdb_genre_" + _slug(g) for g in IGDB_GENRES]
    return df[keep].reset_index(drop=True)


# --------------------------------------------------------------------------
# Metacritic
# --------------------------------------------------------------------------

def load_metacritic() -> pd.DataFrame:
    """PC-Eintraege der Metacritic-Rohdatei laden und je Name/Jahr deduplizieren."""
    path = fetch_source("metacritic_raw")
    df = pd.read_csv(path, keep_default_na=False, na_values=[""])
    record_rows("metacritic_raw", len(df))
    _count("Metacritic", "Rohzeilen (Plattform-Einträge)", len(df))

    df = df[df["platform"].astype(str).str.strip().str.fullmatch("PC", case=False, na=False)].copy()
    _count("Metacritic", "PC-Einträge", len(df))
    n_before = len(df)
    df = df.drop_duplicates()
    _count("Metacritic", "exakte Dublettenzeilen der Quelle entfernt", n_before - len(df))

    for col, label in [("metascore", "Metascore"), ("user_score", "Nutzerscore")]:
        raw = df[col]
        df[col] = pd.to_numeric(raw, errors="coerce")
        tbd = raw.astype(str).str.strip().str.lower().eq("tbd")
        other = raw.notna() & df[col].isna() & ~tbd
        _count("Metacritic", f"{label} 'tbd' -> NaN", int(tbd.sum()))
        if other.any():
            _count("Metacritic", f"{label} sonst nicht numerisch -> NaN", int(other.sum()))
        _count("Metacritic", f"{label} leer", int(raw.isna().sum()))

    dates = pd.to_datetime(df["release_date"], errors="coerce", format="%Y-%m-%d")
    df["mc_release_year"] = dates.dt.year.astype(float)

    df["_norm"] = _normalise_name(df["name"])
    short = df["_norm"].str.len() < MIN_NAME_LEN_EXACT
    _count("Metacritic", f"ohne verwertbaren Namen (< {MIN_NAME_LEN_EXACT} Zeichen) entfernt",
           int(short.sum()))
    df = df[~short].copy()

    df["_has_meta"] = df["metascore"].notna().astype(int)
    df["_has_user"] = df["user_score"].notna().astype(int)
    df, n_dupes = _dedupe_name_year(
        df, "mc_release_year",
        order=["_has_meta", "_has_user", "metascore", "user_score"],
        ascending=[False, False, False, False],
    )
    _count("Metacritic", "Name/Jahr-Dubletten entfernt (Eintrag mit Metascore bevorzugt)", n_dupes)
    _count("Metacritic", "Einträge für das Matching", len(df))
    # Undatierte Eintraege bleiben enthalten (sie zaehlen bei der Eindeutigkeit des
    # Namens mit), koennen aber nicht gematcht werden: Beide Stufen brauchen ein Jahr.
    _count("Metacritic", "davon ohne lesbares Veröffentlichungsdatum (bleiben enthalten, nicht matchbar)",
           int(df["mc_release_year"].isna().sum()))
    _count("Metacritic", "davon mit Metascore", int(df["metascore"].notna().sum()))
    _count("Metacritic", "davon mit Nutzerscore", int(df["user_score"].notna().sum()))

    df["_company_text"] = [
        "; ".join(([d] if isinstance(d, str) else []) + parse_list(p))
        for d, p in zip(df["developer"], df["publisher"])
    ]
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
    df["_src"] = np.arange(len(df))
    return df[["_src", "_norm", "mc_release_year", "_company_text"] + list(rename.values())].copy()


# --------------------------------------------------------------------------
# Preishistorie
# --------------------------------------------------------------------------

def load_price_history() -> pd.DataFrame:
    """Monatliche Preishistorie je App zu normierten Merkmalen verdichten.

    Merkmale (alle je Beobachtungsmonat normiert, weil die Fenster 3 bis 24
    Monate lang sind und Summen vor allem die Fensterlaenge messen):
    - ph_months, ph_full_window: Anzahl Monate; volles Fenster (Beginn Nov. oder Dez. 2023 bis
      Okt. 2025 vollstaendig beobachtet).
    - ph_price_changes_per_month, ph_days_on_sale_per_month,
      ph_share_months_on_sale: Preisaenderungen bzw. Rabatttage je Monat und
      Anteil der Monate mit mindestens einem Rabatttag.
    - ph_max_discount_pct: hoechster Rabatt im Fenster (Tiefstpreis gegenueber dem
      Basispreis); haengt weiter von der Fensterlaenge ab und ist nur bei
      vollem Fenster vergleichbar.
    - ph_mean_monthly_max_discount_pct: Mittel der monatlichen Hoechstrabatte
      (aus dem monatlichen Tiefstpreis; Monate ohne Rabatt zaehlen mit 0);
      kein mittlerer Rabatt je Verkaufstag.
    - ph_lowest_above_base: base_price ist in der Quelle je App statisch und
      folgt keinen Listenpreiserhoehungen. Liegt lowest_price darueber, ist der
      Rabatt des Monats nicht bestimmbar (NaN), und die Quelle zaehlt dort keine
      Rabatttage. Solche Apps sind markiert (fuer Sensitivitaetsanalysen).
    """
    path = fetch_source("price_history")
    raw = pd.read_csv(path)
    record_rows("price_history", len(raw))
    n_apps = raw["app_id"].nunique()
    if n_apps != PH_EXPECTED_APPS:
        raise ValueError(f"Preishistorie: {n_apps} Apps statt {PH_EXPECTED_APPS}")
    _count("Preishistorie", "Rohzeilen (App-Monate)", len(raw))
    _count("Preishistorie", "Apps in der Rohdatei", n_apps)

    df = raw[["app_id", "month"] + PH_PRICE_FIELDS].copy()
    for col in ["app_id"] + PH_PRICE_FIELDS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["month"] = pd.to_datetime(df["month"], errors="coerce", format="%Y-%m-%d")
    invalid = df.isna().any(axis=1)
    _count("Preishistorie", "Zeilen mit fehlenden/ungültigen Werten entfernt", int(invalid.sum()))
    df = df[~invalid].copy()
    df["app_id"] = df["app_id"].astype(int)

    # Fuer 8 AppIDs fuehrt die Quelle jeden Monat mehrfach (Editionen/Bundles
    # unter der Basis-AppID, bei 57650 sogar fremde Spielnamen). Die Preisfelder
    # sind identisch; Summieren wuerde die Preisaenderungen vervielfachen.
    key = ["app_id", "month"]
    dup = df.duplicated(key, keep=False)
    conflicts = int(df[dup].groupby(key)[PH_PRICE_FIELDS].nunique().gt(1).any(axis=1).sum())
    _count("Preishistorie", "doppelte (app_id, month)-Gruppen", int(df[dup].groupby(key).ngroups),
           hinweis=f"{df.loc[dup, 'app_id'].nunique()} Apps; Gruppen mit abweichenden Preisfeldern: {conflicts}")
    if conflicts:
        raise ValueError(f"Preishistorie: {conflicts} doppelte App-Monate mit abweichenden Preisfeldern")
    n_before = len(df)
    df = df.drop_duplicates(key)
    _count("Preishistorie", "doppelte App-Monatszeilen entfernt (nicht summiert)", n_before - len(df))

    late = df["month"] > PH_LAST_COMPLETE_MONTH
    _count("Preishistorie", "unvollständiger Monat 2025-11 entfernt (Zeilen)", int(late.sum()),
           hinweis=f"{df.loc[late, 'app_id'].nunique()} Apps betroffen")
    early = df["month"] < PH_FIRST_MONTH
    if early.any():
        _count("Preishistorie", "Monate vor 2023-11 entfernt", int(early.sum()))
    df = df[~late & ~early].copy()
    _count("Preishistorie", "Zeilen nach Bereinigung", len(df))

    if (df["base_price"] <= 0).any():
        raise ValueError("Preishistorie: Basispreis <= 0 (kostenlose Titel) nicht erwartet")
    n_varying = int((df.groupby("app_id")["base_price"].nunique() > 1).sum())
    if n_varying:
        _count("Preishistorie", "Apps mit wechselndem Basispreis (Median verwendet)", n_varying)

    df["_above"] = df["lowest_price"] > df["base_price"]
    df["discount_pct"] = np.where(
        df["_above"], np.nan, (df["base_price"] - df["lowest_price"]) / df["base_price"] * 100.0
    )
    _count("Preishistorie", "Monate mit lowest_price > base_price (Rabatt -> NaN)",
           int(df["_above"].sum()), hinweis=f"{df.loc[df['_above'], 'app_id'].nunique()} Apps")

    agg = df.groupby("app_id").agg(
        ph_months=("month", "nunique"),
        ph_first_month=("month", "min"),
        ph_last_month=("month", "max"),
        ph_base_price=("base_price", "median"),
        ph_lowest_price=("lowest_price", "min"),
        ph_max_discount_pct=("discount_pct", "max"),
        ph_mean_monthly_max_discount_pct=("discount_pct", "mean"),
        _changes=("price_changes", "sum"),
        _days=("days_on_sale", "sum"),
        _sale_months=("days_on_sale", lambda s: int((s > 0).sum())),
        ph_months_lowest_above_base=("_above", "sum"),
    )
    span = ((agg["ph_last_month"].dt.year - agg["ph_first_month"].dt.year) * 12
            + agg["ph_last_month"].dt.month - agg["ph_first_month"].dt.month + 1)
    n_gaps = int((span != agg["ph_months"]).sum())
    if n_gaps:
        _count("Preishistorie", "Apps mit Lücken im Fenster", n_gaps)
    lost = n_apps - len(agg)
    if lost:
        _count("Preishistorie", "Apps ohne verbleibende Monate entfernt", lost)

    agg["ph_full_window"] = (agg["ph_first_month"] <= PH_FULL_WINDOW_START) & (
        agg["ph_last_month"] >= PH_LAST_COMPLETE_MONTH
    )
    agg["ph_price_changes_per_month"] = agg["_changes"] / agg["ph_months"]
    agg["ph_days_on_sale_per_month"] = agg["_days"] / agg["ph_months"]
    agg["ph_share_months_on_sale"] = agg["_sale_months"] / agg["ph_months"]
    agg["ph_has_sale"] = agg["_sale_months"] > 0
    agg["ph_months_lowest_above_base"] = agg["ph_months_lowest_above_base"].astype(int)
    agg["ph_lowest_above_base"] = agg["ph_months_lowest_above_base"] > 0
    agg = agg.drop(columns=["_changes", "_days", "_sale_months"]).reset_index()

    _count("Preishistorie", "Apps für das Matching", len(agg))
    _count("Preishistorie", "davon volles Fenster (Beginn Nov. oder Dez. 2023 bis Okt. 2025, 24 bzw. 23 Monate)",
           int(agg["ph_full_window"].sum()))
    _count("Preishistorie", "davon kürzeres Fenster (Release im Zeitraum)",
           int((~agg["ph_full_window"]).sum()))
    _count("Preishistorie", "davon mit lowest_price > base_price in mind. einem Monat",
           int(agg["ph_lowest_above_base"].sum()))
    return agg.rename(columns={"app_id": "AppID_int"})


# --------------------------------------------------------------------------
# Matching
# --------------------------------------------------------------------------

def _match_names(steam: pd.DataFrame, src: pd.DataFrame, src_year: str,
                 year_window: tuple[int | None, int | None], undated_stage: bool,
                 label: str, check_gap: int | None = None,
                 scores: tuple[str, str] | None = None) -> pd.DataFrame:
    """Ordnet Quelleintraege (Spalte _src) Steam-AppIDs in bis zu drei Stufen zu.

    steam: AppID, _norm, _year, _norm_unique, _company_text, analytic_sample
    src:   _src, _norm, <src_year>, _company_text (je (Name, Jahr) hoechstens
           ein datierter Eintrag)
    check_gap: ab dieser Jahresdifferenz (Steam minus Quelle) verlangt Stufe 2
           einen gemeinsamen Entwickler/Publisher, sofern beide Seiten Firmen
           angeben; ersatzweise genuegt ein identischer Score (``scores`` =
           Steam-Spalte, Quellspalte), d. h. der Steam-Store verweist auf
           denselben Eintrag.
    """
    analytic = steam.set_index("AppID")["analytic_sample"]
    src = src.copy()
    src["_norm_unique"] = src.groupby("_norm")["_norm"].transform("size") == 1
    links = []

    # Stufe 1: Name + Jahr exakt. Teilen sich mehrere AppIDs denselben
    # Schluessel, entscheidet ein gemeinsamer Entwickler/Publisher, danach die
    # Zugehoerigkeit zur Analysestichprobe; bleibt es offen, wird nicht gematcht.
    s1 = steam[(steam["_norm"].str.len() >= MIN_NAME_LEN_EXACT) & steam["_year"].notna()]
    cand = s1[["AppID", "_norm", "_year", "analytic_sample", "_company_text"]].merge(
        src.loc[src[src_year].notna(), ["_src", "_norm", src_year, "_company_text"]],
        left_on=["_norm", "_year"], right_on=["_norm", src_year], suffixes=("", "_q"),
    )
    size = cand.groupby("_src")["AppID"].transform("size")
    multi = cand[size > 1].copy()
    resolved = cand[size == 1]
    n_dev = n_analytic = n_ambiguous = 0
    if not multi.empty:
        multi["_dev"] = [
            _company_overlap(_company_keys(a), _company_keys(b))
            for a, b in zip(multi["_company_text"], multi["_company_text_q"])
        ]
        multi["_score"] = multi["_dev"].astype(int) * 2 + multi["analytic_sample"].astype(int)
        multi["_n_dev"] = multi.groupby("_src")["_dev"].transform("sum")
        top = multi[multi["_score"] == multi.groupby("_src")["_score"].transform("max")]
        n_top = top.groupby("_src")["AppID"].transform("size")
        winners = top[n_top == 1]
        # Ueber die Firma entschieden ist ein Fall nur, wenn genau ein Kandidat
        # eine Firmenueberlappung hat; teilen sich mehrere diese Ueberlappung,
        # entscheidet erst die Zugehoerigkeit zur Analysestichprobe.
        n_dev = int((winners["_dev"] & (winners["_n_dev"] == 1)).sum())
        n_analytic = int(len(winners) - n_dev)
        n_ambiguous = int(top.loc[n_top > 1, "_src"].nunique())
        resolved = pd.concat([resolved, winners], ignore_index=True)
    links.append(resolved[["AppID", "_src"]].assign(match_stage=1))
    _count(label, "Stufe 1: Quelleinträge mit mehreren AppIDs (gleicher Name und Jahr)",
           int(multi["_src"].nunique()) if not multi.empty else 0,
           hinweis=(f"entschieden über Entwickler/Publisher: {n_dev}, über Analysestichprobe: "
                    f"{n_analytic}, mehrdeutig und nicht gematcht: {n_ambiguous}"))

    def remaining() -> tuple[pd.DataFrame, pd.DataFrame]:
        done = pd.concat(links, ignore_index=True)
        st = steam[~steam["AppID"].isin(done["AppID"]) & steam["_norm_unique"]
                   & (steam["_norm"].str.len() >= MIN_NAME_LEN_FUZZY)]
        sr = src[~src["_src"].isin(done["_src"]) & src["_norm_unique"]]
        return st, sr

    # Stufe 2: beidseitig eindeutiger Name, Jahrestoleranz.
    st, sr = remaining()
    steam_extra = [scores[0]] if scores else []
    src_extra = [scores[1]] if scores else []
    cand = st[["AppID", "_norm", "_year", "_company_text"] + steam_extra].merge(
        sr.loc[sr[src_year].notna(), ["_src", "_norm", src_year, "_company_text"] + src_extra],
        on="_norm", suffixes=("", "_q"),
    )
    diff = cand["_year"] - cand[src_year]
    low, high = year_window
    ok = diff.notna()
    if low is not None:
        ok &= diff >= low
    if high is not None:
        ok &= diff <= high
    if check_gap is not None:
        # Grosse Jahresabstaende entstehen meist durch spaete Portierungen, aber
        # auch durch gleichnamige, voellig andere Spiele (z. B. ein Indie-Titel
        # 2025 und ein Klassiker von 2007). Dort muss ein Firmen- oder
        # Score-Abgleich die Identitaet stuetzen.
        gap = ok & (diff >= check_gap)
        conflict = []
        for idx in cand.index[gap]:
            left = _company_keys(cand.at[idx, "_company_text"])
            right = _company_keys(cand.at[idx, "_company_text_q"])
            same_score = bool(scores) and cand.at[idx, scores[0]] == cand.at[idx, scores[1]]
            conflict.append(bool(left and right) and not _company_overlap(left, right)
                            and not same_score)
        rejected = pd.Series(conflict, index=cand.index[gap], dtype=bool)
        rejected = rejected[rejected].index
        _count(label, f"Stufe 2: Kandidaten mit Jahresabstand >= {check_gap} geprüft",
               int(gap.sum()),
               hinweis=(f"verworfen (Entwickler/Publisher verschieden, Score nicht bestätigt): "
                        f"{len(rejected)}"))
        ok &= ~cand.index.isin(rejected)
    links.append(cand.loc[ok, ["AppID", "_src"]].assign(match_stage=2))

    # Stufe 3: undatierte Quelleintraege mit beidseitig eindeutigem Namen. Ohne
    # Jahr ist der Name allein schwach (z. B. "Storm" oder "Drive" sind ganz
    # verschiedene Spiele); geben beide Seiten Firmen an, muessen sie sich
    # ueberschneiden (wie beim Jahres-Lookup und bei Metacritic-Stufe 2).
    if undated_stage:
        st, sr = remaining()
        cand = st[["AppID", "_norm", "_company_text"]].merge(
            sr.loc[sr[src_year].isna(), ["_src", "_norm", "_company_text"]], on="_norm", suffixes=("", "_q")
        )
        left = cand["_company_text"].map(_company_keys)
        right = cand["_company_text_q"].map(_company_keys)
        both = (left.map(len) > 0) & (right.map(len) > 0)
        conflict = pd.Series([bool(a and b) and not _company_overlap(a, b) for a, b in zip(left, right)],
                             index=cand.index, dtype=bool)
        _count(label, "Stufe 3: Kandidaten mit Firmenangabe auf beiden Seiten geprüft", int(both.sum()),
               int(analytic.reindex(cand.loc[both, "AppID"]).sum()),
               hinweis=(f"verworfen (Entwickler/Publisher verschieden): {int(conflict.sum())} "
                        f"(Analysestichprobe {int(analytic.reindex(cand.loc[conflict, 'AppID']).sum())})"))
        links.append(cand.loc[~conflict, ["AppID", "_src"]].assign(match_stage=3))

    out = pd.concat(links, ignore_index=True)
    if out["AppID"].duplicated().any() or out["_src"].duplicated().any():
        raise AssertionError(f"{label}: Zuordnung ist nicht eins-zu-eins")
    for stage in sorted(out["match_stage"].unique()):
        ids = out.loc[out["match_stage"] == stage, "AppID"]
        _count(label, f"gematcht in Stufe {stage}", len(ids), int(analytic.reindex(ids).sum()))
    _count(label, "gematcht gesamt", len(out), int(analytic.reindex(out["AppID"]).sum()))

    # Plausibilitaet je Stufe: Anteil der Treffer, deren Entwickler/Publisher sich
    # widersprechen (nur wo beide Seiten Firmen angeben). Firmenwechsel und
    # Namensvarianten erzeugen auch bei richtigen Treffern einen Grundanteil.
    comp_steam = steam.set_index("AppID")["_company_text"]
    comp_src = src.set_index("_src")["_company_text"]
    for stage in sorted(out["match_stage"].unique()):
        part = out[out["match_stage"] == stage]
        left = part["AppID"].map(comp_steam).map(_company_keys)
        right = part["_src"].map(comp_src).map(_company_keys)
        both = ((left.map(len) > 0) & (right.map(len) > 0)).to_numpy()
        conflict = np.array([bool(a and b) and not _company_overlap(a, b) for a, b in zip(left, right)])
        in_an = analytic.reindex(part["AppID"]).fillna(False).astype(bool).to_numpy()
        n_both, n_both_an = int(both.sum()), int((both & in_an).sum())
        _count(label, f"Prüfung Stufe {stage}: Treffer mit Firmenangabe auf beiden Seiten, davon Firmen verschieden",
               int(conflict.sum()), int((conflict & in_an).sum()),
               hinweis=(f"von {n_both} ({conflict.sum() / max(n_both, 1):.1%}); Analysestichprobe von {n_both_an} "
                        f"({(conflict & in_an).sum() / max(n_both_an, 1):.1%})"))
    return out


def _igdb_year_lookup(base: pd.DataFrame, igdb: pd.DataFrame) -> pd.Series:
    """IGDB-Erstveroeffentlichungsjahr per Namens-Lookup fuer Spiele ohne datierten Treffer.

    Das Matching laesst hoechstens ein Jahr Abstand zu. Ein Spiel, das auf Steam
    deutlich spaeter erschien als laut IGDB (Neuauflage, Portierung, fruehere
    Veroeffentlichung ausserhalb von Steam oder im Early Access), bleibt deshalb
    ungematcht, und ohne Lookup koennte release_shifted nie zutreffen. Das Jahr
    wird uebernommen, wenn der normalisierte Name auf beiden Seiten eindeutig ist
    (mind. MIN_NAME_LEN_FUZZY Zeichen) und - sofern beide Seiten Firmen angeben -
    Entwickler/Publisher uebereinstimmen. Die uebrigen IGDB-Merkmale werden fuer
    diese Spiele nicht angehaengt (has_igdb bleibt False).

    Rueckgabe: Jahr je Zeile von ``base`` (NaN ohne Lookup-Treffer).
    """
    n_entries = igdb.groupby("_norm")["_norm"].transform("size")
    unique_dated = igdb.loc[(n_entries == 1) & igdb["igdb_release_year"].notna(),
                            ["_norm", "igdb_release_year", "_company_text"]]
    need = (base["igdb_first_release_year"].isna() & base["_norm_unique"]
            & (base["_norm"].str.len() >= MIN_NAME_LEN_FUZZY) & base["_year"].notna())
    cand = base.loc[need, ["_norm", "_year", "_company_text", "analytic_sample"]].reset_index().merge(
        unique_dated, on="_norm", how="inner", suffixes=("", "_q"),
    ).set_index("index")
    conflict = pd.Series(
        [
            bool(left and right) and not _company_overlap(left, right)
            for left, right in zip(cand["_company_text"].map(_company_keys),
                                   cand["_company_text_q"].map(_company_keys))
        ],
        index=cand.index, dtype=bool,
    )
    accepted = cand[~conflict]
    earlier = (accepted["_year"] - accepted["igdb_release_year"]) > 1
    _count("IGDB", "Erstveröffentlichungsjahr per Namens-Lookup (ohne Match, Name beidseitig eindeutig)",
           len(accepted), int(accepted["analytic_sample"].sum()),
           hinweis=(f"verworfen wegen abweichender Entwickler/Publisher: {int(conflict.sum())}; "
                    f"IGDB-Jahr mehr als 1 Jahr vor dem Steam-Jahr: {int(earlier.sum())} "
                    f"(Analysestichprobe {int((earlier & accepted['analytic_sample']).sum())})"))
    return accepted["igdb_release_year"].reindex(base.index)


def _metacritic_overlap(base: pd.DataFrame, mc: pd.DataFrame) -> pd.DataFrame:
    """Abdeckung und Wertuebereinstimmung FronkonGames-Metacritic vs. extern."""
    mc_names = set(mc["_norm"])
    rows = []
    for basis, mask in [("Analysestichprobe", base["analytic_sample"].astype(bool)),
                        ("alle aufbereiteten Spiele", pd.Series(True, index=base.index))]:
        d = base[mask]
        fg = d["metacritic"].notna()
        ext = d["mc_metascore"].notna()
        both = d[fg & ext]
        diff = (both["mc_metascore"] - both["metacritic"]).abs()
        only_fg = d[fg & ~ext]
        rows.append({
            "Basis": basis,
            "N": len(d),
            "N_FG": int(fg.sum()),
            "N_extern": int(ext.sum()),
            "N_beide": len(both),
            "N_beide_Stufe1": int((both["mc_match_stage"] == 1).sum()),
            "N_beide_Stufe2": int((both["mc_match_stage"] == 2).sum()),
            "N_nur_FG": len(only_fg),
            "N_nur_FG_Name_im_MC_PC_Teil": int(only_fg["_norm"].isin(mc_names).sum()),
            "N_nur_extern": int((~fg & ext).sum()),
            "N_exakt_gleich": int((diff == 0).sum()),
            "Anteil_exakt_gleich": float((diff == 0).mean()) if len(both) else np.nan,
            "Spearman_rho": both["mc_metascore"].corr(both["metacritic"], method="spearman")
            if len(both) > 2 else np.nan,
            "Median_abs_Diff": float(diff.median()) if len(both) else np.nan,
            "Max_abs_Diff": float(diff.max()) if len(both) else np.nan,
        })
    return pd.DataFrame(rows)


def attach_external_sources(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Matcht alle externen Quellen an den Basis-Datensatz.

    Erwartet die Spalten AppID, Name, release_year und analytic_sample fuer alle
    aufbereiteten Spiele (die Namenseindeutigkeit wird ueber alle geprueft);
    optional Developers/Publishers (Tie-Break bei gleichem Namen und Jahr,
    Firmenabgleich), metacritic (Score-Bestaetigung, Quellenvergleich) und
    release_date (Zaehlung nach IGDB-Stand). Schreibt external_features.parquet sowie die Tabellen
    source_coverage, source_overlap, source_counts und metacritic_overlap und
    gibt den erweiterten Datensatz sowie Coverage- und Overlap-Tabelle zurueck.
    """
    _COUNTS.clear()
    if df["AppID"].duplicated().any():
        raise ValueError("AppID ist im Basis-Datensatz nicht eindeutig")
    cols = ["AppID", "Name", "release_year", "analytic_sample"]
    cols += [c for c in ["Developers", "Publishers", "metacritic"] if c in df.columns]
    base = df[cols].copy()
    base["analytic_sample"] = base["analytic_sample"].fillna(False).astype(bool)
    base["_norm"] = _normalise_name(base["Name"])
    base["_year"] = pd.to_numeric(base["release_year"], errors="coerce").astype(float)
    norm_size = base.groupby("_norm")["_norm"].transform("size")
    base["_norm_unique"] = (base["_norm"] != "") & (norm_size == 1)
    base["_company_text"] = ""
    for col in ["Developers", "Publishers"]:
        if col in base.columns:
            base["_company_text"] = base["_company_text"] + "," + base[col].fillna("").astype(str)
    base["AppID_int"] = pd.to_numeric(base["AppID"], errors="coerce")
    n_analytic = int(base["analytic_sample"].sum())
    _count("FronkonGames", "Spiele im Basisdatensatz", len(base), n_analytic)
    _count("FronkonGames", f"Name nach Normalisierung kürzer als {MIN_NAME_LEN_EXACT} Zeichen (nicht matchbar)",
           int((base["_norm"].str.len() < MIN_NAME_LEN_EXACT).sum()),
           int((base["analytic_sample"] & (base["_norm"].str.len() < MIN_NAME_LEN_EXACT)).sum()))
    # Namen zwischen beiden Mindestlaengen koennen nur exakt (Stufe 1) gematcht werden.
    exact_only = base["_norm"].str.len().between(MIN_NAME_LEN_EXACT, MIN_NAME_LEN_FUZZY - 1)
    _count("FronkonGames", f"Name nach Normalisierung {MIN_NAME_LEN_EXACT} bis {MIN_NAME_LEN_FUZZY - 1} Zeichen "
                           "(nur Stufe 1 möglich)",
           int(exact_only.sum()), int((base["analytic_sample"] & exact_only).sum()))
    _count("FronkonGames", "Name über alle Spiele eindeutig (Voraussetzung für Stufe 2/3)",
           int(base["_norm_unique"].sum()), int((base["analytic_sample"] & base["_norm_unique"]).sum()))
    # Nicht-lateinische Schrift: im Rohnamen; davon Schluessel auf den lateinischen
    # Teil reduziert (gemischtschriftliche Namen) bzw. weiterhin nicht-lateinisch.
    raw_non_latin = base["Name"].map(has_non_latin_letters)
    non_latin = base["_norm"].map(
        lambda t: any(c.isalpha() and not unicodedata.name(c, "").startswith("LATIN") for c in t)
    )
    reduced_latin = raw_non_latin & ~non_latin

    igdb = load_igdb()
    mc = load_metacritic()
    price = load_price_history()

    steam_cols = ["AppID", "_norm", "_year", "_norm_unique", "_company_text", "analytic_sample"]
    igdb_links = _match_names(base[steam_cols], igdb.rename(columns={"igdb_id": "_src"}),
                              "igdb_release_year", IGDB_YEAR_WINDOW, True, "IGDB")
    matched = set(igdb_links["AppID"])
    lost = base[~base["AppID"].isin(matched)]
    lost_keys = pd.Series(list(zip(lost["_norm"], lost["_year"])), index=lost.index).isin(_IGDB_NON_PC_KEYS)
    _count("IGDB", "ungematcht, Name und Jahr nur bei ausgeschlossenen Nicht-PC-Einträgen",
           int(lost_keys.sum()), int(lost.loc[lost_keys, "analytic_sample"].sum()))
    is_matched = base["AppID"].isin(matched)
    an_flag = base["analytic_sample"]
    for mask, text in [
        (raw_non_latin, "Steam-Namen mit nicht-lateinischen Buchstaben (Rohname), davon gematcht"),
        (reduced_latin, "darunter Schlüssel auf den lateinischen Namensteil reduziert, davon gematcht"),
        (non_latin, "darunter Schlüssel mit nicht-lateinischen Buchstaben, davon gematcht"),
    ]:
        _count("IGDB", text, int((mask & is_matched).sum()), int((mask & is_matched & an_flag).sum()),
               hinweis=f"von {int(mask.sum())} (Analysestichprobe {int((mask & an_flag).sum())})")
    if "release_date" in df.columns:
        late = pd.to_datetime(df["release_date"], errors="coerce").gt(IGDB_SNAPSHOT).to_numpy()
        _count("IGDB", "Release nach dem IGDB-Stand 26.07.2025, davon gematcht",
               int((late & is_matched).sum()), int((late & is_matched & base["analytic_sample"]).sum()),
               hinweis=f"von {int(late.sum())} (Analysestichprobe {int((late & base['analytic_sample']).sum())})")
    if "metacritic" in base.columns:
        # Spiele mit Steam-Metacritic-Score sind praktisch sicher in IGDB erfasst;
        # ihr Trefferanteil schaetzt die Recall-Rate des Matchings.
        ref = base["metacritic"].notna()
        ref_a = ref & base["analytic_sample"]
        _count("IGDB", "Recall-Referenz: Spiele mit Steam-Metacritic-Score, davon gematcht",
               int((ref & is_matched).sum()), int((ref_a & is_matched).sum()),
               hinweis=(f"von {int(ref.sum())} ({(ref & is_matched).sum() / max(int(ref.sum()), 1):.1%}); "
                        f"Analysestichprobe von {int(ref_a.sum())} "
                        f"({(ref_a & is_matched).sum() / max(int(ref_a.sum()), 1):.1%})"))
    igdb_feat = igdb.drop(columns=["_norm", "igdb_release_year", "_company_text"])
    base = base.merge(igdb_links.rename(columns={"_src": "igdb_id", "match_stage": "igdb_match_stage"}),
                      on="AppID", how="left", validate="one_to_one")
    base = base.merge(igdb_feat, on="igdb_id", how="left", validate="many_to_one")
    # Erstveroeffentlichungsjahr: Jahr des datierten Treffers, sonst Namens-Lookup
    # (nur fuer release_year_alt/release_shifted in 01_prepare_data.py).
    lookup_year = _igdb_year_lookup(base, igdb)
    base["igdb_year_from_lookup"] = lookup_year.notna()
    base["igdb_first_release_year"] = base["igdb_first_release_year"].fillna(lookup_year)

    has_fg = "metacritic" in base.columns
    mc_links = _match_names(
        base[steam_cols + (["metacritic"] if has_fg else [])], mc, "mc_release_year",
        MC_YEAR_WINDOW, False, "Metacritic", check_gap=MC_CHECK_GAP,
        scores=("metacritic", "mc_metascore") if has_fg else None,
    )
    base = base.merge(mc_links.rename(columns={"match_stage": "mc_match_stage"}),
                      on="AppID", how="left", validate="one_to_one")
    base = base.merge(mc.drop(columns=["_norm", "_company_text"]), on="_src", how="left",
                      validate="many_to_one").drop(columns=["_src"])
    mc_with_score = base["mc_metascore"].notna()
    _count("Metacritic", "gematcht mit Metascore (has_mc_external)", int(mc_with_score.sum()),
           int((mc_with_score & base["analytic_sample"]).sum()))
    if has_fg:
        # Plausibilitaet je Stufe: Der Steam-Store verlinkt den Metacritic-Eintrag
        # des Spiels; ein abweichender Score deutet auf eine andere Version hin.
        for stage in sorted(base["mc_match_stage"].dropna().unique()):
            m = (base["mc_match_stage"] == stage) & mc_with_score & base["metacritic"].notna()
            same = m & (base["mc_metascore"] == base["metacritic"])
            m_an, same_an = m & base["analytic_sample"], same & base["analytic_sample"]
            _count("Metacritic", f"Prüfung Stufe {int(stage)}: Treffer mit Steam-Metacritic-Score, davon gleicher Metascore",
                   int(same.sum()), int(same_an.sum()),
                   hinweis=(f"von {int(m.sum())} ({same.sum() / max(int(m.sum()), 1):.1%}); Analysestichprobe von "
                            f"{int(m_an.sum())} ({same_an.sum() / max(int(m_an.sum()), 1):.1%})"))

    base = base.merge(price, how="left", on="AppID_int", validate="many_to_one")
    _count("Preishistorie", "gematcht über AppID", int(base["ph_months"].notna().sum()),
           int((base["ph_months"].notna() & base["analytic_sample"]).sum()))
    _count("Preishistorie", "gematcht mit vollem Fenster",
           int(base["ph_full_window"].eq(True).sum()),
           int((base["ph_full_window"].eq(True) & base["analytic_sample"]).sum()))

    for col in ["igdb_match_stage", "mc_match_stage"]:
        base[col] = base[col].astype("Int64")
    for col in ["ph_full_window", "ph_has_sale", "ph_lowest_above_base", "igdb_has_modes",
                "igdb_has_genres"]:
        base[col] = base[col].astype("boolean")
    base["has_igdb"] = base["igdb_id"].notna()
    base["has_mc_external"] = base["mc_metascore"].notna()
    base["has_price_history"] = base["ph_months"].notna()
    base["n_sources"] = (
        1 + base["has_igdb"].astype(int) + base["has_mc_external"].astype(int)
        + base["has_price_history"].astype(int)
    )

    if "metacritic" in base.columns:
        save_table(_metacritic_overlap(base, mc), "metacritic_overlap")
    else:
        print("Hinweis: Spalte 'metacritic' fehlt, metacritic_overlap.csv wird nicht geschrieben.")

    external_cols = [c for c in base.columns if c.startswith(("igdb_", "mc_", "ph_"))]
    flag_cols = ["has_igdb", "has_mc_external", "has_price_history", "n_sources"]
    base[["AppID"] + external_cols + flag_cols].to_parquet(DATA / "external_features.parquet", index=False)

    out = df.merge(base[["AppID"] + external_cols + flag_cols], how="left", on="AppID",
                   validate="one_to_one")

    def cov_row(quelle: str, key: str, n_src: int, has: pd.Series, schluessel: str,
                hinweis: str) -> dict:
        return {
            "Quelle": quelle,
            "Datensatz": SOURCES[key]["slug"],
            "Version": _version(key),
            "Datei": SOURCES[key]["file"],
            "Datenstand": SOURCES[key]["datenstand"],
            "N_Quelle": n_src,
            "N_gematcht_gesamt": int(has.sum()),
            "Match_Rate_gesamt": float(has.mean()),
            "N_gematcht_analytisch": int((has & base["analytic_sample"]).sum()),
            "Match_Rate_analytisch": (has & base["analytic_sample"]).sum() / n_analytic
            if n_analytic else np.nan,
            "Schluessel": schluessel,
            "Hinweis": hinweis,
        }

    def stage_note(col: str) -> str:
        return "Treffer in der Analysestichprobe je Stufe: " + ", ".join(
            f"Stufe {int(s)}: {int(((base[col] == s) & base['analytic_sample']).sum())}"
            for s in sorted(base[col].dropna().unique())
        )
    coverage = pd.DataFrame([
        cov_row("FronkonGames (Steam)", "fronkon", len(df), pd.Series(True, index=base.index),
                "AppID", "Basisdatensatz"),
        cov_row("IGDB", "igdb", len(igdb), base["has_igdb"],
                "normalisierter Name + Jahr (Stufe 1); eindeutiger Name mit |Jahresdifferenz| <= 1 "
                "(Stufe 2) bzw. ohne IGDB-Datum (Stufe 3)",
                "nur Einträge mit PC-naher oder fehlender Plattformangabe; ein Eintrag je Name/Jahr; "
                f"{stage_note('igdb_match_stage')}"),
        cov_row("Metacritic extern", "metacritic_raw", len(mc), base["has_mc_external"],
                "normalisierter Name + Jahr (Stufe 1); eindeutiger Name mit Steam-Jahr >= MC-Jahr - 1, "
                f"ab {MC_CHECK_GAP} Jahren Abstand mit Entwickler- oder Score-Bestätigung (Stufe 2)",
                "nur PC-Einträge der Rohdatei; als gematcht zählen nur Einträge mit Metascore; "
                f"{stage_note('mc_match_stage')} (inkl. Einträge ohne Metascore)"),
        cov_row("Preishistorie", "price_history", len(price), base["has_price_history"],
                "Steam-AppID",
                "Monat 2025-11 unvollständig und verworfen; keine kostenlosen Titel; "
                f"volles Fenster bei {int(price['ph_full_window'].sum())} von {len(price)} Apps"),
    ])
    save_table(coverage, "source_coverage")
    save_table(pd.DataFrame(_COUNTS), "source_counts")

    overlap = (
        out.loc[out["analytic_sample"].fillna(False).astype(bool), "n_sources"]
        .value_counts()
        .sort_index()
        .rename_axis("Anzahl_Datenquellen")
        .reset_index(name="N")
    )
    overlap["Anteil"] = overlap["N"] / max(n_analytic, 1)
    save_table(overlap, "source_overlap")

    return out, coverage, overlap

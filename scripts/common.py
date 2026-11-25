"""Gemeinsame Hilfsfunktionen und Pfade fuer die Analyse.

Projekt: Erfolgsfaktoren von Videospielen auf Steam
Datensatz: FronkonGames Steam Games Dataset (Kaggle), Version 58.

Die urspruengliche Auswertung nannte Version 57, die der Eigentuemer inzwischen
geloescht hat. Ein Neulauf am 12.11.2026 mit Version 58 (hochgeladen am 08.11.2026)
hat alle 10 Parquet-Dateien SHA-256-gleich und 25 von 31 Tabellen bytegleich
reproduziert (die uebrigen nur mit Rundungsrauschen der Logit-Schaetzungen). Belegt
ist damit die Inhaltsgleichheit der damals verarbeiteten Datei mit Version 58,
nicht, welche Version damals tatsaechlich geladen wurde.
Jede Rohdatei wird ueber Groesse und SHA-256 identifiziert (``SOURCES``); jeder
Abruf wird in ``PROVENANCE`` protokolliert und von ``write_provenance`` nach
``outputs/tables/data_provenance.csv`` geschrieben.
"""

from __future__ import annotations

import csv
import hashlib
import math
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
OUT = ROOT / "outputs"
TAB = OUT / "tables"
FIG = ROOT / "figures"

for _p in (DATA, RAW, OUT, TAB, FIG):
    _p.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

# Die FronkonGames-Datei vereint zwei Erhebungsstaende: Store-Felder (Preis,
# Rabatt, Recommendations, App-Liste) von ca. Anfang Maerz 2026 und
# SteamSpy-Felder (Besitzer, Positive/Negative, Peak CCU) aus einem Abruf um
# den 06.-08.06.2025. Spiele, die danach erschienen sind, haben strukturell
# keine Ergebnisdaten. Die Analysestichprobe endet deshalb vor diesem Stichtag;
# 01_prepare_data.py prueft den Bruch empirisch (steamspy_coverage_by_month.csv).
STEAMSPY_CUTOFF = pd.Timestamp("2025-06-01")
MIN_REVIEWS = 10
MIN_REVIEWS_ROBUST = 50
# Serien von Kleinstspielen zu exakt 54,99 USD Listenpreis treten erst ab 2024
# gehaeuft auf (2024: 87 Titel, 0,62 % der Kaufspiele). Vorher sind es hoechstens
# 5 Titel je Jahr (Spitze 2015 mit 4 von 2.176 = 0,18 %, darunter regulaere
# Vollpreistitel; sonst <= 0,05 %). Die Cluster-Titel sind praktisch nie
# erfolgreich. Sie verzerren die oberste Preisklasse und werden dort wie die
# Preisausreisser getrennt ausgewiesen (Merkmal price_cluster).
CLUSTER_PRICE = 54.99
CLUSTER_FROM_YEAR = 2024

# Kaggle-Downloads sind auf feste Dataset-Versionen gepinnt und werden ueber
# Groesse und SHA-256 geprueft. ``rows`` ist die Zahl der Datensaetze nach dem
# Einlesen mit pandas (nicht die Zahl physischer Zeilen).
FRONKON_SLUG = "fronkongames/steam-games-dataset"
FRONKON_VERSION = 58

SOURCES: dict[str, dict] = {
    "fronkon": dict(
        slug=FRONKON_SLUG, version=FRONKON_VERSION, file="games.csv",
        bytes=401140503,
        sha256="1b48008b01a799d82385d6e66ba0cb65dd477b6aa4b6c2ec8c443025ab6880ad",
        rows=125855,
        titel="FronkonGames Steam Games Dataset",
        datenstand=(
            "Store-Felder ca. Anfang März 2026, SteamSpy-Felder ca. Anfang Juni 2025; "
            "Version 58 vom 08.11.2026 (die ursprüngliche Auswertung nannte die nicht mehr öffentliche "
            "Version 57; ihre Daten sind inhaltsgleich mit Version 58)"
        ),
    ),
    "igdb": dict(
        slug="emirshn/igdb-dataset-for-data-mining-projects", version=1,
        file="game_dataset_cleaned.csv", bytes=238529214,
        sha256="1e2e5b3d4e4857a044bbaaffaf088e0a056ca19c0e416e80d2c4551c71915a28",
        rows=232595,
        titel="IGDB Dataset for Data Mining Projects",
        datenstand="Momentaufnahme Juli 2025 (Kaggle-Version 1 vom 26.07.2025)",
    ),
    "metacritic_raw": dict(
        slug="stadynromn/metacritic-video-games-reviews-up-to-sep-2025", version=1,
        file="metacritic_dataset_raw.csv", bytes=3850484,
        sha256="8527832ba07d11df9f6a94292f519450e1ec458b5a65dfe82611b909b1124b14",
        rows=36831,
        titel="Metacritic Video Games Reviews",
        datenstand="Veröffentlichungen bis Anfang Oktober 2025",
    ),
    "price_history": dict(
        slug="andrii222/steam-price-history-nov-2023-oct-2025", version=1,
        file="steam_price_history.csv", bytes=1720080,
        sha256="7650b76172129b0ed775a93f9834b611fc74b0a05b709af8d5df5207eb20d1d8",
        rows=13119,
        titel="Steam Price History",
        datenstand=(
            "rollierende Fenster von bis zu 24 Monaten zwischen Nov. 2023 und Okt. 2025; "
            "kürzer bei Release im Zeitraum"
        ),
    ),
}

# Protokoll aller Abrufe in diesem Lauf (eine Zeile je Quelle).
PROVENANCE: list[dict] = []

# Kopfzeile der Rohdatei: 39 Namen, weil "Discount" und "DLC count" dort zu
# "DiscountDLC count" verschmolzen sind. Die Datenzeilen haben 40 Felder.
RAW_HEADER = [
    "AppID", "Name", "Release date", "Estimated owners", "Peak CCU",
    "Required age", "Price", "DiscountDLC count", "About the game",
    "Supported languages", "Full audio languages", "Reviews", "Header image",
    "Website", "Support url", "Support email", "Windows", "Mac", "Linux",
    "Metacritic score", "Metacritic url", "User score", "Positive", "Negative",
    "Score rank", "Achievements", "Recommendations", "Notes",
    "Average playtime forever", "Average playtime two weeks",
    "Median playtime forever", "Median playtime two weeks", "Developers",
    "Publishers", "Categories", "Genres", "Tags", "Screenshots", "Movies",
]

# Spaltenreihenfolge der Datenzeilen (40 Felder).
RAW_COLUMNS = [
    "AppID", "Name", "Release date", "Estimated owners", "Peak CCU",
    "Required age", "Price", "Discount", "DLC count", "About the game",
    "Supported languages", "Full audio languages", "Reviews", "Header image",
    "Website", "Support url", "Support email", "Windows", "Mac", "Linux",
    "Metacritic score", "Metacritic url", "User score", "Positive", "Negative",
    "Score rank", "Achievements", "Recommendations", "Notes",
    "Average playtime forever", "Average playtime two weeks",
    "Median playtime forever", "Median playtime two weeks", "Developers",
    "Publishers", "Categories", "Genres", "Tags", "Screenshots", "Movies",
]

USECOLS = [
    "AppID", "Name", "Release date", "Estimated owners", "Peak CCU",
    "Price", "Discount", "DLC count", "Supported languages", "Windows", "Mac",
    "Linux", "Metacritic score", "User score", "Positive", "Negative",
    "Recommendations", "Achievements", "Developers", "Publishers",
    "Categories", "Genres", "Tags",
]

STANDARD_GENRES = [
    "Action", "Adventure", "Casual", "Indie", "Massively Multiplayer",
    "RPG", "Racing", "Simulation", "Sports", "Strategy",
]

SOFTWARE_GENRES = {
    "Utilities", "Design & Illustration", "Animation & Modeling",
    "Education", "Video Production", "Game Development", "Audio Production",
    "Software Training", "Photo Editing", "Web Publishing", "Accounting",
    "Movie", "Documentary", "360 Video",
}

# Nicht-Spiel-Erkennung ueber den Namen. Eindeutige Begriffe zaehlen ueberall;
# mehrdeutige Begriffe (Soundtrack, OST, DLC, Expansion) zaehlen mit Trennzeichen
# davor ("Spiel - Soundtrack"). Als letztes Wort (auch "Benchmark") zaehlen sie
# nur, wenn der Name kein Buendel "Spiel + Zusatz" bzw. "Spiel & Zusatz" ist:
# "KINGDOM HEARTS III + Re Mind (DLC)" oder "Sinless + OST" enthalten das Spiel.
# So bleiben auch Spieltitel wie "DLC Quest", "Expansion", "Expansion Core" oder
# "Benchmark" erhalten (Endwort-Regel greift erst nach einem vorangehenden Wort).
# "trial" allein waere zu breit ("Urban Trial Freestyle"), "free trial" ist
# dagegen eine eigene Testversion.
NON_GAME_PATTERN = re.compile(
    r"\b(?:demo|playtest|play test|season pass|expansion pass|expansion pack|"
    r"downloadable content|artbook|art book|wallpapers?|dedicated server|sdk|"
    r"closed beta|open beta|beta test|(?:public )?test server|free edition|free trial)\b"
    r"|\s[-\u2013\u2014:|]\s*(?:original\s+)?(?:soundtrack|ost|dlc|expansion)\b",
    re.IGNORECASE,
)
NON_GAME_SUFFIX_PATTERN = re.compile(
    r"\s[(\[]?(?:original\s+)?(?:soundtrack|ost|dlc|expansion|benchmark)"
    r"(?:\s+(?:pack|vol\.?\s*\d+))?[)\]]?\s*$",
    re.IGNORECASE,
)
BUNDLE_PATTERN = re.compile(r"\s[+&]\s")


def non_game_name(names: pd.Series) -> pd.Series:
    """Nicht-Spiel nach dem Namen: eindeutiger Begriff oder Zusatz am Namensende ohne Buendel."""
    text = names.fillna("").astype(str)
    suffix = text.str.contains(NON_GAME_SUFFIX_PATTERN) & ~text.str.contains(BUNDLE_PATTERN)
    return text.str.contains(NON_GAME_PATTERN) | suffix

# Editionszusaetze, die normalize_name nur am Namensende entfernt. Generische
# Woerter (standard, special, gold, vr ...) zaehlen nur in Verbindung mit
# "edition", damit Titel wie "Gold Rush" oder "Ultimate Chess" erhalten bleiben.
_EDITION_QUALIFIERS = r"standard|special|premium|starter|legacy|gold|digital|limited|steam|vr"
_EDITION_TOKENS = (
    r"edition|goty|game of the year|remastered|definitive|deluxe|ultimate|"
    r"complete|enhanced|anniversary|collector(?:['\u2019]?s)?"
)
EDITION_SUFFIX = re.compile(
    rf"[\W_]+(?:(?:{_EDITION_QUALIFIERS})\s+edition|{_EDITION_TOKENS})[\W_]*$"
)
# Marken- und Ersatzzeichen (TM, SM, R, C, U+FFFD aus defekter Kodierung der Quelle).
_TRADEMARK_CHARS = re.compile("[\u2122\u2120\u00ae\u00a9\ufffd]")


# ---------------------------------------------------------------------------
# Rohdaten: Abruf, Pruefung, Provenienz
# ---------------------------------------------------------------------------

def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_file(path: Path, key: str, origin: str) -> str:
    """Groesse und SHA-256 einer Rohdatei gegen SOURCES pruefen; bei Abweichung abbrechen."""
    spec = SOURCES[key]
    size = path.stat().st_size
    if size != spec["bytes"]:
        raise RuntimeError(
            f"{spec['file']} ({origin}) hat {size:,} Bytes, erwartet {spec['bytes']:,} Bytes "
            f"({spec['slug']}, Version {spec['version']}). Die Datei entspricht nicht der "
            "dokumentierten Version; Lauf abgebrochen."
        )
    digest = sha256_file(path)
    if digest != spec["sha256"]:
        raise RuntimeError(
            f"{spec['file']} ({origin}) hat SHA-256 {digest}, erwartet {spec['sha256']} "
            f"({spec['slug']}, Version {spec['version']}). Lauf abgebrochen."
        )
    return digest


def _version_from_path(path: Path) -> int | None:
    match = re.search(r"[\\/]versions[\\/](\d+)(?:[\\/]|$)", str(path))
    return int(match.group(1)) if match else None


def fetch_source(key: str) -> Path:
    """Pfad zur geprueften Rohdatei einer Quelle aus ``SOURCES`` liefern.

    Eine lokale Kopie unter ``data/raw/<datei>`` hat Vorrang, wird aber nur
    akzeptiert, wenn Groesse und SHA-256 stimmen. Sonst wird genau diese Datei
    der gepinnten Kaggle-Version ueber kagglehub geladen (bzw. aus dessen Cache
    genommen) und ebenfalls geprueft. Massgeblich ist der Hash; die Version im
    Pfad (``.../versions/N/``) wird nur protokolliert, weil kagglehub in Kaggle-
    und Colab-Notebooks Pfade ohne Versionsangabe liefert. Jeder Abruf wird in
    ``PROVENANCE`` eingetragen (Spalte Version_Herkunft).
    """
    if key not in SOURCES:
        raise KeyError(f"Unbekannte Quelle '{key}'. Bekannt: {sorted(SOURCES)}")
    spec = SOURCES[key]
    local = RAW / spec["file"]
    if local.exists():
        _verify_file(local, key, "lokale Kopie data/raw")
        path = local
        version = spec["version"]
        version_note = "aus Hash bestätigt (lokale Kopie)"
        path_type = "lokal (data/raw, Hash geprüft)"
    else:
        handle = f"{spec['slug']}/versions/{spec['version']}"
        try:
            import kagglehub

            downloaded = Path(kagglehub.dataset_download(handle, path=spec["file"]))
        except Exception as exc:  # kagglehub wirft je nach Fehler unterschiedliche Typen
            raise RuntimeError(
                f"Abruf von {handle} (Datei {spec['file']}) fehlgeschlagen: "
                f"{type(exc).__name__}: {exc}\n"
                "Haeufigste Ursache bei HTTP 404: Der Eigentuemer hat diese Version auf "
                "Kaggle geloescht. Abhilfe: die archivierte Originaldatei "
                f"({spec['bytes']:,} Bytes, SHA-256 {spec['sha256']}) unter {local} ablegen; "
                "sie wird dann nach Hashpruefung verwendet."
            ) from exc
        if downloaded.is_dir():
            candidates = sorted(downloaded.rglob(spec["file"]))
            if len(candidates) != 1:
                raise RuntimeError(
                    f"{spec['file']} im Download {handle} {len(candidates)}-mal gefunden, erwartet genau einmal."
                )
            downloaded = candidates[0]
        # Der Inhalt entscheidet: erst Groesse und SHA-256 pruefen. Die Version im Pfad
        # ist nur ein Zusatzhinweis, denn die Kaggle-/Colab-Resolver von kagglehub
        # liefern Mount-Pfade ohne 'versions/N'.
        _verify_file(downloaded, key, "kagglehub")
        path_version = _version_from_path(downloaded)
        version = spec["version"]
        if path_version is None:
            version_note = "aus Hash bestätigt (Pfad ohne versions/N)"
        elif path_version == spec["version"]:
            version_note = "aus Pfad, Hash geprüft"
        else:
            version_note = f"Pfad meldet Version {path_version}; Inhalt hashgleich mit Version {spec['version']}"
            print(f"WARNUNG: kagglehub-Pfad meldet Version {path_version} statt {spec['version']} fuer "
                  f"{spec['slug']}; Groesse und SHA-256 stimmen, die Datei wird verwendet.")
        path = downloaded
        path_type = "kagglehub (Download bzw. Cache, Hash geprüft)"

    entry = {
        "Quelle": key,
        "Titel": spec.get("titel", ""),
        "Kaggle_Handle": spec["slug"],
        "Version": version,
        "Version_Herkunft": version_note,
        "Datei": spec["file"],
        "Bytes": spec["bytes"],
        "SHA256": spec["sha256"],
        "Pfadtyp": path_type,
        "Datenstand": spec.get("datenstand", ""),
        "Zeilen_erwartet": spec["rows"],
        "Zeilen_eingelesen": np.nan,
        "Zeilen_geprueft": False,
    }
    PROVENANCE[:] = [row for row in PROVENANCE if row["Quelle"] != key]
    PROVENANCE.append(entry)
    return path


def record_rows(key: str, n_rows: int) -> None:
    """Eingelesene Datensatzzahl nachtragen und gegen ``SOURCES[key]['rows']`` pruefen."""
    entry = next((row for row in PROVENANCE if row["Quelle"] == key), None)
    if entry is None:
        raise KeyError(f"record_rows('{key}') vor fetch_source('{key}') aufgerufen.")
    expected = SOURCES[key]["rows"]
    entry["Zeilen_eingelesen"] = int(n_rows)
    entry["Zeilen_geprueft"] = int(n_rows) == expected
    if int(n_rows) != expected:
        raise RuntimeError(
            f"{SOURCES[key]['file']}: {n_rows:,} Datensaetze eingelesen, erwartet {expected:,}. "
            "Die Datei ist hashgleich, das Einlesen hat sich also geaendert (z. B. pandas-Version "
            "oder read_csv-Parameter); Lauf abgebrochen."
        )


def write_provenance() -> Path:
    """Protokoll aller Abrufe dieses Laufs als data_provenance.csv speichern."""
    order = {key: i for i, key in enumerate(SOURCES)}
    rows = sorted(PROVENANCE, key=lambda row: order.get(row["Quelle"], len(order)))
    return save_table(pd.DataFrame(rows), "data_provenance")


def _check_raw_layout(path: Path) -> int:
    """Kopfzeile und Feldzahl jedes Datensatzes der FronkonGames-Datei pruefen.

    pandas fuellt mit ``names``/``usecols`` einzelne zu kurze Zeilen still mit NaN
    auf bzw. verwirft ueberzaehlige Felder; deshalb wird jede Zeile mit dem
    csv-Modul gezaehlt. Rueckgabe: Zahl der Datensaetze.
    """
    bad: list[tuple[int, int]] = []
    n_records = 0
    with open(path, encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        if header != RAW_HEADER:
            diff = [(i, a, b) for i, (a, b) in enumerate(zip(header, RAW_HEADER)) if a != b]
            raise RuntimeError(
                f"Kopfzeile von games.csv weicht ab: {len(header)} Namen statt {len(RAW_HEADER)}, "
                f"erste Unterschiede {diff[:5]}"
            )
        for record in reader:
            n_records += 1
            if len(record) != len(RAW_COLUMNS):
                bad.append((reader.line_num, len(record)))
    if bad:
        raise RuntimeError(
            f"{len(bad)} Datensaetze in games.csv haben nicht {len(RAW_COLUMNS)} Felder, "
            f"z. B. (Zeile, Felder) {bad[:5]}"
        )
    return n_records


def load_raw(usecols: list[str] | None = None) -> pd.DataFrame:
    """FronkonGames-Rohdaten geprueft laden (Hash, Kopfzeile, Feldzahl, Datensatzzahl).

    Nur leere Felder gelten als fehlend (``keep_default_na=False``), damit
    woertliche Werte wie "None" oder "N/A" (z. B. in Entwicklernamen) nicht
    still zu NaN werden.
    """
    path = fetch_source("fronkon")
    n_records = _check_raw_layout(path)
    df = pd.read_csv(
        path,
        names=RAW_COLUMNS,
        skiprows=1,
        usecols=usecols or USECOLS,
        dtype=str,
        keep_default_na=False,
        na_values=[""],
        on_bad_lines="error",
        low_memory=False,
    )
    if len(df) != n_records:
        raise RuntimeError(f"pandas las {len(df):,} Datensaetze, das csv-Modul {n_records:,}.")
    record_rows("fronkon", len(df))
    return df


# ---------------------------------------------------------------------------
# Statistik-Hilfen
# ---------------------------------------------------------------------------

def split_list(value: object) -> list[str]:
    if not isinstance(value, str) or not value.strip():
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def wilson_lower_bound(positive: float, negative: float, z: float = 1.96) -> float:
    if not (np.isfinite(positive) and np.isfinite(negative)):
        return np.nan
    n = positive + negative
    if n <= 0:
        return np.nan
    p = positive / n
    denominator = 1.0 + z * z / n
    centre = p + z * z / (2.0 * n)
    margin = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n)
    return max(0.0, (centre - margin) / denominator)


def cliffs_delta_from_mannwhitney(u: float, n1: int, n2: int) -> float:
    """Rank-biserialer Effekt (Cliff's Delta) aus einer Mann-Whitney-U-Statistik."""
    if n1 == 0 or n2 == 0:
        return np.nan
    return 2.0 * u / (n1 * n2) - 1.0


def odds_ratio_ci(table: np.ndarray, alpha: float = 0.05) -> tuple[float, float, float, float]:
    """Odds Ratio, Woolf-Konfidenzintervall und Chi2-p-Wert fuer eine 2x2-Tabelle."""
    from scipy.stats import chi2_contingency

    a, b, c, d = table.ravel()
    if min(a, b, c, d) == 0:
        a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    odds_ratio = (a * d) / (b * c)
    se = math.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    z = 1.959963984540054
    low = math.exp(math.log(odds_ratio) - z * se)
    high = math.exp(math.log(odds_ratio) + z * se)
    p = chi2_contingency(table, correction=False)[1]
    return odds_ratio, low, high, p


def bh_correction(p_values: pd.Series) -> pd.Series:
    """Benjamini-Hochberg FDR-Korrektur."""
    p = pd.to_numeric(p_values, errors="coerce")
    result = pd.Series(np.nan, index=p.index, dtype=float)
    valid = p.dropna()
    if valid.empty:
        return result
    order = valid.sort_values()
    m = len(order)
    adjusted = order * m / np.arange(1, m + 1)
    adjusted = adjusted[::-1].cummin()[::-1].clip(upper=1.0)
    result.loc[adjusted.index] = adjusted
    return result


QUARTILE_LABELS = ("Q1 (niedrig)", "Q2", "Q3", "Q4 (hoch)")


def ranked_quartiles(
    series: pd.Series,
    labels: tuple[str, str, str, str] = QUARTILE_LABELS,
) -> pd.Series:
    """Bindungstreue Quartilsgruppen: gleiche Werte landen immer in derselben Gruppe.

    Jeder Wert wird ueber seinen mittleren Rang (alle Bindungen eines Werts haben
    denselben mittleren Rang) dem Quartil zugeordnet, in das dieser Rang faellt:
    Gruppe = floor(4 * (mittlerer Rang - 0,5) / n). Ohne Bindungen entstehen vier
    (fast) gleich grosse Gruppen; ein Bindungsblock an einer Grenze geht komplett
    in die Gruppe, in der seine Mitte liegt. Damit die Extremgruppen auch bei sehr
    grossen Bindungsbloecken die Extremwerte enthalten, gehoert der kleinste Wert
    immer zu Q1 und der groesste immer zu Q4 (bei mindestens zwei verschiedenen
    Werten). Gruppen koennen ungleich gross oder leer sein; die tatsaechlichen
    Wertebereiche und Groessen liefert ``quartile_ranges``. Fehlende Werte bleiben NaN.
    """
    values = pd.to_numeric(pd.Series(series), errors="coerce")
    result = pd.Series(pd.Categorical([np.nan] * len(values), categories=list(labels), ordered=True),
                       index=values.index)
    valid = values.dropna()
    if valid.empty:
        return result
    midranks = valid.rank(method="average").to_numpy()
    groups = np.floor(4.0 * (midranks - 0.5) / len(valid)).astype(int).clip(0, 3)
    numbers = valid.to_numpy(dtype=float)
    if numbers.min() < numbers.max():
        groups[numbers == numbers.min()] = 0
        groups[numbers == numbers.max()] = 3
    result.loc[valid.index] = pd.Categorical.from_codes(groups, categories=list(labels), ordered=True)
    return result


def quartile_ranges(series: pd.Series, groups: pd.Series) -> pd.DataFrame:
    """Wertebereich (Min, Max) und Gruppengroesse je Quartilsgruppe.

    ``groups`` ist das Ergebnis von ``ranked_quartiles`` fuer dieselbe Reihe.
    Leere Gruppen erscheinen mit n = 0.
    """
    values = pd.to_numeric(pd.Series(series), errors="coerce")
    groups = pd.Series(groups, index=values.index)
    categories = list(groups.cat.categories) if hasattr(groups, "cat") else sorted(groups.dropna().unique())
    rows = []
    for label in categories:
        part = values[groups == label].dropna()
        rows.append(
            {
                "Gruppe": label,
                "n": int(len(part)),
                "min": float(part.min()) if len(part) else np.nan,
                "max": float(part.max()) if len(part) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def save_table(df: pd.DataFrame, name: str, index: bool = False) -> Path:
    # 10 signifikante Stellen: haelt Ergebnisse exakt, unterdrueckt aber
    # plattformabhaengiges Rauschen in der letzten Stelle (z. B. bei Logit-Fits).
    path = TAB / f"{name}.csv"
    df.to_csv(path, index=index, encoding="utf-8-sig", float_format="%.10g")
    return path


def format_p(p: float) -> str:
    if pd.isna(p):
        return "n/a"
    if p < 0.001:
        return "< 0,001"
    return f"{p:.3f}".replace(".", ",")


# ---------------------------------------------------------------------------
# Namensnormalisierung (Duplikaterkennung in 01, Matching in external_data.py)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _is_latin(char: str) -> bool:
    return unicodedata.name(char, "").startswith("LATIN")


def _strip_edition_suffix(text: str) -> str:
    """Editionszusaetze wiederholt vom Namensende entfernen.

    Es bleiben immer mindestens vier Buchstaben/Ziffern stehen, damit kurze Titel
    wie "VR Edition" oder "The Collector's" nicht zu generischen Schluesseln werden.
    """
    while True:
        match = EDITION_SUFFIX.search(text)
        if not match:
            return text
        head = text[: match.start()]
        if len(re.findall(r"[^\W_]", head)) < 4:
            return text
        text = head


def has_non_latin_letters(value: object) -> bool:
    """True, wenn der Name (nach NFKC) mindestens einen nicht-lateinischen Buchstaben enthaelt."""
    if not isinstance(value, str):
        return False
    return any(c.isalpha() and not _is_latin(c) for c in unicodedata.normalize("NFKC", value))


def normalize_name(value: object, latin_part: bool = True) -> str:
    """Vergleichsschluessel fuer Spielnamen.

    Schritte: Markenzeichen entfernen, Unicode-NFKC und casefold, Editionszusaetze
    nur am Namensende entfernen, Akzente lateinischer Buchstaben entfernen
    (z. B. "e" mit Akut -> "e"), danach nur Buchstaben und Ziffern behalten (``\\w`` in
    allen Schriften). Nicht-lateinische Namen bleiben damit erhalten statt leer zu
    werden.

    ``latin_part=True`` (Matching mit externen Quellen): Enthaelt ein Name
    lateinische und andere Buchstaben (z. B. chinesischer Titel plus "Warm Snow")
    mit mindestens vier lateinischen Buchstaben und mindestens einem Drittel so
    vielen lateinischen wie anderen Buchstaben, wird nur der lateinische Teil samt
    Ziffern verwendet, weil externe Quellen meist den englischen Titel fuehren.
    Kurze lateinische Einsprengsel ("etc.", "RPG") bleiben Teil des vollstaendigen
    Schluessels. Die Regel kann generische Schluessel erzeugen (z. B. "online"
    aus zwei verschiedenen chinesischen Titeln mit dem Zusatz "Online"); fuer die
    Duplikaterkennung innerhalb von Steam wird deshalb ``latin_part=False``
    verwendet (vollstaendiger Schluessel in allen Schriften).
    """
    if not isinstance(value, str):
        return ""
    text = _TRADEMARK_CHARS.sub(" ", value)
    text = unicodedata.normalize("NFKC", text).casefold()
    text = _strip_edition_suffix(text)

    decomposed = unicodedata.normalize("NFKD", text)
    kept: list[str] = []
    base_is_latin = False
    for char in decomposed:
        if unicodedata.combining(char):
            if not base_is_latin:
                kept.append(char)
            continue
        base_is_latin = _is_latin(char)
        kept.append(char)
    text = unicodedata.normalize("NFC", "".join(kept))

    chars = [c for c in text if c.isalnum() or unicodedata.category(c).startswith("M")]
    latin_letters = sum(1 for c in chars if c.isalpha() and _is_latin(c))
    other_letters = sum(1 for c in chars if c.isalpha() and not _is_latin(c))
    if latin_part and other_letters and latin_letters >= 4 and 3 * latin_letters >= other_letters:
        chars = [c for c in chars if (c.isalpha() and _is_latin(c)) or c in "0123456789"]
    return "".join(chars)

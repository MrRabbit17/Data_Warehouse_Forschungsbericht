"""Gemeinsame Hilfsfunktionen und Pfade fuer die Analyse.

Projekt: Erfolgsfaktoren von Videospielen auf Steam
Datensatz: FronkonGames Steam Games Dataset (Kaggle), Version 57
"""

from __future__ import annotations

import math
import re
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
REFERENCE_YEAR = 2025  # letztes vollstaendiges Veroeffentlichungsjahr im Datensatz
MIN_REVIEWS = 10
MIN_REVIEWS_ROBUST = 50

# Kaggle-Downloads werden auf feste Dataset-Versionen gepinnt, damit ein
# erneuter Lauf dieselben Rohdaten verwendet.
FRONKON_SLUG = "fronkongames/steam-games-dataset"
FRONKON_VERSION = 57

# Spaltenreihenfolge der Rohdatei. Die Kopfzeile enthaelt nur 39 Namen,
# die Datenzeilen enthalten 40 Felder: "DiscountDLC count" ist in den Daten
# in die beiden Spalten "Discount" und "DLC count" aufgeteilt.
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
    "Price", "Discount", "DLC count", "Windows", "Mac", "Linux",
    "Metacritic score", "User score", "Positive", "Negative",
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

# Sehr breite, aber bewusst konservative Nicht-Spiel-Erkennung.
NON_GAME_PATTERN = re.compile(
    r"\b(?:demo|playtest|play test|soundtrack|original soundtrack|ost|"
    r"season pass|expansion pass|expansion pack|expansion|dlc|"
    r"downloadable content|benchmark|artbook|art book|wallpaper|"
    r"dedicated server|sdk|closed beta|open beta|beta test)\b",
    re.IGNORECASE,
)

EDITION_TOKENS = re.compile(
    r"\b(deluxe|ultimate|definitive|gold|complete|enhanced|remastered|"
    r"goty|game of the year|collector'?s?|special|premium|starter|"
    r"standard|anniversary|legacy|vr|edition|version)\b",
    re.IGNORECASE,
)


def load_raw(usecols: list[str] | None = None) -> pd.DataFrame:
    """Rohdaten laden; bei Bedarf automatisch ueber kagglehub herunterladen."""
    path = RAW / "games.csv"
    if not path.exists():
        import kagglehub

        handle = f"{FRONKON_SLUG}/versions/{FRONKON_VERSION}"
        downloaded = Path(kagglehub.dataset_download(handle))
        candidates = list(downloaded.glob("games.csv"))
        if not candidates:
            raise FileNotFoundError("games.csv im Kaggle-Download nicht gefunden")
        path = candidates[0]

    return pd.read_csv(
        path,
        names=RAW_COLUMNS,
        skiprows=1,
        usecols=usecols or USECOLS,
        dtype=str,
        low_memory=False,
    )


def split_list(value: object) -> list[str]:
    if not isinstance(value, str) or not value.strip():
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def wilson_lower_bound(positive: float, negative: float, z: float = 1.96) -> float:
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


def partial_spearman(x: pd.Series, y: pd.Series, z: pd.Series) -> tuple[float, float, int]:
    """Partielle Spearman-Korrelation von x und y, kontrolliert fuer z."""
    from scipy.stats import pearsonr, rankdata

    frame = pd.DataFrame({"x": x, "y": y, "z": z}).dropna()
    if len(frame) < 10:
        return np.nan, np.nan, len(frame)
    rx = rankdata(frame["x"])
    ry = rankdata(frame["y"])
    rz = rankdata(frame["z"])
    rz_design = np.column_stack([np.ones(len(rz)), rz])
    beta_x = np.linalg.lstsq(rz_design, rx, rcond=None)[0]
    beta_y = np.linalg.lstsq(rz_design, ry, rcond=None)[0]
    res_x = rx - rz_design @ beta_x
    res_y = ry - rz_design @ beta_y
    r, p = pearsonr(res_x, res_y)
    return float(r), float(p), len(frame)


def ranked_quartiles(
    series: pd.Series,
    labels: tuple[str, str, str, str] = ("Q1 (niedrig)", "Q2", "Q3", "Q4 (hoch)"),
    seed: int = RANDOM_STATE,
) -> pd.Series:
    """Rangbasierte Quartile, die Bindungen reproduzierbar zufaellig aufteilen.

    ``pd.qcut`` auf den Rohwerten bricht bei stark gebundenen Merkmalen zusammen
    (z. B. IGDB-Nutzerbewertung, bei der ein grosser Teil der Werte exakt
    70/100 ist); es entstehen Gruppen mit wenigen Beobachtungen. Hier werden die
    Werte zunaechst sortiert und Gleichstaende ueber eine feste Zufallsreihenfolge
    (``seed``) aufgeloest. Dadurch bleibt die inhaltliche Ordnung erhalten und die
    vier Gruppen sind gleich gross. Fehlende Werte bleiben NaN.
    """
    values = pd.Series(series)
    valid = values.notna()
    if not valid.any():
        return pd.Series(np.nan, index=values.index, dtype=object)
    positions = np.flatnonzero(valid.to_numpy())
    numbers = values.to_numpy(dtype=float)[positions]
    rng = np.random.default_rng(seed)
    tiebreak = rng.random(len(positions))
    order = np.lexsort((tiebreak, numbers))
    ranks = np.full(len(values), np.nan)
    ranks[positions[order]] = np.arange(1, len(positions) + 1, dtype=float)
    return pd.qcut(pd.Series(ranks, index=values.index), 4, labels=list(labels))


def save_table(df: pd.DataFrame, name: str, index: bool = False) -> Path:
    path = TAB / f"{name}.csv"
    df.to_csv(path, index=index, encoding="utf-8-sig")
    return path


def format_p(p: float) -> str:
    if pd.isna(p):
        return "n/a"
    if p < 0.001:
        return "< 0,001"
    return f"{p:.3f}".replace(".", ",")


def normalize_name(value: object) -> str:
    if not isinstance(value, str):
        return ""
    text = value.lower()
    text = EDITION_TOKENS.sub(" ", text)
    text = re.sub(r"[^a-z0-9]+", "", text)
    return text

"""Explorative Datenanalyse: deskriptive Tabellen und Abbildungen.

Das Skript liest ausschliesslich die Parquet-Dateien aus data/ (erzeugt von
01_prepare_data.py). Es haengt damit nicht von CSV-Zwischenergebnissen aus 01
ab; source_overlap.csv wird nur zum Abgleich gelesen, falls vorhanden.

"Erfolgsquote" meint durchgehend success_top25 (owners_low >= 20.000). Weil
die SteamSpy-Besitzer nur als Klassen vorliegen, beschreiben die Tabellen die
Besitzerverteilung je Gruppe ueber Anteile oberhalb fester Klassengrenzen
(>= 20.000 = Erfolgsquote, >= 100.000, >= 1 Mio.) statt ueber Mediane oder
Mittelwerte von Klassenmittelpunkten.
"""

from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator
from statsmodels.stats.proportion import proportion_confint

from common import (
    CLUSTER_FROM_YEAR, CLUSTER_PRICE, DATA, FIG, MIN_REVIEWS, QUARTILE_LABELS, STANDARD_GENRES,
    STEAMSPY_CUTOFF, TAB, quartile_ranges, ranked_quartiles, save_table,
)

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "DejaVu Sans",
        "axes.titleweight": "bold",
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "grid.color": "#e3e2de",
        "grid.linewidth": 0.6,
    }
)
# Kategoriale Reihenfolge fest (geprueft auf Farbfehlsichtigkeit fuer alle Paare);
# geordnete Kategorien (Besitzerklassen, Kohorten) nutzen eine einfarbige Rampe.
COLORS = {"main": "#2a78d6", "accent": "#eb6834", "grey": "#9a9993", "ink": "#2b2b2b"}
SEQ_CMAP = "Blues"

SUCCESS_OWNERS = 20_000
SHARE_THRESHOLDS = [("Anteil_ab_100k", 100_000), ("Anteil_ab_1Mio", 1_000_000)]
OWNER_GROUPS = [
    (0, 20_000, "< 20.000"),
    (20_000, 100_000, "20.000 bis < 100.000"),
    (100_000, 1_000_000, "100.000 bis < 1 Mio."),
    (1_000_000, np.inf, "≥ 1 Mio."),
]
PRICE_ORDER = ["Free", "0,01 - 4,99", "5 - 9,99", "10 - 19,99", "20 - 39,99", "40+"]
PRICE_KEYS = {
    "Free": "free", "0,01 - 4,99": "p0_5", "5 - 9,99": "p5_10", "10 - 19,99": "p10_20",
    "20 - 39,99": "p20_40", "40+": "p40plus",
}
DISCOUNT_BINS = [-0.1, 0, 25, 50, 75, 100]
DISCOUNT_LABELS = ["kein Rabatt", "1-25 %", "26-50 %", "51-75 %", "76-100 %"]
DISCOUNT_KEYS = ["d0", "d1_25", "d26_50", "d51_75", "d76_100"]
PLATFORM_LABELS = {
    "W": "nur Windows", "W+M": "Windows + macOS", "W+L": "Windows + Linux",
    "W+M+L": "Windows + macOS + Linux",
}
COHORT_EDGES = [1996, 2009, 2014, 2019, 2022, 2025]
COHORT_LABELS = ["bis 2009", "2010-2014", "2015-2019", "2020-2022", "2023-2025"]
# Direkte Altersstandardisierung: Schichten = Veroeffentlichungsjahre; die
# duenn besetzten Jahre bis 2009 bilden eine gemeinsame Schicht.
STD_FIRST_YEAR = 2009
# Jahre mit weniger Spielen werden in fig_02 zusammengefasst (Tabelle: alle Jahre).
MIN_YEAR_N = 100
MIN_CELL_N = 30
# Das 54,99-USD-Cluster (common.CLUSTER_PRICE, Merkmal price_cluster) wird in der
# Preisuebersicht und in fig_05 wie die Preisausreisser getrennt ausgewiesen; die
# Tests in 03 enthalten es (dort Sensitivitaet ohne Cluster).
QUARTILE_SHORT = dict(zip(QUARTILE_LABELS, ["Q1", "Q2", "Q3", "Q4"]))
# Lesbare Namen der IGDB-Genres (Spaltennamen sind ASCII-Kurzformen der Originallabels).
IGDB_GENRE_LABELS = {
    "card_board_game": "Card & Board Game",
    "hack_and_slash_beat_em_up": "Hack and slash/Beat 'em up",
    "moba": "MOBA",
    "point_and_click": "Point-and-click",
    "quiz_trivia": "Quiz/Trivia",
    "real_time_strategy_rts": "Real Time Strategy (RTS)",
    "role_playing_rpg": "Role-playing (RPG)",
    "turn_based_strategy_tbs": "Turn-based strategy (TBS)",
}

VAR_LABELS = {
    "log_owners": "Besitzer (log)",
    "log_peak_ccu": "Peak CCU (log, inkl. 0)",
    "release_year": "Release-Jahr",
    "n_platforms": "Plattformen",
    "n_genres": "Genres (Anzahl)",
    "dlc_count": "DLCs",
    "list_price": "Listenpreis",
    "discount": "Rabatt",
    "positive_ratio": "Positive Ratio",
    "wilson": "Wilson-Score",
    "igdb_rating": "IGDB-Nutzer",
    "igdb_aggregated_rating": "IGDB-Kritiker",
    "igdb_n_genres": "IGDB-Genres (Anzahl)",
    "mc_metascore": "Metascore",
    "mc_user_score": "MC-Nutzerscore",
    "ph_base_price": "Basispreis",
    "ph_price_changes_per_month": "Preisänd./Monat",
    "ph_share_months_on_sale": "Anteil Rabattmonate",
    "ph_days_on_sale_per_month": "Rabatttage/Monat",
    "ph_mean_monthly_max_discount_pct": "Mittl. Monatsrabatt",
    "ph_max_discount_pct": "Max. Rabatt",
}


# --------------------------------------------------------------------------- Helfer
def de(value: object, decimals: int = 1) -> str:
    """Zahl mit Dezimalkomma und Tausenderpunkt."""
    if value is None or pd.isna(value):
        return "–"
    text = f"{float(value):,.{decimals}f}"
    return text.replace(",", "X").replace(".", ",").replace("X", ".")


def pct(value: object, decimals: int = 1) -> str:
    if value is None or pd.isna(value):
        return "–"
    return f"{de(100 * float(value), decimals)} %"


def de_formatter(decimals: int = 0, suffix: str = "") -> FuncFormatter:
    return FuncFormatter(lambda v, _: f"{de(v, decimals)}{suffix}")


def owners_axis(ax: plt.Axes, axis: str = "y") -> None:
    """log10-Besitzerachse mit lesbaren Klassengrenzen beschriften."""
    ticks = [4, 5, 6, 7, 8]
    labels = ["10.000", "100.000", "1 Mio.", "10 Mio.", "100 Mio."]
    target = ax.yaxis if axis == "y" else ax.xaxis
    target.set_major_locator(FixedLocator(ticks))
    target.set_major_formatter(FuncFormatter(lambda v, _: labels[ticks.index(v)] if v in ticks else ""))


def warn(message: str) -> None:
    print(f"WARNUNG: {message}")


def save(fig: plt.Figure, name: str, created: list[str]) -> None:
    fig.savefig(FIG / f"{name}.png")
    plt.close(fig)
    created.append(f"figures/{name}.png")


def write(df: pd.DataFrame, name: str, created: list[str]) -> pd.DataFrame:
    save_table(df, name)
    created.append(f"outputs/tables/{name}.csv")
    return df


def proportion_ci(success: pd.Series) -> tuple[float, float, float, int]:
    values = pd.Series(success).dropna().astype(bool)
    n = len(values)
    if n == 0:
        return np.nan, np.nan, np.nan, 0
    k = int(values.sum())
    low, high = proportion_confint(k, n, method="wilson")
    return k / n, float(low), float(high), n


def group_stats(frame: pd.DataFrame) -> dict[str, float]:
    """Erfolgsquote mit Wilson-KI, Anteile ueber Besitzergrenzen, mittleres Alter."""
    rate, low, high, n = proportion_ci(frame["success_top25"])
    row: dict[str, float] = {"N": n, "Erfolgsquote": rate, "CI_unten": low, "CI_oben": high}
    for col, threshold in SHARE_THRESHOLDS:
        row[col] = float((frame["owners_low"] >= threshold).mean()) if n else np.nan
    row["Mittleres_Alter"] = float(frame["age_years"].mean()) if n else np.nan
    return row


def year_stratum(frame: pd.DataFrame) -> pd.Series:
    return frame["release_year"].clip(lower=STD_FIRST_YEAR)


def standardized_rate(frame: pd.DataFrame, weights: pd.Series) -> dict[str, float]:
    """Direkt altersstandardisierte Erfolgsquote (Standard: Jahrgaenge der Analysestichprobe).

    Fehlen einer Gruppe einzelne Jahrgangsschichten, wird ueber die vorhandenen
    Schichten neu gewichtet; ``Gewicht_abgedeckt`` nennt den abgedeckten Anteil
    der Standardbevoelkerung. KI per Normalapproximation. Bei Gruppen mit
    weniger als MIN_CELL_N Spielen bleibt die Quote leer (zu instabil), die
    Abdeckung wird trotzdem ausgewiesen.
    """
    empty = {"Erfolgsquote_standardisiert": np.nan, "CI_unten_std": np.nan,
             "CI_oben_std": np.nan, "Gewicht_abgedeckt": np.nan}
    if frame.empty:
        return empty
    cells = frame.groupby(year_stratum(frame))["success_top25"].agg(["mean", "size"])
    common = cells.index.intersection(weights.index)
    if len(common) == 0:
        return empty
    w = weights.loc[common]
    covered = float(w.sum())
    if len(frame) < MIN_CELL_N:
        return {**empty, "Gewicht_abgedeckt": covered}
    w = w / covered
    p = cells.loc[common, "mean"]
    n = cells.loc[common, "size"]
    rate = float((w * p).sum())
    se = float(np.sqrt((w**2 * p * (1 - p) / n).sum()))
    return {
        "Erfolgsquote_standardisiert": rate,
        "CI_unten_std": max(0.0, rate - 1.96 * se),
        "CI_oben_std": min(1.0, rate + 1.96 * se),
        "Gewicht_abgedeckt": covered,
    }


def cohort_group(frame: pd.DataFrame) -> pd.Series:
    return pd.cut(frame["release_year"], bins=COHORT_EDGES, labels=COHORT_LABELS)


def cohort_key(label: str) -> str:
    return label.replace(" ", "_").replace("-", "_")


def cohort_rates(frame: pd.DataFrame) -> dict[str, float]:
    groups = cohort_group(frame)
    out: dict[str, float] = {}
    for label in COHORT_LABELS:
        rate, _, _, n = proportion_ci(frame.loc[groups == label, "success_top25"])
        out[f"N_{cohort_key(label)}"] = n
        out[f"Quote_{cohort_key(label)}"] = rate
    return out


def describe_rows(specs: list[tuple[str, pd.DataFrame, str, str]]) -> pd.DataFrame:
    rows = []
    for col, frame, basis, note in specs:
        s = pd.to_numeric(frame[col], errors="coerce").dropna()
        rows.append(
            {
                "Variable": col,
                "Basis": basis,
                "N": len(s),
                "Mittelwert": s.mean(),
                "SD": s.std(),
                "Median": s.median(),
                "Q1": s.quantile(0.25),
                "Q3": s.quantile(0.75),
                "Min": s.min(),
                "Max": s.max(),
                "Hinweis": note,
            }
        )
    return pd.DataFrame(rows)


def quartile_rows(frame: pd.DataFrame, var: str, label: str, basis: str,
                  key_col: str = "Bewertungsmaß", extra=None) -> list[dict]:
    """Bindungstreue Quartile mit Wertebereich, Erfolgsquote und Besitzeranteilen."""
    d = frame.dropna(subset=[var])
    groups = ranked_quartiles(d[var])
    ranges = quartile_ranges(d[var], groups).set_index("Gruppe")
    rows = []
    for quartile in QUARTILE_LABELS:
        part = d[groups == quartile]
        if part.empty:
            warn(f"{label}: Quartilsgruppe {quartile} ist leer (Bindungsblock an der Grenze).")
        stats = group_stats(part)
        row = {
            key_col: label, "Variable": var, "Basis": basis, "Quartil": quartile,
            "N": stats.pop("N"), "Wert_min": ranges.loc[quartile, "min"],
            "Wert_max": ranges.loc[quartile, "max"], **stats,
        }
        if extra is not None:
            row.update(extra(part))
        rows.append(row)
    return rows


def quartile_ticks(table: pd.DataFrame, decimals: int = 1) -> list[str]:
    """Quartilsbeschriftung mit Wertebereich. Rundet eine Obergrenze auf die
    Untergrenze der naechsten Gruppe, wird sie als "< x" geschrieben."""
    lows = [de(v, decimals) for v in table["Wert_min"]]
    highs = [de(v, decimals) for v in table["Wert_max"]]
    labels = []
    for i, quartile in enumerate(table["Quartil"]):
        high = highs[i]
        if (i + 1 < len(lows) and high == lows[i + 1]
                and table["Wert_max"].iloc[i] < table["Wert_min"].iloc[i + 1]):
            high = f"< {lows[i + 1]}"
        labels.append(f"{QUARTILE_SHORT[quartile]}\n{lows[i]}–{high}")
    return labels


def dash(label: str) -> str:
    """Bindestrich in Bereichsangaben fuer Abbildungen durch Halbgeviertstrich ersetzen."""
    return label.replace(" - ", "–").replace("-", "–")


def rate_bars(ax: plt.Axes, labels: list[str], table: pd.DataFrame, color: str,
              horizontal: bool = False, rate_col: str = "Erfolgsquote",
              low_col: str = "CI_unten", high_col: str = "CI_oben", n_labels: bool = True,
              width: float = 0.7) -> None:
    """Balken der Quoten (in %) mit Wilson-KI und n-Beschriftung."""
    pos = np.arange(len(table))
    rate = table[rate_col].to_numpy() * 100
    err = [rate - table[low_col].to_numpy() * 100, table[high_col].to_numpy() * 100 - rate]
    if horizontal:
        ax.barh(pos, rate, color=color, height=width)
        ax.errorbar(rate, pos, xerr=err, fmt="none", ecolor=COLORS["ink"], elinewidth=1, capsize=3)
        ax.set_yticks(pos, labels)
        ax.xaxis.set_major_formatter(de_formatter(0, " %"))
        if n_labels:
            for y, n in zip(pos, table["N"]):
                ax.text(1, y, f"n = {de(n, 0)}", va="center", ha="left", fontsize=7, color="white")
    else:
        ax.bar(pos, rate, color=color, width=width)
        ax.errorbar(pos, rate, yerr=err, fmt="none", ecolor=COLORS["ink"], elinewidth=1, capsize=3)
        ax.set_xticks(pos, labels)
        ax.yaxis.set_major_formatter(de_formatter(0, " %"))
        if n_labels:
            for x, n in zip(pos, table["N"]):
                ax.text(x, 1, f"n = {de(n, 0)}", ha="center", va="bottom", fontsize=7, color="white")


def heatmap(ax: plt.Axes, corr: pd.DataFrame, title: str) -> object:
    """Unteres Dreieck ohne Diagonale (erste Zeile und letzte Spalte waeren leer)."""
    lower = corr.iloc[1:, :-1]
    mask = np.triu(np.ones_like(lower, dtype=bool), k=1)
    annot = lower.map(lambda v: de(v, 2))
    mesh = sns.heatmap(
        lower, mask=mask, annot=annot, fmt="", cmap="RdBu_r", center=0, vmin=-1, vmax=1,
        square=True, linewidths=0.5, linecolor="white", ax=ax, cbar=False,
        annot_kws={"fontsize": 7.5},
        xticklabels=[VAR_LABELS.get(c, c) for c in lower.columns],
        yticklabels=[VAR_LABELS.get(c, c) for c in lower.index],
    )
    ax.set_title(title, fontsize=10)
    ax.tick_params(axis="both", labelsize=8)
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right", rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=0)
    ax.grid(False)
    return mesh


def correlation_block(frame: pd.DataFrame, variables: list[str], name: str, basis: str,
                      complete: bool) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    data = frame[variables].apply(pd.to_numeric, errors="coerce")
    if complete:
        data = data.dropna()
    corr = data.corr(method="spearman")
    valid = data.notna().astype(int)
    n_matrix = valid.T @ valid
    rows = []
    for i, a in enumerate(variables):
        for b in variables[i + 1:]:
            rows.append({"Matrix": name, "Basis": basis, "Variable_1": a, "Variable_2": b,
                         "rho": corr.loc[a, b], "N": int(n_matrix.loc[a, b])})
    return corr, n_matrix, rows


def n_range_text(n_matrix: pd.DataFrame) -> str:
    values = n_matrix.to_numpy()[np.triu_indices(len(n_matrix), k=1)]
    lo, hi = int(values.min()), int(values.max())
    return f"n = {de(lo, 0)}" if lo == hi else f"n = {de(lo, 0)}–{de(hi, 0)}, paarweise"


# --------------------------------------------------------------------------- Hauptteil
def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    created: list[str] = []

    prepared = pd.read_parquet(DATA / "steam_prepared.parquet")
    df = prepared[prepared["analytic_sample"]].copy()
    rating = df[df["rating_sample"]].copy()
    igdb = df[df["has_igdb"]].copy()
    meta = df[df["has_mc_external"]].copy()
    meta_fronk = df[df["meta_sample"]].copy()
    price_hist = df[df["has_price_history"]].copy()
    zero_sample = prepared[prepared["sample_with_owners_zero"]].copy()

    # Konsistenz mit den Teilstichproben-Parquets und den Definitionen aus 01.
    for name, frame in [("analytic_sample", df), ("rating_sample", rating), ("igdb_sample", igdb),
                        ("mc_external_sample", meta), ("price_history_sample", price_hist)]:
        path = DATA / f"{name}.parquet"
        if path.exists():
            n_file = len(pd.read_parquet(path, columns=["AppID"]))
            if n_file != len(frame):
                warn(f"{name}.parquet hat {n_file:,} Zeilen, aus steam_prepared abgeleitet {len(frame):,}.")
    assert (df["success_top25"] == (df["owners_low"] >= SUCCESS_OWNERS)).all(), \
        "success_top25 entspricht nicht owners_low >= 20.000"
    assert (df["release_date"] < STEAMSPY_CUTOFF).all(), "Analysestichprobe enthaelt Releases ab dem Stichtag"
    assert not zero_sample.loc[zero_sample["owners_zero"], "success_top25"].any(), \
        "'0 - 0'-Zeilen der Sensitivitaetsstichprobe muessen als Nicht-Erfolg zaehlen"
    assert rating["review_count"].ge(MIN_REVIEWS).all()

    # Kostenpflichtig wie in 03_tests.py: bestimmbarer Listenpreis > 0. Rabatt- und
    # Preisniveau-Tabellen ohne Preisausreisser (> 100 USD), damit Deskription und
    # Tests dieselbe Basis haben; paid_all (inkl. Ausreisser) nur fuer den Vergleich
    # kostenlos vs. kostenpflichtig. Spiele ohne bestimmbaren Listenpreis gehoeren
    # zu keiner der beiden Gruppen.
    paid_all = df[df["list_price"] > 0].copy()
    paid = paid_all[~paid_all["price_outlier"]].copy()
    paid_clean = paid
    n_no_list = int(df["list_price"].isna().sum())
    weights = year_stratum(df).value_counts(normalize=True).sort_index()
    overall_rate = df["success_top25"].mean()

    # ================================================================ Tabellen
    # --- Uebersicht numerischer Variablen
    specs = [
        ("owners_mid", df, "Analysestichprobe",
         "Klassenmittelpunkt (13 Werte); Mittelwert von wenigen sehr großen Titeln getrieben"),
        ("log_owners", df, "Analysestichprobe", "log10 des Klassenmittelpunkts"),
        ("peak_ccu", df, "Analysestichprobe",
         "gleichzeitige Spieler am Vortag des SteamSpy-Abrufs (ca. Juni 2025), kein Allzeit-Peak"),
        ("log_peak_ccu", df, "Analysestichprobe", "log10(Peak CCU + 1)"),
        ("list_price", paid_clean, "kostenpflichtig, ohne Preisausreißer",
         "rekonstruierter Listenpreis in USD (Untergrenze, Rabatt teils nicht erfasst)"),
        ("list_price", df, "Analysestichprobe (kostenlos = 0)",
         f"rekonstruierter Listenpreis in USD; {n_no_list} Einträge ohne bestimmbaren Listenpreis "
         "(100 % Rabatt oder Tagespreis 0 trotz Rabatt)"),
        ("price_current", df, "Analysestichprobe", "Tagespreis zum Store-Snapshot (ca. Anfang März 2026)"),
        ("discount", paid, "kostenpflichtig, ohne Preisausreißer", "Rabatt in % zum Store-Snapshot (Untergrenze)"),
        ("dlc_count", df, "Analysestichprobe", ""),
        ("release_year", df, "Analysestichprobe", "Kohorte 2025 nur Januar bis Mai"),
        ("age_years", df, "Analysestichprobe", "Jahre bis zum SteamSpy-Stichtag 01.06.2025"),
        ("n_platforms", df, "Analysestichprobe", ""),
        ("n_genres", df, "Analysestichprobe", "Anzahl Steam-Standardgenres"),
        ("review_count", df, "Analysestichprobe", "Positive + Negative (SteamSpy)"),
        ("positive_ratio", rating, f"Bewertungsstichprobe (>= {MIN_REVIEWS} Reviews)",
         "primäres Maß der Nutzerbewertung"),
        ("wilson", rating, f"Bewertungsstichprobe (>= {MIN_REVIEWS} Reviews)",
         "nur Diagnose (hängt vom Review-Volumen ab)"),
        ("metacritic", meta_fronk, "FronkonGames-Metacritic vorhanden", "Steam-Metacritic-Score"),
    ]
    write(describe_rows(specs), "descriptives_overall", created)

    # --- Besitzerverteilung (inkl. Anteil mindestens dieser Klasse = CCDF)
    owners_dist = (
        df.groupby(["owners_class", "owners_label", "owners_low"], as_index=False)
        .agg(N=("AppID", "size"))
        .sort_values("owners_class")
    )
    owners_dist["Anteil"] = owners_dist["N"] / len(df)
    owners_dist["Anteil_mindestens"] = owners_dist["N"][::-1].cumsum()[::-1] / len(df)
    write(owners_dist, "owners_distribution", created)

    # --- Erfolg vs. Nicht-Erfolg (ohne Besitzer selbst: tautologisch)
    comparison = []
    by_success_specs = [
        ("peak_ccu", df, "Analysestichprobe"),
        ("list_price", paid_clean, "kostenpflichtig, ohne Preisausreißer"),
        ("price_current", paid, "kostenpflichtig, ohne Preisausreißer"),
        ("discount", paid, "kostenpflichtig, ohne Preisausreißer"),
        ("dlc_count", df, "Analysestichprobe"),
        ("release_year", df, "Analysestichprobe"),
        ("age_years", df, "Analysestichprobe"),
        ("n_platforms", df, "Analysestichprobe"),
        ("n_genres", df, "Analysestichprobe"),
        ("review_count", df, "Analysestichprobe"),
        ("positive_ratio", rating, f"Bewertungsstichprobe (>= {MIN_REVIEWS} Reviews)"),
        ("wilson", rating, f"Bewertungsstichprobe (>= {MIN_REVIEWS} Reviews)"),
    ]
    for col, frame, basis in by_success_specs:
        for group, label in [(True, "Erfolgreich (>= 20.000 Besitzer)"), (False, "Nicht erfolgreich")]:
            s = pd.to_numeric(frame.loc[frame["success_top25"] == group, col], errors="coerce").dropna()
            comparison.append(
                {"Variable": col, "Basis": basis, "Gruppe": label, "N": len(s), "Median": s.median(),
                 "Mittelwert": s.mean(), "Q1": s.quantile(0.25), "Q3": s.quantile(0.75)}
            )
    write(pd.DataFrame(comparison), "descriptives_by_success", created)

    # --- Steam-Genres
    def genre_col(genre: str) -> str:
        return "genre_" + genre.lower().replace(" ", "_").replace("&", "").replace("__", "_")

    genre_rows = []
    for genre in STANDARD_GENRES:
        col = genre_col(genre)
        if col not in df.columns:
            warn(f"Genre-Spalte {col} fehlt.")
            continue
        part = df[df[col].astype(bool)]
        if part.empty:
            warn(f"Genre {genre}: keine Spiele in der Analysestichprobe.")
        part_paid = part[(part["list_price"] > 0) & ~part["price_outlier"]]
        genre_rows.append(
            {
                "Genre": genre,
                **group_stats(part),
                "Anteil_an_Analysestichprobe": len(part) / len(df),
                **standardized_rate(part, weights),
                "Median_Listenpreis_kostenpflichtig": part_paid["list_price"].median(),
                "Anteil_kostenlos": part["is_free"].mean(),
                "Anteil_F2P_Tag": part["f2p_tag"].mean(),
            }
        )
    genres = pd.DataFrame(genre_rows).sort_values("Erfolgsquote", ascending=False)
    genre_cols = ["Genre", "N", "Anteil_an_Analysestichprobe", "Erfolgsquote", "CI_unten", "CI_oben",
                  "Erfolgsquote_standardisiert", "CI_unten_std", "CI_oben_std", "Anteil_ab_100k",
                  "Anteil_ab_1Mio", "Mittleres_Alter", "Median_Listenpreis_kostenpflichtig",
                  "Anteil_kostenlos", "Anteil_F2P_Tag"]
    genres = write(genres[genre_cols], "genre_overview", created)

    # --- Spielmodi, Plattformanzahl, Plattformkombination
    mode_rows = []
    for col in ["mode", "n_platforms", "platform_combo"]:
        for value, group in df.groupby(col, dropna=False):
            label = PLATFORM_LABELS.get(str(value), str(value)) if col == "platform_combo" else str(value)
            mode_rows.append({"Merkmal": col, "Ausprägung": label, "Code": str(value), **group_stats(group),
                              **standardized_rate(group, weights)})
    modes = pd.DataFrame(mode_rows)
    write(modes, "mode_platform_overview", created)

    combo_rows = []
    combo_code = df["platform_combo"].where(df["platform_combo"].isin(PLATFORM_LABELS), "ohne Windows")
    for code in list(PLATFORM_LABELS) + ["ohne Windows"]:
        part = df[combo_code == code]
        combo_rows.append(
            {
                "Kennung": code,
                "Kombination": PLATFORM_LABELS.get(code, "ohne Windows (nur macOS/Linux)"),
                **group_stats(part),
                "Anteil_an_Analysestichprobe": len(part) / len(df),
                **standardized_rate(part, weights),
                **cohort_rates(part),
            }
        )
    combos = pd.DataFrame(combo_rows)
    write(combos, "platform_combo_overview", created)

    # --- Preis nach Listenpreis (Tagespreis als Zusatzspalte), Ausreisser und
    # 54,99-Cluster getrennt
    clean = df[~df["price_outlier"] & ~df["price_cluster"]]
    current_rates = {
        value: proportion_ci(group["success_top25"])
        for value, group in clean.groupby("price_class_current")
    }
    price_rows = []
    for value in PRICE_ORDER:
        part = clean[clean["price_class"] == value]
        rate_cur, _, _, n_cur = current_rates.get(value, (np.nan, np.nan, np.nan, 0))
        price_rows.append(
            {
                "Kennung": PRICE_KEYS[value],
                "Preisklasse": value,
                **group_stats(part),
                "Anteil_an_Analysestichprobe": len(part) / len(df),
                **standardized_rate(part, weights),
                "N_Tagespreisklasse": n_cur,
                "Erfolgsquote_Tagespreisklasse": rate_cur,
                "Hinweis": (f"ohne Preisausreißer (Listenpreis > 100 USD) und ohne {de(CLUSTER_PRICE, 2)}-USD-Cluster"
                            if value == "40+" else ""),
            }
        )
    cluster = df[df["price_cluster"]]
    price_rows.append(
        {"Kennung": "cluster_5499", "Preisklasse": f"{de(CLUSTER_PRICE, 2)}-Cluster (ab {CLUSTER_FROM_YEAR})",
         **group_stats(cluster), "Anteil_an_Analysestichprobe": len(cluster) / len(df),
         **standardized_rate(cluster, weights), "N_Tagespreisklasse": np.nan, "Erfolgsquote_Tagespreisklasse": np.nan,
         "Hinweis": f"Listenpreis exakt {de(CLUSTER_PRICE, 2)} USD, Release ab {CLUSTER_FROM_YEAR}; getrennt ausgewiesen"}
    )
    outliers = df[df["price_outlier"]]
    price_rows.append(
        {"Kennung": "outlier", "Preisklasse": "Ausreißer (> 100)", **group_stats(outliers),
         "Anteil_an_Analysestichprobe": len(outliers) / len(df), **standardized_rate(outliers, weights),
         "N_Tagespreisklasse": np.nan, "Erfolgsquote_Tagespreisklasse": np.nan,
         "Hinweis": "Listenpreis > 100 USD, getrennt ausgewiesen"}
    )
    no_list = df[df["list_price"].isna()]
    price_rows.append(
        {"Kennung": "missing", "Preisklasse": "Listenpreis fehlt", **group_stats(no_list),
         "Anteil_an_Analysestichprobe": len(no_list) / len(df), **standardized_rate(no_list, weights),
         "N_Tagespreisklasse": np.nan, "Erfolgsquote_Tagespreisklasse": np.nan,
         "Hinweis": (f"Listenpreis nicht rekonstruierbar: 100 % Rabatt ({int((no_list['discount'] >= 100).sum())}) "
                     f"oder Tagespreis 0 trotz Rabatt 1–99 % ({int(no_list['price_zero_with_discount'].sum())})")}
    )
    prices = pd.DataFrame(price_rows)
    price_cols = ["Kennung", "Preisklasse", "N", "Anteil_an_Analysestichprobe", "Erfolgsquote", "CI_unten",
                  "CI_oben", "Erfolgsquote_standardisiert", "CI_unten_std", "CI_oben_std", "Anteil_ab_100k",
                  "Anteil_ab_1Mio", "Mittleres_Alter", "N_Tagespreisklasse", "Erfolgsquote_Tagespreisklasse",
                  "Hinweis"]
    prices = write(prices[price_cols], "price_overview", created)
    assert prices["N"].sum() == len(df), "Preisklassen decken die Analysestichprobe nicht vollstaendig ab"

    # --- Zusammensetzung der Klasse 40+ (nach Listenpreis)
    c40 = df[df["price_class"] == "40+"].copy()
    c40["publisher1"] = c40["Publishers"].fillna("").astype(str).str.split(",").str[0].str.strip()
    is_cluster = c40["price_cluster"]
    parts = [
        ("outlier", "Listenpreis > 100 USD (Preisausreißer)", c40["price_outlier"]),
        ("cluster_5499", f"Listenpreis exakt {de(CLUSTER_PRICE, 2)} USD, Release ab {CLUSTER_FROM_YEAR}", is_cluster),
        ("rest", "übrige Titel 40–100 USD", ~c40["price_outlier"] & ~is_cluster),
        ("no_outlier", "40–100 USD einschließlich Cluster", ~c40["price_outlier"]),
        ("all", "Klasse 40+ gesamt", pd.Series(True, index=c40.index)),
    ]
    composition = []
    for key, label, mask in parts:
        part = c40[mask]
        pubs = part["publisher1"].replace("", np.nan).dropna()
        top = pubs.value_counts()
        pub_counts = pubs.value_counts()
        composition.append(
            {
                "Kennung": key,
                "Teilgruppe": label,
                **group_stats(part),
                "Anteil_an_Klasse": len(part) / len(c40),
                "Median_Reviews": part["review_count"].median(),
                "Anteil_Release_2024_2025": (part["release_year"] >= 2024).mean(),
                "N_Publisher": int(pubs.nunique()),
                "Anteil_Publisher_ab_3_Titeln": float(pub_counts[pub_counts >= 3].sum() / len(part)) if len(part) else np.nan,
                "Groesster_Publisher": top.index[0] if len(top) else "",
                "Anteil_groesster_Publisher": float(top.iloc[0] / len(part)) if len(top) else np.nan,
            }
        )
    composition = pd.DataFrame(composition)
    comp_cols = ["Kennung", "Teilgruppe", "N", "Anteil_an_Klasse", "Erfolgsquote", "CI_unten", "CI_oben",
                 "Anteil_ab_100k", "Anteil_ab_1Mio", "Mittleres_Alter", "Median_Reviews",
                 "Anteil_Release_2024_2025", "N_Publisher", "Anteil_Publisher_ab_3_Titeln",
                 "Groesster_Publisher", "Anteil_groesster_Publisher"]
    write(composition[comp_cols], "price_40plus_composition", created)
    # Alle Jahrgaenge mit mindestens einem Titel zu exakt 54,99 USD ausgeben, damit
    # die Wahl von CLUSTER_FROM_YEAR an der vollstaendigen Reihe pruefbar bleibt.
    at_cluster = paid_clean["list_price"].round(2).eq(CLUSTER_PRICE)
    cluster_by_year = at_cluster.groupby(paid_clean["release_year"]).agg(["sum", "size", "mean"])
    cluster_years = cluster_by_year[cluster_by_year["sum"] > 0]
    print(f"Kaufspiele mit Listenpreis {de(CLUSTER_PRICE, 2)} USD je Jahr (alle Jahrgaenge mit mind. einem Titel): "
          + ", ".join(f"{y}: {int(r['sum'])}/{int(r['size'])} ({pct(r['mean'], 2)})" for y, r in cluster_years.iterrows()))
    before = cluster_by_year.loc[cluster_by_year.index < CLUSTER_FROM_YEAR]
    print(f"  vor {CLUSTER_FROM_YEAR}: {int(before['sum'].sum())} Titel, hoechstens {int(before['sum'].max())} je Jahr "
          f"(max. Anteil {pct(before['mean'].max(), 2)})")

    # --- Rabatt (nur kostenpflichtige Spiele)
    paid["discount_group"] = pd.cut(paid["discount"], bins=DISCOUNT_BINS, labels=DISCOUNT_LABELS)
    discount_rows = []
    basis_paid = "nur kostenpflichtige Spiele (Listenpreis > 0, ohne Preisausreißer > 100 USD)"
    note_disc = "Rabatt zum Store-Snapshot (ca. März 2026), teils nicht erfasst: Untergrenze"
    for key, label in zip(DISCOUNT_KEYS, DISCOUNT_LABELS):
        part = paid[paid["discount_group"] == label]
        discount_rows.append({"Kennung": key, "Rabattgruppe": label, "Basis": basis_paid, **group_stats(part),
                              "Anteil_an_Basis": len(part) / len(paid), **standardized_rate(part, weights),
                              "Hinweis": note_disc})
    any_disc = paid[paid["has_discount"]]
    discount_rows.append({"Kennung": "d_any", "Rabattgruppe": "Rabatt aktiv (1-100 %)", "Basis": basis_paid,
                          **group_stats(any_disc), "Anteil_an_Basis": len(any_disc) / len(paid),
                          **standardized_rate(any_disc, weights), "Hinweis": note_disc})
    discount_rows.append({"Kennung": "paid_all", "Rabattgruppe": "alle kostenpflichtigen Spiele",
                          "Basis": basis_paid, **group_stats(paid), "Anteil_an_Basis": 1.0,
                          **standardized_rate(paid, weights),
                          "Hinweis": f"Erfolgsquote der Analysestichprobe insgesamt: {pct(overall_rate)}"})
    discounts = pd.DataFrame(discount_rows)
    disc_cols = ["Kennung", "Rabattgruppe", "Basis", "N", "Anteil_an_Basis", "Erfolgsquote", "CI_unten",
                 "CI_oben", "Erfolgsquote_standardisiert", "CI_unten_std", "CI_oben_std", "Anteil_ab_100k",
                 "Anteil_ab_1Mio", "Mittleres_Alter", "Hinweis"]
    discounts = write(discounts[disc_cols], "discount_overview", created)
    assert discounts.loc[discounts["Kennung"].isin(DISCOUNT_KEYS), "N"].sum() == len(paid)

    # --- Kostenlos (Listenpreis 0) mit/ohne F2P-Tag
    free = df[df["is_free"]]
    free_groups = [
        ("all", "Analysestichprobe gesamt", df),
        ("free_all", "kostenlos (Listenpreis 0)", free),
        ("free_f2p", "kostenlos, mit Genre „Free To Play“", free[free["f2p_tag"]]),
        ("free_no_f2p", "kostenlos, ohne Genre „Free To Play“", free[~free["f2p_tag"]]),
        ("paid_all", "kostenpflichtig (Listenpreis > 0)", paid_all),
        ("paid_f2p", "kostenpflichtig, mit Genre „Free To Play“", paid_all[paid_all["f2p_tag"]]),
    ]
    free_rows = [
        {"Kennung": key, "Gruppe": label, **group_stats(part), "Anteil_an_Analysestichprobe": len(part) / len(df),
         **standardized_rate(part, weights)}
        for key, label, part in free_groups
    ]
    free_tbl = pd.DataFrame(free_rows)
    free_cols = ["Kennung", "Gruppe", "N", "Anteil_an_Analysestichprobe", "Erfolgsquote", "CI_unten", "CI_oben",
                 "Erfolgsquote_standardisiert", "CI_unten_std", "CI_oben_std", "Anteil_ab_100k",
                 "Anteil_ab_1Mio", "Mittleres_Alter"]
    free_tbl = write(free_tbl[free_cols], "free_overview", created)

    zero_rows = []
    for key, label, frame, note in [
        ("raw", "Rohdatensatz (alle Einträge)", prepared, "inkl. Nicht-Spiele, Software, Releases ab Stichtag"),
        ("with_owners_zero", "Analysestichprobe + Einträge mit Besitzer „0 - 0“", zero_sample,
         "„0 - 0“ als Nicht-Erfolg gezählt"),
        ("analytic", "Analysestichprobe (ohne „0 - 0“)", df, "Primäranalyse"),
        ("owners_zero_only", "nur Einträge mit Besitzer „0 - 0“", zero_sample[zero_sample["owners_zero"]],
         "in der Primäranalyse ausgeschlossen"),
    ]:
        zero_rows.append(
            {
                "Kennung": key,
                "Basis": label,
                "N": len(frame),
                "N_kostenlos": int(frame["is_free"].sum()),
                "Anteil_kostenlos": frame["is_free"].mean(),
                "N_F2P_Tag": int(frame["f2p_tag"].sum()),
                "Anteil_F2P_Tag": frame["f2p_tag"].mean(),
                "Erfolgsquote": frame["success_top25"].mean() if key != "raw" else np.nan,
                "Hinweis": note,
            }
        )
    write(pd.DataFrame(zero_rows), "free_share_owners_zero", created)

    # --- Kohorten nach Veroeffentlichungsjahr (alle Jahre, auch duenn besetzte)
    zero_by_year = zero_sample.groupby("release_year")
    cohort_rows = []
    for year, part in df.groupby("release_year"):
        zpart = zero_by_year.get_group(year)
        part_paid = part[(part["list_price"] > 0) & ~part["price_outlier"]]
        notes = []
        if len(part) < MIN_YEAR_N:
            notes.append(f"N < {MIN_YEAR_N}")
        if year == STEAMSPY_CUTOFF.year:
            notes.append("nur Januar bis Mai")
        cohort_rows.append(
            {
                "Jahr": int(year),
                "Kohortengruppe": str(cohort_group(part).iloc[0]),
                **group_stats(part),
                "Anteil_kostenlos": part["is_free"].mean(),
                "Anteil_F2P_Tag": part["f2p_tag"].mean(),
                "Median_Listenpreis_kostenpflichtig": part_paid["list_price"].median(),
                "Anteil_Rabatt_kostenpflichtig": part_paid["has_discount"].mean(),
                "Anteil_Multiplayer": part["multi_player"].mean(),
                "N_0_0": int(zpart["owners_zero"].sum()),
                "Anteil_0_0": float(zpart["owners_zero"].mean()),
                "Erfolgsquote_mit_0_0": float(zpart["success_top25"].mean()),
                "Hinweis": "; ".join(notes),
            }
        )
    cohort = write(pd.DataFrame(cohort_rows), "cohort_by_year", created)

    # --- Altersstandardisierte und kohortenstratifizierte Erfolgsquoten
    std_rows = []

    def add_std(merkmal: str, gruppe: str, part: pd.DataFrame, basis: str) -> None:
        std_rows.append({"Merkmal": merkmal, "Gruppe": gruppe, "Basis": basis, **group_stats(part),
                         **standardized_rate(part, weights), **cohort_rates(part)})

    for label in DISCOUNT_LABELS:
        add_std("Rabattgruppe", label, paid[paid["discount_group"] == label], basis_paid)
    add_std("Rabattgruppe", "Rabatt aktiv (1-100 %)", any_disc, basis_paid)
    for code in list(PLATFORM_LABELS) + ["ohne Windows"]:
        add_std("Plattformkombination", PLATFORM_LABELS.get(code, "ohne Windows (nur macOS/Linux)"),
                df[combo_code == code], "Analysestichprobe")
    for genre in STANDARD_GENRES:
        add_std("Genre", genre, df[df[genre_col(genre)].astype(bool)], "Analysestichprobe")
    for value, part in df.groupby("mode"):
        add_std("Spielmodus", str(value), part, "Analysestichprobe")
    add_std("Preis", "kostenlos (Listenpreis 0)", free, "Analysestichprobe")
    add_std("Preis", "kostenpflichtig (Listenpreis > 0)", paid_all, "Analysestichprobe")
    add_std("Gesamt", "Analysestichprobe", df, "Analysestichprobe")
    std_tbl = pd.DataFrame(std_rows)
    std_tbl.insert(3, "Standard", f"Jahrgangsverteilung der Analysestichprobe (Schichten: bis {STD_FIRST_YEAR}, "
                                  f"dann Einzeljahre bis {STEAMSPY_CUTOFF.year})")
    std_tbl = write(std_tbl, "age_standardized_rates", created)
    low_cover = std_tbl[std_tbl["Gewicht_abgedeckt"] < 0.99]
    for _, row in low_cover.iterrows():
        warn(f"Altersstandardisierung {row['Merkmal']} / {row['Gruppe']}: nur "
             f"{pct(row['Gewicht_abgedeckt'])} der Standardgewichte abgedeckt (n = {row['N']}).")

    # --- Bewertungen (Steam auf der Bewertungsstichprobe, Metacritic extern)
    rating_rows = []
    basis_rating = f"Bewertungsstichprobe (>= {MIN_REVIEWS} Reviews)"
    rating_rows += quartile_rows(rating, "positive_ratio", "Positive Ratio", basis_rating)
    rating_rows += quartile_rows(rating, "wilson", "Wilson-Score (Diagnose)", basis_rating)
    rating_rows += quartile_rows(meta, "mc_metascore", "Metacritic (extern)", "externe Metacritic-Teilstichprobe")
    write(pd.DataFrame(rating_rows), "ratings_overview", created)

    meta_coverage = pd.DataFrame(
        [
            {"Kennung": "n_analytic", "Kennzahl": "Analysestichprobe (N)", "Wert": len(df)},
            {"Kennung": "n_mc_external", "Kennzahl": "mit externem Metascore (N)", "Wert": int(df["has_mc_external"].sum())},
            {"Kennung": "cov_mc_external", "Kennzahl": "externe Metacritic-Coverage", "Wert": df["has_mc_external"].mean()},
            {"Kennung": "cov_mc_external_success", "Kennzahl": "externe Metacritic-Coverage unter erfolgreichen Spielen",
             "Wert": df.loc[df["success_top25"], "has_mc_external"].mean()},
            {"Kennung": "cov_mc_external_nonsuccess",
             "Kennzahl": "externe Metacritic-Coverage unter nicht erfolgreichen Spielen",
             "Wert": df.loc[~df["success_top25"], "has_mc_external"].mean()},
            {"Kennung": "n_mc_user", "Kennzahl": "externe Metacritic-Teilstichprobe mit Nutzerscore (N)",
             "Wert": int(meta["mc_user_score"].notna().sum())},
            {"Kennung": "n_fg", "Kennzahl": "mit FronkonGames-Metacritic-Score (N)", "Wert": int(df["has_metacritic"].sum())},
            {"Kennung": "cov_fg", "Kennzahl": "FronkonGames-Metacritic-Coverage", "Wert": df["has_metacritic"].mean()},
        ]
    )
    write(meta_coverage, "metacritic_coverage", created)

    # --- Externe Variablen: Deskription je Teilstichprobe
    full_window = price_hist[price_hist["ph_full_window"].eq(True)]
    igdb_with_genres = igdb[igdb["igdb_has_genres"].eq(True)]
    igdb_with_modes = igdb[igdb["igdb_has_modes"].eq(True)]
    basis_igdb = "IGDB-Teilstichprobe"
    basis_ph = "Preishistorie-Teilstichprobe"
    basis_ph_full = "Preishistorie, volles Fenster (Beginn Nov. oder Dez. 2023 bis Okt. 2025)"
    rated = igdb.dropna(subset=["igdb_rating"])
    integer_share = rated["igdb_rating_integer"].eq(True).mean()
    share_70 = rated["igdb_rating"].eq(70).mean()
    ext_specs = [
        ("igdb_rating", igdb, basis_igdb,
         f"Nutzerbewertung 0–100; {pct(integer_share, 1)} der Werte ganzzahlig, {pct(share_70, 1)} exakt 70"),
        ("igdb_aggregated_rating", igdb, basis_igdb, "Kritikerbewertung 0–100; Platzhalter 0 als fehlend"),
        ("igdb_n_genres", igdb_with_genres, "IGDB-Teilstichprobe mit Genreangabe", ""),
        ("mc_metascore", meta, "externe Metacritic-Teilstichprobe", "Kritikerscore 0–100"),
        ("mc_user_score", meta, "externe Metacritic-Teilstichprobe", "Nutzerscore 0–10, fehlt teils trotz Metascore"),
        ("ph_months", price_hist, basis_ph, "Beobachtungsmonate (Teilmonat 2025-11 verworfen)"),
        ("ph_price_changes_per_month", price_hist, basis_ph, "Preisänderungen je beobachtetem Monat"),
        ("ph_share_months_on_sale", price_hist, basis_ph, "Anteil der Monate mit Rabatt"),
        ("ph_days_on_sale_per_month", price_hist, basis_ph, "Rabatttage je Monat"),
        ("ph_mean_monthly_max_discount_pct", price_hist, basis_ph, "Mittel der monatlichen Höchstrabatte in % (aus dem monatlichen Tiefstpreis)"),
        ("ph_max_discount_pct", full_window, basis_ph_full, "höchster Rabatt (Tiefstpreis ggü. Basispreis); nur bei vollem Fenster vergleichbar"),
        ("ph_base_price", price_hist, basis_ph, "statischer Basispreis der Quelle in USD"),
        ("ph_lowest_price", price_hist, basis_ph, "Tiefstpreis im Fenster in USD"),
    ]
    write(describe_rows(ext_specs), "descriptives_external", created)

    # --- IGDB-Genres (Basis: IGDB-Spiele mit Genreangabe)
    igdb_genre_rows = []
    for col in [c for c in df.columns if c.startswith("igdb_genre_")]:
        part = igdb_with_genres[igdb_with_genres[col].eq(True)]
        key = col.removeprefix("igdb_genre_")
        label = IGDB_GENRE_LABELS.get(key, key.replace("_", " ").title())
        if part.empty:
            warn(f"IGDB-Genre {label}: keine Spiele in der IGDB-Teilstichprobe.")
        part_paid = part[(part["list_price"] > 0) & ~part["price_outlier"]]
        igdb_genre_rows.append(
            {
                "IGDB_Genre": label,
                "Variable": col,
                **group_stats(part),
                "Anteil_an_IGDB_mit_Genreangabe": len(part) / len(igdb_with_genres),
                "Anteil_an_IGDB_Teilstichprobe": len(part) / len(igdb),
                "Median_Listenpreis_kostenpflichtig": part_paid["list_price"].median(),
            }
        )
    igdb_genres = pd.DataFrame(igdb_genre_rows).sort_values("Erfolgsquote", ascending=False)
    igdb_genre_cols = ["IGDB_Genre", "Variable", "N", "Anteil_an_IGDB_mit_Genreangabe",
                       "Anteil_an_IGDB_Teilstichprobe", "Erfolgsquote", "CI_unten", "CI_oben",
                       "Anteil_ab_100k", "Anteil_ab_1Mio", "Mittleres_Alter", "Median_Listenpreis_kostenpflichtig"]
    write(igdb_genres[igdb_genre_cols], "igdb_genre_overview", created)

    # --- IGDB-Spielmodi (Basis: IGDB-Spiele mit Modusangabe) inkl. Referenzzeilen
    basis_modes = "IGDB-Teilstichprobe mit Modusangabe"
    igdb_mode_rows = [
        {"Kennung": "with_modes", "IGDB_Merkmal": "alle mit Modusangabe (Referenz)", "Basis": basis_modes,
         **group_stats(igdb_with_modes), "Anteil_an_Basis": 1.0, "N_ohne_Merkmal": np.nan,
         "Erfolgsquote_ohne_Merkmal": np.nan},
        {"Kennung": "without_modes", "IGDB_Merkmal": "ohne Modusangabe", "Basis": basis_igdb,
         **group_stats(igdb[igdb["igdb_has_modes"].eq(False)]),
         "Anteil_an_Basis": float(igdb["igdb_has_modes"].eq(False).mean()), "N_ohne_Merkmal": np.nan,
         "Erfolgsquote_ohne_Merkmal": np.nan},
    ]
    for col, label in [("igdb_singleplayer", "Singleplayer"), ("igdb_multiplayer", "Multiplayer"),
                       ("igdb_coop", "Kooperation"), ("igdb_mmo", "MMO")]:
        mask = igdb_with_modes[col].eq(True)
        if not mask.any():
            warn(f"IGDB-Modus {label}: n = 0 unter {len(igdb_with_modes):,} Spielen mit Modusangabe.")
        without = igdb_with_modes[igdb_with_modes[col].eq(False)]
        rate_without, _, _, n_without = proportion_ci(without["success_top25"])
        igdb_mode_rows.append(
            {"Kennung": col, "IGDB_Merkmal": label, "Basis": basis_modes, **group_stats(igdb_with_modes[mask]),
             "Anteil_an_Basis": float(mask.mean()), "N_ohne_Merkmal": n_without,
             "Erfolgsquote_ohne_Merkmal": rate_without}
        )
    igdb_modes = pd.DataFrame(igdb_mode_rows)
    igdb_mode_cols = ["Kennung", "IGDB_Merkmal", "Basis", "N", "Anteil_an_Basis", "Erfolgsquote", "CI_unten",
                      "CI_oben", "Anteil_ab_100k", "Anteil_ab_1Mio", "Mittleres_Alter", "N_ohne_Merkmal",
                      "Erfolgsquote_ohne_Merkmal"]
    write(igdb_modes[igdb_mode_cols], "igdb_mode_overview", created)

    # --- IGDB-Bewertungen: Quartile und Struktur der Nutzerbewertung
    def rating_structure(part: pd.DataFrame) -> dict[str, float]:
        if part.empty or part["igdb_rating"].isna().all():
            return {"Anteil_ganzzahlig": np.nan, "Anteil_Wert_70": np.nan, "Median_Reviews": np.nan}
        return {
            "Anteil_ganzzahlig": float(part["igdb_rating_integer"].eq(True).mean()),
            "Anteil_Wert_70": float(part["igdb_rating"].eq(70).mean()),
            "Median_Reviews": float(part["review_count"].median()),
        }

    user_rated = igdb.dropna(subset=["igdb_rating"])
    igdb_rating_rows = []
    igdb_rating_rows += quartile_rows(user_rated, "igdb_rating", "IGDB-Nutzerbewertung",
                                      "IGDB-Teilstichprobe mit Nutzerbewertung", extra=rating_structure)
    non_integer = user_rated[user_rated["igdb_rating_integer"].eq(False)]
    igdb_rating_rows += quartile_rows(non_integer, "igdb_rating", "IGDB-Nutzerbewertung (nicht ganzzahlig)",
                                      "IGDB-Nutzerbewertung nicht ganzzahlig", extra=rating_structure)
    igdb_rating_rows += quartile_rows(igdb, "igdb_aggregated_rating", "IGDB-Kritikerbewertung",
                                      "IGDB-Teilstichprobe mit Kritikerbewertung")
    igdb_ratings = write(pd.DataFrame(igdb_rating_rows), "igdb_rating_overview", created)

    value_rows = []
    integer = user_rated[user_rated["igdb_rating_integer"].eq(True)]
    value_groups = [
        ("all", "alle Nutzerbewertungen", user_rated),
        ("integer", "ganzzahlig", integer),
        ("non_integer", "nicht ganzzahlig", non_integer),
        ("integer_without_70", "ganzzahlig ohne Wert 70", integer[integer["igdb_rating"] != 70]),
    ]
    for value in sorted(integer["igdb_rating"].round().value_counts().loc[lambda s: s >= 100].index):
        value_groups.append((f"value_{int(value)}", f"Wert exakt {int(value)}", user_rated[user_rated["igdb_rating"] == value]))
    for key, label, part in value_groups:
        value_rows.append({"Kennung": key, "Gruppe": label, **group_stats(part),
                           "Anteil_an_Nutzerbewertungen": len(part) / len(user_rated),
                           "Median_Reviews": part["review_count"].median(),
                           "Median_Bewertung": part["igdb_rating"].median()})
    value_tbl = pd.DataFrame(value_rows)
    value_cols = ["Kennung", "Gruppe", "N", "Anteil_an_Nutzerbewertungen", "Erfolgsquote", "CI_unten", "CI_oben",
                  "Anteil_ab_100k", "Anteil_ab_1Mio", "Median_Reviews", "Median_Bewertung", "Mittleres_Alter"]
    value_tbl = write(value_tbl[value_cols], "igdb_rating_values", created)

    # --- Preishistorie: Fensterlaengen und normierte Merkmale (volles Fenster)
    months_rows = []
    for months, part in price_hist.groupby("ph_months"):
        months_rows.append(
            {
                "Monate": int(months),
                **group_stats(part),
                "Anteil": len(part) / len(price_hist),
                "N_volles_Fenster": int(part["ph_full_window"].eq(True).sum()),
                "Median_Release_Jahr": part["release_year"].median(),
                "Erster_Monat_min": part["ph_first_month"].min().strftime("%Y-%m"),
                "Erster_Monat_max": part["ph_first_month"].max().strftime("%Y-%m"),
                "Letzter_Monat_max": part["ph_last_month"].max().strftime("%Y-%m"),
            }
        )
    months_tbl = pd.DataFrame(months_rows)
    months_tbl["Anteil_kumuliert"] = months_tbl["Anteil"].cumsum()
    months_cols = ["Monate", "N", "Anteil", "Anteil_kumuliert", "N_volles_Fenster", "Erfolgsquote", "CI_unten",
                   "CI_oben", "Anteil_ab_100k", "Anteil_ab_1Mio", "Median_Release_Jahr", "Erster_Monat_min",
                   "Erster_Monat_max", "Letzter_Monat_max"]
    months_tbl = write(months_tbl[months_cols], "price_history_months", created)

    ph_features = [
        ("ph_price_changes_per_month", "Preisänderungen je Monat"),
        ("ph_share_months_on_sale", "Anteil Monate mit Rabatt"),
        ("ph_days_on_sale_per_month", "Rabatttage je Monat"),
        ("ph_mean_monthly_max_discount_pct", "Mittlerer monatlicher Maximalrabatt"),
        ("ph_max_discount_pct", "Maximaler Rabatt"),
    ]
    price_hist_rows = []
    for var, label in ph_features:
        price_hist_rows += quartile_rows(
            full_window, var, label, basis_ph_full, key_col="Preismerkmal",
            extra=lambda part: {"Median_Monate": part["ph_months"].median()},
        )
    write(pd.DataFrame(price_hist_rows), "price_history_overview", created)

    # --- Externe Metacritic-Werte nach Quartilen (Nutzerscore nur, wo vorhanden)
    mc_rows = []
    mc_rows += quartile_rows(meta, "mc_metascore", "Metacritic-Kritikerscore", "externe Metacritic-Teilstichprobe")
    mc_rows += quartile_rows(meta, "mc_user_score", "Metacritic-Nutzerscore",
                             "externe Metacritic-Teilstichprobe mit Nutzerscore")
    mc_tbl = write(pd.DataFrame(mc_rows), "external_metacritic_overview", created)

    # --- Korrelationen: Steam einheitlich, externe Quellen je Teilstichprobe
    with_rating = df.assign(positive_ratio=df["positive_ratio"].where(df["rating_sample"]),
                            wilson=df["wilson"].where(df["rating_sample"]))
    steam_vars = ["log_owners", "log_peak_ccu", "release_year", "n_platforms", "n_genres", "dlc_count",
                  "list_price", "discount", "positive_ratio", "wilson"]
    steam_basis = f"Bewertungsstichprobe (>= {MIN_REVIEWS} Reviews), kostenpflichtig, ohne Preisausreißer, vollständige Fälle"
    steam_frame = with_rating[with_rating["rating_sample"] & (with_rating["list_price"] > 0)
                              & ~with_rating["price_outlier"]]
    blocks = [
        ("Steam", steam_frame, steam_vars, steam_basis, True),
        ("IGDB", with_rating[with_rating["has_igdb"]],
         ["log_owners", "release_year", "positive_ratio", "igdb_rating", "igdb_aggregated_rating", "igdb_n_genres"],
         "IGDB-Teilstichprobe, paarweise (Positive Ratio nur ab 10 Reviews)", False),
        ("Metacritic", with_rating[with_rating["has_mc_external"]],
         ["log_owners", "release_year", "positive_ratio", "mc_metascore", "mc_user_score"],
         "externe Metacritic-Teilstichprobe, paarweise (Positive Ratio nur ab 10 Reviews)", False),
        ("Preishistorie", with_rating[with_rating["has_price_history"] & with_rating["ph_full_window"].eq(True)],
         ["log_owners", "release_year", "positive_ratio", "ph_base_price", "ph_price_changes_per_month",
          "ph_share_months_on_sale", "ph_days_on_sale_per_month", "ph_mean_monthly_max_discount_pct",
          "ph_max_discount_pct"],
         "Preishistorie mit vollem Fenster, paarweise (Positive Ratio nur ab 10 Reviews)", False),
    ]
    corr_results = []
    corr_rows: list[dict] = []
    for name, frame, variables, basis, complete in blocks:
        corr, n_matrix, rows = correlation_block(frame, variables, name, basis, complete)
        corr_results.append((name, corr, n_matrix))
        corr_rows += rows
    # log10(1 + Peak CCU) enthaelt in der Steam-Matrix die Nullwerte und mischt damit
    # Aktivitaet am Messtag (Huerde) und Intensitaet. Zusatzzeile: nur Peak > 0 auf
    # denselben vollstaendigen Faellen (nicht Teil der Abbildung).
    steam_complete = steam_frame[steam_vars].apply(pd.to_numeric, errors="coerce").dropna()
    steam_peak = steam_complete[steam_complete["log_peak_ccu"] > 0]
    share_zero = 1 - len(steam_peak) / len(steam_complete)
    _, _, peak_rows = correlation_block(
        steam_peak, ["log_owners", "log_peak_ccu"], "Steam, nur Peak > 0",
        f"wie Steam-Matrix, nur Peak CCU > 0 (in der Steam-Matrix sind {pct(share_zero)} der Fälle Peak = 0)", True)
    corr_rows += peak_rows
    write(pd.DataFrame(corr_rows), "correlation_spearman", created)

    # --- Genre x Kohorte (Daten zu fig_09)
    cohorts = cohort_group(df)
    heat_rows = []
    for genre in genres["Genre"]:
        member = df[genre_col(genre)].astype(bool)
        for label in COHORT_LABELS:
            part = df[member & (cohorts == label)]
            rate, low, high, n = proportion_ci(part["success_top25"])
            heat_rows.append({"Genre": genre, "Kohorte": label, "N": n, "Erfolgsquote": rate, "CI_unten": low,
                              "CI_oben": high, "maskiert": n < MIN_CELL_N})
    heat = write(pd.DataFrame(heat_rows), "genre_cohort_overview", created)

    # ================================================================ Abbildungen
    # 1) Besitzerverteilung: Klassenhaeufigkeiten und CCDF
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw={"wspace": 0.45})
    axes[0].barh(owners_dist["owners_label"], owners_dist["N"], color=COLORS["main"], height=0.7)
    axes[0].set_xscale("log")
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Anzahl Spiele (log-Skala)")
    axes[0].set_title("Spiele je Besitzerklasse (SteamSpy)")
    axes[0].xaxis.set_major_formatter(de_formatter(0))
    for y, n in enumerate(owners_dist["N"]):
        axes[0].text(n * 1.12, y, de(n, 0), va="center", fontsize=7, color=COLORS["ink"])
    axes[0].set_xlim(1, owners_dist["N"].max() * 6)
    axes[0].grid(axis="y", visible=False)
    class_pos = np.arange(len(owners_dist))
    axes[1].plot(class_pos, owners_dist["Anteil_mindestens"] * 100, "-o", color=COLORS["main"], lw=2, ms=4)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Besitzer mindestens (untere Klassengrenze)")
    axes[1].set_ylabel("Anteil Spiele ab dieser Klasse (log-Skala)")
    axes[1].set_title("Komplementäre Verteilung (CCDF)")
    short = {0: "0", 20_000: "20 Tsd.", 50_000: "50 Tsd.", 100_000: "100 Tsd.", 200_000: "200 Tsd.",
             500_000: "500 Tsd.", 1_000_000: "1 Mio.", 2_000_000: "2 Mio.", 5_000_000: "5 Mio.",
             10_000_000: "10 Mio.", 20_000_000: "20 Mio.", 50_000_000: "50 Mio.", 100_000_000: "100 Mio."}
    axes[1].set_xticks(class_pos, [short.get(int(v), de(v, 0)) for v in owners_dist["owners_low"]],
                       rotation=45, ha="right", fontsize=8)
    axes[1].yaxis.set_major_formatter(FuncFormatter(
        lambda v, _: f"{de(v, 0 if v >= 1 else 1 if v >= 0.1 else 2 if v >= 0.01 else 3)} %"))
    for x, share, low in zip(class_pos, owners_dist["Anteil_mindestens"], owners_dist["owners_low"]):
        if low in (20_000, 100_000, 1_000_000):
            axes[1].annotate(pct(share), (x, share * 100), textcoords="offset points", xytext=(6, 4), fontsize=7)
    fig.suptitle(f"Verteilung der geschätzten Besitzer (n = {de(len(df), 0)})", fontweight="bold", y=1.02)
    save(fig, "fig_01_owners_distribution", created)

    # 2) Kohorten: Spiele je Jahr und Erfolgsquote (zwei Panels statt Zwillingsachse)
    plot_cohort = cohort.copy()
    small = plot_cohort["N"] < MIN_YEAR_N
    small_years = plot_cohort[small & (plot_cohort["Jahr"] < 2010)]
    pooled_year = int(small_years["Jahr"].max()) if len(small_years) else None
    if pooled_year is not None:
        pooled = df[df["release_year"] <= pooled_year]
        pooled_zero = zero_sample[zero_sample["release_year"] <= pooled_year]
        rate, low, high, n = proportion_ci(pooled["success_top25"])
        pooled_row = {"Jahr": pooled_year, "N": n, "Erfolgsquote": rate, "CI_unten": low, "CI_oben": high,
                      "Erfolgsquote_mit_0_0": pooled_zero["success_top25"].mean()}
        plot_cohort = pd.concat([pd.DataFrame([pooled_row]),
                                 plot_cohort[plot_cohort["Jahr"] > pooled_year]], ignore_index=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6.4), sharex=True,
                                   gridspec_kw={"height_ratios": [1, 1.4]})
    bar_colors = [COLORS["grey"] if (pooled_year is not None and y == pooled_year) else COLORS["main"]
                  for y in plot_cohort["Jahr"]]
    ax1.bar(plot_cohort["Jahr"], plot_cohort["N"], color=bar_colors, width=0.75)
    ax1.set_ylabel("Anzahl Spiele")
    ax1.yaxis.set_major_formatter(de_formatter(0))
    ax1.set_title("Spiele und Erfolgsquote nach Veröffentlichungsjahr")
    ax1.grid(axis="x", visible=False)
    rate = plot_cohort["Erfolgsquote"] * 100
    ax2.errorbar(plot_cohort["Jahr"], rate,
                 yerr=[rate - plot_cohort["CI_unten"] * 100, plot_cohort["CI_oben"] * 100 - rate],
                 color=COLORS["main"], marker="o", ms=4, lw=2, capsize=2, label="Analysestichprobe (95-%-KI)")
    ax2.plot(plot_cohort["Jahr"], plot_cohort["Erfolgsquote_mit_0_0"] * 100, color=COLORS["accent"],
             marker="s", ms=3.5, lw=1.5, ls="--", label="mit „0 - 0“ als Nicht-Erfolg")
    ax2.set_ylabel("Erfolgsquote (≥ 20.000 Besitzer)")
    ax2.set_xlabel("Veröffentlichungsjahr")
    ax2.yaxis.set_major_formatter(de_formatter(0, " %"))
    ax2.set_ylim(0, 105)
    handles, labels = ax2.get_legend_handles_labels()
    order = [labels.index(lab) for lab in sorted(labels, key=lambda t: not t.startswith("Analyse"))]
    ax2.legend([handles[i] for i in order], [labels[i] for i in order], loc="upper right", fontsize=8)
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    ticks = [int(y) for y in plot_cohort["Jahr"]]
    ax2.set_xticks(ticks)
    ax2.set_xticklabels([f"≤ {y}" if (pooled_year is not None and y == pooled_year) else
                         (f"{y}*" if y == STEAMSPY_CUTOFF.year else str(y)) for y in ticks],
                        rotation=45, ha="right", fontsize=8)
    note = (f"≤ {pooled_year}: {de(int(small_years['N'].sum()), 0)} Spiele aus Jahren mit n < {MIN_YEAR_N} "
            f"zusammengefasst (grau). * {STEAMSPY_CUTOFF.year}: nur Januar bis Mai.") if pooled_year else ""
    fig.text(0.01, -0.02, note, fontsize=8, color=COLORS["ink"])
    fig.tight_layout()
    save(fig, "fig_02_release_cohorts", created)

    # 3) Genres: rohe und altersstandardisierte Erfolgsquote
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    g = genres.sort_values("Erfolgsquote").reset_index(drop=True)
    rate_bars(ax, [f"{name} (n = {de(n, 0)})" for name, n in zip(g["Genre"], g["N"])], g, COLORS["main"],
              horizontal=True, n_labels=False)
    ax.plot(g["Erfolgsquote_standardisiert"] * 100, np.arange(len(g)), "D", color=COLORS["accent"], ms=5.5,
            mec="white", mew=0.8, label="altersstandardisiert")
    ax.axvline(overall_rate * 100, color=COLORS["grey"], ls="--", lw=1, label=f"Gesamt: {pct(overall_rate)}")
    ax.set_xlabel("Erfolgsquote (≥ 20.000 Besitzer), Balken roh mit 95-%-KI")
    ax.set_title("Erfolgsquote nach Steam-Genre")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="y", visible=False)
    save(fig, "fig_03_genre_success", created)

    # 4) Spielmodi und Plattformkombinationen
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), gridspec_kw={"width_ratios": [1, 1.1]})
    mode_data = modes[modes["Merkmal"] == "mode"].sort_values("Erfolgsquote").reset_index(drop=True)
    rate_bars(axes[0], [f"{a}\n(n = {de(n, 0)})" for a, n in zip(mode_data["Ausprägung"], mode_data["N"])],
              mode_data, COLORS["main"], horizontal=True, n_labels=False)
    axes[0].plot(mode_data["Erfolgsquote_standardisiert"] * 100, np.arange(len(mode_data)), "D",
                 color=COLORS["accent"], ms=5.5, mec="white", mew=0.8, label="altersstandardisiert")
    axes[0].set_xlabel("Erfolgsquote (≥ 20.000 Besitzer)")
    axes[0].set_title("Spielmodi (Steam-Kategorien)")
    axes[0].axvline(overall_rate * 100, color=COLORS["grey"], ls="--", lw=1)
    axes[0].legend(loc="lower right", fontsize=8)
    axes[0].grid(axis="y", visible=False)
    combo_plot = combos[combos["Kennung"].isin(PLATFORM_LABELS)].reset_index(drop=True)
    combo_ticks = [PLATFORM_LABELS[c].replace(" + ", " +\n") + f"\n(n = {de(n, 0)})"
                   for c, n in zip(combo_plot["Kennung"], combo_plot["N"])]
    rate_bars(axes[1], combo_ticks, combo_plot, COLORS["main"], n_labels=False)
    axes[1].plot(np.arange(len(combo_plot)), combo_plot["Erfolgsquote_standardisiert"] * 100, "D",
                 color=COLORS["accent"], ms=5.5, mec="white", mew=0.8, label="altersstandardisiert")
    axes[1].axhline(overall_rate * 100, color=COLORS["grey"], ls="--", lw=1)
    axes[1].set_ylabel("Erfolgsquote (≥ 20.000 Besitzer)")
    n_other = int(combos.loc[combos["Kennung"] == "ohne Windows", "N"].iloc[0])
    axes[1].set_title(f"Plattformkombinationen (ohne {n_other} Spiele ohne Windows)")
    axes[1].tick_params(axis="x", labelsize=8)
    axes[1].legend(loc="upper left", fontsize=8)
    axes[1].grid(axis="x", visible=False)
    fig.text(0.01, -0.07, "Balken: rohe Erfolgsquote mit 95-%-Wilson-KI; gestrichelt: Gesamtquote.",
             fontsize=8, color=COLORS["ink"])
    save(fig, "fig_04_modes_platforms", created)

    # 5) Preis: Besitzerklassen je Listenpreisklasse und kostenlos vs. kostenpflichtig
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), gridspec_kw={"width_ratios": [1.6, 1]})
    price_display = {"Free": "kostenlos", "40+": "40–100"}
    price_groups = [(PRICE_KEYS[v], price_display.get(v, dash(v)), clean[clean["price_class"] == v])
                    for v in PRICE_ORDER]
    price_groups.append(("cluster_5499", f"{de(CLUSTER_PRICE, 2)}-\nCluster", cluster))
    price_groups.append(("outlier", "> 100\n(Ausreißer)", outliers))
    owner_colors = plt.get_cmap(SEQ_CMAP)(np.linspace(0.25, 0.95, len(OWNER_GROUPS)))
    bottoms = np.zeros(len(price_groups))
    for (lo, hi, label), color in zip(OWNER_GROUPS, owner_colors):
        shares = np.array([((part["owners_low"] >= lo) & (part["owners_low"] < hi)).mean() * 100
                           for _, _, part in price_groups])
        axes[0].bar(np.arange(len(price_groups)), shares, bottom=bottoms, color=color, width=0.72,
                    edgecolor="white", linewidth=1, label=label)
        bottoms += shares
    axes[0].set_xticks(np.arange(len(price_groups)),
                       [f"{label}\nn = {de(len(part), 0)}" for _, label, part in price_groups], fontsize=8)
    axes[0].set_xlabel("Listenpreisklasse in USD (rekonstruiert, Untergrenze)")
    axes[0].set_ylabel("Anteil der Spiele")
    axes[0].yaxis.set_major_formatter(de_formatter(0, " %"))
    axes[0].set_ylim(0, 100)
    axes[0].set_title("Besitzerklassen nach Listenpreis")
    axes[0].legend(title="Besitzer", fontsize=8, title_fontsize=8, ncol=len(OWNER_GROUPS),
                   loc="upper center", bbox_to_anchor=(0.5, -0.24))
    axes[0].grid(axis="x", visible=False)
    free_plot = free_tbl[free_tbl["Kennung"].isin(["free_f2p", "free_no_f2p", "paid_all"])].reset_index(drop=True)
    free_labels = {"free_f2p": "kostenlos,\nmit F2P-Tag", "free_no_f2p": "kostenlos,\nohne F2P-Tag",
                   "paid_all": "kostenpflichtig\n"}
    rate_bars(axes[1], [f"{free_labels[k]}\nn = {de(n, 0)}" for k, n in zip(free_plot["Kennung"], free_plot["N"])],
              free_plot, COLORS["main"], n_labels=False)
    axes[1].plot(np.arange(len(free_plot)), free_plot["Erfolgsquote_standardisiert"] * 100, "D",
                 color=COLORS["accent"], ms=5.5, mec="white", mew=0.8, label="altersstandardisiert")
    axes[1].set_ylabel("Erfolgsquote (≥ 20.000 Besitzer)")
    axes[1].set_title("Kostenlos (Listenpreis 0) vs. kostenpflichtig")
    axes[1].tick_params(axis="x", labelsize=8)
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, "fig_05_price", created)

    # 6) Rabatt: roh/altersstandardisiert und innerhalb der Kohorten
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    disc_plot = discounts[discounts["Kennung"].isin(DISCOUNT_KEYS)].reset_index(drop=True)
    rate_bars(axes[0], [f"{dash(r)}\nn = {de(n, 0)}" for r, n in zip(disc_plot["Rabattgruppe"], disc_plot["N"])],
              disc_plot, COLORS["main"], n_labels=False)
    axes[0].plot(np.arange(len(disc_plot)), disc_plot["Erfolgsquote_standardisiert"] * 100, "D",
                 color=COLORS["accent"], ms=5.5, mec="white", mew=0.8, label="altersstandardisiert")
    paid_rate = float(discounts.loc[discounts["Kennung"] == "paid_all", "Erfolgsquote"].iloc[0])
    axes[0].axhline(paid_rate * 100, color=COLORS["grey"], ls="--", lw=1,
                    label=f"alle kostenpflichtigen: {pct(paid_rate)}")
    axes[0].set_ylabel("Erfolgsquote (≥ 20.000 Besitzer)")
    axes[0].set_xlabel("Rabatt zum Store-Snapshot (Untergrenze)")
    axes[0].set_title("Erfolgsquote nach Rabatt (nur kostenpflichtige Spiele)")
    axes[0].tick_params(axis="x", labelsize=8)
    axes[0].legend(loc="upper left", fontsize=8)
    axes[0].grid(axis="x", visible=False)
    strata = ["2010-2014", "2015-2019", "2020-2022", "2023-2025"]
    strata_colors = plt.get_cmap(SEQ_CMAP)(np.linspace(0.4, 0.95, len(strata)))
    paid_cohort = cohort_group(paid)
    x = np.arange(len(DISCOUNT_LABELS))
    for offset, (label, color) in zip(np.linspace(-0.15, 0.15, len(strata)), zip(strata, strata_colors)):
        stats = [proportion_ci(paid.loc[(paid_cohort == label) & (paid["discount_group"] == d), "success_top25"])
                 for d in DISCOUNT_LABELS]
        r = np.array([s[0] for s in stats]) * 100
        lo = np.array([s[1] for s in stats]) * 100
        hi = np.array([s[2] for s in stats]) * 100
        n_total = sum(s[3] for s in stats)
        suffix = " (2025: Jan.–Mai)" if label == "2023-2025" else ""
        axes[1].errorbar(x + offset, r, yerr=[r - lo, hi - r], color=color, marker="o", ms=4, lw=1.8,
                         capsize=2, label=f"{label.replace('-', '–')}{suffix}, n = {de(n_total, 0)}")
    axes[1].set_xticks(x, [dash(lab) for lab in DISCOUNT_LABELS], fontsize=8)
    axes[1].set_ylabel("Erfolgsquote (≥ 20.000 Besitzer)")
    axes[1].set_xlabel("Rabatt zum Store-Snapshot (Untergrenze)")
    axes[1].yaxis.set_major_formatter(de_formatter(0, " %"))
    axes[1].set_ylim(0, 100)
    axes[1].set_title("Innerhalb der Veröffentlichungskohorten (nur kostenpflichtig)")
    n_early = int((paid_cohort == COHORT_LABELS[0]).sum())
    axes[1].legend(title=f"Kohorte (bis 2009 nicht gezeigt, n = {de(n_early, 0)})", fontsize=7.5,
                   title_fontsize=8, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    axes[1].grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, "fig_06_discount", created)

    # 7) Bewertungen und Besitzer
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    d = rating.dropna(subset=["positive_ratio", "log_owners"])
    hb = axes[0].hexbin(d["positive_ratio"], d["log_owners"], gridsize=(40, 26), cmap=SEQ_CMAP, mincnt=1, bins="log")
    axes[0].set_xlabel(f"Positive Ratio (Steam, ≥ {MIN_REVIEWS} Reviews)")
    axes[0].set_ylabel("Besitzer (Klassenmittelpunkt, log-Skala)")
    owners_axis(axes[0])
    axes[0].xaxis.set_major_formatter(de_formatter(1))
    axes[0].set_title(f"Steam-Nutzerbewertung und Besitzer (n = {de(len(d), 0)})")
    cb = fig.colorbar(hb, ax=axes[0], label="Anzahl Spiele (log)")
    cb.ax.yaxis.set_major_formatter(de_formatter(0))
    d = meta.dropna(subset=["mc_metascore", "log_owners"])
    axes[1].scatter(d["mc_metascore"], d["log_owners"], s=8, alpha=0.25, color=COLORS["accent"], linewidths=0)
    axes[1].set_xlabel("Metacritic-Kritikerscore (extern)")
    axes[1].set_ylabel("Besitzer (Klassenmittelpunkt, log-Skala)")
    owners_axis(axes[1])
    axes[1].set_title(f"Externes Metacritic und Besitzer (n = {de(len(d), 0)})")
    fig.tight_layout()
    save(fig, "fig_07_ratings", created)

    # 8) Korrelationen je Teilstichprobe (Spearman)
    fig, axes = plt.subplots(2, 2, figsize=(14, 13), gridspec_kw={"hspace": 0.45, "wspace": 0.45})
    mesh = None
    for ax, (name, corr, n_matrix) in zip(axes.ravel(), corr_results):
        mesh = heatmap(ax, corr, f"{name} ({n_range_text(n_matrix)})")
    cbar = fig.colorbar(mesh.collections[0], ax=axes.ravel().tolist(), shrink=0.45, label="Spearman ρ")
    cbar.ax.yaxis.set_major_formatter(de_formatter(1))
    fig.suptitle("Spearman-Korrelationen: Steam einheitlich (Bewertungsstichprobe ≥ 10 Reviews, kostenpflichtig, "
                 "ohne Preisausreißer),\nexterne Quellen je Teilstichprobe (Preishistorie: volles Fenster)",
                 fontweight="bold", y=0.95)
    save(fig, "fig_08_correlation_heatmap", created)

    # 9) Genre x Kohorte (alle 10 Genres, Zellen mit n < 30 maskiert)
    pivot = heat.pivot(index="Genre", columns="Kohorte", values="Erfolgsquote").loc[genres["Genre"], COHORT_LABELS]
    n_pivot = heat.pivot(index="Genre", columns="Kohorte", values="N").loc[genres["Genre"], COHORT_LABELS]
    masked = n_pivot < MIN_CELL_N
    annot = pivot.copy().astype(object)
    for gi in pivot.index:
        for ci in pivot.columns:
            n = int(n_pivot.loc[gi, ci])
            annot.loc[gi, ci] = f"n = {n}" if masked.loc[gi, ci] else f"{de(pivot.loc[gi, ci] * 100, 1)}\nn = {de(n, 0)}"
    fig, ax = plt.subplots(figsize=(9.5, 6.4))
    ax.set_facecolor("#ecebe7")
    sns.heatmap(pivot * 100, mask=masked, annot=annot, fmt="", cmap=SEQ_CMAP, vmin=0, vmax=100, ax=ax,
                linewidths=1, linecolor="white", annot_kws={"fontsize": 7.5},
                cbar_kws={"label": "Erfolgsquote in %", "format": de_formatter(0)})
    for (gi, ci), is_masked in masked.stack().items():
        if is_masked:
            ax.text(list(pivot.columns).index(ci) + 0.5, list(pivot.index).index(gi) + 0.5,
                    annot.loc[gi, ci], ha="center", va="center", fontsize=7.5, color=COLORS["ink"])
    ax.grid(False)
    ax.set_title(f"Erfolgsquote nach Genre und Veröffentlichungskohorte (grau: n < {MIN_CELL_N})")
    ax.set_xlabel("Veröffentlichungskohorte (2025: nur Januar bis Mai)")
    ax.set_xticklabels([c.replace("-", "–") for c in COHORT_LABELS])
    ax.set_ylabel("")
    save(fig, "fig_09_genre_cohort", created)

    # 10) Kritiker- vs. Nutzerbewertung
    d = meta.dropna(subset=["mc_metascore"])
    d = d[d["rating_sample"]].dropna(subset=["positive_ratio"])
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    ax.scatter(d["mc_metascore"], d["positive_ratio"], s=8, alpha=0.3, color=COLORS["main"], linewidths=0)
    ax.set_xlabel("Metacritic-Kritikerscore (extern)")
    ax.set_ylabel(f"Positive Ratio (Steam, ≥ {MIN_REVIEWS} Reviews)")
    ax.yaxis.set_major_formatter(de_formatter(1))
    ax.set_title(f"Kritiker- vs. Nutzerbewertung (n = {de(len(d), 0)})")
    save(fig, "fig_10_rating_agreement", created)

    # 11) IGDB-Bewertungen: Wertestruktur und Erfolgsquote nach Quartilen
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), gridspec_kw={"width_ratios": [1.2, 1.1, 1]})
    bins = np.arange(10, 101, 2)
    axes[0].hist([integer["igdb_rating"], non_integer["igdb_rating"]], bins=bins, stacked=True,
                 color=[COLORS["accent"], COLORS["main"]], edgecolor="white", linewidth=0.5,
                 label=[f"ganzzahlig (n = {de(len(integer), 0)})", f"nicht ganzzahlig (n = {de(len(non_integer), 0)})"])
    share70 = float(value_tbl.loc[value_tbl["Kennung"] == "value_70", "Anteil_an_Nutzerbewertungen"].iloc[0]) \
        if (value_tbl["Kennung"] == "value_70").any() else np.nan
    axes[0].annotate(f"Wert 70: {pct(share70)}\naller Nutzerbewertungen", xy=(71, (user_rated["igdb_rating"] == 70).sum()),
                     xytext=(80, (user_rated["igdb_rating"] == 70).sum() * 0.85), fontsize=8,
                     arrowprops={"arrowstyle": "->", "color": COLORS["ink"], "lw": 0.8})
    axes[0].set_xlabel("IGDB-Nutzerbewertung (Klassenbreite 2)")
    axes[0].set_ylabel("Anzahl Spiele")
    axes[0].yaxis.set_major_formatter(de_formatter(0))
    axes[0].set_title(f"Verteilung der IGDB-Nutzerbewertung (n = {de(len(user_rated), 0)})")
    axes[0].legend(fontsize=8, loc="upper left")
    user_q = ranked_quartiles(user_rated["igdb_rating"])
    user_tbl = igdb_ratings[igdb_ratings["Bewertungsmaß"] == "IGDB-Nutzerbewertung"].reset_index(drop=True)
    width = 0.38
    for offset, (flag, label, color) in zip([-width / 2, width / 2],
                                            [(True, "ganzzahlig", COLORS["accent"]),
                                             (False, "nicht ganzzahlig", COLORS["main"])]):
        stats = [proportion_ci(user_rated.loc[(user_q == q) & user_rated["igdb_rating_integer"].eq(flag), "success_top25"])
                 for q in QUARTILE_LABELS]
        r = np.array([s[0] for s in stats]) * 100
        lo = np.array([s[1] for s in stats]) * 100
        hi = np.array([s[2] for s in stats]) * 100
        pos = np.arange(len(QUARTILE_LABELS)) + offset
        axes[1].bar(pos, r, width=width * 0.95, color=color, label=label)
        axes[1].errorbar(pos, r, yerr=[r - lo, hi - r], fmt="none", ecolor=COLORS["ink"], elinewidth=1, capsize=2)
        for xp, s in zip(pos, stats):
            axes[1].text(xp, 1, f"{de(s[3], 0)}", ha="center", va="bottom", fontsize=6.5, color="white", rotation=90)
    axes[1].set_xticks(np.arange(len(QUARTILE_LABELS)), quartile_ticks(user_tbl, 1), fontsize=8)
    axes[1].yaxis.set_major_formatter(de_formatter(0, " %"))
    axes[1].set_ylim(0, 105)
    axes[1].set_ylabel("Erfolgsquote (≥ 20.000 Besitzer)")
    axes[1].set_title(f"Nutzerbewertung: Quartile nach Wertetyp (n = {de(len(user_rated), 0)})")
    axes[1].legend(fontsize=8, loc="upper left")
    axes[1].grid(axis="x", visible=False)
    crit_tbl = igdb_ratings[igdb_ratings["Bewertungsmaß"] == "IGDB-Kritikerbewertung"].reset_index(drop=True)
    rate_bars(axes[2], quartile_ticks(crit_tbl, 1), crit_tbl, COLORS["main"])
    axes[2].set_ylim(0, 105)
    axes[2].set_ylabel("Erfolgsquote (≥ 20.000 Besitzer)")
    axes[2].set_title(f"IGDB-Kritikerbewertung (n = {de(crit_tbl['N'].sum(), 0)})")
    axes[2].tick_params(axis="x", labelsize=8)
    axes[2].grid(axis="x", visible=False)
    fig.text(0.01, -0.04, "Quartile bindungstreu (gleiche Werte in derselben Gruppe), Wertebereich unter dem "
             "Quartil; Fehlerbalken: 95-%-Wilson-KI; Zahlen in den Balken: n.", fontsize=8, color=COLORS["ink"])
    fig.tight_layout()
    save(fig, "fig_11_igdb_ratings", created)

    # 12) Preishistorie: Fensterlaengen und normierte Merkmale bei vollem Fenster
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), gridspec_kw={"width_ratios": [1.1, 1, 1]})
    colors_months = [COLORS["main"] if v > 0 else COLORS["grey"] for v in months_tbl["N_volles_Fenster"]]
    axes[0].bar(months_tbl["Monate"], months_tbl["N"], color=colors_months, width=0.75)
    axes[0].set_xlabel("Beobachtete Monate (bis Okt. 2025)")
    axes[0].set_ylabel("Anzahl Apps")
    axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    n_full = len(full_window)
    axes[0].set_title(f"Fensterlänge (n = {de(len(price_hist), 0)}; blau: volles Fenster, n = {de(n_full, 0)})",
                      fontsize=10)
    axes[0].grid(axis="x", visible=False)
    for ax, var, label in [(axes[1], "ph_price_changes_per_month", "Preisänderungen je Monat"),
                           (axes[2], "ph_share_months_on_sale", "Anteil Monate mit Rabatt")]:
        d = full_window.dropna(subset=[var, "log_owners"])
        ax.scatter(d[var], d["log_owners"], s=12, alpha=0.4, color=COLORS["main"], linewidths=0)
        ax.set_xlabel(f"{label} (volles Fenster)")
        ax.set_ylabel("Besitzer (Klassenmittelpunkt, log-Skala)")
        owners_axis(ax)
        ax.xaxis.set_major_formatter(de_formatter(1))
        ax.set_title(f"{label} und Besitzer (n = {de(len(d), 0)})", fontsize=10)
    fig.tight_layout()
    save(fig, "fig_12_price_history", created)

    # 13) Externe Metacritic-Daten: Nutzer- vs. Kritikerscore, Quartile
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    d = meta.dropna(subset=["mc_metascore", "mc_user_score"])
    axes[0].scatter(d["mc_metascore"], d["mc_user_score"], s=8, alpha=0.3, color=COLORS["accent"], linewidths=0)
    axes[0].set_xlabel("Metacritic-Kritikerscore (0–100)")
    axes[0].set_ylabel("Metacritic-Nutzerscore (0–10)")
    axes[0].yaxis.set_major_formatter(de_formatter(0))
    axes[0].set_title(f"Nutzer- vs. Kritikerscore (n = {de(len(d), 0)})")
    meta_q_tbl = mc_tbl[mc_tbl["Bewertungsmaß"] == "Metacritic-Kritikerscore"].reset_index(drop=True)
    meta_groups = ranked_quartiles(meta["mc_metascore"])
    width = 0.38
    for offset, (col, label, color) in zip([-width / 2, width / 2],
                                           [("success", "≥ 20.000 Besitzer (Erfolg)", COLORS["main"]),
                                            ("mio", "≥ 1 Mio. Besitzer", COLORS["accent"])]):
        stats = []
        for q in QUARTILE_LABELS:
            part = meta[meta_groups == q]
            mask = part["success_top25"] if col == "success" else part["owners_low"] >= 1_000_000
            stats.append(proportion_ci(mask))
        r = np.array([s[0] for s in stats]) * 100
        lo = np.array([s[1] for s in stats]) * 100
        hi = np.array([s[2] for s in stats]) * 100
        pos = np.arange(len(QUARTILE_LABELS)) + offset
        axes[1].bar(pos, r, width=width * 0.95, color=color, label=label)
        axes[1].errorbar(pos, r, yerr=[r - lo, hi - r], fmt="none", ecolor=COLORS["ink"], elinewidth=1, capsize=2)
    axes[1].set_xticks(np.arange(len(QUARTILE_LABELS)), quartile_ticks(meta_q_tbl, 0), fontsize=8)
    axes[1].yaxis.set_major_formatter(de_formatter(0, " %"))
    axes[1].set_ylim(0, 105)
    axes[1].set_ylabel("Anteil der Spiele (95-%-KI)")
    axes[1].set_title(f"Besitzeranteile nach Metascore-Quartil (n = {de(len(meta), 0)})")
    axes[1].legend(fontsize=8, loc="upper left")
    axes[1].grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, "fig_13_external_metacritic", created)

    # 14) Datenquellen-Overlap (direkt aus der Analysestichprobe)
    overlap = (df["n_sources"].value_counts().sort_index().rename_axis("Anzahl_Datenquellen")
               .reset_index(name="N"))
    overlap["Anteil"] = overlap["N"] / len(df)
    overlap_path = TAB / "source_overlap.csv"
    if overlap_path.exists():
        reference = pd.read_csv(overlap_path)
        merged = overlap.merge(reference, on="Anzahl_Datenquellen", how="outer", suffixes=("", "_01"))
        if (merged["N"] != merged["N_01"]).any():
            warn("source_overlap.csv (aus 01) weicht von den Parquet-Daten ab; Abbildung nutzt die Parquet-Daten.")
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.bar(overlap["Anzahl_Datenquellen"].astype(str), overlap["N"], color=COLORS["main"], width=0.7)
    for xpos, (n, share) in enumerate(zip(overlap["N"], overlap["Anteil"])):
        ax.text(xpos, n, f"{de(n, 0)}\n({pct(share)})", ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, overlap["N"].max() * 1.18)
    ax.yaxis.set_major_formatter(de_formatter(0))
    ax.set_xlabel("Anzahl verfügbarer Datenquellen (Steam, IGDB, Metacritic extern, Preishistorie)")
    ax.set_ylabel("Anzahl Spiele")
    ax.set_title(f"Abdeckung der Analysestichprobe nach Datenquellen (n = {de(len(df), 0)})")
    ax.grid(axis="x", visible=False)
    save(fig, "fig_14_source_overlap", created)

    # ================================================================ Konsolenausgabe
    print("EDA abgeschlossen.")
    print(f"Analysestichprobe: {de(len(df), 0)}; Erfolgsquote (>= 20.000 Besitzer): {pct(overall_rate)}")
    print(genres[["Genre", "N", "Erfolgsquote", "Erfolgsquote_standardisiert"]].to_string(index=False))
    coverage = pd.DataFrame(
        [
            {"Quelle": name, "N_gematcht_analytisch": int(df[col].sum()), "Match_Rate_analytisch": df[col].mean()}
            for name, col in [("IGDB", "has_igdb"), ("Metacritic extern (Metascore)", "has_mc_external"),
                              ("Preishistorie", "has_price_history")]
        ]
    )
    print(coverage.to_string(index=False))
    print(f"IGDB mit Modusangabe: {len(igdb_with_modes):,}, mit Genreangabe: {len(igdb_with_genres):,}; "
          f"Preishistorie mit vollem Fenster: {len(full_window):,} von {len(price_hist):,}")
    print(f"Erzeugt: {len(created)} Dateien")
    for path in created:
        print(f"  {path}")


if __name__ == "__main__":
    main()

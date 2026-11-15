"""Statistische Tests fuer H1 (Spieleigenschaften), H2 (Preis), H3 (Bewertungen)
und H4 (externe Quellen) sowie die regelbasierten Hypothesenentscheidungen.

Grundsaetze:
- Erfolg primaer ``success_top25`` (owners_low >= 20.000); jede zentrale
  Kennzahl wird zusaetzlich mit ``success_50k`` und ``success_100k`` gerechnet
  (robustness_success_definitions.csv).
- Alterskontrolle ueber Jahr-Fixed-Effects (seltene aufeinanderfolgende
  Jahrgaenge zusammengefasst). Die quadratische Jahresspezifikation mit
  zentriertem Jahr ist nur Sensitivitaet. Jede Logit-Schaetzung wird auf
  Konvergenz geprueft (Spalte ``konvergiert``).
- Peak CCU (gleichzeitige Spieler am Vortag des SteamSpy-Abrufs, ca. Anfang
  Juni 2025) wird zweiteilig ausgewertet: P(Peak > 0) und Intensitaet nur bei
  Peak > 0.
- p-Werte werden je Hypothesenfamilie (Tabelle) per Benjamini-Hochberg
  korrigiert; neben p-Werten stehen Effektstaerken mit Konfidenzintervall.
- Quartile sind bindungstreu (common.ranked_quartiles); Wertebereiche und n je
  Gruppe stehen in quartile_groups.csv.
- Die Entscheidungsregeln stehen in ``build_decisions``.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2 as chi2_dist
from scipy.stats import chi2_contingency, fisher_exact, kruskal, mannwhitneyu, norm, rankdata, spearmanr
from scipy.stats import t as t_dist
from scipy.stats.contingency import odds_ratio as scipy_odds_ratio
from statsmodels.tools.sm_exceptions import (
    ConvergenceWarning,
    HessianInversionWarning,
    PerfectSeparationError,
    PerfectSeparationWarning,
)

from common import (
    DATA,
    QUARTILE_LABELS,
    RANDOM_STATE,
    STANDARD_GENRES,
    bh_correction,
    cliffs_delta_from_mannwhitney,
    odds_ratio_ci,
    quartile_ranges,
    ranked_quartiles,
    save_table,
)

# ---------------------------------------------------------------------------
# Konstanten
# ---------------------------------------------------------------------------

OUTCOME = "success_top25"
ALT_OUTCOMES = ("success_50k", "success_100k")
ALL_OUTCOMES = (OUTCOME, *ALT_OUTCOMES)

# Jahr-FE: aufeinanderfolgende Jahrgaenge werden zusammengefasst, bis jede
# Gruppe diese Mindestgroesse hat. In der Analysestichprobe ergibt das fuer die
# Logits die Gruppe 1997-2008 und danach Einzeljahre. Fuer partielle
# Rangkorrelationen (lineare Residualisierung, keine Separationsgefahr) genuegt
# eine kleinere Mindestgroesse.
YEAR_FE_MIN_LOGIT = 200
YEAR_FE_MIN_PARTIAL = 30
YEAR_CENTER = 2015
YEAR_SCALE = 10.0

# Kohorten fuer Stratifizierung und Interaktionstests (2025 nur Januar-Mai).
COHORTS = [
    (1997, 2014, "bis 2014"),
    (2015, 2018, "2015–2018"),
    (2019, 2022, "2019–2022"),
    (2023, 2025, "2023–2025"),
]

# Schwellen fuer praktische Relevanz (Entscheidungsregeln).
REL_OR = 1.5
REL_RHO = 0.10
REL_DELTA = 0.147
REL_EPS2 = 0.01
# Quartils-ORs gelten als instabil, wenn der Erfolgsanteil der Teilstichprobe
# ueber diesem Wert liegt oder eine Zelle kleiner als MIN_CELL ist.
UNSTABLE_SHARE = 0.85
MIN_CELL = 5
N_BOOT = 2000

GENRE_COLS = {
    genre: "genre_" + genre.lower().replace(" ", "_").replace("&", "").replace("__", "_")
    for genre in STANDARD_GENRES
}
MODE_COLS = [
    ("single_player", "Singleplayer"),
    ("multi_player", "Multiplayer"),
    ("coop", "Koop"),
    ("pvp", "PvP"),
    ("mmo", "MMO"),
]
PLATFORM_FLAGS = [
    ("mac", "Plattform: macOS"),
    ("linux", "Plattform: Linux"),
    ("multi_platform", "Mehr als eine Plattform"),
]
COMBO_REF = "W"
COMBO_LEVELS = [
    ("W+M", "Windows + macOS"),
    ("W+L", "Windows + Linux"),
    ("W+M+L", "Windows + macOS + Linux"),
]

# Steam-analoge IGDB-Merkmale (Modi auf Spielen mit Modusangabe, Genres auf
# Spielen mit Genreangabe). Action hat in IGDB kein eigenes Genre; die
# Zuordnung ueber Shooter/Fighting/Hack and slash/Platform ist naeherungsweise.
IGDB_MODE_PAIRS = [
    ("igdb_singleplayer", "single_player", "Singleplayer"),
    ("igdb_multiplayer", "multi_player", "Multiplayer"),
    ("igdb_coop", "coop", "Koop"),
    ("igdb_mmo", "mmo", "MMO"),
]
IGDB_GENRE_PAIRS = [
    ("igdb_genre_adventure", "genre_adventure", "Adventure", ["igdb_genre_adventure"]),
    ("igdb_genre_indie", "genre_indie", "Indie", ["igdb_genre_indie"]),
    ("igdb_genre_racing", "genre_racing", "Racing", ["igdb_genre_racing"]),
    ("igdb_genre_role_playing_rpg", "genre_rpg", "RPG", ["igdb_genre_role_playing_rpg"]),
    ("igdb_genre_simulator", "genre_simulation", "Simulation/Simulator", ["igdb_genre_simulator"]),
    ("igdb_genre_sport", "genre_sports", "Sports/Sport", ["igdb_genre_sport"]),
    ("igdb_strategy_any", "genre_strategy", "Strategy (Strategy, RTS, TBS)",
     ["igdb_genre_strategy", "igdb_genre_real_time_strategy_rts", "igdb_genre_turn_based_strategy_tbs"]),
    ("igdb_action_like", "genre_action", "Action (näherungsweise: Shooter, Fighting, Hack and slash, Platform)",
     ["igdb_genre_shooter", "igdb_genre_fighting", "igdb_genre_hack_and_slash_beat_em_up", "igdb_genre_platform"]),
]

PH_FEATURES = [
    ("ph_price_changes_per_month", "Preisänderungen je Monat"),
    ("ph_share_months_on_sale", "Anteil Monate mit Rabatt"),
    ("ph_days_on_sale_per_month", "Rabatttage je Monat"),
    ("ph_mean_monthly_max_discount_pct", "mittlerer monatlicher Maximalrabatt"),
    ("ph_max_discount_pct", "maximaler Rabatt (nur volles Fenster)"),
]
# Der Maximalrabatt waechst mit der Fensterlaenge und ist nur bei vollem
# Fenster vergleichbar.
PH_FULL_WINDOW_ONLY = {"ph_max_discount_pct"}

S_AN = "Analysestichprobe"
S_AN_PRICE = "Analysestichprobe, Listenpreis bestimmbar"
S_PAID ="kostenpflichtig (Listenpreis > 0), ohne Preisausreißer (> 100 USD)"
S_PAID_ALL = "kostenpflichtig (Listenpreis > 0), inkl. Preisausreißer"
S_PAID_CURRENT = "Tagespreis > 0, ohne Preisausreißer"
S_PEAK = "Peak > 0"
S_R10 = "mind. 10 Reviews"
S_R50 = "mind. 50 Reviews"
S_MC = "Metacritic extern (Metascore vorhanden)"
S_IGDB_MODES = "IGDB-Treffer mit Modusangabe"
S_IGDB_GENRES = "IGDB-Treffer mit Genreangabe"
S_PH = "Preishistorie"
S_PH_FULL = "Preishistorie, volles Fenster"
S_ZERO = "Analysestichprobe + „0 - 0“ (als Nicht-Erfolg), Listenpreis bestimmbar"
S_LE2020 = "Analysestichprobe, Releases bis 2020, Listenpreis bestimmbar"

ROW_COLUMNS = [
    "Kennung", "Familie", "Rolle", "Modell", "Test", "Merkmal", "Gruppe", "Stichprobe", "N",
    "Erfolgsdefinition", "Erfolgsquote_Gruppe", "Erfolgsquote_Referenz", "Effekt", "Effekt_typ",
    "CI_unten", "CI_oben", "p", "Zusatz_Effekt", "Zusatz_Effekt_Typ", "Median_Gruppe",
    "Median_Referenz", "Bereich_Gruppe", "Bereich_Referenz", "konvergiert", "instabil", "Hinweis",
    "p_adj", "signifikant",
]

LOGIT_LOG: list[dict] = []
QUARTILE_ROWS: list[dict] = []


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------

def nan_row(**kw) -> dict:
    row = {"Effekt": np.nan, "CI_unten": np.nan, "CI_oben": np.nan, "p": np.nan}
    row.update(kw)
    return row


def fmt(value: float, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "n/a"
    return f"{value:.{digits}g}"


def spearman_ci(rho: float, n: int, controls: int = 0) -> tuple[float, float]:
    if not np.isfinite(rho) or n <= 3 + controls or abs(rho) >= 1:
        return np.nan, np.nan
    z = np.arctanh(rho)
    se = 1.0 / math.sqrt(n - 3 - controls)
    return float(np.tanh(z - 1.96 * se)), float(np.tanh(z + 1.96 * se))


def year_groups(years: pd.Series, min_n: int) -> pd.Series:
    """Fasst aufeinanderfolgende Jahrgaenge zusammen, bis jede Gruppe >= min_n Faelle hat."""
    years = pd.Series(years).astype(int)
    counts = years.value_counts().sort_index()
    bins: list[list[int]] = []
    current: list[int] = []
    total = 0
    for year, n in counts.items():
        current.append(int(year))
        total += int(n)
        if total >= min_n:
            bins.append(current)
            current, total = [], 0
    if current:
        if bins:
            bins[-1].extend(current)
        else:
            bins.append(current)
    mapping = {}
    for group in bins:
        label = str(group[0]) if len(group) == 1 else f"{group[0]}-{group[-1]}"
        for year in group:
            mapping[year] = label
    return years.map(mapping)


def cohort_labels(years: pd.Series) -> pd.Series:
    years = pd.Series(years).astype(int)
    out = pd.Series(pd.NA, index=years.index, dtype="object")
    for start, end, label in COHORTS:
        out[(years >= start) & (years <= end)] = label
    return out


def p_text(p: float, prefix: str = "p") -> str:
    """p-Wert fuer Hinweistexte; ein Unterlauf auf 0 wird als '< 1e-300' ausgewiesen."""
    return f"{prefix} < 1e-300" if p == 0 else f"{prefix} = {p:.2g}"


def open_cohort_labels(years: pd.Series) -> pd.Series:
    """Kohorten wie COHORTS, die frueheste nach unten offen (fuer release_year_alt < 1997)."""
    years = np.floor(pd.Series(years).astype(float))
    out = pd.Series(pd.NA, index=years.index, dtype="object")
    for i, (start, end, label) in enumerate(COHORTS):
        lower = -np.inf if i == 0 else start
        out[(years >= lower) & (years <= end)] = label
    return out


def year_dummies(years: pd.Series, min_n: int = YEAR_FE_MIN_LOGIT) -> pd.DataFrame:
    """Dummy-Kodierung der zusammengefassten Jahrgaenge (Referenz: fruehste Gruppe)."""
    groups = year_groups(np.floor(pd.Series(years).astype(float)).astype(int), min_n)
    return pd.get_dummies(groups, prefix="jahr", drop_first=True, dtype=float)


def fit_logit(y: pd.Series, X: pd.DataFrame, context: str):
    """Logit mit Konvergenzpruefung; bei Nicht-Konvergenz zweiter Versuch mit BFGS.

    Die Designmatrix wird direkt uebergeben (Konstante wird ergaenzt); das ist
    deutlich schneller als die Formelschnittstelle und rechnet dasselbe Modell.
    """
    X = sm.add_constant(X.astype(float), has_constant="add")
    notes: list[str] = []
    model = None
    converged = False
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            model = sm.Logit(y.astype(float), X).fit(disp=0, method="newton", maxiter=100)
            converged = bool(model.mle_retvals.get("converged", False))
            if not converged:
                notes.append("Newton nicht konvergiert, BFGS verwendet")
                model = sm.Logit(y.astype(float), X).fit(disp=0, method="bfgs", maxiter=5000)
                converged = bool(model.mle_retvals.get("converged", False))
        except (PerfectSeparationError, np.linalg.LinAlgError, ValueError) as exc:
            notes.append(f"Schätzung fehlgeschlagen ({type(exc).__name__})")
            model = None
    for w in caught:
        if issubclass(w.category, (ConvergenceWarning, PerfectSeparationWarning, HessianInversionWarning)):
            notes.append(w.category.__name__)
    if model is not None and converged:
        # Quasi-Separation zeigt sich in riesigen Koeffizienten bzw. Standardfehlern.
        if (model.params.abs() > 15).any() or (model.bse > 50).any():
            notes.append("Hinweis auf (Quasi-)Separation")
    note = "; ".join(dict.fromkeys(notes))
    terms = [c for c in X.columns if not c.startswith("jahr_")]
    n_year = sum(c.startswith("jahr_") for c in X.columns)
    LOGIT_LOG.append({"Kontext": context, "Terme": " + ".join(terms) + (f" + {n_year} Jahr-Dummies" if n_year else ""),
                      "N": len(X), "konvergiert": converged, "Hinweis": note})
    if not converged:
        print(f"WARNUNG: Logit nicht konvergiert ({context}): {note}")
    return model, converged, note


def direction(effect: float, effect_type: str) -> int:
    if effect is None or not np.isfinite(effect):
        return 0
    if "Odds Ratio" in str(effect_type):
        if effect <= 0:
            return 0
        value = math.log(effect)
    else:
        value = effect
    return int(np.sign(value))


def is_relevant(effect: float, effect_type: str) -> bool:
    if effect is None or not np.isfinite(effect):
        return False
    effect_type = str(effect_type)
    if "Odds Ratio" in effect_type:
        return effect >= REL_OR or effect <= 1.0 / REL_OR
    if "Spearman" in effect_type:
        return abs(effect) >= REL_RHO
    if "Cliff" in effect_type:
        return abs(effect) >= REL_DELTA
    if "Epsilon" in effect_type or "ε²" in effect_type:
        return effect >= REL_EPS2
    return False


# ---------------------------------------------------------------------------
# Testbausteine (liefern je eine Ergebniszeile als dict)
# ---------------------------------------------------------------------------

def chi2_indicator(df: pd.DataFrame, flag: str, label: str, kennung: str, outcome: str = OUTCOME,
                   stichprobe: str = S_AN, rolle: str = "Haupttest") -> dict:
    d = df[[flag, outcome]].dropna()
    x = d[flag].astype(bool)
    y = d[outcome].astype(bool)
    tab = pd.crosstab(x, y).reindex(index=[False, True], columns=[False, True], fill_value=0)
    n = int(tab.to_numpy().sum())
    row = nan_row(Kennung=kennung, Rolle=rolle, Modell="unadjustiert", Test="Chi² / Odds Ratio",
                  Merkmal=label, Gruppe=f"{flag} = ja vs. nein", Stichprobe=stichprobe, N=n,
                  Erfolgsdefinition=outcome, Effekt_typ="Odds Ratio", Zusatz_Effekt_Typ="Cramers V")
    if (tab.sum(axis=1) == 0).any() or (tab.sum(axis=0) == 0).any():
        row["Hinweis"] = "keine Variation in Merkmal oder Ergebnis"
        return row
    chi2, p, _, _ = chi2_contingency(tab.to_numpy(), correction=False)
    odds, low, high, _ = odds_ratio_ci(tab.to_numpy())
    row.update(
        Erfolgsquote_Gruppe=tab.loc[True, True] / tab.loc[True].sum(),
        Erfolgsquote_Referenz=tab.loc[False, True] / tab.loc[False].sum(),
        Effekt=odds, CI_unten=low, CI_oben=high, p=p,
        Zusatz_Effekt=math.sqrt(chi2 / n) if n else np.nan,
    )
    if tab.to_numpy().min() < MIN_CELL:
        row["Hinweis"] = f"kleinste Zelle {int(tab.to_numpy().min())} < {MIN_CELL}"
    return row


def chi2_categorical(df: pd.DataFrame, var: str, label: str, kennung: str, outcome: str = OUTCOME,
                     stichprobe: str = S_AN, rolle: str = "Zusatz") -> dict:
    d = df[[var, outcome]].dropna()
    tab = pd.crosstab(d[var], d[outcome].astype(bool))
    chi2, p, dof, _ = chi2_contingency(tab.to_numpy(), correction=False)
    n = int(tab.to_numpy().sum())
    v = math.sqrt(chi2 / (n * (min(tab.shape) - 1))) if n and min(tab.shape) > 1 else np.nan
    rates = (tab[True] / tab.sum(axis=1)) if True in tab.columns else pd.Series(dtype=float)
    return nan_row(Kennung=kennung, Rolle=rolle, Modell="unadjustiert", Test="Chi² (kategorial)",
                   Merkmal=label, Gruppe=var, Stichprobe=stichprobe, N=n, Erfolgsdefinition=outcome,
                   Effekt=chi2, Effekt_typ=f"Chi2 (df={dof})", p=p, Zusatz_Effekt=v,
                   Zusatz_Effekt_Typ="Cramers V",
                   Hinweis="Erfolgsquote je Gruppe: " + "; ".join(f"{k}: {v_:.3f}" for k, v_ in rates.items()))


def adjusted_or(df: pd.DataFrame, flag: str, label: str, kennung: str, outcome: str = OUTCOME,
                spec: str = "fe", year_col: str = "release_year", covars: tuple[str, ...] = (),
                stichprobe: str = S_AN, rolle: str | None = None, modell: str | None = None) -> dict:
    """Logit-OR von ``flag`` mit Alterskontrolle (Jahr-FE oder quadratisch) und optionalen Kovariaten."""
    cols = list(dict.fromkeys([flag, outcome, year_col, *covars]))
    d = df[cols].dropna()
    y = d[outcome].astype(bool).astype(int)
    X = pd.DataFrame({"_x": d[flag].astype(bool).astype(int)}, index=d.index)
    for i, c in enumerate(covars):
        X[f"_c{i}"] = (d[c].astype(bool).astype(int) if pd.api.types.is_bool_dtype(d[c])
                       else d[c].astype(float))
    year = np.floor(d[year_col].astype(float)).astype(int)
    if spec == "fe":
        X = X.join(year_dummies(year))
        modell = modell or ("Jahr-FE" if year_col == "release_year" else f"Jahr-FE ({year_col})")
        rolle = rolle or "Adjustiert"
    elif spec == "quad":
        X["_yc"] = (year - YEAR_CENTER) / YEAR_SCALE
        X["_yc2"] = X["_yc"] ** 2
        modell = modell or "Jahr quadratisch, zentriert ((Jahr − 2015)/10)"
        rolle = rolle or "Sensitivität"
    else:
        raise ValueError(spec)
    if covars:
        modell += " + " + ", ".join(covars)
    row = nan_row(Kennung=kennung, Rolle=rolle, Modell=modell, Test="Logistische Regression",
                  Merkmal=label, Gruppe=f"{flag} = ja vs. nein", Stichprobe=stichprobe, N=len(d),
                  Erfolgsdefinition=outcome, Effekt_typ="adjustierte Odds Ratio")
    if X["_x"].nunique() < 2 or y.nunique() < 2:
        row["Hinweis"] = "keine Variation in Merkmal oder Ergebnis"
        return row
    row["Erfolgsquote_Gruppe"] = y[X["_x"] == 1].mean()
    row["Erfolgsquote_Referenz"] = y[X["_x"] == 0].mean()
    model, converged, note = fit_logit(y, X, kennung)
    row["konvergiert"] = converged
    row["Hinweis"] = note
    if model is None:
        return row
    ci = model.conf_int().loc["_x"]
    row.update(Effekt=math.exp(model.params["_x"]), CI_unten=math.exp(ci.iloc[0]),
               CI_oben=math.exp(ci.iloc[1]), p=float(model.pvalues["_x"]))
    return row


def adjusted_or_levels(df: pd.DataFrame, var: str, ref: str, levels: list[tuple[str, str]],
                       kennung_prefix: str, outcome: str = OUTCOME, stichprobe: str = S_AN) -> list[dict]:
    """Gemeinsames Logit mit Dummies fuer mehrere Kategorien gegen eine Referenz (Jahr-FE)."""
    keep = df[var].isin([ref] + [lvl for lvl, _ in levels])
    d = df.loc[keep, [var, outcome, "release_year"]].dropna()
    y = d[outcome].astype(bool).astype(int)
    X = pd.DataFrame(index=d.index)
    for i, (lvl, _) in enumerate(levels):
        X[f"_l{i}"] = (d[var] == lvl).astype(int)
    X = X.join(year_dummies(d["release_year"]))
    model, converged, note = fit_logit(y, X, kennung_prefix)
    rows = []
    for i, (lvl, label) in enumerate(levels):
        mask = d[var] == lvl
        row = nan_row(Kennung=f"{kennung_prefix}_{lvl.replace('+', '').lower()}_aor", Rolle="Adjustiert",
                      Modell="Jahr-FE, gemeinsames Modell der Kombinationen", Test="Logistische Regression",
                      Merkmal=f"Plattformkombination: {label}", Gruppe=f"{lvl} vs. {ref} (nur Windows)",
                      Stichprobe=stichprobe, N=len(d), Erfolgsdefinition=outcome,
                      Erfolgsquote_Gruppe=y[mask].mean(),
                      Erfolgsquote_Referenz=y[d[var] == ref].mean(),
                      Effekt_typ="adjustierte Odds Ratio", konvergiert=converged, Hinweis=note)
        if model is not None:
            ci = model.conf_int().loc[f"_l{i}"]
            row.update(Effekt=math.exp(model.params[f"_l{i}"]), CI_unten=math.exp(ci.iloc[0]),
                       CI_oben=math.exp(ci.iloc[1]), p=float(model.pvalues[f"_l{i}"]))
        rows.append(row)
    return rows


def cliff_row(values: pd.Series, group: pd.Series, label: str, kennung: str, group_label: str,
              stichprobe: str, rolle: str = "Haupttest", outcome: str = "") -> dict:
    """Mann-Whitney-U mit Cliff's Delta: Werte der Gruppe (True) gegen die uebrigen."""
    d = pd.DataFrame({"v": values, "g": group}).dropna()
    g = d["g"].astype(bool)
    x = d.loc[g, "v"].astype(float)
    y = d.loc[~g, "v"].astype(float)
    row = nan_row(Kennung=kennung, Rolle=rolle, Modell="unadjustiert", Test="Mann-Whitney-U",
                  Merkmal=label, Gruppe=f"{group_label} = ja (n={len(x)}) vs. nein (n={len(y)})",
                  Stichprobe=stichprobe, N=len(x) + len(y), Erfolgsdefinition=outcome,
                  Effekt_typ="Cliff's Delta")
    if len(x) < MIN_CELL or len(y) < MIN_CELL:
        row["Hinweis"] = f"zu wenige Fälle in einer Gruppe (n={len(x)}/{len(y)})"
        return row
    u, p = mannwhitneyu(x, y, alternative="two-sided")
    row.update(Effekt=cliffs_delta_from_mannwhitney(u, len(x), len(y)), p=p,
               Median_Gruppe=x.median(), Median_Referenz=y.median())
    return row


def kruskal_row(df: pd.DataFrame, value: str, group: str, label: str, kennung: str,
                stichprobe: str, rolle: str = "Zusatz", order: list[str] | None = None) -> dict:
    d = df[[value, group, OUTCOME]].dropna(subset=[value, group])
    keys = order if order is not None else sorted(d[group].unique())
    groups = [d.loc[d[group] == k, value].to_numpy() for k in keys]
    keys = [k for k, g in zip(keys, groups) if len(g) > 0]
    groups = [g for g in groups if len(g) > 0]
    h, p = kruskal(*groups)
    n = sum(len(g) for g in groups)
    rates = d.groupby(group)[OUTCOME].mean()
    return nan_row(Kennung=kennung, Rolle=rolle, Modell="unadjustiert", Test="Kruskal-Wallis",
                   Merkmal=label, Gruppe=group, Stichprobe=stichprobe, N=n, Effekt=h,
                   Effekt_typ="Kruskal-Wallis H", p=p,
                   # Epsilon-Quadrat nach Tomczak & Tomczak (2014): H / (n - 1).
                   Zusatz_Effekt=h / (n - 1) if n > 1 else np.nan, Zusatz_Effekt_Typ="Epsilon-Quadrat (ε²)",
                   Hinweis="Erfolgsquote (success_top25) je Gruppe: "
                           + "; ".join(f"{k}: {rates.get(k, np.nan):.3f} (n={len(g)})" for k, g in zip(keys, groups)))


def control_matrix(d: pd.DataFrame, year_col: str | None, extra_controls: tuple[str, ...]) -> np.ndarray:
    cols = [np.ones(len(d))]
    if year_col is not None:
        years = np.floor(d[year_col].astype(float)).astype(int)
        groups = year_groups(years, YEAR_FE_MIN_PARTIAL)
        dummies = pd.get_dummies(groups, drop_first=True, dtype=float).to_numpy()
        if dummies.size:
            cols.append(dummies)
    for c in extra_controls:
        cols.append(rankdata(d[c].astype(float))[:, None])
    return np.column_stack(cols)


def partial_rank_corr(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[float, float, int]:
    """Partielle Rangkorrelation: Raenge von x und y auf die Kontrollmatrix z residualisiert."""
    rx = rankdata(x)
    ry = rankdata(y)
    ex = rx - z @ np.linalg.lstsq(z, rx, rcond=None)[0]
    ey = ry - z @ np.linalg.lstsq(z, ry, rcond=None)[0]
    k = int(np.linalg.matrix_rank(z)) - 1
    if ex.std() == 0 or ey.std() == 0:
        return np.nan, np.nan, k
    r = float(np.corrcoef(ex, ey)[0, 1])
    dof = len(x) - 2 - k
    if dof <= 0 or abs(r) >= 1:
        return r, np.nan, k
    t = r * math.sqrt(dof / (1 - r * r))
    return r, float(2 * t_dist.sf(abs(t), dof)), k


def spearman_row(df: pd.DataFrame, x: str, y: str, label: str, kennung: str, partial: bool = False,
                 extra_controls: tuple[str, ...] = (), stichprobe: str = S_AN, rolle: str | None = None,
                 year_col: str = "release_year") -> dict:
    cols = list(dict.fromkeys([x, y, year_col, *extra_controls]))
    d = df[cols].dropna()
    if partial:
        modell = "partiell: Jahr-FE" + "".join(f" + {c} (Rang)" for c in extra_controls)
    else:
        modell = "unadjustiert"
    row = nan_row(Kennung=kennung, Rolle=rolle or ("Adjustiert" if partial else "Haupttest"),
                  Modell=modell, Test="Spearman" + (" (partiell)" if partial else ""), Merkmal=label,
                  Gruppe=f"{x} vs. {y}", Stichprobe=stichprobe, N=len(d), Effekt_typ="Spearman rho")
    if len(d) < 10 or d[x].nunique() < 2 or d[y].nunique() < 2:
        row["Hinweis"] = "zu wenige Fälle oder keine Variation"
        return row
    xv = d[x].astype(float).to_numpy()
    yv = d[y].astype(float).to_numpy()
    if partial:
        z = control_matrix(d, year_col, extra_controls)
        rho, p, k = partial_rank_corr(xv, yv, z)
    else:
        rho, p = spearmanr(xv, yv)
        k = 0
    low, high = spearman_ci(rho, len(d), k)
    row.update(Effekt=float(rho), CI_unten=low, CI_oben=high, p=float(p))
    return row


def _range_text(r: pd.Series) -> str:
    if r["n"] == 0:
        return "leer"
    return f"{fmt(r['min'], 4)}–{fmt(r['max'], 4)} (n={int(r['n'])})"


def quartile_groups(df: pd.DataFrame, value: str) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    d = df.dropna(subset=[value])
    groups = ranked_quartiles(d[value])
    ranges = quartile_ranges(d[value], groups).set_index("Gruppe")
    return d, groups, ranges


def record_quartiles(df: pd.DataFrame, value: str, label: str, kennung: str, stichprobe: str) -> None:
    d, groups, ranges = quartile_groups(df, value)
    for lab in QUARTILE_LABELS:
        part = d[groups == lab]
        entry = {"Kennung": kennung, "Variable": value, "Merkmal": label, "Stichprobe": stichprobe,
                 "Gruppe": lab, "n": int(ranges.loc[lab, "n"]), "min": ranges.loc[lab, "min"],
                 "max": ranges.loc[lab, "max"]}
        for outcome in ALL_OUTCOMES:
            entry[f"Erfolgsquote_{outcome}"] = part[outcome].astype(bool).mean() if len(part) else np.nan
        QUARTILE_ROWS.append(entry)


def quartile_or(df: pd.DataFrame, value: str, label: str, kennung: str, outcome: str = OUTCOME,
                stichprobe: str = S_AN, rolle: str = "Zusatz") -> dict:
    """Q4 vs. Q1 mit bindungstreuen Quartilen; bei kleinen Zellen exaktes bedingtes OR."""
    d, groups, ranges = quartile_groups(df[[value, outcome]], value)
    lo, hi = QUARTILE_LABELS[0], QUARTILE_LABELS[-1]
    top = d.loc[groups == hi, outcome].astype(bool)
    bottom = d.loc[groups == lo, outcome].astype(bool)
    share = d[outcome].astype(bool).mean()
    row = nan_row(Kennung=kennung, Rolle=rolle, Modell="unadjustiert", Test="Chi² / Odds Ratio (Q4 vs. Q1)",
                  Merkmal=label, Gruppe=f"{value}: Q4 vs. Q1 (bindungstreue Quartile)",
                  Stichprobe=stichprobe, N=len(top) + len(bottom), Erfolgsdefinition=outcome,
                  Effekt_typ="Odds Ratio Q4/Q1", Bereich_Gruppe=_range_text(ranges.loc[hi]),
                  Bereich_Referenz=_range_text(ranges.loc[lo]))
    if len(top) == 0 or len(bottom) == 0:
        row["Hinweis"] = "Q1 oder Q4 leer"
        row["instabil"] = True
        return row
    table = np.array([[top.sum(), (~top).sum()], [bottom.sum(), (~bottom).sum()]])
    min_cell = int(table.min())
    reasons = []
    if share > UNSTABLE_SHARE:
        reasons.append(f"Erfolgsanteil der Teilstichprobe {share:.1%} > {UNSTABLE_SHARE:.0%}")
    if min_cell < MIN_CELL:
        reasons.append(f"kleinste Zelle {min_cell} < {MIN_CELL}")
    if min_cell < MIN_CELL:
        res = scipy_odds_ratio(table, kind="conditional")
        ci = res.confidence_interval(confidence_level=0.95)
        odds, low, high = float(res.statistic), float(ci.low), float(ci.high)
        p = float(fisher_exact(table)[1])
        row["Test"] = "Fisher / bedingtes OR, exaktes KI (Q4 vs. Q1)"
        if not np.isfinite(odds):
            reasons.append("bedingtes ML-OR unendlich (Nullzelle), nur das exakte KI ist interpretierbar")
    else:
        odds, low, high, p = odds_ratio_ci(table)
    row.update(Erfolgsquote_Gruppe=top.mean(), Erfolgsquote_Referenz=bottom.mean(), Effekt=odds,
               CI_unten=low, CI_oben=high, p=p, instabil=bool(reasons), Hinweis="; ".join(reasons))
    return row


def quartile_shape_rows(df: pd.DataFrame, value: str, label: str, kennung: str, stichprobe: str,
                        outcome: str = OUTCOME) -> list[dict]:
    """Nichtlinearitaet ueber Quartile: Chi-Quadrat (4x2), Cochran-Armitage-Trend und Abweichung vom Trend."""
    d, groups, _ = quartile_groups(df[[value, outcome]], value)
    tab = pd.crosstab(groups, d[outcome].astype(bool)).reindex(
        index=list(QUARTILE_LABELS), columns=[False, True], fill_value=0)
    scores = np.arange(1, len(QUARTILE_LABELS) + 1, dtype=float)
    keep = tab.sum(axis=1).to_numpy() > 0
    tab, scores = tab[keep], scores[keep]
    n_i = tab.sum(axis=1).to_numpy(float)
    r_i = tab[True].to_numpy(float)
    total = n_i.sum()
    chi2, p, dof, _ = chi2_contingency(tab.to_numpy(), correction=False)
    pbar = r_i.sum() / total
    t_stat = float(np.sum(scores * (r_i - n_i * pbar)))
    var = pbar * (1 - pbar) * (np.sum(n_i * scores ** 2) - np.sum(n_i * scores) ** 2 / total)
    z = t_stat / math.sqrt(var) if var > 0 else np.nan
    p_trend = float(2 * norm.sf(abs(z))) if np.isfinite(z) else np.nan
    chi2_dev = max(chi2 - z ** 2, 0.0) if np.isfinite(z) else np.nan
    dof_dev = dof - 1
    p_dev = float(chi2_dist.sf(chi2_dev, dof_dev)) if dof_dev > 0 else np.nan
    rates = " / ".join(f"{r:.3f}" for r in r_i / n_i)
    base = dict(Rolle="Nichtlinearität", Modell="unadjustiert", Merkmal=label, Stichprobe=stichprobe,
                N=int(total), Erfolgsdefinition=outcome, Gruppe=f"{value}: Quartile Q1–Q4",
                Hinweis=f"Erfolgsquote Q1–Q4: {rates}")
    return [
        nan_row(Kennung=f"{kennung}_qchi2", Test="Chi² über Quartile (4×2)", Effekt=chi2,
                Effekt_typ=f"Chi2 (df={dof})", p=p, Zusatz_Effekt=math.sqrt(chi2 / total),
                Zusatz_Effekt_Typ="Cramers V", **base),
        nan_row(Kennung=f"{kennung}_qtrend", Test="Cochran-Armitage-Trendtest", Effekt=z,
                Effekt_typ="z (linearer Trend)", p=p_trend, **base),
        nan_row(Kennung=f"{kennung}_qnonlin", Test="Abweichung vom linearen Trend", Effekt=chi2_dev,
                Effekt_typ=f"Chi2 (df={dof_dev})", p=p_dev, **base),
    ]


def finalize(rows: list[dict], family: str) -> pd.DataFrame:
    out = pd.DataFrame(rows)
    for col in ROW_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    out["Familie"] = family
    out["p_adj"] = bh_correction(out["p"])
    out["signifikant"] = out["p_adj"] < 0.05
    dup = out["Kennung"][out["Kennung"].duplicated()]
    assert dup.empty, f"doppelte Kennungen in {family}: {list(dup)}"
    return out[ROW_COLUMNS]


CORE_COLUMNS = {"Kennung", "Familie", "Rolle", "Modell", "Test", "Merkmal", "Gruppe", "Stichprobe", "N", "Effekt",
                "Effekt_typ", "CI_unten", "CI_oben", "p", "p_adj", "signifikant"}


def save(df: pd.DataFrame, name: str) -> None:
    """Speichert eine Ergebnistabelle ohne optionale Spalten, die in dieser Tabelle leer sind."""
    empty = [c for c in df.columns
             if c not in CORE_COLUMNS and (df[c].isna() | (df[c].astype(str) == "")).all()]
    save_table(df.drop(columns=empty), name)


# ---------------------------------------------------------------------------
# Bausteine fuer Robustheit, Sensitivitaet, Selektion, Uebereinstimmung
# ---------------------------------------------------------------------------

ROBUST_ROWS: list[dict] = []


def robust_binary(df: pd.DataFrame, flag: str, label: str, key: str, stichprobe: str = S_AN) -> None:
    for outcome in ALL_OUTCOMES:
        ROBUST_ROWS.append(chi2_indicator(df, flag, label, f"rob_{key}_{outcome}_or", outcome=outcome,
                                          stichprobe=stichprobe, rolle="Robustheit"))
        ROBUST_ROWS.append(adjusted_or(df, flag, label, f"rob_{key}_{outcome}_aor", outcome=outcome,
                                       stichprobe=stichprobe, rolle="Robustheit"))


def robust_combo(df: pd.DataFrame) -> None:
    sub = df[df["platform_combo"].isin([COMBO_REF] + [lvl for lvl, _ in COMBO_LEVELS])]
    for lvl, label in COMBO_LEVELS:
        part = sub[sub["platform_combo"].isin([COMBO_REF, lvl])].assign(_combo=lambda t: t["platform_combo"] == lvl)
        key = f"combo_{lvl.replace('+', '').lower()}"
        for outcome in ALL_OUTCOMES:
            row = chi2_indicator(part, "_combo", f"Plattformkombination: {label}", f"rob_{key}_{outcome}_or",
                                 outcome=outcome, rolle="Robustheit")
            row["Gruppe"] = f"{lvl} vs. {COMBO_REF} (nur Windows)"
            ROBUST_ROWS.append(row)
    for outcome in ALL_OUTCOMES:
        for row in adjusted_or_levels(df, "platform_combo", COMBO_REF, COMBO_LEVELS, "tmp", outcome=outcome):
            lvl = row["Kennung"].split("_")[1]
            row["Kennung"] = f"rob_combo_{lvl}_{outcome}_aor"
            row["Rolle"] = "Robustheit"
            ROBUST_ROWS.append(row)


def robust_continuous(df: pd.DataFrame, value: str, label: str, key: str, stichprobe: str) -> None:
    """Richtung eines stetigen Merkmals unter allen Erfolgsdefinitionen (Cliff's Delta)."""
    for outcome in ALL_OUTCOMES:
        ROBUST_ROWS.append(cliff_row(df[value], df[outcome], label, f"rob_{key}_{outcome}_delta",
                                     outcome, stichprobe, rolle="Robustheit", outcome=outcome))


def robust_quartile(df: pd.DataFrame, value: str, label: str, key: str, stichprobe: str) -> None:
    for outcome in ALL_OUTCOMES:
        ROBUST_ROWS.append(quartile_or(df, value, label, f"rob_{key}_{outcome}_q", outcome=outcome,
                                       stichprobe=stichprobe, rolle="Robustheit"))


def interaction_lr(df: pd.DataFrame, flag: str, outcome: str = OUTCOME) -> tuple[float, int, float, bool]:
    """LR-Test der Interaktion flag x Kohorte im Logit mit Jahr-FE."""
    d = df[[flag, outcome, "release_year"]].dropna()
    y = d[outcome].astype(bool).astype(int)
    x = d[flag].astype(bool).astype(int)
    X0 = pd.DataFrame({"_x": x}, index=d.index).join(year_dummies(d["release_year"]))
    cohorts = pd.get_dummies(cohort_labels(d["release_year"]), drop_first=True, dtype=float)
    X1 = X0.join(cohorts.mul(x, axis=0).add_prefix("_x_mal_"))
    m0, c0, _ = fit_logit(y, X0, f"interaktion_{flag}_basis")
    m1, c1, _ = fit_logit(y, X1, f"interaktion_{flag}_kohorte")
    if m0 is None or m1 is None:
        return np.nan, 0, np.nan, False
    lr = 2 * (m1.llf - m0.llf)
    dof = int(len(m1.params) - len(m0.params))
    return float(lr), dof, float(chi2_dist.sf(lr, dof)), bool(c0 and c1)


def cohen_kappa(a: pd.Series, b: pd.Series) -> dict:
    a = a.astype(bool).to_numpy()
    b = b.astype(bool).to_numpy()
    n = len(a)
    po = float(np.mean(a == b))
    pa, pb = a.mean(), b.mean()
    pe = pa * pb + (1 - pa) * (1 - pb)
    kappa = (po - pe) / (1 - pe) if pe < 1 else np.nan
    return {"N": n, "Anteil_IGDB": pa, "Anteil_Steam": pb, "N_beide": int((a & b).sum()),
            "N_nur_IGDB": int((a & ~b).sum()), "N_nur_Steam": int((~a & b).sum()),
            "Anteil_identisch": po, "Kappa": kappa, "PABAK": 2 * po - 1}


def bootstrap_difference(d: pd.DataFrame, a: str, b: str, y: str, partial: bool, rng_seed: int) -> dict:
    """Differenz rho(a, y) - rho(b, y) in identischer Stichprobe mit Bootstrap-Perzentil-KI."""
    xa = d[a].to_numpy(float)
    xb = d[b].to_numpy(float)
    yy = d[y].to_numpy(float)
    # Partialisierung nach Jahr-FE = Abzug der Gruppenmittel der Raenge (identisch
    # zur Regression auf Konstante + Jahr-Dummies, aber schnell genug fuer den Bootstrap).
    years = np.floor(d["release_year"].astype(float)).astype(int)
    codes = pd.factorize(year_groups(years, YEAR_FE_MIN_PARTIAL))[0] if partial else np.zeros(len(d), dtype=int)

    def demean(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
        sums = np.bincount(groups, weights=values)
        counts = np.bincount(groups)
        means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
        return values - means[groups]

    def rho(xv: np.ndarray, yv: np.ndarray, groups: np.ndarray) -> float:
        ex = demean(rankdata(xv), groups)
        ey = demean(rankdata(yv), groups)
        if ex.std() == 0 or ey.std() == 0:
            return np.nan
        return float(np.corrcoef(ex, ey)[0, 1])

    ra, rb = rho(xa, yy, codes), rho(xb, yy, codes)
    rng = np.random.default_rng(rng_seed)
    diffs = np.empty(N_BOOT)
    n = len(d)
    for i in range(N_BOOT):
        idx = rng.integers(0, n, n)
        gi = codes[idx]
        diffs[i] = rho(xa[idx], yy[idx], gi) - rho(xb[idx], yy[idx], gi)
    low, high = np.nanpercentile(diffs, [2.5, 97.5])
    # Zweiseitiger Bootstrap-p-Wert mit (+1)-Korrektur, damit p nie exakt 0 wird.
    p_boot = 2 * min((np.sum(diffs <= 0) + 1) / (N_BOOT + 1), (np.sum(diffs >= 0) + 1) / (N_BOOT + 1))
    return {"N": n, "rho_A": ra, "rho_B": rb, "Differenz": ra - rb, "CI_unten": float(low),
            "CI_oben": float(high), "p_bootstrap": float(min(p_boot, 1.0)), "Bootstrap_n": N_BOOT,
            "Seed": rng_seed}


# ---------------------------------------------------------------------------
# Hauptprogramm
# ---------------------------------------------------------------------------

def main() -> None:
    LOGIT_LOG.clear()
    QUARTILE_ROWS.clear()
    ROBUST_ROWS.clear()

    df = pd.read_parquet(DATA / "analytic_sample.parquet")
    assert df["analytic_sample"].all()
    df["multi_platform"] = df["n_platforms"] >= 2
    df["peak_positive"] = df["peak_ccu"] > 0
    df["paid"] = df["list_price"] > 0
    df["log_review_count"] = np.log10(1 + df["review_count"])

    rating = df[df["rating_sample"]].copy()
    rating50 = df[df["rating_sample_50"]].copy()
    peak = df[df["peak_sample"]].copy()
    paid = df[df["paid"] & ~df["price_outlier"]].copy()
    igdb = df[df["has_igdb"]].copy()
    meta = df[df["has_mc_external"]].copy()
    ph = df[df["has_price_history"]].copy()

    non_windows = int((~df["platform_combo"].isin([COMBO_REF] + [lvl for lvl, _ in COMBO_LEVELS])).sum())
    print(f"Analysestichprobe: {len(df)}; ohne Windows (nicht in Kombinationsvergleich): {non_windows}")
    print(f"Jahr-FE-Gruppen (Logit): {sorted(year_groups(df['release_year'], YEAR_FE_MIN_LOGIT).unique())}")
    print(f"Kostenpflichtig ohne Ausreißer: {len(paid)}; Preisausreißer (> 100 USD) ausgeschlossen: "
          f"{int((df['paid'] & df['price_outlier']).sum())}; ohne Listenpreis (100 % Rabatt): "
          f"{int(df['list_price'].isna().sum())}")

    # ------------------------------------------------------------------ H1
    h1_rows: list[dict] = []
    for genre, col in GENRE_COLS.items():
        key = col.removeprefix("genre_")
        label = f"Genre: {genre}"
        h1_rows.append(chi2_indicator(df, col, label, f"h1_genre_{key}_or"))
        h1_rows.append(adjusted_or(df, col, label, f"h1_genre_{key}_aor"))
        h1_rows.append(adjusted_or(df, col, label, f"h1_genre_{key}_aor_quad", spec="quad"))
        robust_binary(df, col, label, f"genre_{key}")
    for col, name in MODE_COLS:
        label = f"Spielmodus: {name}"
        h1_rows.append(chi2_indicator(df, col, label, f"h1_mode_{col}_or"))
        h1_rows.append(adjusted_or(df, col, label, f"h1_mode_{col}_aor"))
        h1_rows.append(adjusted_or(df, col, label, f"h1_mode_{col}_aor_quad", spec="quad"))
        robust_binary(df, col, label, f"mode_{col}")
    h1_rows.append(chi2_categorical(df, "mode", "Spielmodus (4 Kategorien)", "h1_mode_categorical"))
    h1_rows.append(chi2_categorical(df, "n_platforms", "Anzahl Plattformen (1/2/3)", "h1_n_platforms_categorical"))
    for col, label in PLATFORM_FLAGS:
        h1_rows.append(chi2_indicator(df, col, label, f"h1_platform_{col}_or"))
        h1_rows.append(adjusted_or(df, col, label, f"h1_platform_{col}_aor"))
        h1_rows.append(adjusted_or(df, col, label, f"h1_platform_{col}_aor_quad", spec="quad"))
        robust_binary(df, col, label, f"platform_{col}")
    # macOS und Linux gegenseitig adjustiert (Linux-Ports sind meist auch Mac-Ports).
    h1_rows.append(adjusted_or(df, "mac", "Plattform: macOS", "h1_platform_mac_aor_joint", covars=("linux",),
                               rolle="Sensitivität"))
    h1_rows.append(adjusted_or(df, "linux", "Plattform: Linux", "h1_platform_linux_aor_joint", covars=("mac",),
                               rolle="Sensitivität"))
    combo_df = df[df["platform_combo"].isin([COMBO_REF] + [lvl for lvl, _ in COMBO_LEVELS])]
    h1_rows.append(chi2_categorical(combo_df, "platform_combo", "Plattformkombination (4 Kategorien mit Windows)",
                                    "h1_platform_combo_categorical"))
    for lvl, label in COMBO_LEVELS:
        part = combo_df[combo_df["platform_combo"].isin([COMBO_REF, lvl])].assign(
            _combo=lambda t: t["platform_combo"] == lvl)
        row = chi2_indicator(part, "_combo", f"Plattformkombination: {label}",
                             f"h1_combo_{lvl.replace('+', '').lower()}_or")
        row["Gruppe"] = f"{lvl} vs. {COMBO_REF} (nur Windows)"
        h1_rows.append(row)
    h1_rows.extend(adjusted_or_levels(df, "platform_combo", COMBO_REF, COMBO_LEVELS, "h1_combo"))
    robust_combo(df)
    h1 = finalize(h1_rows, "H1 - Spieleigenschaften")
    save(h1, "h1_tests")

    # Gemeinsames multivariables Modell (Genres + Modi + Plattformen + Jahr-FE).
    joint_rows = []
    joint_vars = list(GENRE_COLS.values()) + [c for c, _ in MODE_COLS] + ["mac", "linux"]
    joint_labels = {**{c: f"Genre: {g}" for g, c in GENRE_COLS.items()},
                    **{c: f"Spielmodus: {n}" for c, n in MODE_COLS},
                    "mac": "Plattform: macOS", "linux": "Plattform: Linux"}
    for outcome in ALL_OUTCOMES:
        d = df[joint_vars + [outcome, "release_year"]].dropna()
        X = d[joint_vars].astype(int).join(year_dummies(d["release_year"]))
        model, converged, note = fit_logit(d[outcome].astype(int), X, f"joint_{outcome}")
        ci = model.conf_int()
        for var in joint_vars:
            joint_rows.append({
                "Kennung": f"joint_{var}_{outcome}", "Variable": var, "Merkmal": joint_labels[var],
                "Erfolgsdefinition": outcome, "N": len(d), "Effekt": math.exp(model.params[var]),
                "Effekt_typ": "adjustierte Odds Ratio (gemeinsames Modell)",
                "CI_unten": math.exp(ci.loc[var].iloc[0]), "CI_oben": math.exp(ci.loc[var].iloc[1]),
                "p": float(model.pvalues[var]), "Pseudo_R2": float(model.prsquared),
                "konvergiert": converged, "Hinweis": note,
                "Modell": "Logit: 10 Genres + 5 Modi + macOS + Linux + Jahr-FE",
            })
    joint = pd.DataFrame(joint_rows)
    joint["p_adj"] = joint.groupby("Erfolgsdefinition")["p"].transform(bh_correction)
    save(joint, "h1_joint_model")

    # ------------------------------------------------------------------ H2
    df["discount_group"] = pd.cut(
        df["discount"], bins=[-0.1, 0, 25, 50, 75, 100],
        labels=["kein Rabatt", "1-25 %", "26-50 %", "51-75 %", "76-100 %"],
    ).astype(str)
    paid = df[df["paid"] & ~df["price_outlier"]].copy()
    paid_all = df[df["paid"]].copy()
    paid_current = df[(df["price_current"] > 0) & ~df["price_outlier"]].copy()
    paid_peak = paid[paid["peak_sample"]]

    h2_rows: list[dict] = []
    free_label = "kostenlos (Listenpreis 0) vs. kostenpflichtig"
    # Kostenlos gegen kostenpflichtig nur bei bestimmbarem Listenpreis: Spiele mit
    # 100 % Rabatt oder Tagespreis 0 trotz Rabatt sind weder das eine noch das andere.
    df_price = df[df["list_price"].notna()].copy()
    peak_price = peak[peak["list_price"].notna()]
    h2_rows.append(chi2_indicator(df_price, "is_free", f"Erfolgsquote: {free_label}", "h2_free_or",
                                  stichprobe=S_AN_PRICE))
    h2_rows.append(adjusted_or(df_price, "is_free", f"Erfolgsquote: {free_label}", "h2_free_aor",
                               stichprobe=S_AN_PRICE))
    h2_rows.append(adjusted_or(df_price, "is_free", f"Erfolgsquote: {free_label}", "h2_free_aor_quad", spec="quad",
                               stichprobe=S_AN_PRICE))
    h2_rows.append(cliff_row(df_price["owners_mid"], df_price["is_free"], f"Besitzerklasse: {free_label}",
                             "h2_free_owners_delta", "is_free", S_AN_PRICE, rolle="Zusatz"))
    h2_rows.append(chi2_indicator(df, "f2p_tag", "Erfolgsquote: Steam-Tag „Free To Play“ vs. ohne", "h2_f2ptag_or",
                                  rolle="Sensitivität"))
    h2_rows.append(adjusted_or(df, "f2p_tag", "Erfolgsquote: Steam-Tag „Free To Play“ vs. ohne", "h2_f2ptag_aor",
                               rolle="Sensitivität"))
    # Kostenlose Spiele mit bzw. ohne F2P-Tag jeweils gegen kostenpflichtige.
    for has_tag, key, name in [(True, "with_tag", "mit F2P-Tag"), (False, "without_tag", "ohne F2P-Tag")]:
        sub = df_price[~df_price["is_free"] | (df_price["f2p_tag"] == has_tag)]
        lab = f"Erfolgsquote: kostenlos {name} vs. kostenpflichtig"
        h2_rows.append(chi2_indicator(sub, "is_free", lab, f"h2_free_{key}_or", stichprobe=S_AN_PRICE,
                                      rolle="Sensitivität"))
        h2_rows.append(adjusted_or(sub, "is_free", lab, f"h2_free_{key}_aor", stichprobe=S_AN_PRICE,
                                   rolle="Sensitivität"))
    # Peak CCU als Hurdle: P(Peak > 0) und Intensitaet nur bei Peak > 0.
    h2_rows.append(chi2_indicator(df_price, "is_free", f"P(Peak > 0): {free_label}", "h2_free_peakpos_or",
                                  outcome="peak_positive", stichprobe=S_AN_PRICE, rolle="Hurdle (Peak > 0)"))
    h2_rows.append(adjusted_or(df_price, "is_free", f"P(Peak > 0): {free_label}", "h2_free_peakpos_aor",
                               outcome="peak_positive", stichprobe=S_AN_PRICE, rolle="Hurdle (Peak > 0)"))
    h2_rows.append(cliff_row(peak_price["peak_ccu"], peak_price["is_free"], f"Peak CCU (nur Peak > 0): {free_label}",
                             "h2_free_peak_delta", "is_free", f"{S_PEAK}, Listenpreis bestimmbar",
                             rolle="Intensität (Peak > 0)"))
    robust_binary(df_price, "is_free", f"Erfolgsquote: {free_label}", "free", stichprobe=S_AN_PRICE)

    price_label = "Listenpreis vs. Besitzer"
    h2_rows.append(spearman_row(paid, "list_price", "log_owners", price_label, "h2_price_rho", stichprobe=S_PAID))
    h2_rows.append(spearman_row(paid, "list_price", "log_owners", price_label, "h2_price_rho_partial",
                                partial=True, stichprobe=S_PAID))
    h2_rows.append(kruskal_row(paid, "owners_mid", "price_class", "Besitzerklasse nach Preisklasse (Listenpreis)",
                               "h2_price_class_kw", S_PAID,
                               order=["0,01 - 4,99", "5 - 9,99", "10 - 19,99", "20 - 39,99", "40+"]))
    h2_rows.append(spearman_row(paid_all, "list_price", "log_owners", f"{price_label} (inkl. Ausreißer)",
                                "h2_price_rho_with_outliers", stichprobe=S_PAID_ALL, rolle="Sensitivität"))
    # Das 54,99-USD-Cluster bleibt in der Hauptanalyse (echte Listenpreise); hier ohne.
    paid_nocluster = paid[~paid["price_cluster"]]
    h2_rows.append(spearman_row(paid_nocluster, "list_price", "log_owners", f"{price_label} (ohne 54,99-Cluster)",
                                "h2_price_rho_nocluster_partial", partial=True,
                                stichprobe=f"{S_PAID}, ohne 54,99-USD-Cluster ab 2024", rolle="Sensitivität"))
    h2_rows.append(kruskal_row(paid_nocluster, "owners_mid", "price_class",
                               "Besitzerklasse nach Preisklasse (Listenpreis, ohne 54,99-Cluster)",
                               "h2_price_class_kw_nocluster", f"{S_PAID}, ohne 54,99-USD-Cluster ab 2024",
                               rolle="Sensitivität",
                               order=["0,01 - 4,99", "5 - 9,99", "10 - 19,99", "20 - 39,99", "40+"]))
    # Getrennt nach Rabattstatus: Haengt der Listenpreis-Zusammenhang am aktuellen
    # Rabatt (Effektmodifikation), und bleibt er unter rabattierten Spielen auch bei
    # Kontrolle der Rabatthoehe bestehen?
    for has_disc, key, name in [(False, "nodiscount", "ohne aktuellen Rabatt"), (True, "discount", "mit aktuellem Rabatt")]:
        part = paid[paid["has_discount"] == has_disc]
        s_part = f"{S_PAID}, {name}"
        h2_rows.append(spearman_row(part, "list_price", "log_owners", f"{price_label} ({name})",
                                    f"h2_price_rho_{key}", stichprobe=s_part, rolle="Sensitivität"))
        h2_rows.append(spearman_row(part, "list_price", "log_owners", f"{price_label} ({name})",
                                    f"h2_price_rho_{key}_partial", partial=True, stichprobe=s_part,
                                    rolle="Sensitivität"))
        if has_disc:
            h2_rows.append(spearman_row(part, "list_price", "log_owners", f"{price_label} ({name})",
                                        f"h2_price_rho_{key}_partial_depth", partial=True,
                                        extra_controls=("discount",), stichprobe=s_part, rolle="Sensitivität"))
    h2_rows.append(spearman_row(paid_current, "price_current", "log_owners", "Tagespreis vs. Besitzer",
                                "h2_price_current_rho", stichprobe=S_PAID_CURRENT, rolle="Sensitivität"))
    h2_rows.append(spearman_row(paid_current, "price_current", "log_owners", "Tagespreis vs. Besitzer",
                                "h2_price_current_rho_partial", partial=True, stichprobe=S_PAID_CURRENT,
                                rolle="Sensitivität"))
    h2_rows.append(kruskal_row(paid_current, "owners_mid", "price_class_current",
                               "Besitzerklasse nach Preisklasse (Tagespreis)", "h2_price_class_current_kw",
                               S_PAID_CURRENT, rolle="Sensitivität",
                               order=["0,01 - 4,99", "5 - 9,99", "10 - 19,99", "20 - 39,99", "40+"]))
    h2_rows.append(cliff_row(paid["list_price"], paid["peak_positive"], "Listenpreis: Peak > 0 vs. Peak = 0",
                             "h2_price_peakpos_delta", "peak_positive", S_PAID, rolle="Hurdle (Peak > 0)"))
    h2_rows.append(spearman_row(paid_peak, "list_price", "log_peak_ccu", "Listenpreis vs. Peak CCU (nur Peak > 0)",
                                "h2_price_peak_rho", stichprobe=f"{S_PAID}, Peak > 0", rolle="Intensität (Peak > 0)"))
    # Vergleichswert auf derselben Teilstichprobe (Peak > 0), damit Besitzer und
    # Aktivitaet nicht ueber verschiedene Stichproben verglichen werden.
    h2_rows.append(spearman_row(paid_peak, "list_price", "log_owners", "Listenpreis vs. Besitzer (nur Peak > 0)",
                                "h2_price_rho_peaksample", stichprobe=f"{S_PAID}, Peak > 0",
                                rolle="Intensität (Peak > 0)"))
    h2_rows.append(spearman_row(paid_peak, "list_price", "log_peak_ccu", "Listenpreis vs. Peak CCU (nur Peak > 0)",
                                "h2_price_peak_rho_partial", partial=True, stichprobe=f"{S_PAID}, Peak > 0",
                                rolle="Intensität (Peak > 0)"))
    robust_continuous(paid, "list_price", "Listenpreis", "price", S_PAID)

    disc_label = "Rabatt aktiv vs. kein Rabatt"
    h2_rows.append(chi2_indicator(paid, "has_discount", f"Erfolgsquote: {disc_label}", "h2_discount_or",
                                  stichprobe=S_PAID))
    h2_rows.append(adjusted_or(paid, "has_discount", f"Erfolgsquote: {disc_label}", "h2_discount_aor",
                               stichprobe=S_PAID))
    h2_rows.append(adjusted_or(paid, "has_discount", f"Erfolgsquote: {disc_label}", "h2_discount_aor_quad",
                               spec="quad", stichprobe=S_PAID))
    h2_rows.append(spearman_row(paid, "discount", "log_owners", "Rabatthöhe vs. Besitzer", "h2_discount_rho",
                                stichprobe=S_PAID, rolle="Zusatz"))
    h2_rows.append(spearman_row(paid, "discount", "log_owners", "Rabatthöhe vs. Besitzer", "h2_discount_rho_partial",
                                partial=True, stichprobe=S_PAID, rolle="Zusatz"))
    h2_rows.append(kruskal_row(paid, "owners_mid", "discount_group", "Besitzerklasse nach Rabattgruppe",
                               "h2_discount_group_kw", S_PAID,
                               order=["kein Rabatt", "1-25 %", "26-50 %", "51-75 %", "76-100 %"]))
    h2_rows.append(chi2_indicator(paid, "has_discount", f"P(Peak > 0): {disc_label}", "h2_discount_peakpos_or",
                                  outcome="peak_positive", stichprobe=S_PAID, rolle="Hurdle (Peak > 0)"))
    h2_rows.append(adjusted_or(paid, "has_discount", f"P(Peak > 0): {disc_label}", "h2_discount_peakpos_aor",
                               outcome="peak_positive", stichprobe=S_PAID, rolle="Hurdle (Peak > 0)"))
    h2_rows.append(spearman_row(paid_peak, "discount", "log_peak_ccu", "Rabatthöhe vs. Peak CCU (nur Peak > 0)",
                                "h2_discount_peak_rho", stichprobe=f"{S_PAID}, Peak > 0", rolle="Intensität (Peak > 0)"))
    h2_rows.append(spearman_row(paid_peak, "discount", "log_peak_ccu", "Rabatthöhe vs. Peak CCU (nur Peak > 0)",
                                "h2_discount_peak_rho_partial", partial=True, stichprobe=f"{S_PAID}, Peak > 0",
                                rolle="Intensität (Peak > 0)"))
    robust_binary(paid, "has_discount", f"Erfolgsquote: {disc_label}", "discount", stichprobe=S_PAID)

    h2_rows.append(spearman_row(df, "dlc_count", "log_owners", "DLC-Anzahl vs. Besitzer", "h2_dlc_rho", rolle="Zusatz"))
    h2_rows.append(spearman_row(df, "dlc_count", "log_owners", "DLC-Anzahl vs. Besitzer", "h2_dlc_rho_partial",
                                partial=True, rolle="Zusatz"))
    h2 = finalize(h2_rows, "H2 - Preis")
    save(h2, "h2_tests")

    # Kostenlos: Kohorten, Interaktion und Sensitivitaeten ("0 - 0", bis 2020, F2P-Tag).
    prepared = pd.read_parquet(DATA / "steam_prepared.parquet",
                               columns=["sample_with_owners_zero", "owners_zero", "is_free", "f2p_tag",
                                        "list_price", "release_year", *ALL_OUTCOMES])
    zero_sample = prepared[prepared["sample_with_owners_zero"] & prepared["list_price"].notna()].copy()
    assert not zero_sample.loc[zero_sample["owners_zero"], list(ALL_OUTCOMES)].any().any()
    print(f"Sensitivitätsstichprobe mit „0 - 0“: {len(zero_sample)} "
          f"(davon „0 - 0“: {int(zero_sample['owners_zero'].sum())})")
    free_rows = []
    samples = [(S_AN_PRICE, df_price), (S_ZERO, zero_sample), (S_LE2020, df_price[df_price["release_year"] <= 2020])]
    flags = [("is_free", "kostenlos (Listenpreis 0)"), ("f2p_tag", "Steam-Tag „Free To Play“")]
    for s_label, data in samples:
        s_key = {S_AN_PRICE: "an", S_ZERO: "zero", S_LE2020: "le2020"}[s_label]
        for flag, f_label in flags:
            last_year = int(data["release_year"].max())
            strata = [("gesamt", data)] + [
                (lab if b <= last_year else f"{a}–{last_year}",
                 data[(data["release_year"] >= a) & (data["release_year"] <= b)])
                for a, b, lab in COHORTS if a <= last_year]
            for c_idx, (c_label, part) in enumerate(strata):
                if part.empty:
                    continue
                c_key = "all" if c_label == "gesamt" else f"c{c_idx}"
                raw = chi2_indicator(part, flag, f_label, "tmp")
                adj = adjusted_or(part, flag, f_label, "tmp")
                free_rows.append({
                    "Kennung": f"free_{s_key}_{flag}_{c_key}", "Stichprobe": s_label, "Merkmal": f_label,
                    "Kohorte": c_label, "N": raw["N"], "N_Gruppe": int(part[flag].sum()),
                    "Anteil_Gruppe": float(part[flag].mean()),
                    "Erfolgsquote_Gruppe": raw.get("Erfolgsquote_Gruppe"),
                    "Erfolgsquote_Referenz": raw.get("Erfolgsquote_Referenz"),
                    "OR": raw["Effekt"], "OR_CI_unten": raw["CI_unten"], "OR_CI_oben": raw["CI_oben"],
                    "p_OR": raw["p"], "aOR_JahrFE": adj["Effekt"], "aOR_CI_unten": adj["CI_unten"],
                    "aOR_CI_oben": adj["CI_oben"], "p_aOR": adj["p"], "konvergiert": adj.get("konvergiert"),
                    "Hinweis": "; ".join(h for h in [raw.get("Hinweis"), adj.get("Hinweis")]
                                         if isinstance(h, str) and h),
                })
            lr, dof, p_int, conv = interaction_lr(data, flag)
            free_rows.append({
                "Kennung": f"free_{s_key}_{flag}_interaction", "Stichprobe": s_label, "Merkmal": f_label,
                "Kohorte": "Interaktion Merkmal × Kohorte (LR-Test, Jahr-FE)", "N": len(data),
                "LR_Chi2": lr, "df": dof, "p_Interaktion": p_int, "konvergiert": conv,
            })
    free_table = pd.DataFrame(free_rows)
    save(free_table, "h2_free_sensitivity")

    # ------------------------------------------------------------------ H3
    h3_rows: list[dict] = []
    for data, suffix, s_label in [(rating, "mind. 10 Reviews", S_R10), (rating50, "mind. 50 Reviews", S_R50)]:
        tag = "r10" if data is rating else "r50"
        rolle_main = "Haupttest" if data is rating else "Sensitivität"
        data_peak = data[data["peak_sample"]]
        for var, name, rolle in [("positive_ratio", "Positive Ratio", rolle_main), ("wilson", "Wilson-Score", "Sensitivität")]:
            key = f"h3_{'pr' if var == 'positive_ratio' else 'wilson'}_{tag}"
            h3_rows.append(spearman_row(data, var, "log_owners", f"{name} vs. Besitzer ({suffix})", f"{key}_rho",
                                        stichprobe=s_label, rolle=rolle))
            h3_rows.append(spearman_row(data, var, "log_owners", f"{name} vs. Besitzer ({suffix})", f"{key}_rho_partial",
                                        partial=True, stichprobe=s_label,
                                        rolle="Adjustiert" if rolle == "Haupttest" else "Sensitivität"))
            h3_rows.append(cliff_row(data[var], data["peak_positive"], f"{name}: Peak > 0 vs. Peak = 0 ({suffix})",
                                     f"{key}_peakpos_delta", "peak_positive", s_label, rolle="Hurdle (Peak > 0)"))
            h3_rows.append(spearman_row(data_peak, var, "log_peak_ccu", f"{name} vs. Peak CCU (nur Peak > 0, {suffix})",
                                        f"{key}_peak_rho", stichprobe=f"{s_label}, Peak > 0",
                                        rolle="Intensität (Peak > 0)"))
            h3_rows.append(spearman_row(data_peak, var, "log_peak_ccu", f"{name} vs. Peak CCU (nur Peak > 0, {suffix})",
                                        f"{key}_peak_rho_partial", partial=True, stichprobe=f"{s_label}, Peak > 0",
                                        rolle="Intensität (Peak > 0)"))
            h3_rows.append(quartile_or(data, var, f"Erfolgsquote {name} Q4 vs. Q1 ({suffix})", f"{key}_q",
                                       stichprobe=s_label))
            h3_rows.extend(quartile_shape_rows(data, var, f"{name}: Erfolgsquote über Quartile ({suffix})", key,
                                               s_label))
            record_quartiles(data, var, f"{name} ({suffix})", key, s_label)
    # Breiter Nullwert vs. dokumentierte Spiele: Positive Ratio mit und ohne Metacritic-Match.
    h3_rows.append(spearman_row(rating[~rating["has_mc_external"]], "positive_ratio", "log_owners",
                                "Positive Ratio vs. Besitzer (mind. 10 Reviews, ohne Metacritic-Match)",
                                "h3_pr_r10_nomc_rho", stichprobe=f"{S_R10}, ohne Metascore", rolle="Zusatz"))
    robust_continuous(rating, "positive_ratio", "Positive Ratio (mind. 10 Reviews)", "pr_r10", S_R10)
    robust_quartile(rating, "positive_ratio", "Positive Ratio Q4 vs. Q1 (mind. 10 Reviews)", "pr_r10", S_R10)
    robust_quartile(rating50, "positive_ratio", "Positive Ratio Q4 vs. Q1 (mind. 50 Reviews)", "pr_r50", S_R50)
    robust_quartile(rating, "wilson", "Wilson-Score Q4 vs. Q1 (mind. 10 Reviews)", "wilson_r10", S_R10)

    meta_peak = meta[meta["peak_sample"]]
    h3_rows.append(spearman_row(meta, "mc_metascore", "log_owners", "Metascore (extern) vs. Besitzer", "h3_mc_rho",
                                stichprobe=S_MC))
    h3_rows.append(spearman_row(meta, "mc_metascore", "log_owners", "Metascore (extern) vs. Besitzer",
                                "h3_mc_rho_partial", partial=True, stichprobe=S_MC))
    h3_rows.append(cliff_row(meta["mc_metascore"], meta["peak_positive"], "Metascore: Peak > 0 vs. Peak = 0",
                             "h3_mc_peakpos_delta", "peak_positive", S_MC, rolle="Hurdle (Peak > 0)"))
    h3_rows.append(spearman_row(meta_peak, "mc_metascore", "log_peak_ccu", "Metascore vs. Peak CCU (nur Peak > 0)",
                                "h3_mc_peak_rho", stichprobe=f"{S_MC}, Peak > 0", rolle="Intensität (Peak > 0)"))
    h3_rows.append(spearman_row(meta_peak, "mc_metascore", "log_peak_ccu", "Metascore vs. Peak CCU (nur Peak > 0)",
                                "h3_mc_peak_rho_partial", partial=True, stichprobe=f"{S_MC}, Peak > 0",
                                rolle="Intensität (Peak > 0)"))
    h3_rows.append(quartile_or(meta, "mc_metascore", "Erfolgsquote Metascore Q4 vs. Q1", "h3_mc_q", stichprobe=S_MC))
    h3_rows.extend(quartile_shape_rows(meta, "mc_metascore", "Metascore: Erfolgsquote über Quartile", "h3_mc", S_MC))
    record_quartiles(meta, "mc_metascore", "Metascore (extern)", "h3_mc", S_MC)
    robust_continuous(meta, "mc_metascore", "Metascore (extern)", "mc", S_MC)
    robust_quartile(meta, "mc_metascore", "Metascore Q4 vs. Q1", "mc", S_MC)
    meta_user = meta[meta["mc_user_score"].notna()]
    s_mc_user = f"{S_MC}, Nutzerscore vorhanden"
    h3_rows.append(spearman_row(meta_user, "mc_user_score", "log_owners", "Metacritic-Nutzerscore vs. Besitzer",
                                "h3_mcuser_rho", stichprobe=s_mc_user, rolle="Zusatz"))
    h3_rows.append(spearman_row(meta_user, "mc_user_score", "log_owners", "Metacritic-Nutzerscore vs. Besitzer",
                                "h3_mcuser_rho_partial", partial=True, stichprobe=s_mc_user, rolle="Zusatz"))
    h3_rows.append(spearman_row(meta_user[meta_user["peak_sample"]], "mc_user_score", "log_peak_ccu",
                                "Metacritic-Nutzerscore vs. Peak CCU (nur Peak > 0)", "h3_mcuser_peak_rho",
                                stichprobe=f"{s_mc_user}, Peak > 0", rolle="Intensität (Peak > 0)"))
    # Identische Stichprobe fuer den Vergleich Kritiker vs. Nutzer (Metascore und mind. 10 Reviews).
    meta_r10 = meta[meta["rating_sample"]]
    s_meta_r10 = f"{S_MC}, mind. 10 Reviews"
    h3_rows.append(spearman_row(meta_r10, "mc_metascore", "log_owners", "Metascore vs. Besitzer (identische Stichprobe)",
                                "h3_ident_mc_rho", stichprobe=s_meta_r10, rolle="Zusatz"))
    h3_rows.append(spearman_row(meta_r10, "positive_ratio", "log_owners",
                                "Positive Ratio vs. Besitzer (identische Stichprobe)", "h3_ident_pr_rho",
                                stichprobe=s_meta_r10, rolle="Zusatz"))
    h3_rows.append(spearman_row(meta_r10, "positive_ratio", "log_owners",
                                "Positive Ratio vs. Besitzer (identische Stichprobe)", "h3_ident_pr_rho_partial",
                                partial=True, stichprobe=s_meta_r10, rolle="Zusatz"))
    robust_continuous(meta_r10, "positive_ratio", "Positive Ratio (identische Metacritic-Stichprobe)", "pr_ident",
                      s_meta_r10)
    # Positive Ratio und Wilson nur ab 10 Reviews (wie die Korrelationsmatrizen und fig_10).
    h3_rows.append(spearman_row(meta_r10, "positive_ratio", "mc_metascore",
                                "Positive Ratio vs. Metascore (Übereinstimmung)", "h3_agree_pr_mc",
                                stichprobe=s_meta_r10, rolle="Übereinstimmung"))
    h3_rows.append(spearman_row(meta_r10, "wilson", "mc_metascore", "Wilson vs. Metascore (Übereinstimmung)",
                                "h3_agree_wilson_mc", stichprobe=s_meta_r10, rolle="Übereinstimmung"))
    both_meta = meta[meta["metacritic"].notna()]
    h3_rows.append(spearman_row(both_meta, "mc_metascore", "metacritic",
                                "Externer vs. FronkonGames-Metascore (Übereinstimmung)", "h3_agree_mc_fg",
                                stichprobe=f"{S_MC}, FG-Metacritic vorhanden", rolle="Übereinstimmung"))
    h3 = finalize(h3_rows, "H3 - Bewertungen")
    save(h3, "h3_tests")

    # Kritiker vs. Nutzer in identischen Stichproben: Bootstrap-Differenz der Korrelationen.
    ident_rows = []
    igdb_both = igdb[igdb["igdb_aggregated_rating"].notna() & igdb["igdb_rating"].notna()]
    comparisons = [
        ("mc_metascore", "positive_ratio", "Metascore", "Positive Ratio", meta_r10, s_meta_r10, ""),
        ("mc_metascore", "mc_user_score", "Metascore", "Metacritic-Nutzerscore",
         meta[meta["mc_user_score"].notna()], s_mc_user, ""),
        ("igdb_aggregated_rating", "positive_ratio", "IGDB-Kritikerbewertung", "Positive Ratio",
         igdb[igdb["rating_sample"] & igdb["igdb_aggregated_rating"].notna()],
         "IGDB-Kritikerbewertung vorhanden, mind. 10 Reviews", ""),
        ("igdb_rating", "positive_ratio", "IGDB-Nutzerbewertung", "Positive Ratio",
         igdb[igdb["rating_sample"] & igdb["igdb_rating"].notna()],
         "IGDB-Nutzerbewertung vorhanden, mind. 10 Reviews", ""),
        # Die Einzelkorrelationen der beiden IGDB-Bewertungen stammen aus verschiedenen
        # Teilstichproben; der Vergleich der Staerke braucht dieselben Spiele.
        ("igdb_aggregated_rating", "igdb_rating", "IGDB-Kritikerbewertung", "IGDB-Nutzerbewertung",
         igdb_both, "IGDB-Kritiker- und -Nutzerbewertung vorhanden", ""),
        ("igdb_aggregated_rating", "igdb_rating", "IGDB-Kritikerbewertung", "IGDB-Nutzerbewertung (nicht ganzzahlig)",
         igdb_both[igdb_both["igdb_rating_integer"].eq(False)],
         "IGDB-Kritiker- und -Nutzerbewertung vorhanden, Nutzerbewertung nicht ganzzahlig", "_nonint"),
    ]
    for i, (a, b, la, lb, data, s_label, tag) in enumerate(comparisons):
        d = data[[a, b, "log_owners", "release_year"]].dropna()
        for partial in (False, True):
            res = bootstrap_difference(d, a, b, "log_owners", partial, RANDOM_STATE)
            ident_rows.append({
                "Kennung": f"ident_{a}_vs_{b}{tag}{'_partial' if partial else ''}",
                "Vergleich": f"{la} vs. {lb}", "A": a, "B": b, "Zielgroesse": "log_owners",
                "Modell": "partiell: Jahr-FE" if partial else "unadjustiert", "Stichprobe": s_label,
                "Erfolgsquote": float(data.loc[d.index, OUTCOME].mean()), **res,
            })
    ident = pd.DataFrame(ident_rows)
    save(ident, "h3_identical_sample")

    # Kohortenstratifizierung der Bewertungszusammenhaenge mit Heterogenitaetstest.
    cohort_rows = []
    cohort_specs = [
        ("positive_ratio", "Positive Ratio", rating, S_R10),
        ("positive_ratio", "Positive Ratio", rating50, S_R50),
        ("wilson", "Wilson-Score", rating, S_R10),
        ("mc_metascore", "Metascore", meta_r10, s_meta_r10),
        ("positive_ratio", "Positive Ratio", meta_r10, s_meta_r10),
    ]
    for spec_idx, (var, name, data, s_label) in enumerate(cohort_specs):
        zs, ws = [], []
        for c_idx, (start, end, c_label) in enumerate(COHORTS):
            part = data[(data["release_year"] >= start) & (data["release_year"] <= end)][[var, "log_owners"]].dropna()
            if len(part) < 10:
                rho, p, low, high = np.nan, np.nan, np.nan, np.nan
            else:
                rho, p = spearmanr(part[var], part["log_owners"])
                low, high = spearman_ci(rho, len(part))
                zs.append(np.arctanh(rho))
                ws.append(len(part) - 3)
            cohort_rows.append({"Kennung": f"cohort_{spec_idx}_c{c_idx}",
                                "Merkmal": name, "Stichprobe": s_label, "Kohorte": c_label, "N": len(part),
                                "Erfolgsquote": float(data.loc[part.index, OUTCOME].mean()) if len(part) else np.nan,
                                "Effekt": float(rho), "Effekt_typ": "Spearman rho (vs. log_owners)",
                                "CI_unten": low, "CI_oben": high, "p": float(p)})
        zs_a, ws_a = np.array(zs), np.array(ws, dtype=float)
        z_bar = np.sum(ws_a * zs_a) / ws_a.sum()
        q_stat = float(np.sum(ws_a * (zs_a - z_bar) ** 2))
        cohort_rows.append({"Kennung": f"cohort_{spec_idx}_heterogeneity",
                            "Merkmal": name, "Stichprobe": s_label, "Kohorte": "Heterogenität (Cochran-Q, Fisher-z)",
                            "N": int(ws_a.sum() + 3 * len(ws_a)), "Effekt": q_stat,
                            "Effekt_typ": f"Q (df={len(zs_a) - 1})",
                            "p": float(chi2_dist.sf(q_stat, len(zs_a) - 1))})
    cohort_table = pd.DataFrame(cohort_rows)
    cohort_table["p_adj"] = bh_correction(cohort_table["p"])
    save(cohort_table, "h3_cohort_stratified")

    # ------------------------------------------------------------------ H4
    h4_rows: list[dict] = []
    igdb_modes = igdb[igdb["igdb_has_modes"].eq(True)].copy()
    igdb_genres = igdb[igdb["igdb_has_genres"].eq(True)].copy()
    for name, _, _, members in IGDB_GENRE_PAIRS:
        if name not in igdb_genres.columns:
            igdb_genres[name] = igdb_genres[members].eq(True).any(axis=1)
    agreement_rows = []
    pair_specs = [(ig, st, lab, igdb_modes, S_IGDB_MODES, "Modus") for ig, st, lab in IGDB_MODE_PAIRS]
    pair_specs += [(ig, st, lab, igdb_genres, S_IGDB_GENRES, "Genre") for ig, st, lab, _ in IGDB_GENRE_PAIRS]
    pair_specs.append(("igdb_mmo", "genre_massively_multiplayer", "MMO (IGDB-Modus) vs. Steam-Genre Massively Multiplayer",
                       igdb_modes, S_IGDB_MODES, "Modus/Genre"))
    for ig_col, st_col, label, data, s_label, kind in pair_specs:
        d = data[data[ig_col].notna()]
        n_true = int(d[ig_col].eq(True).sum())
        if n_true == 0:
            print(f"WARNUNG: IGDB-Merkmal {ig_col} ohne positive Fälle in {s_label}")
        agreement_rows.append({"Kennung": f"agree_{ig_col}_{st_col}", "Art": kind, "Merkmal": label,
                               "IGDB": ig_col, "Steam": st_col, "Stichprobe": s_label,
                               **cohen_kappa(d[ig_col], d[st_col])})
        if kind == "Modus/Genre":
            continue
        key = ig_col.removeprefix("igdb_")
        # Action hat in IGDB keine Entsprechung (kappa niedrig); nur als Zusatz ausgewiesen.
        rolle_main = "Zusatz" if ig_col == "igdb_action_like" else "Haupttest"
        h4_rows.append(chi2_indicator(d, ig_col, f"IGDB-{kind}: {label}", f"h4_igdb_{key}_or", stichprobe=s_label,
                                      rolle=rolle_main))
        h4_rows.append(adjusted_or(d, ig_col, f"IGDB-{kind}: {label}", f"h4_igdb_{key}_aor", stichprobe=s_label))
        h4_rows.append(adjusted_or(d, ig_col, f"IGDB-{kind}: {label}", f"h4_igdb_{key}_aor_quad", spec="quad",
                                   stichprobe=s_label))
        h4_rows.append(chi2_indicator(d, st_col, f"Steam-{kind} (gleiche Teilstichprobe): {label}",
                                      f"h4_steam_{key}_or", stichprobe=s_label, rolle="Vergleich Steam"))
        h4_rows.append(adjusted_or(d, st_col, f"Steam-{kind} (gleiche Teilstichprobe): {label}",
                                   f"h4_steam_{key}_aor", stichprobe=s_label, rolle="Vergleich Steam"))
        # Zusatzinformation der IGDB-Angabe ueber das Steam-Merkmal hinaus.
        h4_rows.append(adjusted_or(d, ig_col, f"IGDB-{kind} gegeben Steam-Merkmal: {label}",
                                   f"h4_igdb_{key}_aor_given_steam", covars=(st_col,), stichprobe=s_label,
                                   rolle="Zusatzinformation"))
        robust_binary(d, ig_col, f"IGDB-{kind}: {label}", f"igdb_{key}", stichprobe=s_label)
    agreement = pd.DataFrame(agreement_rows)
    save(agreement, "igdb_steam_agreement")

    igdb_user = igdb[igdb["igdb_rating"].notna()]
    # IGDB liefert keine Stimmenzahl; ganzzahlige Bewertungen stammen meist von wenigen Stimmen.
    igdb_nonint = igdb_user[igdb_user["igdb_rating_integer"].eq(False)]
    s_nonint = "IGDB-Nutzerbewertung nicht ganzzahlig"
    rating_strata = [
        ("all", "alle", igdb_user, "IGDB-Nutzerbewertung vorhanden", "Zusatz"),
        ("nonint", "nicht ganzzahlig (mehr Stimmen)", igdb_nonint, s_nonint, "Haupttest"),
        ("int", "ganzzahlig (wenige Stimmen)", igdb_user[igdb_user["igdb_rating_integer"].eq(True)],
         "IGDB-Nutzerbewertung ganzzahlig", "Zusatz"),
    ]
    for key, name, data, s_label, rolle in rating_strata:
        lab = f"IGDB-Nutzerbewertung ({name}) vs. Besitzer"
        h4_rows.append(spearman_row(data, "igdb_rating", "log_owners", lab, f"h4_igdbuser_{key}_rho",
                                    stichprobe=s_label, rolle=rolle))
        h4_rows.append(spearman_row(data, "igdb_rating", "log_owners", lab, f"h4_igdbuser_{key}_rho_partial",
                                    partial=True, stichprobe=s_label,
                                    rolle="Adjustiert" if rolle == "Haupttest" else "Zusatz"))
        h4_rows.append(spearman_row(data[data["peak_sample"]], "igdb_rating", "log_peak_ccu",
                                    f"IGDB-Nutzerbewertung ({name}) vs. Peak CCU (nur Peak > 0)",
                                    f"h4_igdbuser_{key}_peak_rho", stichprobe=f"{s_label}, Peak > 0",
                                    rolle="Intensität (Peak > 0)"))
        h4_rows.append(quartile_or(data, "igdb_rating", f"Erfolgsquote IGDB-Nutzerbewertung ({name}) Q4 vs. Q1",
                                   f"h4_igdbuser_{key}_q", stichprobe=s_label))
        record_quartiles(data, "igdb_rating", f"IGDB-Nutzerbewertung ({name})", f"h4_igdbuser_{key}", s_label)
    h4_rows.append(spearman_row(igdb_user, "igdb_rating_integer", "log_owners",
                                "Ganzzahlige IGDB-Nutzerbewertung (Indikator wenige Stimmen) vs. Besitzer",
                                "h4_igdbuser_integerflag_rho", stichprobe="IGDB-Nutzerbewertung vorhanden",
                                rolle="Diagnose"))
    robust_continuous(igdb_nonint, "igdb_rating", "IGDB-Nutzerbewertung (nicht ganzzahlig)", "igdbuser_nonint",
                      s_nonint)
    robust_quartile(igdb_nonint, "igdb_rating", "IGDB-Nutzerbewertung (nicht ganzzahlig) Q4 vs. Q1",
                    "igdbuser_nonint", s_nonint)

    igdb_crit = igdb[igdb["igdb_aggregated_rating"].notna()]
    s_crit = "IGDB-Kritikerbewertung vorhanden"
    h4_rows.append(spearman_row(igdb_crit, "igdb_aggregated_rating", "log_owners", "IGDB-Kritikerbewertung vs. Besitzer",
                                "h4_igdbcrit_rho", stichprobe=s_crit))
    h4_rows.append(spearman_row(igdb_crit, "igdb_aggregated_rating", "log_owners", "IGDB-Kritikerbewertung vs. Besitzer",
                                "h4_igdbcrit_rho_partial", partial=True, stichprobe=s_crit))
    h4_rows.append(spearman_row(igdb_crit[igdb_crit["peak_sample"]], "igdb_aggregated_rating", "log_peak_ccu",
                                "IGDB-Kritikerbewertung vs. Peak CCU (nur Peak > 0)", "h4_igdbcrit_peak_rho",
                                stichprobe=f"{s_crit}, Peak > 0", rolle="Intensität (Peak > 0)"))
    h4_rows.append(quartile_or(igdb_crit, "igdb_aggregated_rating", "Erfolgsquote IGDB-Kritikerbewertung Q4 vs. Q1",
                               "h4_igdbcrit_q", stichprobe=s_crit))
    record_quartiles(igdb_crit, "igdb_aggregated_rating", "IGDB-Kritikerbewertung", "h4_igdbcrit", s_crit)
    robust_continuous(igdb_crit, "igdb_aggregated_rating", "IGDB-Kritikerbewertung", "igdbcrit", s_crit)
    robust_quartile(igdb_crit, "igdb_aggregated_rating", "IGDB-Kritikerbewertung Q4 vs. Q1", "igdbcrit", s_crit)
    h4_rows.append(spearman_row(igdb_genres, "igdb_n_genres", "log_owners",
                                "Anzahl IGDB-Genres vs. Besitzer (auch Dokumentationstiefe)", "h4_igdb_ngenres_rho",
                                stichprobe=S_IGDB_GENRES, rolle="Zusatz"))
    # Positive Ratio nur ab 10 Reviews (wie die Korrelationsmatrizen in 02).
    h4_rows.append(spearman_row(igdb_user[igdb_user["rating_sample"]], "igdb_rating", "positive_ratio",
                                "IGDB-Nutzerbewertung vs. Positive Ratio (Übereinstimmung)", "h4_agree_igdbuser_pr",
                                stichprobe="IGDB-Nutzerbewertung vorhanden, mind. 10 Reviews", rolle="Übereinstimmung"))
    h4_rows.append(spearman_row(igdb_crit[igdb_crit["rating_sample"]], "igdb_aggregated_rating", "positive_ratio",
                                "IGDB-Kritikerbewertung vs. Positive Ratio (Übereinstimmung)", "h4_agree_igdbcrit_pr",
                                stichprobe=f"{s_crit}, mind. 10 Reviews", rolle="Übereinstimmung"))
    both_crit = igdb_crit[igdb_crit["mc_metascore"].notna()]
    h4_rows.append(spearman_row(both_crit, "igdb_aggregated_rating", "mc_metascore",
                                "IGDB-Kritikerbewertung vs. Metascore (Übereinstimmung)", "h4_agree_igdbcrit_mc",
                                stichprobe="IGDB-Kritikerbewertung und Metascore vorhanden", rolle="Übereinstimmung"))

    # Preishistorie: nur normierte Merkmale; wegen Erfolgsquote nahe 100 % Rangkorrelationen mit log_owners.
    ph_full = ph[ph["ph_full_window"].eq(True)]
    ph_clean = ph[~ph["ph_lowest_above_base"].eq(True)]
    print(f"Preishistorie: {len(ph)} Apps, volles Fenster {len(ph_full)}, ohne lowest > base {len(ph_clean)}, "
          f"Erfolgsquote {ph[OUTCOME].mean():.3f}")
    for var, name in PH_FEATURES:
        key = var.removeprefix("ph_")
        main = ph_full if var in PH_FULL_WINDOW_ONLY else ph
        s_main = S_PH_FULL if var in PH_FULL_WINDOW_ONLY else S_PH
        h4_rows.append(spearman_row(main, var, "log_owners", f"{name} vs. Besitzer", f"h4_ph_{key}_rho",
                                    stichprobe=s_main))
        h4_rows.append(spearman_row(main, var, "log_owners", f"{name} vs. Besitzer", f"h4_ph_{key}_rho_partial",
                                    partial=True, extra_controls=("ph_months",), stichprobe=s_main))
        if var not in PH_FULL_WINDOW_ONLY:
            h4_rows.append(spearman_row(ph_full, var, "log_owners", f"{name} vs. Besitzer (volles Fenster)",
                                        f"h4_ph_{key}_full_rho", stichprobe=S_PH_FULL, rolle="Sensitivität"))
            h4_rows.append(spearman_row(ph_full, var, "log_owners", f"{name} vs. Besitzer (volles Fenster)",
                                        f"h4_ph_{key}_full_rho_partial", partial=True, extra_controls=("ph_months",),
                                        stichprobe=S_PH_FULL, rolle="Sensitivität"))
        clean_main = main[~main["ph_lowest_above_base"].eq(True)]
        h4_rows.append(spearman_row(clean_main, var, "log_owners", f"{name} vs. Besitzer (ohne Tiefstpreis > Basispreis)",
                                    f"h4_ph_{key}_clean_rho_partial", partial=True, extra_controls=("ph_months",),
                                    stichprobe=f"{s_main}, ohne lowest_price > base_price", rolle="Sensitivität"))
        h4_rows.append(spearman_row(main[main["peak_sample"]], var, "log_peak_ccu",
                                    f"{name} vs. Peak CCU (nur Peak > 0)", f"h4_ph_{key}_peak_rho",
                                    stichprobe=f"{s_main}, Peak > 0", rolle="Intensität (Peak > 0)"))
        robust_continuous(main, var, name, f"ph_{key}", s_main)
    for var, name in [("ph_max_discount_pct", "maximaler Rabatt"), ("ph_share_months_on_sale", "Anteil Monate mit Rabatt")]:
        main = ph_full if var in PH_FULL_WINDOW_ONLY else ph
        s_main = S_PH_FULL if var in PH_FULL_WINDOW_ONLY else S_PH
        key = var.removeprefix("ph_")
        h4_rows.append(quartile_or(main, var, f"Erfolgsquote {name} Q4 vs. Q1", f"h4_ph_{key}_q", stichprobe=s_main))
        record_quartiles(main, var, name, f"h4_ph_{key}", s_main)
    h4 = finalize(h4_rows, "H4 - Externe Datenquellen")
    save(h4, "h4_external_tests")

    # ----------------------------------------------- Diagnose Review-Volumen
    # Die Review-Anzahl ist selbst ein Popularitaetsmass. Die Partialisierung ist
    # deshalb eine Ueberkontrolle und dient nur der Diagnose, wie stark ein
    # Bewertungsmass vom Volumen abhaengt - einheitlich fuer alle Bewertungsmasse.
    diag_rows = []
    diag_specs = [
        ("positive_ratio", "Positive Ratio", rating, S_R10),
        ("positive_ratio", "Positive Ratio", rating50, S_R50),
        ("wilson", "Wilson-Score", rating, S_R10),
        ("wilson", "Wilson-Score", rating50, S_R50),
        ("mc_metascore", "Metascore (extern)", meta, S_MC),
        ("mc_user_score", "Metacritic-Nutzerscore", meta_user, s_mc_user),
        ("igdb_rating", "IGDB-Nutzerbewertung", igdb_user, "IGDB-Nutzerbewertung vorhanden"),
        ("igdb_rating", "IGDB-Nutzerbewertung (nicht ganzzahlig)", igdb_nonint, s_nonint),
        ("igdb_aggregated_rating", "IGDB-Kritikerbewertung", igdb_crit, s_crit),
    ]
    for i, (var, name, data, s_label) in enumerate(diag_specs):
        base = f"diag_{i}_{var}"
        diag_rows.append(spearman_row(data, "review_count", "log_owners", f"Review-Anzahl vs. Besitzer ({s_label})",
                                      f"{base}_reviews_owners", stichprobe=s_label, rolle="Diagnose"))
        diag_rows.append(spearman_row(data, var, "review_count", f"{name} vs. Review-Anzahl",
                                      f"{base}_vs_reviews", stichprobe=s_label, rolle="Diagnose"))
        diag_rows.append(spearman_row(data, var, "log_owners", f"{name} vs. Besitzer (roh)",
                                      f"{base}_raw", stichprobe=s_label, rolle="Diagnose"))
        d = data[[var, "log_owners", "log_review_count"]].dropna()
        rho, p, k = partial_rank_corr(d[var].to_numpy(float), d["log_owners"].to_numpy(float),
                                      np.column_stack([np.ones(len(d)), rankdata(d["log_review_count"])]))
        low, high = spearman_ci(rho, len(d), k)
        diag_rows.append(nan_row(Kennung=f"{base}_partial_reviews", Rolle="Diagnose (Überkontrolle)",
                                 Modell="partiell: Review-Anzahl (Rang)", Test="Spearman (partiell)",
                                 Merkmal=f"{name} vs. Besitzer, Review-Anzahl herauspartialisiert",
                                 Gruppe=f"{var} vs. log_owners | log_review_count", Stichprobe=s_label, N=len(d),
                                 Effekt=rho, Effekt_typ="Spearman rho", CI_unten=low, CI_oben=high, p=p,
                                 Hinweis="Überkontrolle: Review-Anzahl ist selbst ein Popularitätsmaß; "
                                         "nur Diagnose der Volumenabhängigkeit, kein inhaltlicher Befund"))
    diag = finalize(diag_rows, "Diagnose - Bewertungsvolumen")
    save(diag, "review_volume_diagnostic")

    # --------------------------------------------- Selektion der Teilstichproben
    selection_rows = []
    availability = [
        ("has_igdb", "IGDB-Treffer", df["has_igdb"]),
        ("igdb_rating", "IGDB-Nutzerbewertung", df["igdb_rating"].notna()),
        ("igdb_aggregated_rating", "IGDB-Kritikerbewertung", df["igdb_aggregated_rating"].notna()),
        ("has_mc_external", "Metascore extern", df["has_mc_external"]),
        ("has_price_history", "Preishistorie", df["has_price_history"]),
        ("has_metacritic", "Metacritic (FronkonGames)", df["has_metacritic"]),
    ]
    # Kennung = Verfuegbarkeit + feste Schichtkennung (nicht die laufende Zeilennummer),
    # damit 05 die Zeilen auch nach dem Hinzufuegen weiterer Schichten eindeutig findet.
    strata = [("gesamt", "gesamt", pd.Series(True, index=df.index))]
    strata += [(f"kohorte{i}", f"Kohorte {lab}", (df["release_year"] >= a) & (df["release_year"] <= b))
               for i, (a, b, lab) in enumerate(COHORTS)]
    strata += [("englisch", "Englisch unterstützt", df["languages_listed"] & df["english_supported"]),
               ("ohne_englisch", "ohne Englisch (mit Sprachangabe)", df["languages_listed"] & ~df["english_supported"])]
    for key, name, avail in availability:
        own_strata = list(strata)
        if key in ("igdb_rating", "igdb_aggregated_rating"):
            own_strata.append(("in_igdb", "innerhalb der IGDB-Treffer", df["has_igdb"]))
        for s_key, s_name, mask in own_strata:
            part = df[mask]
            av = avail[mask]
            raw = chi2_indicator(part.assign(_avail=av), "_avail", name, "tmp")
            cl = cliff_row(part["owners_mid"], av, name, "tmp", key, s_name)
            selection_rows.append({
                "Kennung": f"sel_{key}_{s_key}", "Merkmal": f"Besitzer: {name} vorhanden vs. nicht",
                "Verfuegbarkeit": name, "Schicht": s_name, "N": len(part), "N_verfuegbar": int(av.sum()),
                "Anteil_verfuegbar": float(av.mean()) if len(part) else np.nan,
                "Erfolgsquote_verfuegbar": raw.get("Erfolgsquote_Gruppe"),
                "Erfolgsquote_nicht_verfuegbar": raw.get("Erfolgsquote_Referenz"),
                "Effekt": cl["Effekt"], "Effekt_typ": "Cliff's Delta (Besitzerklasse)", "p": cl["p"],
                "OR_Erfolg": raw["Effekt"], "OR_CI_unten": raw["CI_unten"], "OR_CI_oben": raw["CI_oben"],
                "p_OR": raw["p"], "Hinweis": cl.get("Hinweis", "") or raw.get("Hinweis", ""),
            })
    # IGDB-Treffer nach Match-Stufe gegen Spiele ohne Treffer.
    for stage in (1, 2, 3):
        mask = df["igdb_match_stage"].eq(stage).fillna(False).astype(bool) | ~df["has_igdb"]
        part = df[mask]
        av = part["igdb_match_stage"].eq(stage).fillna(False).astype(bool)
        raw = chi2_indicator(part.assign(_avail=av), "_avail", "IGDB-Stufe", "tmp")
        cl = cliff_row(part["owners_mid"], av, "IGDB-Stufe", "tmp", "stage", "")
        selection_rows.append({
            "Kennung": f"sel_igdb_stage{stage}", "Merkmal": f"Besitzer: IGDB-Treffer Stufe {stage} vs. kein Treffer",
            "Verfuegbarkeit": f"IGDB-Treffer Stufe {stage}", "Schicht": "Stufe vs. kein Treffer", "N": len(part),
            "N_verfuegbar": int(av.sum()), "Anteil_verfuegbar": float(av.mean()),
            "Erfolgsquote_verfuegbar": raw.get("Erfolgsquote_Gruppe"),
            "Erfolgsquote_nicht_verfuegbar": raw.get("Erfolgsquote_Referenz"), "Effekt": cl["Effekt"],
            "Effekt_typ": "Cliff's Delta (Besitzerklasse)", "p": cl["p"], "OR_Erfolg": raw["Effekt"],
            "OR_CI_unten": raw["CI_unten"], "OR_CI_oben": raw["CI_oben"], "p_OR": raw["p"], "Hinweis": "",
        })
    selection = pd.DataFrame(selection_rows)
    assert selection["Kennung"].is_unique, "Kennungen der Selektionstests nicht eindeutig"
    selection["p_adj"] = bh_correction(selection["p"])
    selection["p_adj_OR"] = bh_correction(selection["p_OR"])
    save(selection, "external_selection_tests")

    # --------------------------------------------- Sensitivitaet Release-Datum
    release_rows = []
    shifted = df[~df["release_shifted"]]
    release_flags = ([(c, f"Genre: {g}", df) for g, c in GENRE_COLS.items()]
                     + [(c, f"Spielmodus: {n}", df) for c, n in MODE_COLS]
                     + [(c, lab, df) for c, lab in PLATFORM_FLAGS]
                     + [("is_free", "kostenlos (Listenpreis 0)", df_price),
                        ("has_discount", "Rabatt aktiv (kostenpflichtig)", paid)])
    for flag, label, data in release_flags:
        variants = [
            ("Hauptmodell (Jahr-FE, Steam-Jahr)", data, "release_year"),
            ("ohne release_shifted (IGDB-Jahr < Steam-Jahr − 1)", data[~data["release_shifted"]], "release_year"),
            ("Jahr-FE mit release_year_alt", data, "release_year_alt"),
            ("ohne EA-Absolventen-Proxy (früherer Markteintritt)", data[~data["ea_graduate_proxy"]], "release_year"),
            ("Jahr-FE mit market_entry_year", data, "market_entry_year"),
        ]
        for v_label, part, year_col in variants:
            r = adjusted_or(part, flag, label, "tmp", year_col=year_col)
            release_rows.append({"Kennung": f"rel_{flag}_{len(release_rows)}", "Merkmal": label, "Variante": v_label,
                                 "N": r["N"], "Effekt": r["Effekt"], "Effekt_typ": "adjustierte Odds Ratio",
                                 "CI_unten": r["CI_unten"], "CI_oben": r["CI_oben"], "p": r["p"],
                                 "konvergiert": r.get("konvergiert"), "Hinweis": r.get("Hinweis", "")})
    release = pd.DataFrame(release_rows)
    save(release, "release_date_sensitivity")

    # Kohortenanalysen mit alternativem Release-Jahr: Erfolgsquote je Kohorte,
    # Positive Ratio je Kohorte (mit Heterogenitaetstest), kostenlos je Kohorte und
    # die partiellen Preishistorien-Korrelationen. release_year_alt kann vor 1997
    # liegen; solche Spiele zaehlen zur fruehesten Kohorte.
    rc_rows: list[dict] = []
    rc_variants = [
        ("main", "Steam-Jahr (Hauptanalyse)", df, "release_year"),
        ("noshift", "ohne release_shifted", df[~df["release_shifted"]], "release_year"),
        ("alt", "release_year_alt", df, "release_year_alt"),
        ("noea", "ohne EA-Absolventen-Proxy", df[~df["ea_graduate_proxy"]], "release_year"),
        ("entry", "market_entry_year", df, "market_entry_year"),
    ]
    for v_key, v_label, data, ycol in rc_variants:
        coh = open_cohort_labels(data[ycol])
        rated = data[data["rating_sample"]]
        coh_r = open_cohort_labels(rated[ycol])
        priced = data[data["list_price"].notna()]
        coh_p = open_cohort_labels(priced[ycol])
        zs, ws = [], []
        for c_idx, (_, _, c_label) in enumerate(COHORTS):
            part = data[coh == c_label]
            rc_rows.append({"Kennung": f"rc_{v_key}_success_c{c_idx}", "Analyse": "Erfolgsquote", "Kohorte": c_label,
                            "Variante": v_label, "N": len(part), "Effekt": float(part[OUTCOME].mean()),
                            "Effekt_typ": "Anteil"})
            pr = rated[coh_r == c_label][["positive_ratio", "log_owners"]].dropna()
            rho, p = spearmanr(pr["positive_ratio"], pr["log_owners"])
            low, high = spearman_ci(rho, len(pr))
            zs.append(np.arctanh(rho))
            ws.append(len(pr) - 3)
            rc_rows.append({"Kennung": f"rc_{v_key}_pr_c{c_idx}", "Analyse": "Positive Ratio vs. Besitzer (mind. 10 Reviews)",
                            "Kohorte": c_label, "Variante": v_label, "N": len(pr), "Effekt": float(rho),
                            "Effekt_typ": "Spearman rho", "CI_unten": low, "CI_oben": high, "p": float(p)})
            fr = adjusted_or(priced[coh_p == c_label], "is_free", "kostenlos (Listenpreis 0)", "tmp", year_col=ycol)
            rc_rows.append({"Kennung": f"rc_{v_key}_free_c{c_idx}", "Analyse": "kostenlos (Listenpreis 0), aOR Jahr-FE",
                            "Kohorte": c_label, "Variante": v_label, "N": fr["N"], "Effekt": fr["Effekt"],
                            "Effekt_typ": "adjustierte Odds Ratio", "CI_unten": fr["CI_unten"],
                            "CI_oben": fr["CI_oben"], "p": fr["p"], "konvergiert": fr.get("konvergiert")})
        zs_a, ws_a = np.array(zs), np.array(ws, dtype=float)
        z_bar = np.sum(ws_a * zs_a) / ws_a.sum()
        q_stat = float(np.sum(ws_a * (zs_a - z_bar) ** 2))
        rc_rows.append({"Kennung": f"rc_{v_key}_pr_heterogeneity", "Analyse": "Positive Ratio vs. Besitzer (mind. 10 Reviews)",
                        "Kohorte": "Heterogenität (Cochran-Q, Fisher-z)", "Variante": v_label,
                        "N": int(ws_a.sum() + 3 * len(ws_a)), "Effekt": q_stat, "Effekt_typ": f"Q (df={len(zs_a) - 1})",
                        "p": float(chi2_dist.sf(q_stat, len(zs_a) - 1)), "Mittel_Kohorten_rho": float(np.tanh(z_bar))})
        ph_v = data[data["has_price_history"]]
        for var, name in PH_FEATURES:
            main = ph_v[ph_v["ph_full_window"].eq(True)] if var in PH_FULL_WINDOW_ONLY else ph_v
            r = spearman_row(main, var, "log_owners", f"{name} vs. Besitzer", "tmp", partial=True,
                             extra_controls=("ph_months",), year_col=ycol)
            rc_rows.append({"Kennung": f"rc_{v_key}_ph_{var.removeprefix('ph_')}", "Analyse": f"Preishistorie partiell: {name}",
                            "Kohorte": "alle", "Variante": v_label, "N": r["N"], "Effekt": r["Effekt"],
                            "Effekt_typ": "Spearman rho (partiell: Jahr-FE + Fensterlänge)", "CI_unten": r["CI_unten"],
                            "CI_oben": r["CI_oben"], "p": r["p"]})
    release_cohorts = pd.DataFrame(rc_rows)
    save(release_cohorts, "release_cohort_sensitivity")
    print(f"release_shifted ausgeschlossen: {int(df['release_shifted'].sum())}; "
          f"release_year_alt < release_year: {int((df['release_year_alt'] < df['release_year']).sum())}; "
          f"Analysestichprobe ohne release_shifted: {len(shifted)}; EA-Absolventen-Proxy ausgeschlossen: "
          f"{int(df['ea_graduate_proxy'].sum())}; market_entry_year < release_year: "
          f"{int((df['market_entry_year'] < df['release_year']).sum())}")

    # --------------------------------------------- Robustheit, Quartile, Gesamt
    robust = pd.DataFrame(ROBUST_ROWS)
    for col in ROW_COLUMNS:
        if col not in robust.columns:
            robust[col] = np.nan
    robust = robust[ROW_COLUMNS].drop(columns=["Familie", "p_adj", "signifikant"])
    assert not robust["Kennung"].duplicated().any(), "doppelte Kennungen in der Robustheitstabelle"
    save(robust, "robustness_success_definitions")

    quartiles = pd.DataFrame(QUARTILE_ROWS)
    save(quartiles, "quartile_groups")

    all_tests = pd.concat([h1, h2, h3, h4], ignore_index=True, sort=False)
    assert not all_tests["Kennung"].duplicated().any()
    save(all_tests, "tests_all")

    # --------------------------------------------- Hypothesenentscheidungen
    lookup = pd.concat([all_tests.set_index("Kennung"), robust.set_index("Kennung")], sort=False)
    assert not lookup.index.duplicated().any()
    context = {"mc_success": float(meta[OUTCOME].mean()), "mc_n": len(meta),
               "rating_success": float(rating[OUTCOME].mean())}
    decisions = build_decisions(lookup, free_table, cohort_table, ident, release, context)
    save(decisions, "hypothesis_decisions")

    # --------------------------------------------- Konsole
    logits = pd.DataFrame(LOGIT_LOG)
    n_bad = int((~logits["konvergiert"]).sum())
    print(f"\nLogit-Schätzungen: {len(logits)}, davon nicht konvergiert: {n_bad}; "
          f"mit Hinweis: {int((logits['Hinweis'] != '').sum())}")
    if n_bad:
        print(logits[~logits["konvergiert"]].to_string(index=False))
    cols = ["Kennung", "N", "Effekt", "CI_unten", "CI_oben", "p_adj"]
    for name, table in [("H1", h1), ("H2", h2), ("H3", h3), ("H4", h4)]:
        main = table[table["Rolle"].isin(["Haupttest", "Adjustiert"])]
        print(f"\n{name} (Haupttests und adjustierte Modelle):")
        print(main[cols].to_string(index=False))
    print("\nHypothesenentscheidungen:")
    summary = decisions.drop_duplicates(["Hypothese", "Komponente"])[
        ["Hypothese", "Komponente", "Kernkomponente", "n_gestützt", "n_teiltests", "Komponente_gestützt",
         "Komponente_gestützt_streng", "Entscheidung_Hypothese", "Entscheidung_streng"]]
    print(summary.to_string(index=False))


# ---------------------------------------------------------------------------
# Entscheidungsregeln
# ---------------------------------------------------------------------------

HYPOTHESES = {
    "H1": ("Spieleigenschaften: Genre, Spielmodus und Plattformangebot hängen mit dem Erfolg "
           "(≥ 20.000 Besitzer) zusammen; bestimmte Ausprägungen sind unter erfolgreichen Spielen "
           "über- bzw. unterrepräsentiert."),
    "H2": ("Preis: Die Preisstrategie hängt mit der Popularität zusammen – kostenlose Spiele (Listenpreis 0), "
           "das Preisniveau kostenpflichtiger Spiele (Listenpreis) und aktive Rabatte gehen mit einer anderen "
           "Erfolgsquote bzw. Besitzerzahl einher."),
    "H3": ("Bewertungen: Höhere Kritikerbewertungen (Metascore) und höhere Nutzerbewertungen "
           "(Steam Positive Ratio) gehen mit mehr Besitzern einher."),
    "H4": ("Externe Quellen: IGDB-Spielmodi und -Genres zeigen dieselbe Richtung wie die Steam-Merkmale "
           "(Konsistenzprüfung, keine unabhängige Bestätigung), IGDB-Bewertungen hängen positiv mit den Besitzern "
           "zusammen, und normierte Preishistorien-Merkmale hängen mit den Besitzern zusammen."),
}

RULE_TEXT = (f"Teiltest gestützt, wenn BH-q < 0,05, praktisch relevant (OR ≥ {REL_OR} bzw. ≤ 1/{REL_OR}; "
             f"|ρ| ≥ {REL_RHO}; |Cliff's δ| ≥ {REL_DELTA}; ε² ≥ {REL_EPS2}) und richtungskonsistent: gleiche Richtung "
             "im adjustierten Modell (falls vorhanden; die Richtung muss dort statistisch gesichert sein, q < 0,05) "
             "sowie unter success_50k und success_100k (bei H4 zusätzlich: IGDB-Merkmal in derselben Richtung wie "
             "das Steam-Merkmal bzw. gleiches Vorzeichen der rohen Korrelation im vollen Preisfenster). Komponente "
             "gestützt, wenn mindestens ein Teiltest gestützt ist; Hypothese gestützt = alle Kernkomponenten, "
             "teilweise = mindestens eine, nicht gestützt = keine. Streng: zusätzlich ist der adjustierte Effekt "
             "praktisch relevant. Sensitivitäten (Release-Datum einschließlich geschätztem Markteintritt, „0 - 0“, "
             "Releases bis 2020, Tagespreis, 54,99-Cluster, partielle Korrelation im vollen Preisfenster) werden "
             "ausgewiesen, gehen aber nicht in die Entscheidung ein; adjustierte bzw. partielle Sensitivitäten werden "
             "mit dem adjustierten Haupteffekt verglichen, rohe mit dem rohen.")


def build_decisions(lookup: pd.DataFrame, free_table: pd.DataFrame, cohort_table: pd.DataFrame,
                    ident: pd.DataFrame, release: pd.DataFrame, context: dict) -> pd.DataFrame:
    """Wendet die vorab festgelegten Entscheidungsregeln auf die Testergebnisse an."""

    def release_sens(label: str | None) -> list[tuple]:
        if label is None:
            return []
        part = release[(release["Merkmal"] == label) & ~release["Variante"].str.startswith("Hauptmodell")]
        return [(r["Variante"], r["Effekt"], r["p"], r["Effekt_typ"]) for _, r in part.iterrows()]

    def lookup_sens(keys: list[tuple[str, str]]) -> list[tuple]:
        out = []
        for key, desc in keys:
            r = lookup.loc[key]
            out.append((desc, r["Effekt"], r["p"], r["Effekt_typ"]))
        return out

    def binary(hyp, komp, teil, key_main, key_adj, key_rob, kern=True, kriterium=None, extra=(), hinweis="",
               sens=()):
        return dict(hyp=hyp, komp=komp, teil=teil, kern=kern, primary=key_main, adjusted=key_adj,
                    alt=[f"rob_{key_rob}_{o}_or" for o in ALT_OUTCOMES], extra=list(extra), hinweis=hinweis,
                    sens=list(sens),
                    kriterium=kriterium or "OR (χ², success_top25); adjustiert: Logit mit Jahr-FE; "
                                           "Richtung auch bei success_50k/success_100k (OR)")

    def continuous(hyp, komp, teil, key_main, key_adj, key_rob, kern=True, kriterium=None, extra=(), hinweis="",
                   sens=()):
        alt = [f"rob_{key_rob}_{o}_delta" for o in ALT_OUTCOMES] if key_rob else []
        return dict(hyp=hyp, komp=komp, teil=teil, kern=kern, primary=key_main, adjusted=key_adj,
                    alt=alt, extra=list(extra), hinweis=hinweis, sens=list(sens),
                    kriterium=kriterium or "Spearman ρ mit log_owners; adjustiert: partielle Rangkorrelation (Jahr-FE); "
                                           "Richtung auch bei success_50k/success_100k (Cliff's δ)")

    specs = []
    # H1
    for genre, col in GENRE_COLS.items():
        key = col.removeprefix("genre_")
        specs.append(binary("H1", "Genres", f"Genre: {genre}", f"h1_genre_{key}_or", f"h1_genre_{key}_aor",
                            f"genre_{key}", sens=release_sens(f"Genre: {genre}")))
    for col, name in MODE_COLS:
        specs.append(binary("H1", "Spielmodi", f"Spielmodus: {name}", f"h1_mode_{col}_or", f"h1_mode_{col}_aor",
                            f"mode_{col}", sens=release_sens(f"Spielmodus: {name}")))
    for col, label in PLATFORM_FLAGS:
        specs.append(binary("H1", "Plattformen", label, f"h1_platform_{col}_or", f"h1_platform_{col}_aor",
                            f"platform_{col}", sens=release_sens(label)))
    for lvl, label in COMBO_LEVELS:
        k = lvl.replace("+", "").lower()
        specs.append(binary("H1", "Plattformen", f"Plattformkombination: {label} vs. nur Windows",
                            f"h1_combo_{k}_or", f"h1_combo_{k}_aor", f"combo_{k}"))

    # H2
    free_hint = []
    an = free_table[(free_table["Stichprobe"] == S_AN_PRICE) & (free_table["Merkmal"] == "kostenlos (Listenpreis 0)")]
    for _, r in an[an["Kohorte"].isin([c for _, _, c in COHORTS])].iterrows():
        free_hint.append(f"{r['Kohorte']}: aOR {r['aOR_JahrFE']:.2f} [{r['aOR_CI_unten']:.2f}; {r['aOR_CI_oben']:.2f}]")
    inter = an[an["Kohorte"].str.startswith("Interaktion")]
    if not inter.empty:
        free_hint.append(f"Interaktion × Kohorte {p_text(inter['p_Interaktion'].iloc[0])}")
    free_sens = release_sens("kostenlos (Listenpreis 0)")
    for s_label, kohorte, desc in [(S_ZERO, "gesamt", "„0 - 0“ als Nicht-Erfolg einbezogen"),
                                   (S_LE2020, "gesamt", "nur Releases bis 2020"),
                                   (S_AN_PRICE, COHORTS[-1][2], f"nur Kohorte {COHORTS[-1][2]}")]:
        r = free_table[(free_table["Stichprobe"] == s_label) & (free_table["Merkmal"] == "kostenlos (Listenpreis 0)")
                       & (free_table["Kohorte"] == kohorte)]
        if not r.empty:
            free_sens.append((desc, r["aOR_JahrFE"].iloc[0], r["p_aOR"].iloc[0], "adjustierte Odds Ratio"))
    free_sens += lookup_sens([("h2_f2ptag_aor", "Steam-Tag „Free To Play“ statt Listenpreis 0"),
                              ("h2_free_without_tag_aor", "kostenlos ohne F2P-Tag vs. kostenpflichtig")])
    specs.append(binary("H2", "kostenlos vs. kostenpflichtig", "kostenlos (Listenpreis 0)", "h2_free_or", "h2_free_aor",
                        "free", hinweis="; ".join(free_hint), sens=free_sens))
    kw = lookup.loc["h2_price_class_kw"]
    specs.append(continuous("H2", "Preisniveau (Listenpreis)", "Listenpreis vs. Besitzer (kostenpflichtig, ohne Ausreißer)",
                            "h2_price_rho", "h2_price_rho_partial", "price",
                            hinweis=f"Kruskal-Wallis nach Preisklasse ε² = {kw['Zusatz_Effekt']:.3f}; {kw['Hinweis']}",
                            sens=lookup_sens([("h2_price_rho_with_outliers", "inkl. Preisausreißer"),
                                              ("h2_price_rho_nocluster_partial", "ohne 54,99-Cluster (partiell)"),
                                              ("h2_price_current_rho_partial", "Tagespreis statt Listenpreis (partiell)"),
                                              ("h2_price_rho_nodiscount_partial", "nur Spiele ohne aktuellen Rabatt (partiell)"),
                                              ("h2_price_rho_discount_partial", "nur Spiele mit aktuellem Rabatt (partiell)")])))
    specs.append(binary("H2", "Rabatt", "Rabatt aktiv vs. kein Rabatt (kostenpflichtig)", "h2_discount_or",
                        "h2_discount_aor", "discount",
                        hinweis="Rabatt zum Store-Snapshot (ca. März 2026), teils fehlend und damit Untergrenze; "
                                "SteamSpy-Besitzer stammen von ca. Juni 2025",
                        sens=release_sens("Rabatt aktiv (kostenpflichtig)")))
    specs.append(continuous("H2", "DLC-Anzahl (Zusatz)", "DLC-Anzahl vs. Besitzer", "h2_dlc_rho", "h2_dlc_rho_partial",
                            None, kern=False, kriterium="Spearman ρ; adjustiert: partiell Jahr-FE (keine "
                                                       "Robustheitsprüfung, nicht entscheidungsrelevant)"))

    # H3
    def ident_text(a, b, tag=""):
        part = ident[ident["Kennung"].isin([f"ident_{a}_vs_{b}{tag}", f"ident_{a}_vs_{b}{tag}_partial"])]
        out = []
        for _, r in part.iterrows():
            out.append(f"{r['Modell']}: ρ {r['rho_A']:.2f} vs. {r['rho_B']:.2f}, Δ {r['Differenz']:.3f} "
                       f"[{r['CI_unten']:.3f}; {r['CI_oben']:.3f}] (N={int(r['N'])})")
        return "; ".join(out)

    def cohort_text(var, s_label):
        part = cohort_table[(cohort_table["Merkmal"] == var) & (cohort_table["Stichprobe"] == s_label)]
        out = [f"{r['Kohorte']}: {r['Effekt']:.2f}" for _, r in part.iterrows() if not r["Kohorte"].startswith("Hetero")]
        het = part[part["Kohorte"].str.startswith("Hetero")]
        if not het.empty:
            out.append(f"Heterogenität {p_text(het['p_adj'].iloc[0], 'q (BH)')}")
        return "; ".join(out)

    shape = lookup.loc["h3_pr_r10_qchi2"]
    specs.append(continuous("H3", "Kritikerbewertung", "Metascore vs. Besitzer", "h3_mc_rho", "h3_mc_rho_partial", "mc",
                            hinweis=f"selektive Teilstichprobe ({context['mc_n']} Spiele, Erfolgsquote "
                                    f"{context['mc_success']:.1%} vs. {context['rating_success']:.1%} in der "
                                    "Bewertungsstichprobe); identische Stichprobe mit Positive Ratio: "
                                    f"{ident_text('mc_metascore', 'positive_ratio')}"))
    specs.append(continuous("H3", "Nutzerbewertung (Positive Ratio)", "Positive Ratio vs. Besitzer (mind. 10 Reviews)",
                            "h3_pr_r10_rho", "h3_pr_r10_rho_partial", "pr_r10",
                            hinweis=f"nicht monoton ({shape['Hinweis']}); Kohorten: "
                                    f"{cohort_text('Positive Ratio', S_R10)}",
                            sens=lookup_sens([("h3_pr_r50_rho", "mind. 50 Reviews (roh)"),
                                              ("h3_pr_r50_rho_partial", "mind. 50 Reviews (partiell)")])))
    specs.append(continuous("H3", "Nutzerbewertung in identischer Metacritic-Stichprobe (Zusatz)",
                            "Positive Ratio vs. Besitzer (Metascore vorhanden, mind. 10 Reviews)", "h3_ident_pr_rho",
                            "h3_ident_pr_rho_partial", "pr_ident", kern=False,
                            hinweis="Vergleichbarkeit mit der Kritikerbewertung; nicht entscheidungsrelevant"))

    # H4 (Action hat in IGDB keine Entsprechung; die naeherungsweise Zuordnung geht nicht in die Entscheidung ein)
    for ig, st, label in IGDB_MODE_PAIRS:
        key = ig.removeprefix("igdb_")
        specs.append(binary("H4", "IGDB-Modi/-Genres richtungsgleich mit Steam", f"IGDB-Modus: {label}",
                            f"h4_igdb_{key}_or", f"h4_igdb_{key}_aor", f"igdb_{key}",
                            extra=[(f"h4_steam_{key}_or", "Richtung wie Steam-Merkmal")],
                            sens=lookup_sens([(f"h4_igdb_{key}_aor_given_steam",
                                               "IGDB-Merkmal gegeben Steam-Merkmal (Zusatzinformation)")]),
                            kriterium="OR (χ², nur Spiele mit Modusangabe); adjustiert: Jahr-FE; Richtung auch bei "
                                      "success_50k/success_100k und gleich dem Steam-Merkmal in derselben "
                                      "Teilstichprobe (Konsistenz, keine unabhängige Bestätigung)"))
    for ig, st, label, _ in IGDB_GENRE_PAIRS:
        if ig == "igdb_action_like":
            continue
        key = ig.removeprefix("igdb_")
        specs.append(binary("H4", "IGDB-Modi/-Genres richtungsgleich mit Steam", f"IGDB-Genre: {label}",
                            f"h4_igdb_{key}_or", f"h4_igdb_{key}_aor", f"igdb_{key}",
                            extra=[(f"h4_steam_{key}_or", "Richtung wie Steam-Merkmal")],
                            sens=lookup_sens([(f"h4_igdb_{key}_aor_given_steam",
                                               "IGDB-Merkmal gegeben Steam-Merkmal (Zusatzinformation)")]),
                            kriterium="OR (χ², nur Spiele mit Genreangabe); adjustiert: Jahr-FE; Richtung auch bei "
                                      "success_50k/success_100k und gleich dem Steam-Genre in derselben "
                                      "Teilstichprobe (Konsistenz, keine unabhängige Bestätigung)"))
    specs.append(continuous("H4", "IGDB-Bewertungen", "IGDB-Nutzerbewertung (nicht ganzzahlig) vs. Besitzer",
                            "h4_igdbuser_nonint_rho", "h4_igdbuser_nonint_rho_partial", "igdbuser_nonint",
                            hinweis="Ganzzahlige Bewertungen (wenige Stimmen, Häufung bei 70) getrennt: ρ = "
                                    f"{lookup.loc['h4_igdbuser_int_rho', 'Effekt']:.3f}; stark selektiert "
                                    "(siehe external_selection_tests.csv)"))
    specs.append(continuous("H4", "IGDB-Bewertungen", "IGDB-Kritikerbewertung vs. Besitzer", "h4_igdbcrit_rho",
                            "h4_igdbcrit_rho_partial", "igdbcrit",
                            hinweis="stark selektiert (siehe external_selection_tests.csv); identische Stichprobe "
                                    "mit IGDB-Nutzerbewertung: "
                                    f"{ident_text('igdb_aggregated_rating', 'igdb_rating')}"))
    for var, name in PH_FEATURES:
        key = var.removeprefix("ph_")
        full_only = var in PH_FULL_WINDOW_ONLY
        extra = [] if full_only else [(f"h4_ph_{key}_full_rho", "Richtung im vollen Fenster")]
        sens_keys = [(f"h4_ph_{key}_clean_rho_partial", "ohne Apps mit Tiefstpreis > Basispreis (partiell)")]
        if not full_only:
            sens_keys.insert(0, (f"h4_ph_{key}_full_rho_partial", "volles Fenster (partiell)"))
        specs.append(continuous("H4", "Preishistorie (normierte Merkmale)", f"{name} vs. Besitzer",
                                f"h4_ph_{key}_rho", f"h4_ph_{key}_rho_partial", f"ph_{key}", extra=extra,
                                sens=lookup_sens(sens_keys),
                                kriterium="Spearman ρ mit log_owners; adjustiert: partiell Jahr-FE + ph_months; "
                                          "Richtung auch im vollen Fenster und bei success_50k/success_100k "
                                          "(Cliff's δ); keine Erfolgs-ORs, da Erfolgsquote nahe 100 %"))

    rows = []
    for s in specs:
        pr = lookup.loc[s["primary"]]
        sign = direction(pr["Effekt"], pr["Effekt_typ"])
        relevant = is_relevant(pr["Effekt"], pr["Effekt_typ"])
        q = pr["p_adj"]
        checks = []
        # Woertliche Lesart der Regel ("gleiche Richtung" = nur gleiches Vorzeichen im
        # adjustierten Modell, ohne gesicherte Richtung); nur zur Offenlegung, welche
        # Urteile von der Auslegung abhaengen.
        checks_literal = []
        adj_eff = adj_q = np.nan
        rel_adj = None
        if s["adjusted"]:
            ar = lookup.loc[s["adjusted"]]
            adj_eff, adj_q = ar["Effekt"], ar["p_adj"]
            rel_adj = is_relevant(adj_eff, ar["Effekt_typ"])
            # Gleiche Richtung heisst: gleiches Vorzeichen und im adjustierten Modell
            # statistisch gesichert; ein Nulleffekt nach Alterskontrolle hat keine Richtung.
            checks.append(direction(adj_eff, ar["Effekt_typ"]) == sign and np.isfinite(adj_q) and adj_q < 0.05)
            checks_literal.append(direction(adj_eff, ar["Effekt_typ"]) == sign)
            conv = ar.get("konvergiert")
            if isinstance(conv, (bool, np.bool_)) and not conv:
                checks.append(False)
                checks_literal.append(False)
        alt_values = []
        for key in s["alt"]:
            ar = lookup.loc[key]
            alt_values.append(ar["Effekt"])
            checks.append(direction(ar["Effekt"], ar["Effekt_typ"]) == sign)
            checks_literal.append(direction(ar["Effekt"], ar["Effekt_typ"]) == sign)
        extra_notes = []
        for key, desc in s["extra"]:
            er = lookup.loc[key]
            ok = direction(er["Effekt"], er["Effekt_typ"]) == sign
            checks.append(ok)
            checks_literal.append(ok)
            extra_notes.append(f"{desc}: {'ja' if ok else 'nein'} ({er['Effekt_typ']} {er['Effekt']:.3f})")
        consistent = bool(sign != 0 and all(checks))
        supported = bool(np.isfinite(q) and q < 0.05 and relevant and consistent)
        strict = bool(supported and (not s["adjusted"] or rel_adj))
        supported_literal = bool(np.isfinite(q) and q < 0.05 and relevant and sign != 0 and all(checks_literal))
        strict_literal = bool(supported_literal and (not s["adjusted"] or rel_adj))
        sens_texts, sens_ok = [], []
        # Adjustierte bzw. partielle Sensitivitaeten werden mit dem adjustierten
        # Haupteffekt verglichen, rohe mit dem rohen; sonst wuerde z. B. eine
        # partielle Korrelation an einem rohen Nulleffekt gemessen.
        sign_adj = direction(adj_eff, lookup.loc[s["adjusted"], "Effekt_typ"]) if s["adjusted"] else sign
        p_main = pr["p"]
        p_adj_main = lookup.loc[s["adjusted"], "p"] if s["adjusted"] else p_main
        for desc, eff, p, typ in s["sens"]:
            is_adj = "adjustiert" in str(typ) or "partiell" in desc
            ref, ref_p = (sign_adj, p_adj_main) if is_adj else (sign, p_main)
            same = direction(eff, typ) == ref and ref != 0
            ok = bool(same and np.isfinite(p) and p < 0.05)
            sens_ok.append(ok)
            if ref == 0:
                suffix = " – Haupteffekt ohne Richtung"
            elif not same:
                # Wechselt das Vorzeichen zwischen zwei Nulleffekten, ist das kein Widerspruch.
                both_ns = not (np.isfinite(p) and p < 0.05) and not (np.isfinite(ref_p) and ref_p < 0.05)
                suffix = " – Vorzeichen entgegengesetzt (beide n. s.)" if both_ns else " – Richtung entgegengesetzt"
            elif not ok:
                suffix = " – gleiche Richtung, nicht signifikant"
            elif not is_relevant(eff, typ):
                suffix = " – nicht praktisch relevant"
            else:
                suffix = ""
            sens_texts.append(f"{desc}: {eff:.3f} ({p_text(p, 'p unkorr.')}){suffix}")
        hint = "; ".join([h for h in [s["hinweis"], *extra_notes] if h])
        rows.append({
            "Hypothese": s["hyp"], "Hypothese_Formulierung": HYPOTHESES[s["hyp"]], "Komponente": s["komp"],
            "Kernkomponente": s["kern"], "Teiltest": s["teil"], "Kriterium/Test": s["kriterium"],
            "Kennung_Haupttest": s["primary"], "Stichprobe": pr["Stichprobe"], "N": pr["N"],
            "Effekt_typ": pr["Effekt_typ"], "Effekt": pr["Effekt"], "CI_unten": pr["CI_unten"],
            "CI_oben": pr["CI_oben"], "q": q, "relevant": relevant, "Kennung_adjustiert": s["adjusted"],
            "Effekt_adjustiert": adj_eff, "q_adjustiert": adj_q, "relevant_adjustiert": rel_adj,
            "Effekt_success_50k": alt_values[0] if alt_values else np.nan,
            "Effekt_success_100k": alt_values[1] if len(alt_values) > 1 else np.nan,
            "richtung_konsistent": consistent, "gestützt": supported, "gestützt_streng": strict,
            "gestützt_wörtlich": supported_literal, "gestützt_wörtlich_streng": strict_literal,
            "Sensitivitaeten": "; ".join(sens_texts),
            "Sensitivitaeten_bestaetigt": (all(sens_ok) if sens_ok else pd.NA),
            "Hinweis": hint,
        })
    out = pd.DataFrame(rows)
    comp = out.groupby(["Hypothese", "Komponente"]).agg(
        Komponente_gestützt=("gestützt", "any"), Komponente_gestützt_streng=("gestützt_streng", "any"),
        Komponente_gestützt_wörtlich=("gestützt_wörtlich", "any"),
        Komponente_gestützt_wörtlich_streng=("gestützt_wörtlich_streng", "any"),
        n_teiltests=("gestützt", "size"), n_gestützt=("gestützt", "sum"),
        n_gestützt_wörtlich=("gestützt_wörtlich", "sum")).reset_index()
    out = out.merge(comp, on=["Hypothese", "Komponente"], how="left")

    def verdict(flags: pd.Series) -> str:
        if flags.all():
            return "gestützt"
        if flags.any():
            return "teilweise gestützt"
        return "nicht gestützt"

    core = comp.merge(out[["Hypothese", "Komponente", "Kernkomponente"]].drop_duplicates(), on=["Hypothese", "Komponente"])
    core = core[core["Kernkomponente"]]
    verdicts = core.groupby("Hypothese")["Komponente_gestützt"].apply(verdict).rename("Entscheidung_Hypothese")
    verdicts_strict = core.groupby("Hypothese")["Komponente_gestützt_streng"].apply(verdict).rename("Entscheidung_streng")
    verdicts_lit = (core.groupby("Hypothese")["Komponente_gestützt_wörtlich"].apply(verdict)
                    .rename("Entscheidung_wörtlich"))
    verdicts_lit_strict = (core.groupby("Hypothese")["Komponente_gestützt_wörtlich_streng"].apply(verdict)
                           .rename("Entscheidung_wörtlich_streng"))
    out = (out.merge(verdicts, on="Hypothese").merge(verdicts_strict, on="Hypothese")
           .merge(verdicts_lit, on="Hypothese").merge(verdicts_lit_strict, on="Hypothese"))
    out["Regel"] = RULE_TEXT
    order = ["Hypothese", "Hypothese_Formulierung", "Komponente", "Kernkomponente", "Teiltest", "Kriterium/Test",
             "Kennung_Haupttest", "Stichprobe", "N", "Effekt_typ", "Effekt", "CI_unten", "CI_oben", "q", "relevant",
             "Kennung_adjustiert", "Effekt_adjustiert", "q_adjustiert", "relevant_adjustiert", "Effekt_success_50k",
             "Effekt_success_100k", "richtung_konsistent", "gestützt", "gestützt_streng", "n_teiltests", "n_gestützt",
             "Komponente_gestützt", "Komponente_gestützt_streng", "Entscheidung_Hypothese", "Entscheidung_streng",
             "gestützt_wörtlich", "gestützt_wörtlich_streng", "n_gestützt_wörtlich", "Komponente_gestützt_wörtlich",
             "Komponente_gestützt_wörtlich_streng", "Entscheidung_wörtlich", "Entscheidung_wörtlich_streng",
             "Sensitivitaeten", "Sensitivitaeten_bestaetigt", "Hinweis", "Regel"]
    return out[order]


if __name__ == "__main__":
    main()

"""Erzeugt den Ergebnisbericht outputs/Ergebnisse.md aus den Tabellen und Parquets.

Grundsaetze:
- Keine Zahlliterale fuer Ergebnisse oder Quellenkennzahlen: Jede Zahl im Text
  stammt aus einer Tabelle in outputs/tables (Zeilenwahl ueber die Spalte
  'Kennung' bzw. eindeutige Schluessel) oder aus einem Parquet in data/.
- Die Hypothesenentscheidungen stammen aus hypothesis_decisions.csv (03).
- Qualitative Aussagen im Text (Richtung, 'staerker als', 'nicht signifikant',
  'Plateau', 'nicht monoton') werden mit check() gegen die Tabellen geprueft.
  Aendern sich die Ergebnisse, bricht das Skript ab, statt einen falschen Text
  zu erzeugen.
- Methodenparameter (Mindestgroessen der Jahr-FE-Gruppen, Kohorten, Relevanz-
  schwellen, Bootstrap) werden aus den Quelltexten von 02 und 03 gelesen
  (ohne sie auszufuehren), damit Bericht und Rechnung nicht auseinanderlaufen.
"""

from __future__ import annotations

import ast
import math
import re
from types import SimpleNamespace

import numpy as np
import pandas as pd

from common import (DATA, FIG, MIN_REVIEWS, MIN_REVIEWS_ROBUST, OUT, RANDOM_STATE, RAW_COLUMNS, RAW_HEADER, ROOT,
                    STEAMSPY_CUTOFF, TAB)

# Auswertungsstand des Berichts (Datum der letzten Pipeline-Laeufe).
REPORT_DATE = "15. November 2026"
# Datum des Neulaufs, der Version 58 gegen die urspruengliche Version 57 geprueft hat.
VERSION_CHECK_DATE = "12.11.2026"
# Ergebnis dieses Neulaufs (Vergleich mit den Dateien der urspruenglichen Auswertung;
# nicht Teil der Pipeline): Parquets per SHA-256, Tabellen bytegleich bzw. nur in den
# letzten Stellen verschieden (adjustierte Logit-Schaetzungen in h1/h4/tests_all mit
# 17 Zeilen ausserhalb rtol 1e-9; h2, h3 und review_volume_diagnostic nur um
# Gleitkomma-Rundung von hoechstens 1,1e-16 absolut), Abbildungen pixelgleich.
VERSION_CHECK = {
    "parquet_equal": 10, "parquet_total": 10, "tables_equal": 25, "tables_total": 31,
    "tables_logit": ["h1_tests", "h4_external_tests", "tests_all"], "logit_rows": 17,
    "tables_ulp": ["h2_tests", "h3_tests", "review_volume_diagnostic"],
    "figures": 14,
}
# Determinismuspruefung des finalen Codestands (nicht Teil der Pipeline, manuell per
# MD5-Vergleich zweier vollstaendiger Laeufe 01 -> 02 -> 03 -> 05). Die Dateizahlen werden
# im Lauf gegen die tatsaechlich vorhandenen Ausgaben geprueft; aendert sich die Menge der
# Ausgaben, muss die Pruefung wiederholt und die Konstante angepasst werden.
DETERMINISM_CHECK = {"date": "15. November 2026", "steps": "01 → 02 → 03 → 05", "csv": 54, "parquet": 10,
                     "png": 14}
ALPHA = 0.05
MINUS = "−"
_SUPERSCRIPT = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


class ReportCheckError(AssertionError):
    """Eine Aussage im Berichtstext wird von den Tabellen nicht gedeckt."""


def check(condition: object, message: str) -> None:
    if not bool(condition):
        raise ReportCheckError(f"Berichtsaussage passt nicht zu den Tabellen: {message}")


# ---------------------------------------------------------------------------
# Formatierung (Dezimalkomma, Tausenderpunkt, echtes Minuszeichen)
# ---------------------------------------------------------------------------

def _isna(value: object) -> bool:
    return value is None or (not isinstance(value, str) and pd.isna(value))


def _fmt(value: float, decimals: int) -> str:
    if math.isinf(value):
        return "∞" if value > 0 else MINUS + "∞"
    text = f"{value:,.{decimals}f}".replace(",", "X").replace(".", ",").replace("X", ".")
    if text.startswith("-"):
        text = text[1:] if set(text[1:]) <= set("0,.") else MINUS + text[1:]
    return text


def de(value: object, decimals: int = 1) -> str:
    return "–" if _isna(value) else _fmt(float(value), decimals)


def n0(value: object) -> str:
    return de(value, 0)


def pct(value: object, decimals: int = 1) -> str:
    return "–" if _isna(value) else _fmt(float(value) * 100, decimals) + " %"


def pval(p: object) -> str:
    if _isna(p):
        return "–"
    p = float(p)
    return "< 0,001" if p < 0.001 else _fmt(p, 3)


def psci(p: object) -> str:
    """p-Wert, sehr kleine Werte in Zehnerpotenz-Schreibweise."""
    if _isna(p):
        return "–"
    p = float(p)
    if p == 0:
        return "< 10⁻³⁰⁰"
    if p >= 0.001:
        return _fmt(p, 3)
    exponent = int(math.floor(math.log10(p)))
    mantissa = p / 10 ** exponent
    if round(mantissa, 1) >= 10:
        mantissa /= 10
        exponent += 1
    return f"{_fmt(mantissa, 1)} · 10{str(exponent).translate(_SUPERSCRIPT)}"


def ci(value: object, low: object, high: object, decimals: int = 2) -> str:
    if _isna(value):
        return "–"
    if _isna(low) or _isna(high):
        return de(value, decimals)
    return f"{de(value, decimals)} [{de(low, decimals)}; {de(high, decimals)}]"


def kip(low: object, high: object, decimals: int = 1) -> str:
    """Konfidenzintervall eines Anteils in Prozentpunkten (ohne %-Zeichen)."""
    if _isna(low) or _isna(high):
        return "–"
    return f"[{_fmt(float(low) * 100, decimals)}; {_fmt(float(high) * 100, decimals)}]"


def yn(value: object) -> str:
    if _isna(value):
        return "–"
    return "ja" if str(value) == "True" or value is True else "nein"


def is_true(value: object) -> bool:
    return (not _isna(value)) and (value is True or str(value) == "True")


def de_text(text: object, english: bool = False) -> str:
    """Hinweistexte der Tabellen fuer den Bericht aufbereiten.

    Einige Tabellen schreiben Zahlen in Hinweistexten englisch (03: '1.102',
    '3.2e-09', '94.6%'), andere bereits deutsch (01/02: '20.000', '27,6 %').
    Nur bei ``english=True`` werden Dezimalpunkte in Kommas umgesetzt; Datumsangaben
    (TT.MM.JJJJ) bleiben in jedem Fall unveraendert.
    """
    if _isna(text):
        return ""
    s = str(text).replace("-> NaN", "→ fehlend").replace("->", "→")
    s = s.replace("auf NaN gesetzt", "als fehlend gesetzt").replace("NaN", "fehlend")
    if english:
        def sci(match: re.Match) -> str:
            mantissa = match.group(1).replace(".", ",")
            exponent = int(match.group(2))
            return f"{mantissa} · 10{str(exponent).translate(_SUPERSCRIPT)}"

        s = re.sub(r"(\d+(?:\.\d+)?)e([+-]?\d+)", sci, s)
        # Gleitkomma-Unterlauf (03 schreibt '< 1e-300') wie psci() ausweisen.
        s = s.replace("< 1 · 10⁻³⁰⁰", "< 10⁻³⁰⁰")
        s = re.sub(r"(?<![\d.])(\d+)\.(\d+)(?![\d.]*\.\d)", r"\1,\2", s)

        def thousands(match: re.Match) -> str:
            value = int(match.group(1))
            before = match.string[max(0, match.start() - 20):match.start()]
            is_count = before.endswith(("von ", "Analysestichprobe ", "n=", "n = "))
            # Vierstellige Zahlen im Bereich von Jahreszahlen bleiben unveraendert,
            # ausser sie stehen erkennbar als Anzahl ("von 1992", "n=1992").
            if len(match.group(1)) == 4 and 1990 <= value <= 2035 and not is_count:
                return match.group(1)
            return f"{value:,}".replace(",", ".")

        s = re.sub(r"(?<![\d.,])(\d{4,})(?![\d.,]*\d)", thousands, s)
        s = re.sub(r"(?<=\d)%", " %", s)
        s = re.sub(r"(?<![\w\)])-(?=\d)", MINUS, s)
    return s.replace("|", "∣")


MONTHS_DE = ["Januar", "Februar", "März", "April", "Mai", "Juni", "Juli", "August", "September",
             "Oktober", "November", "Dezember"]


def month_de(key: str) -> str:
    """'2025-05' -> 'Mai 2025'."""
    year, month = key.split("-")
    return f"{MONTHS_DE[int(month) - 1]} {year}"


EFFECT_LABELS = {
    "Spearman rho": "Spearman ρ",
    "Spearman rho (vs. log_owners)": "Spearman ρ",
    "Cliff's Delta": "Cliff's δ",
    "Odds Ratio": "OR",
    "adjustierte Odds Ratio": "aOR",
    "adjustierte Odds Ratio (gemeinsames Modell)": "aOR (gemeinsames Modell)",
    "Odds Ratio Q4/Q1": "OR Q4/Q1",
}
EXTRA_LABELS = {"Cramers V": "Cramér's V", "Epsilon-Quadrat (ε²)": "ε² = H/(n − 1)"}


def etype(text: object) -> str:
    s = str(text)
    s = EFFECT_LABELS.get(s, s)
    return s.replace("Chi2", "χ²")


def decimals_for(effect_type: object) -> int:
    s = str(effect_type)
    if "rho" in s or "Delta" in s or "ρ" in s or "δ" in s:
        return 3
    if s.startswith("Chi2") or s.startswith("Kruskal") or s.startswith("z ") or s.startswith("Q (df"):
        return 1
    return 2


def md(df: pd.DataFrame) -> str:
    clean = df.copy()
    for col in clean.columns:
        clean[col] = clean[col].map(lambda v: "" if _isna(v) else str(v).replace("|", "∣"))
    return clean.to_markdown(index=False, disable_numparse=True)


def join_de(items: list[str]) -> str:
    """Aufzaehlung mit 'und' vor dem letzten Element."""
    items = list(items)
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " und " + items[-1]


def drop_empty(df: pd.DataFrame) -> pd.DataFrame:
    keep = [c for c in df.columns if not df[c].map(lambda v: _isna(v) or v in ("", "–")).all()]
    return df[keep]


# ---------------------------------------------------------------------------
# Daten laden
# ---------------------------------------------------------------------------

def read(name: str) -> pd.DataFrame:
    path = TAB / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} fehlt; zuerst 01_prepare_data.py, 02_eda.py und 03_tests.py ausführen.")
    return pd.read_csv(path, encoding="utf-8-sig")


def keyed(df: pd.DataFrame, col: str = "Kennung") -> pd.DataFrame:
    check(df[col].is_unique, f"Schlüsselspalte {col} ist nicht eindeutig")
    return df.set_index(col, drop=False)


def constants_from(script: str, names: list[str]) -> dict:
    """Literale Modulkonstanten eines Skripts lesen, ohne es auszufuehren."""
    tree = ast.parse((ROOT / "scripts" / script).read_text(encoding="utf-8"))
    found = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in names:
                found[name] = ast.literal_eval(node.value)
    missing = set(names) - set(found)
    check(not missing, f"Konstanten {sorted(missing)} fehlen in {script}")
    return found


def load() -> SimpleNamespace:
    D = SimpleNamespace()
    D.prov = keyed(read("data_provenance"), "Quelle")
    D.flow = keyed(read("sample_flow"))
    D.dq = keyed(read("data_quality_summary"))
    D.thr = keyed(read("success_thresholds"))
    D.counts = read("source_counts")
    D.cov = keyed(read("source_coverage"), "Quelle")
    D.mco = keyed(read("metacritic_overlap"), "Basis")
    D.overlap = read("source_overlap")
    D.ssc = keyed(read("steamspy_coverage_by_month"), "Release_Monat")
    D.excl = read("exclusion_samples")

    D.desc = read("descriptives_overall")
    D.desc_succ = read("descriptives_by_success")
    D.owners = read("owners_distribution")
    D.cohort = keyed(read("cohort_by_year"), "Jahr")
    D.agestd = read("age_standardized_rates")
    D.genres = keyed(read("genre_overview"), "Genre")
    D.genre_cohort = read("genre_cohort_overview")
    D.modes = read("mode_platform_overview")
    D.combo = keyed(read("platform_combo_overview"))
    D.prices = keyed(read("price_overview"))
    D.p40 = keyed(read("price_40plus_composition"))
    D.disc = keyed(read("discount_overview"))
    D.free = keyed(read("free_overview"))
    D.free_zero = keyed(read("free_share_owners_zero"))
    D.ratings = read("ratings_overview")
    D.mc_cov = keyed(read("metacritic_coverage"))
    D.desc_ext = keyed(read("descriptives_external"), "Variable")
    D.igdb_genres = read("igdb_genre_overview")
    D.igdb_modes = keyed(read("igdb_mode_overview"))
    D.igdb_ratings = read("igdb_rating_overview")
    D.igdb_values = keyed(read("igdb_rating_values"))
    D.ph_months = read("price_history_months")
    D.ph_overview = read("price_history_overview")
    D.mc_ext = read("external_metacritic_overview")
    D.corr = read("correlation_spearman")

    D.h1 = keyed(read("h1_tests"))
    D.h2 = keyed(read("h2_tests"))
    D.h3 = keyed(read("h3_tests"))
    D.h4 = keyed(read("h4_external_tests"))
    D.tests_all = keyed(read("tests_all"))
    D.T = D.tests_all
    D.joint = read("h1_joint_model")
    D.free_sens = keyed(read("h2_free_sensitivity"))
    D.ident = keyed(read("h3_identical_sample"))
    D.cohort_h3 = keyed(read("h3_cohort_stratified"))
    D.agree = keyed(read("igdb_steam_agreement"))
    D.release = read("release_date_sensitivity")
    D.rc = keyed(read("release_cohort_sensitivity"))
    D.rob = keyed(read("robustness_success_definitions"))
    D.qgroups = read("quartile_groups")
    D.sel = keyed(read("external_selection_tests"))
    for key in ["sel_has_igdb_gesamt", "sel_igdb_rating_gesamt", "sel_igdb_aggregated_rating_gesamt",
                "sel_has_mc_external_gesamt", "sel_has_price_history_gesamt"]:
        check(key in D.sel.index and D.sel.loc[key, "Schicht"] == "gesamt", f"{key} ist die Gesamtzeile")
    D.diag = keyed(read("review_volume_diagnostic"))
    D.dec = read("hypothesis_decisions")

    D.analytic = pd.read_parquet(
        DATA / "analytic_sample.parquet",
        columns=["AppID", "Name", "rating_sample", "success_top25", "n_sources", "release_year", "discount",
                 "list_price", "price_zero_with_discount", "owners_low", "has_price_history",
                 "ea_graduate_proxy", "release_shifted", "entry_proxy_ph", "has_igdb"],
    )
    D.igdb = pd.read_parquet(
        DATA / "igdb_sample.parquet",
        columns=["igdb_match_stage", "igdb_has_modes", "igdb_rating", "igdb_aggregated_rating"],
    )
    D.mc = pd.read_parquet(
        DATA / "mc_external_sample.parquet",
        columns=["mc_match_stage", "mc_metascore", "rating_sample", "positive_ratio", "Name", "metacritic",
                 "release_year", "mc_release_year"],
    )

    D.c03 = constants_from(
        "03_tests.py",
        ["YEAR_FE_MIN_LOGIT", "YEAR_FE_MIN_PARTIAL", "YEAR_CENTER", "YEAR_SCALE", "COHORTS",
         "REL_OR", "REL_RHO", "REL_DELTA", "REL_EPS2", "UNSTABLE_SHARE", "MIN_CELL", "N_BOOT"],
    )
    D.c02 = constants_from("02_eda.py", ["MIN_CELL_N", "COHORT_LABELS", "STD_FIRST_YEAR", "MIN_YEAR_N"])
    D.cc = constants_from("common.py", ["CLUSTER_PRICE", "CLUSTER_FROM_YEAR"])
    D.c01 = constants_from("01_prepare_data.py", ["APPID_PROXY_YEARS", "APPID_REF_MIN_N"])
    return D


def sc(D: SimpleNamespace, quelle: str, prefix: str, col: str = "N") -> object:
    part = D.counts[(D.counts["Quelle"] == quelle) & D.counts["Schritt"].str.startswith(prefix)]
    check(len(part) == 1, f"source_counts: Schritt '{prefix}' ({quelle}) {len(part)}-mal gefunden")
    return part.iloc[0][col]


def sc_hint_int(D: SimpleNamespace, quelle: str, prefix: str, pattern: str, group: int = 1) -> int:
    hint = str(sc(D, quelle, prefix, "Hinweis"))
    match = re.search(pattern, hint)
    check(match is not None, f"source_counts: Hinweis zu '{prefix}' enthält '{pattern}' nicht")
    return int(match.group(group))


def sc_share_an(D: SimpleNamespace, quelle: str, prefix: str) -> tuple[int, int]:
    """Zaehler (N_analytisch) und Basis ('Analysestichprobe von N' im Hinweis) einer Pruefzeile."""
    return int(sc(D, quelle, prefix, "N_analytisch")), sc_hint_int(D, quelle, prefix, r"Analysestichprobe von (\d+)")


def T(D: SimpleNamespace, key: str, col: str = "Effekt") -> object:
    check(key in D.T.index, f"Kennung {key} fehlt in tests_all.csv")
    return D.T.loc[key, col]


def tci(D: SimpleNamespace, key: str, decimals: int | None = None) -> str:
    r = D.T.loc[key]
    d = decimals if decimals is not None else decimals_for(r["Effekt_typ"])
    return ci(r["Effekt"], r["CI_unten"], r["CI_oben"], d)


def sig(D: SimpleNamespace, key: str) -> bool:
    return float(T(D, key, "p_adj")) < ALPHA


def covers(low: object, high: object, null: float) -> bool:
    return float(low) <= null <= float(high)


def ci_covers(D: SimpleNamespace, key: str, null: float) -> bool:
    return covers(T(D, key, "CI_unten"), T(D, key, "CI_oben"), null)


def rob(D: SimpleNamespace, key: str, col: str = "Effekt") -> object:
    check(key in D.rob.index, f"Kennung {key} fehlt in robustness_success_definitions.csv")
    return D.rob.loc[key, col]


# ---------------------------------------------------------------------------
# Tabellenbausteine
# ---------------------------------------------------------------------------

def test_table(D: SimpleNamespace, keys: list[str], show_model: bool = True, source: pd.DataFrame | None = None) -> str:
    table = D.T if source is None else source
    rows = []
    for key in keys:
        check(key in table.index, f"Kennung {key} fehlt")
        r = table.loc[key]
        d = decimals_for(r["Effekt_typ"])
        extra = ""
        if not _isna(r.get("Zusatz_Effekt")) and not _isna(r.get("Zusatz_Effekt_Typ")):
            extra = f"{EXTRA_LABELS.get(r['Zusatz_Effekt_Typ'], r['Zusatz_Effekt_Typ'])} = {de(r['Zusatz_Effekt'], 3)}"
        notes = []
        if is_true(r.get("instabil")):
            notes.append("instabil")
        if not _isna(r.get("Bereich_Gruppe")):
            notes.append(f"Q4: {de_text(r['Bereich_Gruppe'], english=True)}; Q1: {de_text(r['Bereich_Referenz'], english=True)}")
        if not _isna(r.get("Hinweis")):
            notes.append(de_text(r["Hinweis"], english=True))
        rows.append({
            "Merkmal": r["Merkmal"],
            "Modell": r["Modell"] if show_model else "",
            "Stichprobe": r["Stichprobe"],
            "N": n0(r["N"]),
            "Maß": etype(r["Effekt_typ"]),
            "Wert [95-%-KI]": ci(r["Effekt"], r.get("CI_unten"), r.get("CI_oben"), d),
            "Zusatzmaß": extra,
            "p (BH)": pval(r["p_adj"]),
            "Hinweis": "; ".join(notes),
        })
    return md(drop_empty(pd.DataFrame(rows)))


def decision_table(D: SimpleNamespace, hyp: str) -> str:
    part = D.dec[D.dec["Hypothese"] == hyp]
    rows = []
    for _, r in part.iterrows():
        d = decimals_for(r["Effekt_typ"])
        adj_key = r["Kennung_adjustiert"]
        adj = D.T.loc[adj_key] if adj_key in D.T.index else None
        quad_key = f"{adj_key}_quad"
        quad = D.T.loc[quad_key, "Effekt"] if quad_key in D.T.index else np.nan
        komp = r["Komponente"]
        if not is_true(r["Kernkomponente"]):
            komp = komp.replace("(Zusatz)", "").strip() + " (Zusatz, nicht entscheidungsrelevant)"
        rows.append({
            "Komponente": komp,
            "Teiltest": r["Teiltest"],
            "N": n0(r["N"]),
            "Maß": etype(r["Effekt_typ"]),
            "roh [95-%-KI]": ci(r["Effekt"], r["CI_unten"], r["CI_oben"], d),
            "q": pval(r["q"]),
            "adjustiert [95-%-KI]": ci(r["Effekt_adjustiert"],
                                        adj["CI_unten"] if adj is not None else np.nan,
                                        adj["CI_oben"] if adj is not None else np.nan, d),
            "q (adj.)": pval(r["q_adjustiert"]),
            "Jahr² (Sens.)": de(quad, d),
            "≥ 50k": de(r["Effekt_success_50k"], d),
            "≥ 100k": de(r["Effekt_success_100k"], d),
            "relevant": yn(r["relevant"]),
            "Richtung konsistent": yn(r["richtung_konsistent"]),
            "gestützt (Haupt / streng)": f"{yn(r['gestützt'])} / {yn(r['gestützt_streng'])}",
        })
    return md(drop_empty(pd.DataFrame(rows)))


def sensitivity_table(D: SimpleNamespace, hyp: str) -> str:
    part = D.dec[(D.dec["Hypothese"] == hyp) & (D.dec["Sensitivitaeten"].notna() | D.dec["Hinweis"].notna())]
    rows = []
    for _, r in part.iterrows():
        rows.append({
            "Teiltest": r["Teiltest"],
            "Sensitivitäten (Effekt, p unkorr.; Vergleich mit dem Haupteffekt gleicher Art)": de_text(r["Sensitivitaeten"], english=True),
            "alle bestätigt": yn(r["Sensitivitaeten_bestaetigt"]),
            "Hinweis": de_text(r["Hinweis"], english=True),
        })
    return md(drop_empty(pd.DataFrame(rows)))


def discount_cohort_pattern(D: SimpleNamespace, min_n: int = 100) -> SimpleNamespace:
    """Verlauf der Erfolgsquote ueber die Rabattstufen innerhalb der Kohorten.

    Die gepoolt altersstandardisierte Quote ist ueber die Stufen flach; innerhalb
    der Kohorten koennen die Verlaeufe aber gegenlaeufig sein. Kohorten, in denen
    eine Rabattgruppe weniger als ``min_n`` Spiele hat, werden nicht beschrieben.
    """
    ast = D.agestd[D.agestd["Merkmal"] == "Rabattgruppe"].set_index("Gruppe")
    groups = ["kein Rabatt", "1-25 %", "26-50 %", "51-75 %", "76-100 %"]
    check(all(g in ast.index for g in groups), "Rabattgruppen in age_standardized_rates.csv")
    cohorts = [c.removeprefix("Quote_") for c in ast.columns if c.startswith("Quote_")]
    items, trends, skipped = [], {}, []
    for c in cohorts:
        label = c.replace("bis_", "bis ").replace("_", "–")
        n = ast.loc[groups, f"N_{c}"].astype(float)
        q = ast.loc[groups, f"Quote_{c}"].astype(float).to_numpy()
        if (n < min_n).any():
            skipped.append((label, int(n.min())))
            continue
        steps = np.diff(q[1:])
        trend = "steigt" if (steps > 0).all() else "fällt" if (steps < 0).all() else "uneinheitlich"
        trends[label] = trend
        chain = " → ".join(pct(v) for v in q[1:])
        if trend == "steigt":
            items.append(f"in {label} steigt die Quote mit der Rabatthöhe monoton (ohne Rabatt {pct(q[0])}; 1–25 % bis 76–100 %: {chain})")
        elif trend == "fällt":
            items.append(f"in {label} fällt die Quote ab der Stufe 1–25 % mit der Rabatthöhe (ohne Rabatt {pct(q[0])}; {chain})")
        else:
            items.append(f"in {label} verläuft die Quote ohne klares Muster (ohne Rabatt {pct(q[0])}; {chain})")
    rising = [k for k, v in trends.items() if v == "steigt"]
    falling = [k for k, v in trends.items() if v == "fällt"]
    opposite = bool(rising and falling)
    skip_text = (" Die Kohorten " + join_de([lab for lab, _ in skipped]) + " haben zu wenige Spiele mit niedrigem "
                 f"Rabatt (kleinste Gruppe n = {n0(min(m for _, m in skipped))}, Schwelle {min_n})." if skipped else "")
    long = ("Das gepoolte Plateau verdeckt " + ("gegenläufige" if opposite else "unterschiedliche")
            + " Verläufe innerhalb der Kohorten (`age_standardized_rates.csv`, `fig_06` rechts): "
            + "; ".join(items) + "." + skip_text)
    short = (f"innerhalb der Kohorten verlaufen die Quoten gegenläufig (in {join_de(rising)} mit der Rabatthöhe "
             f"steigend, in {join_de(falling)} fallend)" if opposite
             else "innerhalb der Kohorten verlaufen die Quoten unterschiedlich")
    return SimpleNamespace(long=long, short=short, opposite=opposite, rising=rising, falling=falling)


def fig(name: str, caption: str) -> str:
    path = FIG / f"{name}.png"
    check(path.exists(), f"Abbildung {path} fehlt")
    return f"![{caption}](../figures/{name}.png)\n\n*Abbildung `{name}`: {caption}*"


def rate_cell(rate: object, low: object, high: object) -> str:
    return f"{pct(rate)} {kip(low, high)}" if not _isna(rate) else "–"


# ---------------------------------------------------------------------------
# Abschnitte
# ---------------------------------------------------------------------------

def context(D: SimpleNamespace) -> SimpleNamespace:
    """Zentrale Kennzahlen, die in mehreren Abschnitten vorkommen."""
    C = SimpleNamespace()
    C.cutoff = STEAMSPY_CUTOFF.strftime("%d.%m.%Y")
    C.n = int(D.flow.loc["analytic_sample", "N"])
    check(C.n == len(D.analytic), "sample_flow und analytic_sample.parquet haben verschiedene N")
    C.n_raw = int(D.flow.loc["raw", "N"])
    for key in ["rating_sample", "rating_sample_50", "meta_sample", "peak_sample", "igdb_sample",
                "mc_external_sample", "price_history_sample", "multi_source_sample", "sample_with_owners_zero"]:
        setattr(C, f"n_{key}", int(D.flow.loc[key, "N"]))
    C.n_two = int(D.dq.loc["an_two_sources", "Wert"])
    fr = D.prov.loc["fronkon"]
    C.fr = fr
    stand, version_text = str(fr["Datenstand"]).split("; ", 1)
    C.store_stand, C.steamspy_stand = [s.strip() for s in stand.split(", ", 1)]
    C.version_text = version_text
    check(version_text.startswith(f"Version {int(fr['Version'])} "), "Versionstext passt zur Provenienz")
    C.thr = {k: D.thr.loc[k] for k in D.thr.index}
    C.succ = float(D.thr.loc["success_top25", "Anteil"])
    C.thr_label = {
        "success_top25": f"≥ {n0(D.thr.loc['success_top25', 'Schwelle'])}",
        "success_50k": f"≥ {n0(D.thr.loc['success_50k', 'Schwelle'])}",
        "success_100k": f"≥ {n0(D.thr.loc['success_100k', 'Schwelle'])}",
    }
    comp = D.dec.drop_duplicates(["Hypothese", "Komponente"])
    C.comp = comp
    hyp = D.dec.drop_duplicates("Hypothese").set_index("Hypothese")
    C.decision = hyp["Entscheidung_Hypothese"].to_dict()
    C.decision_strict = hyp["Entscheidung_streng"].to_dict()
    C.formulation = hyp["Hypothese_Formulierung"].to_dict()
    C.rule = de_text(hyp["Regel"].iloc[0], english=True).replace("adjustierte Effekt", "adjustierte Zusammenhang")
    check(D.dec["Regel"].nunique() == 1, "Entscheidungsregel ist nicht für alle Hypothesen gleich")
    C.cohort_labels_03 = [lab for _, _, lab in D.c03["COHORTS"]]
    C.kappa = D.agree[~D.agree["Merkmal"].str.contains("näherungsweise")]["Kappa"]
    C.kappa_action = D.agree[D.agree["Merkmal"].str.contains("näherungsweise")]["Kappa"]
    check(len(C.kappa_action) == 1, "genau eine näherungsweise Zuordnung (Action) erwartet")
    # Kostenlos in der juengsten Kohorte (Kennung statt Stichprobentext).
    C.free_last = D.free_sens.loc[f"free_an_is_free_c{len(C.cohort_labels_03)}"]
    check(C.free_last["Kohorte"] == C.cohort_labels_03[-1], "free_an_is_free_c<n> ist die jüngste Kohorte")
    # Woertliche Regel (nur gleiches Vorzeichen im adjustierten Modell): welche Urteile aendern sich?
    C.decision_literal = hyp["Entscheidung_wörtlich"].to_dict()
    C.decision_literal_strict = hyp["Entscheidung_wörtlich_streng"].to_dict()
    C.literal_changes = [h for h in sorted(C.decision) if C.decision_literal[h] != C.decision[h]]
    C.literal_changes_strict = [h for h in sorted(C.decision)
                                if C.decision_literal_strict[h] != C.decision_strict[h]]
    # Preishistorie: Hauptspezifikation und volles Fenster.
    ph = SimpleNamespace()
    ph.rows = D.dec[D.dec["Komponente"].str.startswith("Preishistorie")]
    ph.feats = ["price_changes_per_month", "share_months_on_sale", "days_on_sale_per_month",
                "mean_monthly_max_discount_pct"]
    ph.label = {"price_changes_per_month": "Preisänderungen je Monat",
                "share_months_on_sale": "Rabatthäufigkeit (Anteil der Monate mit Rabatt)",
                "days_on_sale_per_month": "Rabatttage je Monat",
                "mean_monthly_max_discount_pct": "mittlerer monatlicher Maximalrabatt"}
    ph.label_short = {**ph.label, "share_months_on_sale": "Rabatthäufigkeit"}
    ph.label_art = {"price_changes_per_month": "die Zahl der Preisänderungen je Monat",
                    "share_months_on_sale": "die Rabatthäufigkeit", "days_on_sale_per_month": "die Zahl der Rabatttage je Monat",
                    "mean_monthly_max_discount_pct": "der mittlere monatliche Maximalrabatt",
                    "max_discount_pct": "der maximale Rabatt"}
    ph.full_sig = [f for f in ph.feats if sig(D, f"h4_ph_{f}_full_rho_partial")]
    check(set(ph.full_sig) >= {"share_months_on_sale", "days_on_sale_per_month"},
          "im vollen Fenster kleine positive Partialkorrelationen")
    check(all(T(D, f"h4_ph_{f}_full_rho_partial") > 0 for f in ph.full_sig), "volles Fenster: positiv")
    ph.full_min = min(T(D, f"h4_ph_{f}_full_rho_partial") for f in ph.full_sig)
    ph.full_max = max(T(D, f"h4_ph_{f}_full_rho_partial") for f in ph.full_sig)
    ph.full_rel = all(T(D, f"h4_ph_{f}_full_rho_partial") >= D.c03["REL_RHO"] for f in ph.full_sig)
    ph.full_n = int(T(D, f"h4_ph_{ph.full_sig[0]}_full_rho_partial", "N"))
    ph.full_text = join_de([f"{ph.label[f]} {tci(D, f'h4_ph_{f}_full_rho_partial', 3)}" for f in ph.full_sig])
    ph.full_text_short = join_de([f"{ph.label_short[f]} {de(T(D, f'h4_ph_{f}_full_rho_partial'), 3)}"
                                  for f in ph.full_sig])
    check((ph.rows["q_adjustiert"] >= ALPHA).all(), "kein Preishistorien-Merkmal adjustiert signifikant")
    ph.clean_sig = [f for f in ph.feats + ["max_discount_pct"] if sig(D, f"h4_ph_{f}_clean_rho_partial")]
    C.ph = ph
    # Markteintritts-Proxy (Early Access): groesste relative aOR-Aenderung und
    # Preishistorien-Merkmale, die mit market_entry_year signifikant und relevant werden.
    rl = D.release
    main_eff = rl[rl["Variante"].str.startswith("Hauptmodell")].set_index("Merkmal")["Effekt"]
    ea_rows = rl[rl["Variante"].str.contains("EA-Absolventen|market_entry_year")]
    C.ea_max_change = float((ea_rows["Effekt"] / ea_rows["Merkmal"].map(main_eff) - 1).abs().max())
    C.entry_ph_sig = [f for f in ph.feats + ["max_discount_pct"]
                      if D.rc.loc[f"rc_entry_ph_{f}", "p"] < ALPHA
                      and abs(D.rc.loc[f"rc_entry_ph_{f}", "Effekt"]) >= D.c03["REL_RHO"]]
    # Sensitivitaeten, in denen ein Preishistorien-Merkmal nach Alterskontrolle signifikant ist
    # (zusaetzlich zum vollen Fenster); fuer die Kurzfassung.
    sens = {}
    for f in ph.clean_sig:
        sens.setdefault(f, []).append("ohne Apps mit Tiefstpreis über dem Basispreis ρ "
                                      + de(T(D, f"h4_ph_{f}_clean_rho_partial"), 3))
    for f in C.entry_ph_sig:
        sens.setdefault(f, []).append("mit geschätztem Markteintritt als Jahreskontrolle ρ "
                                      + de(D.rc.loc[f"rc_entry_ph_{f}", "Effekt"], 3))
    ph.sens_text = (f"; in Sensitivitäten {'ist' if len(sens) == 1 else 'sind'} außerdem "
                    + join_de([f"{ph.label_art[f]} ({', '.join(v)})" for f, v in sens.items()])
                    + " signifikant (Abschnitte 3.4 und 3.7)") if sens else ""
    # Positive Ratio: gepoolt vs. innerhalb der Kohorten (Konfundierung ueber die Jahrgaenge).
    steam = D.corr[D.corr["Matrix"] == "Steam"]

    def srho(a: str, b: str) -> float:
        part = steam[((steam["Variable_1"] == a) & (steam["Variable_2"] == b))
                     | ((steam["Variable_1"] == b) & (steam["Variable_2"] == a))]
        check(len(part) == 1, f"Steam-Korrelation {a}/{b}")
        return float(part.iloc[0]["rho"])

    # IGDB-Kritiker- vs. -Nutzerbewertung: Staerke nur in identischer Stichprobe vergleichbar.
    C.igdb_ident = D.ident.loc["ident_igdb_aggregated_rating_vs_igdb_rating"]
    C.igdb_ident_nonint = D.ident.loc["ident_igdb_aggregated_rating_vs_igdb_rating_nonint"]
    for r in (C.igdb_ident, C.igdb_ident_nonint, D.ident.loc["ident_igdb_aggregated_rating_vs_igdb_rating_partial"]):
        check(covers(r["CI_unten"], r["CI_oben"], 0.0) and r["rho_A"] > 0 and r["rho_B"] > 0,
              "IGDB-Kritiker- und -Nutzerbewertung in identischer Stichprobe positiv und etwa gleich stark")
    C.igdb_ident_text = (f"ρ {de(C.igdb_ident['rho_A'], 3)} gegen {de(C.igdb_ident['rho_B'], 3)}, Δρ "
                         f"{ci(C.igdb_ident['Differenz'], C.igdb_ident['CI_unten'], C.igdb_ident['CI_oben'], 3)}")
    C.rho_year_pr = srho("release_year", "positive_ratio")
    C.rho_year_owners = srho("release_year", "log_owners")
    C.pr_cohort_mean = float(D.rc.loc["rc_main_pr_heterogeneity", "Mittel_Kohorten_rho"])
    check(C.rho_year_pr > 0 and C.rho_year_owners < 0, "neuere Jahrgänge besser bewertet, weniger Besitzer")
    check(C.pr_cohort_mean > 0 and T(D, "h3_pr_r10_rho_partial") > 0 and ci_covers(D, "h3_pr_r10_rho", 0.0),
          "Positive Ratio gepoolt null, innerhalb der Kohorten im Mittel positiv")
    check(abs(C.pr_cohort_mean - T(D, "h3_pr_r10_rho_partial")) < 0.03,
          "gewichtetes Kohortenmittel nahe am partiellen ρ")
    return C


def header(D, C) -> str:
    fr = C.fr
    return f"""# Umsetzung der Analyse: Erfolgsfaktoren von Videospielen auf Steam

*Ergebnisbericht zur empirischen Umsetzung des Forschungsvorhabens.*
*Datengrundlage: {fr['Titel']} (Kaggle `{fr['Kaggle_Handle']}`, Version {n0(fr['Version'])}, SHA-256 `{fr['SHA256'][:12]}…`) sowie IGDB-, Metacritic- und Preishistorien-Daten (Provenienz in Abschnitt 1.1).*
*Erhebungsstände: {C.store_stand}, {C.steamspy_stand} (Stichtag der Analysestichprobe: Veröffentlichung vor dem {C.cutoff}). Auswertungsstand: {REPORT_DATE}.*
"""


def section0(D, C) -> str:
    h = D.T
    # Kernzahlen und Pruefungen fuer die Kurzfassung
    keys_pos = ["h1_genre_massively_multiplayer_aor", "h1_genre_rpg_aor", "h1_mode_coop_aor",
                "h1_mode_multi_player_aor", "h1_mode_mmo_aor", "h1_platform_mac_aor"]
    for k in keys_pos:
        check(T(D, k) > 1 and sig(D, k), f"{k} positiv und signifikant")
    keys_neg = ["h1_genre_casual_aor", "h1_genre_sports_aor", "h1_genre_indie_aor", "h1_genre_racing_aor"]
    for k in keys_neg:
        check(T(D, k) < 1 and sig(D, k), f"{k} negativ und signifikant")
    check(sig(D, "h1_genre_simulation_or") and not sig(D, "h1_genre_simulation_aor"),
          "Simulation roh signifikant, adjustiert nicht")
    check(ci_covers(D, "h1_combo_wl_aor", 1.0) and ci_covers(D, "h1_platform_linux_aor_joint", 1.0),
          "Linux ohne macOS nicht gesichert")
    jm = D.joint[D.joint["Erfolgsdefinition"] == "success_top25"].set_index("Variable")
    check(covers(jm.loc["genre_massively_multiplayer", "CI_unten"], jm.loc["genre_massively_multiplayer", "CI_oben"], 1.0),
          "MMO-Genre im gemeinsamen Modell nicht gesichert")

    fs = D.free_sens
    c_last = C.cohort_labels_03[-1]
    free_last = C.free_last
    free_zero = fs.loc["free_zero_is_free_all"]
    check(covers(free_last["aOR_CI_unten"], free_last["aOR_CI_oben"], 1.0), "kostenlos in der jüngsten Kohorte n. s.")
    check(covers(free_zero["aOR_CI_unten"], free_zero["aOR_CI_oben"], 1.0), "kostenlos mit '0 - 0' n. s.")
    check(ci_covers(D, "h2_free_without_tag_aor", 1.0), "kostenlos ohne F2P-Tag n. s.")
    check(ci_covers(D, "h2_price_rho_nodiscount_partial", 0.0) and T(D, "h2_price_rho_discount_partial") >= D.c03["REL_RHO"],
          "Listenpreis-Zusammenhang nur bei rabattierten Spielen")
    ph = C.ph
    check(C.literal_changes == ["H4"], "wörtliche Regel ändert nur das H4-Urteil")
    di = D.disc
    std_disc = di.loc[["d1_25", "d26_50", "d51_75", "d76_100"], "Erfolgsquote_standardisiert"].to_numpy()
    check(not (np.diff(std_disc) > 0).all() and not (np.diff(std_disc) < 0).all(),
          "standardisierte Rabattquoten ohne monotonen Gradienten")
    dcp = discount_cohort_pattern(D)
    q_all, q_nonint, q_int = (T(D, "h4_igdbuser_all_q"), T(D, "h4_igdbuser_nonint_q"), T(D, "h4_igdbuser_int_q"))
    check(max(q_nonint, q_int) < q_all / 1.5 and min(q_nonint, q_int) > 1, "Quartils-OR innerhalb der Wertetypen kleiner, aber > 1")
    # Anteil der Gesamt-OR, der innerhalb der Wertetypen erhalten bleibt (Log-Skala).
    q_keep = {"nonint": math.log(q_nonint) / math.log(q_all), "int": math.log(q_int) / math.log(q_all)}
    check(0.2 < min(q_keep.values()) and max(q_keep.values()) < 0.7,
          "Zusammensetzung erklärt einen großen Teil, aber nicht durchweg den überwiegenden Teil der Quartils-OR")
    pr_raw, pr_part = T(D, "h3_pr_r10_rho"), T(D, "h3_pr_r10_rho_partial")
    check(abs(pr_raw) < D.c03["REL_RHO"] and pr_part > pr_raw and abs(pr_part) < D.c03["REL_RHO"],
          "Positive Ratio: Jahr-FE verschiebt den Wert, beide unter der Relevanzschwelle")
    ident = D.ident.loc["ident_mc_metascore_vs_positive_ratio"]
    check(ident["CI_unten"] > 0 and ident["rho_B"] > 0, "Metascore in identischer Stichprobe stärker, beide positiv")
    pr_q = D.ratings[D.ratings["Variable"] == "positive_ratio"].set_index("Quartil")["Erfolgsquote"].to_numpy()
    check(min(pr_q[1], pr_q[2]) > max(pr_q[0], pr_q[3]), "Positive Ratio umgekehrt U-förmig")
    coh = D.cohort_h3
    c_first, c_end = coh.loc["cohort_0_c0"], coh.loc["cohort_0_c3"]
    check(c_first["Effekt"] > 0 and c_first["p_adj"] < ALPHA and c_end["Effekt"] < 0 and c_end["p_adj"] < ALPHA,
          "Positive Ratio: älteste Kohorte positiv, jüngste negativ")
    pr_q_rob = [rob(D, f"rob_pr_r10_{o}_q") for o in ["success_top25", "success_50k", "success_100k"]]
    check(pr_q_rob[0] < 1 < pr_q_rob[2], "Q4/Q1 der Positive Ratio wechselt mit der Schwelle die Seite")

    cy = D.cohort[D.cohort["N"] >= D.c02["MIN_YEAR_N"]]
    peak_year = int(cy["Erfolgsquote"].idxmax())
    last_year = int(D.cohort.index.max())
    check(last_year == STEAMSPY_CUTOFF.year, "letzter Jahrgang = Stichtagsjahr")

    dec_rows = []
    for hyp in sorted(C.decision):
        dec_rows.append({
            "Hypothese": hyp,
            "Formulierung (ausführlich in Abschnitt 3.0)": C.formulation[hyp],
            "Entscheidung (Hauptregel)": C.decision[hyp],
            "streng": C.decision_strict[hyp],
        })
    dec_tbl = md(pd.DataFrame(dec_rows))

    return f"""## 0. Kurzfassung

Die Analyse verbindet **vier Datenquellen**: den FronkonGames-Steam-Datensatz (Basis), IGDB-Spielemetadaten, einen externen Metacritic-Datensatz und eine Steam-Preishistorie. Die **Analysestichprobe** umfasst **{n0(C.n)} Spiele**, die vor dem SteamSpy-Stichtag ({C.cutoff}) erschienen sind. Besitzer, Reviews und Peak CCU stammen aus dem SteamSpy-Stand ({C.steamspy_stand.replace('SteamSpy-Felder ', '')}), Preise und Rabatte aus dem Store-Stand ({C.store_stand.replace('Store-Felder ', '')}). Mit IGDB sind {n0(C.n_igdb_sample)} Spiele verknüpft ({pct(C.n_igdb_sample / C.n)}), mit einem externen Metascore {n0(C.n_mc_external_sample)} ({pct(C.n_mc_external_sample / C.n)}) und mit der Preishistorie {n0(C.n_price_history_sample)} ({pct(C.n_price_history_sample / C.n)}). Für {n0(C.n_two)} Spiele liegen mindestens zwei Quellen vor, für {n0(C.n_multi_source_sample)} alle vier.

Als **Erfolg** gilt primär eine untere Besitzergrenze von {C.thr_label['success_top25']} ({pct(C.succ)} der Analysestichprobe; {de_text(D.thr.loc['success_top25', 'Hinweis'])}). Zur Robustheit dienen {C.thr_label['success_50k']} ({pct(D.thr.loc['success_50k', 'Anteil'])}) und {C.thr_label['success_100k']} Besitzer ({pct(D.thr.loc['success_100k', 'Anteil'])}). Es werden **keine Prognose- oder Klassifikationsmodelle** geschätzt. Berichtet werden Zusammenhänge im Querschnitt mit Effektmaßen, Konfidenzintervallen und BH-korrigierten Tests; kausale Wirkungen lassen sich daraus nicht ableiten.

**Hypothesenentscheidungen** nach festen Regeln (Abschnitt 3.0), die allerdings erst während der Auswertung und in Kenntnis erster Ergebnisse festgelegt wurden (nicht präregistriert). Die strenge Variante verlangt zusätzlich, dass der altersadjustierte Zusammenhang praktisch relevant ist:

{dec_tbl}

**Zentrale Befunde:**

- **Spieleigenschaften (F1):** Auch nach Kontrolle des Veröffentlichungsjahres (Jahr-Fixed-Effects) sind unter erfolgreichen Spielen überrepräsentiert: das Genre Massively Multiplayer (aOR {tci(D, 'h1_genre_massively_multiplayer_aor')}), RPG ({tci(D, 'h1_genre_rpg_aor')}), Koop ({tci(D, 'h1_mode_coop_aor')}), Multiplayer ({tci(D, 'h1_mode_multi_player_aor')}) und der Steam-Modus MMO ({tci(D, 'h1_mode_mmo_aor')}). Unterrepräsentiert sind Casual ({tci(D, 'h1_genre_casual_aor')}), Sports, Indie und Racing. Simulation ist nicht robust (roh OR {tci(D, 'h1_genre_simulation_or')}, adjustiert {tci(D, 'h1_genre_simulation_aor')}, n. s.). Im gemeinsamen Modell geht der Genre-Zusammenhang von Massively Multiplayer im MMO-Modus auf (aOR {ci(jm.loc['genre_massively_multiplayer', 'Effekt'], jm.loc['genre_massively_multiplayer', 'CI_unten'], jm.loc['genre_massively_multiplayer', 'CI_oben'])}, n. s.). macOS-Unterstützung ist positiv assoziiert (aOR {tci(D, 'h1_platform_mac_aor')}), Linux ohne macOS dagegen nicht (Windows + Linux gegen nur Windows aOR {tci(D, 'h1_combo_wl_aor')}; Linux gegenseitig mit macOS adjustiert {tci(D, 'h1_platform_linux_aor_joint')}). IGDB-Modi und -Genres zeigen dieselben Richtungen, soweit das Steam-Merkmal eine gesicherte Richtung hat. Sie stimmen aber weitgehend mit den Steam-Kategorien überein (Cohens κ {de(C.kappa.min(), 2)}–{de(C.kappa.max(), 2)}) und sind deshalb eine Konsistenzprüfung, keine unabhängige Bestätigung.
- **Preis (F2):** Kostenlose Spiele (Listenpreis 0) sind häufiger erfolgreich (OR {tci(D, 'h2_free_or')}, aOR {tci(D, 'h2_free_aor')}). Der Zusammenhang hängt aber an der Kohorte: In {c_last} liegt die aOR bei {ci(free_last['aOR_JahrFE'], free_last['aOR_CI_unten'], free_last['aOR_CI_oben'])} (n. s.). Zählt man die Einträge mit Besitzerangabe „0 - 0“ als Nicht-Erfolg mit, verschwindet er ({ci(free_zero['aOR_JahrFE'], free_zero['aOR_CI_unten'], free_zero['aOR_CI_oben'])}), ebenso bei kostenlosen Spielen ohne Steam-Tag „Free To Play“ ({tci(D, 'h2_free_without_tag_aor')}). Das Preisniveau kostenpflichtiger Spiele (rekonstruierter Listenpreis) korreliert schwach positiv mit den Besitzern (ρ {tci(D, 'h2_price_rho')}, partiell {tci(D, 'h2_price_rho_partial')}). Dieser Zusammenhang entsteht fast nur unter aktuell rabattierten Spielen (ohne Rabatt partiell {tci(D, 'h2_price_rho_nodiscount_partial')}, mit Rabatt {tci(D, 'h2_price_rho_discount_partial')}). Ein aktiver Rabatt geht mit einer höheren Erfolgsquote einher (OR {tci(D, 'h2_discount_or')}, aOR {tci(D, 'h2_discount_aor')}); nach Altersstandardisierung liegen die Rabattstufen eng beieinander ({pct(std_disc.min())} bis {pct(std_disc.max())}), gepoolt zeigt sich kein monotoner Gradient über die Rabatthöhe, {dcp.short}. In der Preishistorie ist nach Kontrolle von Jahr und Beobachtungsdauer in der Hauptspezifikation kein normiertes Merkmal signifikant; im vollen Beobachtungsfenster zeigen sich kleine positive Partialkorrelationen (ρ {de(ph.full_min, 2)} bis {de(ph.full_max, 2)}){ph.sens_text}.
- **Bewertungen (F3):** Der externe Metascore hängt moderat positiv mit den Besitzern zusammen (ρ {tci(D, 'h3_mc_rho')}, partiell {tci(D, 'h3_mc_rho_partial')}). In derselben Teilstichprobe ist der Zusammenhang stärker als bei der Steam-Positive-Ratio (ρ {de(ident['rho_A'], 2)} gegen {de(ident['rho_B'], 2)}, Δρ {ci(ident['Differenz'], ident['CI_unten'], ident['CI_oben'])}), beide sind aber positiv. In der breiten Bewertungsstichprobe ist die Positive Ratio gepoolt praktisch unkorreliert (ρ {tci(D, 'h3_pr_r10_rho', 3)}). Die Konfundierung über die Jahrgänge (neuere Jahrgänge werden besser bewertet und haben weniger Besitzer) drückt den Wert von {de(T(D, 'h3_pr_r10_rho_partial'), 3)} (partiell mit Jahr-FE) auf {de(T(D, 'h3_pr_r10_rho'), 3)}; auch innerhalb der Jahrgänge bleibt er im Mittel unter der Relevanzschwelle von {de(D.c03['REL_RHO'], 2)}. Bestimmend ist die Heterogenität über die Kohorten: bei älteren Titeln deutlich positiv ({de(c_first['Effekt'], 2)}, {c_first['Kohorte']}), bei Neuerscheinungen negativ ({de(c_end['Effekt'], 2)}, {c_end['Kohorte']}). Der Verlauf über die Quartile ist umgekehrt U-förmig. Das Q4/Q1-Verhältnis wechselt je nach Erfolgsschwelle die Seite ({' / '.join(de(v, 2) for v in pr_q_rob)}). Der Wilson-Score enthält das Review-Volumen und dient nur als Diagnose.
- **Externe Quellen (H4):** IGDB-Bewertungen hängen positiv mit den Besitzern zusammen: nicht ganzzahlige Nutzerbewertungen mit ρ {tci(D, 'h4_igdbuser_nonint_rho')}, Kritikerbewertungen mit ρ {tci(D, 'h4_igdbcrit_rho')} (verschiedene Teilstichproben). In identischer Stichprobe (n = {n0(C.igdb_ident['N'])}) sind beide etwa gleich stark ({C.igdb_ident_text}). Beide Teilstichproben sind stark selektiert. Die Quartils-OR der IGDB-Nutzerbewertung über alle Werte ({de(q_all, 2)}) ist zu einem großen Teil durch die Zusammensetzung nach Stimmenvolumen aufgebläht; innerhalb der Wertetypen liegt sie bei {de(q_nonint, 2)} (nicht ganzzahlig) bzw. {de(q_int, 2)} (ganzzahlig), auf der Log-Skala also bei {pct(q_keep['nonint'], 0)} bzw. {pct(q_keep['int'], 0)} des Gesamtwerts (bei unterschiedlichen Quartilsgrenzen nur eingeschränkt vergleichbar). In der Preishistorie (n = {n0(C.n_price_history_sample)}, Erfolgsquote {pct(D.sel.loc['sel_has_price_history_gesamt', 'Erfolgsquote_verfuegbar'])}) korrelieren die normierten Rabattmerkmale roh positiv mit den Besitzern. Nach Kontrolle von Jahr und Fensterlänge ist in der Hauptspezifikation kein Merkmal signifikant (partiell {de(ph.rows['Effekt_adjustiert'].min(), 2)} bis {de(ph.rows['Effekt_adjustiert'].max(), 2)}). Im vollen Beobachtungsfenster (n = {n0(ph.full_n)}) bleiben dagegen kleine positive Partialkorrelationen{' oberhalb der Relevanzschwelle' if ph.full_rel else ''}: {ph.full_text_short}. Ein Rabattzusammenhang ist dort also schwach und hängt von der Spezifikation ab. Nach der festgelegten Regel ist die Preishistorie nicht gestützt; bei wörtlicher Anwendung der Richtungsregel wäre H4 „{C.decision_literal['H4']}“ (Abschnitt 3.0).
- **Zentrale Einschränkung:** Die Besitzerzahl ist kumuliert und hängt stark vom Alter ab. Die Erfolgsquote fällt von {pct(D.cohort.loc[peak_year, 'Erfolgsquote'])} (Jahrgang {peak_year}) auf {pct(D.cohort.loc[last_year, 'Erfolgsquote'])} (Jahrgang {last_year}, nur Januar bis Mai). Zugleich steigt der Anteil fehlender Besitzerangaben („0 - 0“) in den jüngsten Jahrgängen stark an. Alle Hypothesentests kontrollieren deshalb das Veröffentlichungsjahr. Das Store-Datum ist bei Early-Access-Absolventen aber das Datum der Vollversion; ein grober Markteintritts-Proxy ändert die adjustierten ORs um höchstens {de(C.ea_max_change * 100, 1)} %{'; in der Preishistorie ' + ('wird' if len(C.entry_ph_sig) == 1 else 'werden') + ' mit ihm aber ' + join_de([C.ph.label_art[f] for f in C.entry_ph_sig]) + ' signifikant' if C.entry_ph_sig else ''} (Abschnitt 3.7). Restkonfundierung durch nicht erkannte Early-Access-Phasen bleibt möglich.
- **Scheinbare Widersprüche zwischen den Quellen:** IGDB-Singleplayer zeigt dieselbe Richtung wie Steam, weil fehlende IGDB-Modusangaben als unbekannt und nicht als „nein“ behandelt werden (Abschnitt 3.4). Dass die Steam-Positive-Ratio in der Breite nicht, IGDB- und Metacritic-Nutzerbewertungen dagegen positiv korrelieren, liegt an der Stichprobe: In der identischen Metacritic-Teilstichprobe korreliert auch die Positive Ratio positiv. Die schwachen, nach Alterskontrolle nur im vollen Fenster{' oder in einzelnen Sensitivitäten' if ph.sens_text else ''} nachweisbaren Preishistorien-Zusammenhänge beruhen auf einer kleinen, fast vollständig erfolgreichen Teilstichprobe mit anderem Konstrukt (Rabattverlauf statt Rabatt zum Snapshot) und lassen sich nicht auf die Vollstichprobe übertragen.
"""


def section1(D, C) -> str:
    out = []
    # ---- 1.1 Quellen und Provenienz
    prov = D.prov
    check(prov["Zeilen_geprueft"].map(is_true).all(), "alle Rohdateien zeilengeprüft")
    check((prov["Zeilen_erwartet"] == prov["Zeilen_eingelesen"]).all(), "Zeilenzahlen erwartet = eingelesen")
    prov_tbl = pd.DataFrame({
        "Quelle": prov["Titel"],
        "Kaggle-Handle": prov["Kaggle_Handle"].map(lambda s: f"`{s}`"),
        "Version": prov["Version"].map(n0),
        "Datei": prov["Datei"].map(lambda s: f"`{s}`"),
        "Bytes": prov["Bytes"].map(n0),
        "Datensätze (erwartet = eingelesen)": prov["Zeilen_eingelesen"].map(n0),
        "geprüft": prov["Zeilen_geprueft"].map(yn),
        "Datenstand": prov["Datenstand"],
    })
    hashes = "\n".join(f"- `{r['Datei']}` (Version {n0(r['Version'])}): `{r['SHA256']}`" for _, r in prov.iterrows())
    path_types = "; ".join(sorted(prov["Pfadtyp"].unique()))
    version_notes = "; ".join(sorted(prov["Version_Herkunft"].unique()))

    before = (STEAMSPY_CUTOFF - pd.DateOffset(months=1)).strftime("%Y-%m")
    after = STEAMSPY_CUTOFF.strftime("%Y-%m")
    s_before, s_after = D.ssc.loc[before], D.ssc.loc[after]
    check(s_before["Anteil_Reviews_gt0"] > 5 * s_after["Anteil_Reviews_gt0"], "Bruch der Review-Abdeckung am Stichtag")
    check(abs(s_before["Anteil_Recommendations_gt0"] - s_after["Anteil_Recommendations_gt0"]) < 0.1,
          "Store-Recommendations ohne Bruch am Stichtag")

    ig = {
        "raw": sc(D, "IGDB", "Rohzeilen"),
        "crit0": sc(D, "IGDB", "Kritikerbewertung = 0"),
        "user0": sc(D, "IGDB", "Nutzerbewertung = 0"),
        "noname": sc(D, "IGDB", "ohne verwertbaren Namen"),
        "nonpc": sc(D, "IGDB", "Plattformangabe ohne PC-nahe"),
        "dups": sc(D, "IGDB", "Name/Jahr-Dubletten"),
        "entries": sc(D, "IGDB", "Einträge für das Matching"),
        "undated": sc(D, "IGDB", "davon ohne Veröffentlichungsdatum"),
        "multi": sc(D, "IGDB", "Stufe 1: Quelleinträge mit mehreren AppIDs"),
        "s1": sc(D, "IGDB", "gematcht in Stufe 1", "N_analytisch"),
        "s2": sc(D, "IGDB", "gematcht in Stufe 2", "N_analytisch"),
        "s3": sc(D, "IGDB", "gematcht in Stufe 3", "N_analytisch"),
        "all_matched": sc(D, "IGDB", "gematcht gesamt"),
    }
    multi_firm = sc_hint_int(D, "IGDB", "Stufe 1: Quelleinträge mit mehreren AppIDs", r"Entwickler/Publisher: (\d+)")
    multi_an = sc_hint_int(D, "IGDB", "Stufe 1: Quelleinträge mit mehreren AppIDs", r"Analysestichprobe: (\d+)")
    multi_none = sc_hint_int(D, "IGDB", "Stufe 1: Quelleinträge mit mehreren AppIDs", r"nicht gematcht: (\d+)")
    recall_n = sc(D, "IGDB", "Recall-Referenz", "N_analytisch")
    recall_base = int(D.flow.loc["meta_sample", "N"])
    nl = {}
    for key, prefix in [("raw", "Steam-Namen mit nicht-lateinischen Buchstaben (Rohname)"),
                        ("reduced", "darunter Schlüssel auf den lateinischen Namensteil reduziert"),
                        ("key", "darunter Schlüssel mit nicht-lateinischen Buchstaben")]:
        nl[key] = SimpleNamespace(
            matched=int(sc(D, "IGDB", prefix, "N_analytisch")),
            total=sc_hint_int(D, "IGDB", prefix, r"von (\d+)"),
            total_an=sc_hint_int(D, "IGDB", prefix, r"Analysestichprobe (\d+)"),
        )
    check(nl["reduced"].total_an + nl["key"].total_an == nl["raw"].total_an
          and nl["reduced"].matched + nl["key"].matched == nl["raw"].matched,
          "nicht-lateinische Namen: reduziert + nicht-lateinischer Schlüssel = alle")
    check(nl["reduced"].matched / nl["reduced"].total_an > 5 * nl["key"].matched / nl["key"].total_an,
          "reduzierte Namen werden viel häufiger gematcht")
    s3_checked = int(sc(D, "IGDB", "Stufe 3: Kandidaten mit Firmenangabe", "N"))
    s3_checked_an = int(sc(D, "IGDB", "Stufe 3: Kandidaten mit Firmenangabe", "N_analytisch"))
    s3_rejected = sc_hint_int(D, "IGDB", "Stufe 3: Kandidaten mit Firmenangabe", r"verschieden\): (\d+)")
    s3_rejected_an = sc_hint_int(D, "IGDB", "Stufe 3: Kandidaten mit Firmenangabe", r"Analysestichprobe (\d+)")
    firm = {}
    for quelle, stages in [("IGDB", (1, 2, 3)), ("Metacritic", (1, 2))]:
        for st in stages:
            firm[(quelle, st)] = sc_share_an(D, quelle, f"Prüfung Stufe {st}: Treffer mit Firmenangabe")
    check(firm[("IGDB", 3)][0] == 0, "Stufe 3 nach Firmenprüfung ohne Firmenwiderspruch")
    check(s3_rejected_an / s3_checked_an > 2 * firm[("IGDB", 1)][0] / firm[("IGDB", 1)][1],
          "Stufe-3-Kandidaten vor der Prüfung deutlich häufiger widersprüchlich als Stufe 1")
    score_eq = {st: sc_share_an(D, "Metacritic", f"Prüfung Stufe {st}: Treffer mit Steam-Metacritic-Score")
                for st in (1, 2)}
    check(score_eq[2][0] / score_eq[2][1] < score_eq[1][0] / score_eq[1][1], "Stufe 2 seltener scoregleich als Stufe 1")
    check(abs(firm[("IGDB", 2)][0] / firm[("IGDB", 2)][1] - firm[("IGDB", 1)][0] / firm[("IGDB", 1)][1]) < 0.02,
          "IGDB Stufe 2 nahe am Firmenwiderspruchs-Anteil von Stufe 1")
    mc2 = D.mc[(D.mc["mc_match_stage"] == 2) & D.mc["metacritic"].notna()]
    mc2_ex = mc2.loc[(mc2["metacritic"] - mc2["mc_metascore"]).abs().idxmax()]
    check(abs(mc2_ex["metacritic"] - mc2_ex["mc_metascore"]) > 10, "Beispiel einer Versionsverwechslung in Stufe 2")

    def share_text(pair: tuple[int, int]) -> str:
        return f"{pct(pair[0] / pair[1])} (n = {n0(pair[1])})"

    C.match = SimpleNamespace(firm=firm, score_eq=score_eq, nl=nl, s3=(s3_rejected_an, s3_checked_an))
    after_igdb_total_an = sc_hint_int(D, "IGDB", "Release nach dem IGDB-Stand", r"Analysestichprobe (\d+)")
    short_names = sc(D, "FronkonGames", "Name nach Normalisierung kürzer", "N_analytisch")
    exact_only_names = sc(D, "FronkonGames", "Name nach Normalisierung 2 bis 3 Zeichen", "N_analytisch")
    unique_names = sc(D, "FronkonGames", "Name über alle Spiele eindeutig")
    stage_doc = D.igdb.groupby("igdb_match_stage").agg(
        modes=("igdb_has_modes", lambda s: s.eq(True).mean()),
        user=("igdb_rating", lambda s: s.notna().mean()),
        critic=("igdb_aggregated_rating", lambda s: s.notna().mean()),
    )
    check(stage_doc.loc[3, "modes"] < stage_doc.loc[1, "modes"] / 2, "Stufe-3-Treffer spärlich dokumentiert")
    check(abs(stage_doc.loc[3, "modes"] - D.dq.loc["igdb_stage3_has_modes", "Anteil"]) < 1e-9,
          "Stufe-3-Modusanteil Parquet = data_quality_summary")

    mc = {
        "raw": sc(D, "Metacritic", "Rohzeilen"),
        "pc": sc(D, "Metacritic", "PC-Einträge"),
        "exact": sc(D, "Metacritic", "exakte Dublettenzeilen"),
        "tbd": sc(D, "Metacritic", "Metascore 'tbd'"),
        "empty": sc(D, "Metacritic", "Metascore leer"),
        "nodate": sc(D, "Metacritic", "davon ohne lesbares Veröffentlichungsdatum"),
        "noname": sc(D, "Metacritic", "ohne verwertbaren Namen"),
        "dups": sc(D, "Metacritic", "Name/Jahr-Dubletten"),
        "entries": sc(D, "Metacritic", "Einträge für das Matching"),
        "with_score": sc(D, "Metacritic", "davon mit Metascore"),
        "s1": sc(D, "Metacritic", "gematcht in Stufe 1", "N_analytisch"),
        "s2": sc(D, "Metacritic", "gematcht in Stufe 2", "N_analytisch"),
        "gap_checked": sc(D, "Metacritic", "Stufe 2: Kandidaten mit Jahresabstand"),
    }
    mc_gap_rejected = sc_hint_int(D, "Metacritic", "Stufe 2: Kandidaten mit Jahresabstand", r"nicht bestätigt\): (\d+)")
    check(mc["pc"] - mc["exact"] - mc["noname"] - mc["dups"] == mc["entries"],
          "Metacritic-Zählkette: PC − Dubletten − ohne Namen − Name/Jahr-Dubletten = Einträge für das Matching")
    mc_stage = D.mc["mc_match_stage"].value_counts()
    check(int(mc_stage.sum()) == C.n_mc_external_sample, "Stufen der MC-Teilstichprobe vollständig")

    ph = {
        "raw": sc(D, "Preishistorie", "Rohzeilen"),
        "apps": sc(D, "Preishistorie", "Apps in der Rohdatei"),
        "invalid": sc(D, "Preishistorie", "Zeilen mit fehlenden"),
        "dup_groups": sc(D, "Preishistorie", "doppelte (app_id, month)-Gruppen"),
        "dup_rows": sc(D, "Preishistorie", "doppelte App-Monatszeilen"),
        "partial": sc(D, "Preishistorie", "unvollständiger Monat"),
        "clean": sc(D, "Preishistorie", "Zeilen nach Bereinigung"),
        "above_months": sc(D, "Preishistorie", "Monate mit lowest_price > base_price"),
        "full": sc(D, "Preishistorie", "davon volles Fenster"),
        "matched": sc(D, "Preishistorie", "gematcht über AppID"),
    }
    ph_dup_apps = sc_hint_int(D, "Preishistorie", "doppelte (app_id, month)-Gruppen", r"(\d+) Apps")
    ph_dup_diff = sc_hint_int(D, "Preishistorie", "doppelte (app_id, month)-Gruppen", r"abweichenden Preisfeldern: (\d+)")
    check(ph_dup_diff == 0, "doppelte Preishistorien-Monate haben identische Preisfelder")
    ph_above_apps = sc_hint_int(D, "Preishistorie", "Monate mit lowest_price > base_price", r"(\d+) Apps")

    counts_tbl = D.counts.copy()
    counts_tbl = pd.DataFrame({
        "Quelle": counts_tbl["Quelle"],
        "Schritt": counts_tbl["Schritt"].map(de_text),
        "N": counts_tbl["N"].map(n0),
        "N Analysestichprobe": counts_tbl["N_analytisch"].map(n0),
        "Hinweis": counts_tbl["Hinweis"].map(lambda v: de_text(v, english=True)),
    })

    cov = D.cov
    cov_tbl = pd.DataFrame({
        "Quelle": cov["Quelle"],
        "Version": cov["Version"].map(n0),
        "Einträge für das Matching": cov["N_Quelle"].map(n0),
        "Steam-Spiele mit Treffer (alle aufbereiteten)": cov["N_gematcht_gesamt"].map(n0),
        "Anteil (alle)": cov["Match_Rate_gesamt"].map(pct),
        "Treffer in der Analysestichprobe": cov["N_gematcht_analytisch"].map(n0),
        "Match-Rate (Analysestichprobe)": cov["Match_Rate_analytisch"].map(pct),
        "Matching-Schlüssel": cov["Schluessel"].map(de_text),
        "Hinweis": cov["Hinweis"].map(lambda v: de_text(v, english=True)),
    })

    mo = D.mco.loc["Analysestichprobe"]
    check(mo["Anteil_exakt_gleich"] > 0.9 and mo["N_nur_FG"] > 0.25 * mo["N_FG"],
          "Werte decken sich, Abdeckung nicht")
    ov = D.overlap.copy()
    ov_tbl = pd.DataFrame({"Anzahl Datenquellen": ov["Anzahl_Datenquellen"].map(n0), "N": ov["N"].map(n0),
                           "Anteil": ov["Anteil"].map(pct)})

    out.append(f"""## 1. Datengrundlage, Datenqualität und Stichprobe

### 1.1 Datenquellen, Datenstände und Provenienz

Alle vier Rohdateien werden über `kagglehub` als genau die benötigte Datei einer festen Kaggle-Version bezogen und vor der Verarbeitung über Größe und SHA-256 identifiziert; bei einer Abweichung bricht der Lauf ab. Nach dem Einlesen wird zusätzlich die Zahl der Datensätze geprüft. Bezugsweg in diesem Lauf: {path_types}; Versionsangabe: {version_notes}. Das Protokoll steht in `outputs/tables/data_provenance.csv`:

{md(prov_tbl)}

SHA-256 der Rohdateien:

{hashes}

**Version des Basisdatensatzes:** {C.version_text}. Ein Neulauf der gesamten Pipeline am {VERSION_CHECK_DATE} mit dieser Version hat {f"alle {VERSION_CHECK['parquet_total']}" if VERSION_CHECK['parquet_equal'] == VERSION_CHECK['parquet_total'] else f"{VERSION_CHECK['parquet_equal']} von {VERSION_CHECK['parquet_total']}"} Parquet-Dateien der ursprünglichen Auswertung SHA-256-gleich und {VERSION_CHECK['tables_equal']} von {VERSION_CHECK['tables_total']} Tabellen bytegleich reproduziert; die übrigen {len(VERSION_CHECK['tables_logit']) + len(VERSION_CHECK['tables_ulp'])} wichen nur in den letzten Stellen ab: {join_de([f"`{t}`" for t in VERSION_CHECK['tables_logit']])} in den adjustierten Logit-Schätzungen ({VERSION_CHECK['logit_rows']} Zeilen), {join_de([f"`{t}`" for t in VERSION_CHECK['tables_ulp']])} nur durch Gleitkomma-Rundung (±1 ULP). Keine Signifikanzentscheidung änderte sich; die {VERSION_CHECK['figures']} Abbildungen waren pixel-, aber nicht bytegleich. Belegt ist damit, dass die damals verarbeitete Datei in allen genutzten Spalten inhaltsgleich mit Version 58 ist, nicht, welche Version damals tatsächlich geladen wurde. Weil der Eigentümer ältere Kaggle-Versionen löscht, sollte `{C.fr['Datei']}` mit dem oben genannten Hash dauerhaft archiviert werden (Abschnitt 6).

**Erhebungsstände und Stichtag.** Die FronkonGames-Datei vereint zwei Stände: {C.store_stand} (Preis, Rabatt, Recommendations, App-Liste) und {C.steamspy_stand} (Besitzer, Positive/Negative, Peak CCU). Den SteamSpy-Stand belegt `steamspy_coverage_by_month.csv`: Von den Spielen mit Release im {month_de(before)} haben {pct(s_before['Anteil_Reviews_gt0'])} Reviews, von denen mit Release im {month_de(after)} nur {pct(s_after['Anteil_Reviews_gt0'])}. Die Store-Recommendations zeigen keinen solchen Bruch ({pct(s_before['Anteil_Recommendations_gt0'])} gegen {pct(s_after['Anteil_Recommendations_gt0'])}). Spiele ab dem {C.cutoff} haben daher strukturell kein beobachtbares Ergebnis; die Analysestichprobe endet vor diesem Stichtag, und der Jahrgang {STEAMSPY_CUTOFF.year} umfasst nur Januar bis Mai. Das Alter (`age_years`) wird relativ zum Stichtag gemessen. Die externen Quellen haben eigene Stände: IGDB „{D.prov.loc['igdb', 'Datenstand']}“, Metacritic „{D.prov.loc['metacritic_raw', 'Datenstand']}“, Preishistorie „{D.prov.loc['price_history', 'Datenstand']}“.

**Aufbereitung der externen Quellen** (alle Zählungen in `source_counts.csv`):

- **IGDB:** {n0(ig['raw'])} Rohzeilen. {n0(ig['crit0'])} Kritikerbewertungen mit dem Platzhalter 0 werden als fehlend gesetzt (Nutzerbewertung 0: {n0(ig['user0'])} Fälle). {n0(ig['noname'])} Einträge ohne verwertbaren Namen und {n0(ig['nonpc'])} Einträge, die eine Plattformangabe, aber keine PC-nahe Plattform (Windows, Mac, Linux, SteamVR) haben, werden entfernt. Bei {n0(ig['dups'])} Name/Jahr-Dubletten wird das Hauptspiel (Kategorie `main_game`) mit PC-Plattform bevorzugt. Für das Matching bleiben {n0(ig['entries'])} Einträge, davon {n0(ig['undated'])} ohne Veröffentlichungsdatum. Leere Spielmodus- bzw. Genrelisten gelten als fehlend, nicht als „nein“. Die IGDB-Datei enthält keine Stimmenzahlen; ganzzahlige Nutzerbewertungen deuten auf sehr wenige Stimmen hin (Merkmal `igdb_rating_integer`).
- **Metacritic:** Verwendet wird die Rohdatei mit {n0(mc['raw'])} Plattform-Einträgen, davon {n0(mc['pc'])} für PC. Entfernt werden {n0(mc['exact'])} exakte Dublettenzeilen, {n0(mc['noname'])} Einträge ohne verwertbaren Namen und {n0(mc['dups'])} Name/Jahr-Dubletten. Ein Metascore „tbd“ ({n0(mc['tbd'])}) oder leer ({n0(mc['empty'])}) gilt als fehlend. Für das Matching bleiben {n0(mc['entries'])} PC-Einträge, davon {n0(mc['with_score'])} mit Metascore. {n0(mc['nodate'])} dieser Einträge haben kein lesbares Datum: Sie bleiben enthalten (sie zählen bei der Eindeutigkeit des Namens mit), sind aber nicht matchbar, weil beide Stufen ein Jahr verlangen. Als Treffer (`has_mc_external`) zählt nur ein vorhandener Metascore; der Nutzerscore fehlt teils und wird nur dort ausgewertet, wo er vorliegt. Die ebenfalls veröffentlichte „clean“-Datei der Quelle ist ein Complete-Case-Auszug (sie verlangt auch einen Nutzerscore) und wird nicht verwendet.
- **Preishistorie:** {n0(ph['raw'])} App-Monate zu {n0(ph['apps'])} Apps (ungültige Zeilen: {n0(ph['invalid'])}). {n0(ph['dup_groups'])} doppelte (App, Monat)-Gruppen bei {n0(ph_dup_apps)} Apps haben identische Preisfelder und werden dedupliziert, nicht summiert ({n0(ph['dup_rows'])} Zeilen). Der unvollständige Monat 2025-11 wird verworfen ({n0(ph['partial'])} Zeilen). Es bleiben {n0(ph['clean'])} Zeilen. {n0(ph['full'])} Apps haben das volle Fenster. In {n0(ph['above_months'])} Monaten ({n0(ph_above_apps)} Apps) liegt der Tiefstpreis über dem statischen Basispreis der Quelle; der Rabatt ist dort fehlend statt auf 0 gekappt. Der Basispreis ist in der Quelle je App konstant, Rabatte und Rabatttage beziehen sich also auf einen statischen Referenzpreis. Kostenlose Titel enthält die Quelle nicht. Einzelne Namensangaben passen nicht zur AppID (z. B. AppID 57650, in der Quelle „Dungeons & Dragons: Dark Sun Series“, im Steam-Datensatz „Dungeons“). Der Join erfolgt deshalb ausschließlich über die AppID.

**Matching.** IGDB und Metacritic werden über einen normalisierten Namen verknüpft. Die Normalisierung entfernt ™/®/© und wendet Unicode-NFKC, Casefolding und eine `\\w`-basierte Bereinigung an; Akzente lateinischer Buchstaben werden entfernt, nicht-lateinische Buchstaben bleiben erhalten. Gemischtschriftliche Namen werden für das Matching auf ihren lateinischen Teil samt Ziffern reduziert, wenn dieser mindestens vier Buchstaben und mindestens ein Drittel so viele Buchstaben wie der übrige Teil umfasst (z. B. chinesischer Titel plus „Warm Snow“), weil externe Quellen meist den englischen Titel führen. Das betrifft {n0(nl['reduced'].total)} Steam-Einträge ({n0(nl['reduced'].total_an)} in der Analysestichprobe). Die Regel kann generische Schlüssel erzeugen (z. B. „online“ oder „playtest“); solche Schlüssel sind auf Steam-Seite nicht eindeutig und können deshalb nur in Stufe 1 (Name und Jahr exakt) einen Treffer erhalten. Für die Duplikaterkennung innerhalb von Steam (Abschnitt 1.3) gilt die Reduktion nicht. Editionszusätze (edition, goty, game of the year, remastered, definitive, deluxe, ultimate, complete, enhanced, anniversary, collector's; generische Wörter wie standard, special, gold oder vr nur als „… edition“) werden nur am Namensende entfernt; mindestens vier Zeichen bleiben stehen. Für Stufe 1 muss der Name mindestens zwei, für Stufe 2/3 mindestens vier Zeichen haben. In der Analysestichprobe sind {n0(short_names)} Namen nach der Normalisierung kürzer als zwei Zeichen (nicht matchbar), weitere {n0(exact_only_names)} haben zwei bis drei Zeichen und können nur in Stufe 1 einen Treffer erhalten. Das Matching erfolgt in Stufen:

1. **Stufe 1:** normalisierter Name und Veröffentlichungsjahr exakt. Hat ein Quelleintrag mehrere passende AppIDs ({n0(ig['multi'])} IGDB-Einträge), entscheiden Entwickler/Publisher ({n0(multi_firm)}) bzw. die Zugehörigkeit zur Analysestichprobe ({n0(multi_an)}); sonst gibt es keinen Treffer ({n0(multi_none)}).
2. **Stufe 2** (nur für noch ungematchte Spiele und unbenutzte Quelleinträge; der Name muss auf beiden Seiten eindeutig sein, auf Steam-Seite geprüft über alle {n0(C.n_raw)} aufbereiteten Einträge, von denen {n0(unique_names)} einen eindeutigen Namen haben): bei IGDB |Steam-Jahr − IGDB-Jahr| ≤ 1; bei Metacritic Steam-Jahr ≥ MC-Jahr − 1, weil das Metacritic-Datum das spielweite Erstveröffentlichungsdatum ist und späte PC-Portierungen sonst systematisch fehlen. Ab zwei Jahren Abstand verlangt Metacritic eine Bestätigung über Entwickler/Publisher oder einen übereinstimmenden Score ({n0(mc['gap_checked'])} Kandidaten geprüft, {n0(mc_gap_rejected)} verworfen).
3. **Stufe 3** (nur IGDB): undatierte IGDB-Einträge mit beidseitig eindeutigem Namen. Ohne Jahr ist der Name allein schwach (z. B. „Storm“ oder „Drive“ sind verschiedene Spiele); geben beide Seiten Entwickler/Publisher an, müssen sie sich deshalb überschneiden ({n0(s3_checked)} Kandidaten mit beidseitiger Firmenangabe geprüft, {n0(s3_rejected)} verworfen; in der Analysestichprobe {n0(s3_rejected_an)} von {n0(s3_checked_an)}).

Jeder Quelleintrag wird höchstens einer AppID zugeordnet. IGDB-Treffer in der Analysestichprobe: Stufe 1 {n0(ig['s1'])}, Stufe 2 {n0(ig['s2'])}, Stufe 3 {n0(ig['s3'])}. Die Stufe-3-Treffer sind spärlich dokumentiert: Spielmodi liegen bei {pct(stage_doc.loc[3, 'modes'])} vor (Stufe 1: {pct(stage_doc.loc[1, 'modes'])}, Stufe 2: {pct(stage_doc.loc[2, 'modes'])}), eine Nutzerbewertung bei {pct(stage_doc.loc[3, 'user'])}, eine Kritikerbewertung bei {pct(stage_doc.loc[3, 'critic'])}. Als **Recall-Referenz** dienen die Spiele mit Steam-Metacritic-Score, die fast sicher in IGDB erfasst sind: Davon werden {n0(recall_n)} von {n0(recall_base)} gematcht ({pct(recall_n / recall_base)}). Von {n0(nl['raw'].total_an)} Spielen der Analysestichprobe mit nicht-lateinischen Buchstaben im Namen werden {n0(nl['raw'].matched)} gematcht ({pct(nl['raw'].matched / nl['raw'].total_an)}): von den {n0(nl['reduced'].total_an)} auf den lateinischen Teil reduzierten Namen {n0(nl['reduced'].matched)} ({pct(nl['reduced'].matched / nl['reduced'].total_an)}), von den {n0(nl['key'].total_an)} Namen mit nicht-lateinischem Schlüssel nur {n0(nl['key'].matched)}. Kein Spiel der Analysestichprobe erschien nach dem IGDB-Stand ({n0(after_igdb_total_an)} Fälle). Metacritic-Treffer in der Analysestichprobe (inkl. Einträgen ohne Metascore): Stufe 1 {n0(mc['s1'])}, Stufe 2 {n0(mc['s2'])}. In der Teilstichprobe mit Metascore stammen {n0(mc_stage.get(1, 0))} Spiele aus Stufe 1 und {n0(mc_stage.get(2, 0))} aus Stufe 2. Die Preishistorie wird exakt über die AppID verknüpft ({n0(ph['matched'])} von {n0(ph['apps'])} Apps im Steam-Datensatz).

**Plausibilität der Treffer** (Analysestichprobe). Widersprechen sich Entwickler/Publisher bei beidseitiger Firmenangabe, ist ein Treffer verdächtig; wegen Firmenwechseln und Namensvarianten gibt es aber auch bei richtigen Treffern einen Grundanteil. Anteil widersprüchlicher Firmen: IGDB Stufe 1 {share_text(firm[('IGDB', 1)])}, Stufe 2 {share_text(firm[('IGDB', 2)])}, Stufe 3 nach der Firmenprüfung definitionsgemäß {pct(0)} (vor der Prüfung {pct(s3_rejected_an / s3_checked_an)} der Kandidaten); Metacritic Stufe 1 {share_text(firm[('Metacritic', 1)])}, Stufe 2 {share_text(firm[('Metacritic', 2)])}. Wo ein FronkonGames-Metacritic-Score vorliegt, stimmt der externe Metascore in Stufe 1 bei {share_text(score_eq[1])} und in Stufe 2 bei {share_text(score_eq[2])} der Treffer exakt überein. Metacritic-Stufe 2 enthält demnach einzelne Versionsverwechslungen (z. B. „{mc2_ex['Name']}“, Steam {int(mc2_ex['release_year'])}, mit dem Metacritic-Eintrag von {int(mc2_ex['mc_release_year'])}: Metascore {n0(mc2_ex['mc_metascore'])} statt {n0(mc2_ex['metacritic'])}). Bei IGDB liegt der Anteil in Stufe 2 nahe am Grundanteil von Stufe 1. Eine vollständige manuelle Prüfung der Treffer fehlt.

{md(cov_tbl)}

Vollständige Zählungen je Quelle (Laden, Filter, Dubletten, Treffer je Stufe):

{md(counts_tbl)}

**Zwei Metacritic-Quellen.** Der FronkonGames-Datensatz enthält selbst einen Metacritic-Score (FG). In der Analysestichprobe haben {n0(mo['N_FG'])} Spiele einen FG-Score und {n0(mo['N_extern'])} einen externen Metascore; beide liegen für {n0(mo['N_beide'])} Spiele vor (Stufe 1: {n0(mo['N_beide_Stufe1'])}, Stufe 2: {n0(mo['N_beide_Stufe2'])}). Auf dieser Schnittmenge stimmen die **Werte** praktisch überein ({pct(mo['Anteil_exakt_gleich'])} exakt gleich, Spearman ρ = {de(mo['Spearman_rho'], 3)}). Die **Abdeckung** unterscheidet sich dagegen deutlich: {n0(mo['N_nur_FG'])} Spiele haben nur den FG-Score, {n0(mo['N_nur_extern'])} nur den externen. Von den Spielen mit nur FG-Score kommt der normalisierte Name lediglich bei {n0(mo['N_nur_FG_Name_im_MC_PC_Teil'])} überhaupt im PC-Teil der externen Datei vor. Der externe Datensatz dient deshalb als Quellenabgleich und PC-spezifische Ergänzung, nicht als unabhängige Validierung.

Überlappung der Quellen in der Analysestichprobe (alle Spiele stammen aus dem Basisdatensatz):

{md(ov_tbl)}
""")

    # ---- 1.2 Datenqualitaet
    dq = D.dq
    base_main = dq[dq["Basis"].isin(["Rohdatensatz", "Analysestichprobe"])]
    base_sub = dq[~dq["Basis"].isin(["Rohdatensatz", "Analysestichprobe"])]

    def dq_table(part: pd.DataFrame) -> str:
        return md(pd.DataFrame({
            "Prüfung": part["Pruefung"],
            "Basis": part["Basis"],
            "N Basis": part["N_Basis"].map(n0),
            "Wert": part["Wert"].map(n0),
            "Anteil an Basis": part["Anteil"].map(pct),
            "Hinweis": part["Hinweis"].map(de_text),
        }))

    lowest = D.owners.sort_values("owners_class").iloc[0]
    cy_pre = D.cohort[D.cohort.index <= 2020]
    zero_max_pre = cy_pre["Anteil_0_0"].max()
    zero_21 = D.cohort.loc[2021, "Anteil_0_0"]
    zero_last = D.cohort.loc[STEAMSPY_CUTOFF.year, "Anteil_0_0"]
    check(zero_21 > 5 * zero_max_pre, "'0 - 0'-Anteil springt ab 2021")
    out.append(f"""### 1.2 Datenqualität

Die Kopfzeile der FronkonGames-Datei enthält {len(RAW_HEADER)} Spaltennamen, die Datenzeilen {len(RAW_COLUMNS)} Felder: `DiscountDLC count` ist in den Daten auf `Discount` und `DLC count` verteilt. Die Datei wird daher mit einer expliziten Liste von {len(RAW_COLUMNS)} Spalten eingelesen; geprüft wird, dass die Kopfzeile den {len(RAW_HEADER)} erwarteten Namen entspricht und jede Zeile genau {len(RAW_COLUMNS)} Felder hat. Nur leere Felder gelten als fehlend, damit wörtliche Werte wie „None“ oder „N/A“ in Entwickler- und Publishernamen erhalten bleiben. Die Spalte „Basis“ gibt an, worauf sich ein Anteil bezieht:

{dq_table(base_main)}

Kennzahlen innerhalb der Teilstichproben (eigene Basis):

{dq_table(base_sub)}

Die wichtigsten Einschränkungen: Die Besitzerangabe „0 - 0“ fehlt bei {pct(dq.loc['owners_zero', 'Anteil'])} des Rohdatensatzes. Dieser Ausfall ist nicht zufällig: Bis zum Jahrgang 2020 liegt er in der um „0 - 0“ erweiterten Stichprobe bei höchstens {pct(zero_max_pre)} je Jahrgang, 2021 bei {pct(zero_21)} und {STEAMSPY_CUTOFF.year} bei {pct(zero_last)} (`cohort_by_year.csv`). Bei {pct(dq.loc['steamspy_missing', 'Anteil'])} des Rohdatensatzes fehlen die SteamSpy-Felder, obwohl der Store Recommendations ausweist (SteamSpy-Lücke). Die Besitzer liegen nur als Klassen vor; {pct(lowest['Anteil'])} der Analysestichprobe liegen in der untersten Klasse ({lowest['owners_label']}). In der Analysestichprobe haben {pct(dq.loc['an_no_reviews', 'Anteil'])} keine Reviews und {pct(dq.loc['an_lt10_reviews', 'Anteil'])} weniger als {MIN_REVIEWS}; Peak CCU ist bei {pct(dq.loc['an_peak_zero', 'Anteil'])} gleich 0. Nullwerte bei Peak CCU bedeuten „keine Aktivität am Messtag“, nicht „fehlend“.
""")

    # ---- 1.3 Bereinigung
    fl = D.flow
    seq = fl[fl["Basis"] == "sequenzielle Bereinigung"]
    info = fl[fl["Basis"].str.startswith("Information")]
    subs = fl[~fl["Basis"].isin(["sequenzielle Bereinigung"]) & ~fl["Basis"].str.startswith("Information")]
    flow_tbl = pd.DataFrame({
        "Schritt": pd.concat([seq, info, subs])["Schritt"],
        "Basis": pd.concat([seq, info, subs])["Basis"],
        "N": pd.concat([seq, info, subs])["N"].map(n0),
        "Anteil am Rohdatensatz": pd.concat([seq, info, subs])["Anteil_am_Rohdatensatz"].map(pct),
        "entfernt gegenüber Vorzeile": pd.concat([seq, info, subs])["Entfernt_gegenueber_Vorzeile"].map(n0),
        "Anteil an der Analysestichprobe": pd.concat([seq, info, subs])["Anteil_an_Analysestichprobe"].map(pct),
    })
    succ_rating = D.analytic.loc[D.analytic["rating_sample"], "success_top25"].mean()
    is_counter = D.excl["Ausschlussgrund"].str.startswith("Gegenprobe")
    excl_reasons = D.excl[~is_counter].drop_duplicates("Ausschlussgrund")
    n_counter_sample = int(is_counter.sum())
    kept_kw = dq.loc["an_kept_keyword"]
    kept_kw_genre = dq.loc["an_kept_keyword_genre"]
    kept_names = {"DLC Quest", "Expansion", "Expansion Core", "Benchmark", "KINGDOM HEARTS III + Re Mind (DLC)"}
    dropped_names = {"Evolve Benchmark", "OpenVR Benchmark"}
    an_names = set(D.analytic["Name"].dropna())
    check(kept_names <= an_names and not (dropped_names & an_names),
          "Beispiele des Namensfilters stimmen mit der Analysestichprobe überein")
    check(len(VERSION_CHECK["tables_logit"]) + len(VERSION_CHECK["tables_ulp"])
          == VERSION_CHECK["tables_total"] - VERSION_CHECK["tables_equal"], "Neulauf: Tabellenzahlen konsistent")
    out.append(f"""### 1.3 Bereinigung und Analysestichprobe

Die Bereinigung erfolgt sequenziell (`sample_flow.csv`):

1. Einträge mit Besitzerangabe „0 - 0“ werden entfernt ({n0(fl.loc['owners_present', 'Entfernt_gegenueber_Vorzeile'])}). Sie bleiben als Merkmal `owners_zero` erhalten; eine Sensitivitätsstichprobe zählt sie als Nicht-Erfolg ({n0(C.n_sample_with_owners_zero)} Einträge).
2. Nur Veröffentlichungen ab 1997 und vor dem SteamSpy-Stichtag bleiben ({n0(fl.loc['released_before_cutoff', 'Entfernt_gegenueber_Vorzeile'])} entfernt). Davon haben {n0(fl.loc['removed_after_cutoff_with_steamspy', 'N'])} trotz späteren Store-Datums SteamSpy-Daten. Das sind vor allem Early-Access-Absolventen, deren Store-Datum das Datum der Vollversion ist.
3. Einträge mit SteamSpy-Lücke (0 Reviews trotz Store-Recommendations) werden entfernt ({n0(fl.loc['steamspy_present', 'Entfernt_gegenueber_Vorzeile'])}); Besitzer, Reviews und Peak CCU sind dort fehlend gesetzt.
4. Nicht-Spiele nach dem Namen ({n0(fl.loc['no_non_game_name', 'Entfernt_gegenueber_Vorzeile'])}): Demo, Playtest, Testserver, „Free Edition“, „Free Trial“, Wallpaper-Tools u. Ä. „Soundtrack“, „OST“, „DLC“ und „Expansion“ zählen mit Trennzeichen davor („Spiel - Soundtrack“); diese vier und „Benchmark“ zählen außerdem als letztes Wort nach mindestens einem weiteren Wort (z. B. „Evolve Benchmark“, „OpenVR Benchmark“), aber nicht bei Bündeln der Form „Spiel + Zusatz“ (z. B. „KINGDOM HEARTS III + Re Mind (DLC)“). So bleiben Spieltitel wie „DLC Quest“, „Expansion“, „Expansion Core“ oder „Benchmark“ erhalten. „Free Editions“ gelten als Nicht-Spiel, obwohl einzelne spielbare F2P-Produkte sind; „Trial“ allein zählt nicht, weil es in vielen Spieltiteln vorkommt.
5. Software ({n0(fl.loc['no_software', 'Entfernt_gegenueber_Vorzeile'])}): Software-Genre ohne Steam-Standardgenre, auch wenn der Eintrag als Free to Play oder Early Access markiert ist.
6. Duplikate ({n0(fl.loc['analytic_sample', 'Entfernt_gegenueber_Vorzeile'])}): Schlüssel aus normalisiertem Namen (vollständig in allen Schriften, ohne die Reduktion auf den lateinischen Teil), Veröffentlichungsjahr und erstem Entwickler (ersatzweise Publisher); behalten wird der Eintrag mit den meisten Besitzern. Der Entwickler im Schlüssel verhindert, dass verschiedene Spiele mit generischem Namen zusammengelegt werden.

Je Ausschlussgrund liegen bis zu {n0(D.excl[~is_counter].groupby('Ausschlussgrund').size().max())} zufällige Beispiele in `exclusion_samples.csv` ({len(excl_reasons)} Gründe). Als Gegenprobe zum Namensfilter enthält die Datei {n0(n_counter_sample)} zufällige behaltene Titel mit einem Schlüsselwort im Namen (Edition, Test, Pack, Beta, Demo u. Ä.; insgesamt {n0(kept_kw['Wert'])} Titel, davon {pct(kept_kw_genre['Anteil'])} mit Steam-Standardgenre). So lassen sich die Filter in beide Richtungen stichprobenartig prüfen. Software ohne Genreangabe erkennt keiner der Filter.

{md(flow_tbl)}

Die **Analysestichprobe** umfasst **{n0(C.n)} Spiele**. Für Bewertungsanalysen werden mindestens {MIN_REVIEWS} Reviews verlangt ({n0(C.n_rating_sample)} Spiele; Robustheit mit {MIN_REVIEWS_ROBUST} Reviews: {n0(C.n_rating_sample_50)}). Peak-CCU-Intensitäten werden nur auf Spielen mit Peak CCU > 0 ausgewertet ({n0(C.n_peak_sample)}). Für IGDB-Analysen stehen {n0(C.n_igdb_sample)}, für externe Metacritic-Analysen {n0(C.n_mc_external_sample)} und für Preishistorien-Analysen {n0(C.n_price_history_sample)} Spiele zur Verfügung. Die FronkonGames-Metacritic-Spalte deckt {n0(C.n_meta_sample)} Spiele ab und dient nur dem Vergleich.

**Selektion durch die Review-Schwelle:** Die Erfolgsquote beträgt in der Analysestichprobe {pct(C.succ)}, in der Bewertungsstichprobe {pct(succ_rating)}. Die externen Teilstichproben sind noch stärker selektiert (Abschnitt 3.5).

**Veröffentlichungsdatum.** `release_year` ist das aktuelle Store-Datum; bei Early-Access-Absolventen ist das das Datum der Vollversion, bei Neuauflagen und Portierungen das Steam-Datum. Für {n0(dq.loc['an_release_shifted', 'Wert'])} Spiele ({pct(dq.loc['an_release_shifted', 'Anteil'])}) liegt die IGDB-Erstveröffentlichung mehr als ein Jahr vor dem Steam-Jahr (`release_shifted`). Das IGDB-Jahr stammt aus dem Treffer oder, ohne Treffer, aus einem Namens-Lookup bei beidseitig eindeutigem Namen und nicht widersprüchlichen Firmen ({n0(sc(D, 'IGDB', 'Erstveröffentlichungsjahr per Namens-Lookup', 'N_analytisch'))} Lookups in der Analysestichprobe). Weil IGDB-Treffer der Stufen 1 und 2 höchstens ein Jahr vom Steam-Jahr abweichen und Stufe-3-Treffer undatiert sind, stammen alle `release_shifted`-Fälle aus dem Namens-Lookup ({n0(dq.loc['an_release_shifted_lookup', 'Wert'])} von {n0(dq.loc['an_release_shifted_lookup', 'N_Basis'])}); Spiele mit IGDB-Treffer können nie `release_shifted` sein. Die Alterskontrolle wird ohne diese Spiele und mit dem früheren Jahr (`release_year_alt`) wiederholt (Abschnitt 3.7). Für Portierungen ist `release_year_alt` eine obere Schranke des Alters auf Steam. Early-Access-Phasen erfasst `release_shifted` nur teilweise: IGDB datiert Early-Access-Absolventen teils auf den Early-Access-Start, teils auf die Vollversion, und erfasst werden nur Phasen ab zwei Jahren bei Spielen ohne IGDB-Treffer. Dafür markiert `ea_graduate_proxy` zusätzlich Spiele ohne aktuelle Early-Access-Markierung mit Hinweis auf einen früheren Markteintritt (AppID unter dem Median des Jahrgangs {D.c01['APPID_PROXY_YEARS']} Jahre vor dem Store-Jahr oder erster Preismonat mehr als einen Monat vor dem Store-Datum; {n0(dq.loc['an_ea_graduate_proxy', 'Wert'])} Spiele, davon {n0(dq.loc['an_entry_proxy_shifted', 'Wert'])} zugleich `release_shifted`), `market_entry_year` schätzt daraus das Jahr des Markteintritts; beide dienen nur als Sensitivität (Abschnitt 3.7).
""")

    # ---- 1.4 Operationalisierung
    th = D.thr
    th_tbl = pd.DataFrame({
        "Kriterium": th["Kriterium"],
        "Variable": th["Variable"].map(lambda s: f"`{s}`"),
        "Schwelle": th["Schwelle"].map(n0),
        "Basis": th["Basis"],
        "N Basis": th["N_Basis"].map(n0),
        "N erfolgreich": th["N_erfolgreich"].map(n0),
        "Anteil an Basis": th["Anteil"].map(pct),
        "Anteil an Analysestichprobe": th["Anteil_an_Analysestichprobe"].map(pct),
        "Hinweis": th["Hinweis"].map(de_text),
    })
    peak = th.loc["success_peak_top25"]
    an = D.analytic
    lp_none = int(an["list_price"].isna().sum())
    lp_100 = int((an["list_price"].isna() & (an["discount"] >= 100)).sum())
    lp_zero_rows = an[an["price_zero_with_discount"]]
    lp_zero = len(lp_zero_rows)
    check(lp_100 + lp_zero == lp_none and lp_none == int(dq.loc["an_list_price_undefined", "Wert"]),
          "Listenpreis fehlt = 100 % Rabatt + Tagespreis 0 trotz Rabatt")
    check(not lp_zero_rows["list_price"].notna().any(), "Tagespreis 0 trotz Rabatt: Listenpreis fehlend")
    lp_ex = lp_zero_rows.sort_values(["owners_low", "AppID"], ascending=[False, True]).iloc[0]
    out.append(f"""### 1.4 Operationalisierung

**Erfolg.** Die Besitzerangabe liegt nur als Spanne vor ({len(D.owners)} Klassen in der Analysestichprobe). Für Schwellen wird die untere Grenze (`owners_low`) verwendet. Spearman-Korrelationen werden mit dem logarithmierten Klassenmittelpunkt (`log_owners`) berichtet, Mann-Whitney-U/Cliff's δ und Kruskal-Wallis mit dem Klassenmittelpunkt (`owners_mid`). Alle diese Verfahren sind rangbasiert: Maßgeblich ist nur die Rangfolge der Besitzerklassen (mit vielen Bindungen), die Log-Transformation ändert an den Ergebnissen nichts. Erfolgsschwellen und ihre Prävalenz:

{md(th_tbl)}

Weil {pct(lowest['Anteil'])} der Spiele in der untersten Klasse liegen, bedeutet „Erfolg“ praktisch, diese Klasse zu verlassen. Unterschiede innerhalb der untersten Klasse sind nicht beobachtbar. Die Besitzerzahl ist zudem **kumuliert**: Ältere Spiele hatten mehr Zeit, Besitzer zu sammeln. Dieser Alters- bzw. Kohorteneffekt ist die zentrale Einschränkung der Erfolgsdefinition (Abschnitt 2.2); alle Hypothesentests kontrollieren deshalb das Veröffentlichungsjahr.

**Peak CCU** ist die höchste Zahl gleichzeitiger Spieler am Vortag des SteamSpy-Abrufs ({C.steamspy_stand.replace('SteamSpy-Felder ', '')}), kein Allzeit-Peak. Die Peak-Erfolgsdefinition in der Tabelle (Schwelle {n0(peak['Schwelle'])}) ist nur innerhalb der Spiele mit Peak CCU > 0 definiert: {pct(peak['Anteil'])} von {n0(peak['N_Basis'])} Spielen, also {pct(peak['Anteil_an_Analysestichprobe'])} der Analysestichprobe. Sie wird nur dokumentiert und in keiner Analyse verwendet. Peak-Ergebnisse werden stattdessen zweiteilig ausgewertet: P(Peak > 0) als Hürde und die Intensität nur bei Peak > 0. Sie sind eine Momentaufnahme der aktuellen Aktivität.

**Preis.** `price_current` ist der Tagespreis zum Store-Snapshot ({C.store_stand.replace('Store-Felder ', '')}), bei laufendem Rabatt also der rabattierte Preis. Das Preisniveau wird über den rekonstruierten **Listenpreis** gemessen: Tagespreis / (1 − Rabatt/100), auf übliche Preispunkte (x,99 bzw. ganze Dollar) gesetzt. Widerspricht die Rabattangabe dem Tagespreis (Tagespreis exakt x,99 bzw. ganzer Dollar, der rekonstruierte Wert aber neben jedem Preispunkt, z. B. 39,99 USD bei 85 % → 266,60 USD), gilt der Tagespreis als Listenpreis ({n0(dq.loc['an_list_price_conflict', 'Wert'])} Spiele der Analysestichprobe). Weil der Rabatt in den Rohdaten teils nicht erfasst ist, ist der Listenpreis eine **Untergrenze**, und die Rabattbefunde sind entsprechend verunreinigt. „Kostenlos“ heißt **Listenpreis 0** („kostenlos (Listenpreis 0)“), nicht Tagespreis 0 durch einen Rabatt von 100 %. Nicht bestimmbar ist der Listenpreis bei {n0(lp_100)} Spielen der Analysestichprobe mit 100 % Rabatt und bei {n0(lp_zero)} Spielen mit Tagespreis 0 trotz eines Rabatts zwischen 1 und 99 % (z. B. „{lp_ex['Name']}“ mit −{n0(lp_ex['discount'])} %). Letzteres ist ein Datenfehler der Quelle, denn die Formel ergäbe dort wieder 0. Diese {n0(lp_none)} Spiele gelten weder als kostenlos noch als kostenpflichtig und fehlen in allen Preisvergleichen. Die Gruppe „kostenlos“ mischt echte Free-to-Play-Titel mit sonstigen Gratisangeboten (z. B. delisteten oder werbefinanzierten Titeln); das Steam-Genre „Free To Play“ (`f2p_tag`) dient als Sensitivität. Listenpreise über 100 USD gelten als Preisausreißer und werden in Preisniveau-Analysen getrennt ausgewiesen bzw. ausgeschlossen ({n0(dq.loc['an_price_outlier', 'Wert'])} Spiele); das Cluster zu {de(D.cc['CLUSTER_PRICE'], 2)} USD ab {D.cc['CLUSTER_FROM_YEAR']} ({n0(dq.loc['an_price_cluster', 'Wert'])} Spiele) wird in der Preisübersicht getrennt ausgewiesen, bleibt in den Tests aber enthalten.

**Spielmodus.** Multiplayer umfasst die Steam-Kategorien Multi-player, PvP, Co-op, Shared/Split Screen, MMO, LAN, Cross-Platform und Remote Play Together; Koop, PvP und MMO werden zusätzlich einzeln ausgewertet.

**Nutzerbewertung.** Primäres Maß ist die **Positive Ratio** (Anteil positiver Reviews). Der Wilson-Score (Untergrenze des Anteils) hängt stark vom Review-Volumen ab und dient nur als Diagnose. Offenlegung: Ursprünglich war der Wilson-Score als Hauptmaß und die Positive Ratio als Robustheitsmaß vorgesehen. Die Rollen wurden nach der Diagnose der Volumenabhängigkeit getauscht (Abschnitt 3.6), also datengeleitet. Beide Maße werden vollständig berichtet. Die Rollen aller Variablen stehen in `variable_overview.csv`.
""")
    return "\n".join(out)


def section2(D, C) -> str:
    out = ["## 2. Deskriptive Ergebnisse\n"]
    # ---- 2.1 Besitzer
    ow = D.owners.sort_values("owners_class")
    ow_tbl = pd.DataFrame({
        "Besitzerklasse": ow["owners_label"],
        "untere Grenze": ow["owners_low"].map(n0),
        "N": ow["N"].map(n0),
        "Anteil": ow["Anteil"].map(lambda v: pct(v, 2)),
        "Anteil mindestens dieser Klasse": ow["Anteil_mindestens"].map(lambda v: pct(v, 2)),
    })
    mio = ow[ow["owners_low"] == 1_000_000].iloc[0]
    top = ow.iloc[-1]
    out.append(f"""### 2.1 Verteilung der Besitzer

{md(ow_tbl)}

Die Verteilung ist extrem rechtsschief: {pct(ow.iloc[0]['Anteil'])} der Spiele liegen in der untersten Klasse, mindestens eine Million Besitzer erreichen {pct(mio['Anteil_mindestens'], 2)}, die oberste Klasse ({top['owners_label']}) {n0(top['N'])} Spiele. Mediane und Mittelwerte aus Klassenmittelpunkten sind deshalb wenig informativ. Gruppen werden über die Anteile mit mindestens {n0(D.thr.loc['success_top25', 'Schwelle'])}, 100.000 und 1 Mio. Besitzern verglichen.

{fig('fig_01_owners_distribution', 'Häufigkeit der Besitzerklassen (links) und Anteil der Spiele mit mindestens der jeweiligen Klasse (CCDF, rechts).')}
""")

    # ---- 2.2 Kohorten
    co = D.cohort.reset_index(drop=True)
    co_tbl = pd.DataFrame({
        "Jahr": co["Jahr"].astype(int).astype(str),
        "Kohorte": co["Kohortengruppe"].str.replace("-", "–"),
        "N": co["N"].map(n0),
        "Erfolgsquote [95-%-KI]": [rate_cell(r, lo, hi) for r, lo, hi in zip(co["Erfolgsquote"], co["CI_unten"], co["CI_oben"])],
        "≥ 100k": co["Anteil_ab_100k"].map(pct),
        "kostenlos": co["Anteil_kostenlos"].map(pct),
        "Rabatt (kostenpfl.)": co["Anteil_Rabatt_kostenpflichtig"].map(pct),
        "N „0 - 0“": co["N_0_0"].map(n0),
        "Anteil „0 - 0“": co["Anteil_0_0"].map(pct),
        "Erfolgsquote mit „0 - 0“": co["Erfolgsquote_mit_0_0"].map(pct),
        "Hinweis": co["Hinweis"].map(de_text),
    })
    cy = D.cohort[D.cohort["N"] >= D.c02["MIN_YEAR_N"]]
    peak_year = int(cy["Erfolgsquote"].idxmax())
    last = STEAMSPY_CUTOFF.year
    recent = D.cohort.loc[2020:last, "Erfolgsquote"].to_numpy()
    check((np.diff(recent) < 0).all(), "Erfolgsquote sinkt ab 2020 mit jedem Jahrgang")
    fz = D.free_zero
    check(fz.loc["raw", "Anteil_kostenlos"] > fz.loc["with_owners_zero", "Anteil_kostenlos"]
          > fz.loc["analytic", "Anteil_kostenlos"], "Anteil kostenlos: Rohdaten > mit '0 - 0' > Analysestichprobe")
    age_s =D.desc_succ[D.desc_succ["Variable"] == "age_years"].set_index("Gruppe")["Median"]
    check(age_s.iloc[0] > age_s.iloc[1], "erfolgreiche Spiele älter")
    out.append(f"""### 2.2 Veröffentlichungskohorten und Alter

{md(co_tbl)}

Die Erfolgsquote hängt stark vom Jahrgang ab: Sie erreicht {pct(D.cohort.loc[peak_year, 'Erfolgsquote'])} im Jahrgang {peak_year} (unter Jahrgängen mit mindestens {D.c02['MIN_YEAR_N']} Spielen) und sinkt seit 2020 mit jedem Jahrgang bis auf {pct(D.cohort.loc[last, 'Erfolgsquote'])} im Jahrgang {last} (nur Januar bis Mai). Erfolgreiche Spiele sind im Median {de(age_s.iloc[0], 1)} Jahre alt, nicht erfolgreiche {de(age_s.iloc[1], 1)} Jahre (`descriptives_by_success.csv`). Der Anteil der Einträge mit „0 - 0“ steigt ab 2021 sprunghaft. Zählt man sie als Nicht-Erfolg, sinkt die Gesamtquote von {pct(fz.loc['analytic', 'Erfolgsquote'])} auf {pct(fz.loc['with_owners_zero', 'Erfolgsquote'])}. Auch der Anteil kostenloser Spiele hängt an diesem Ausschluss: Unter den Spielen, die alle übrigen Kriterien erfüllen, sind mit den „0 - 0“-Einträgen {pct(fz.loc['with_owners_zero', 'Anteil_kostenlos'])} kostenlos, ohne sie (Analysestichprobe) {pct(fz.loc['analytic', 'Anteil_kostenlos'])}; von den „0 - 0“-Einträgen selbst sind {pct(fz.loc['owners_zero_only', 'Anteil_kostenlos'])} kostenlos. Nur dieser Schritt geht auf den „0 - 0“-Ausschluss zurück. Der höhere Anteil im Rohdatensatz ({pct(fz.loc['raw', 'Anteil_kostenlos'])}) lässt sich ihm dagegen nicht zuordnen, weil der Rohdatensatz zusätzlich Nicht-Spiele, Software, Releases ab dem Stichtag und Einträge der SteamSpy-Lücke enthält. Aussagen zu Kohorten, zur Erfolgsquote und zu kostenlosen Spielen gelten deshalb nur bedingt auf die Spiele mit Besitzerangabe.

Damit Gruppen mit unterschiedlicher Altersstruktur vergleichbar werden, weist `age_standardized_rates.csv` neben der rohen auch eine **altersstandardisierte Erfolgsquote** aus (direkte Standardisierung auf die Jahrgangsverteilung der Analysestichprobe; Schichten: bis {D.c02['STD_FIRST_YEAR']}, danach Einzeljahre) sowie Quoten innerhalb der Kohorten. Für Gruppen unter {D.c02['MIN_CELL_N']} Spielen bleibt die standardisierte Quote leer. Die deskriptiven Kohorten ({', '.join(lab.replace('-', '–') for lab in D.c02['COHORT_LABELS'])}) weichen von den vier Kohorten der Tests ab (Abschnitt 3.0).

{fig('fig_02_release_cohorts', 'Spiele je Veröffentlichungsjahr und Erfolgsquote mit 95-%-KI; gestrichelt die Quote, wenn „0 - 0“-Einträge als Nicht-Erfolg zählen. Jahrgang ' + str(last) + ' nur Januar bis Mai.')}
""")

    # ---- 2.3 Genres
    g = D.genres.reset_index(drop=True)
    g_tbl = pd.DataFrame({
        "Genre": g["Genre"],
        "N": g["N"].map(n0),
        "Anteil": g["Anteil_an_Analysestichprobe"].map(pct),
        "Erfolgsquote [95-%-KI]": [rate_cell(r, lo, hi) for r, lo, hi in zip(g["Erfolgsquote"], g["CI_unten"], g["CI_oben"])],
        "altersstandardisiert [95-%-KI]": [rate_cell(r, lo, hi) for r, lo, hi in zip(g["Erfolgsquote_standardisiert"], g["CI_unten_std"], g["CI_oben_std"])],
        "≥ 100k": g["Anteil_ab_100k"].map(pct),
        "≥ 1 Mio.": g["Anteil_ab_1Mio"].map(pct),
        "mittl. Alter (J.)": g["Mittleres_Alter"].map(lambda v: de(v, 1)),
        "Median Listenpreis (kostenpfl., USD)": g["Median_Listenpreis_kostenpflichtig"].map(lambda v: de(v, 2)),
        "kostenlos": g["Anteil_kostenlos"].map(pct),
        "F2P-Tag": g["Anteil_F2P_Tag"].map(pct),
    })
    gs = D.genres.sort_values("Erfolgsquote", ascending=False)
    below = gs[gs["Erfolgsquote"] < C.succ]["Genre"].tolist()
    std_rank = D.genres.sort_values("Erfolgsquote_standardisiert", ascending=False).index.tolist()
    check(std_rank[0] == gs.index[0] and std_rank[-1] == gs.index[-1], "Standardisierung ändert Spitze und Schluss nicht")
    max_shift = (D.genres["Erfolgsquote_standardisiert"] - D.genres["Erfolgsquote"]).abs().max()
    check(D.genres["Anteil_kostenlos"].idxmax() == "Massively Multiplayer", "MMO mit höchstem Anteil kostenloser Spiele")
    check(D.genre_cohort["Genre"].nunique() == len(D.genres), "fig_09 enthält alle Genres")
    check(D.genre_cohort.loc[D.genre_cohort["N"] < D.c02["MIN_CELL_N"], "maskiert"].map(is_true).all(),
          "fig_09 maskiert alle Zellen unter der Mindestgröße")
    out.append(f"""### 2.3 Steam-Genres

{md(g_tbl)}

Die höchsten Erfolgsquoten haben {gs.index[0]} ({pct(gs.iloc[0]['Erfolgsquote'])}), {gs.index[1]} und {gs.index[2]}. Unter der Gesamtquote von {pct(C.succ)} liegen {', '.join(below[:-1])} und {below[-1]}. Die Altersstandardisierung verschiebt die Quoten um höchstens {de(max_shift * 100, 1)} Prozentpunkte; Spitze ({std_rank[0]}) und Schluss ({std_rank[-1]}, standardisiert {pct(D.genres.loc[std_rank[-1], 'Erfolgsquote_standardisiert'])}) bleiben gleich. Massively Multiplayer hat mit {pct(D.genres.loc['Massively Multiplayer', 'Anteil_kostenlos'])} den höchsten Anteil kostenloser Spiele.

{fig('fig_03_genre_success', 'Erfolgsquote nach Steam-Genre mit 95-%-KI, roh und altersstandardisiert.')}

{fig('fig_09_genre_cohort', f"Erfolgsquote nach Genre und Veröffentlichungskohorte für alle {len(D.genres)} Genres; Zellen mit n < {D.c02['MIN_CELL_N']} grau maskiert, n annotiert.")}
""")

    # ---- 2.4 Modi und Plattformen
    m = D.modes[D.modes["Merkmal"].isin(["mode", "n_platforms"])]
    m_tbl = pd.DataFrame({
        "Merkmal": m["Merkmal"].map({"mode": "Spielmodus", "n_platforms": "Anzahl Plattformen"}),
        "Ausprägung": m["Ausprägung"].astype(str),
        "N": m["N"].map(n0),
        "Erfolgsquote [95-%-KI]": [rate_cell(r, lo, hi) for r, lo, hi in zip(m["Erfolgsquote"], m["CI_unten"], m["CI_oben"])],
        "altersstandardisiert": m["Erfolgsquote_standardisiert"].map(pct),
        "≥ 100k": m["Anteil_ab_100k"].map(pct),
        "≥ 1 Mio.": m["Anteil_ab_1Mio"].map(pct),
        "mittl. Alter (J.)": m["Mittleres_Alter"].map(lambda v: de(v, 1)),
    })
    cb = D.combo
    cohort_cols = [c for c in cb.columns if c.startswith("Quote_")]
    cb_tbl = pd.DataFrame({
        "Kombination": cb["Kombination"],
        "N": cb["N"].map(n0),
        "Anteil": cb["Anteil_an_Analysestichprobe"].map(pct),
        "Erfolgsquote [95-%-KI]": [rate_cell(r, lo, hi) for r, lo, hi in zip(cb["Erfolgsquote"], cb["CI_unten"], cb["CI_oben"])],
        "altersstandardisiert": cb["Erfolgsquote_standardisiert"].map(pct),
        "mittl. Alter (J.)": cb["Mittleres_Alter"].map(lambda v: de(v, 1)),
        **{f"Quote {c.replace('Quote_', '').replace('bis_', 'bis ').replace('_', '–')}": [
            f"{pct(q)} (n = {n0(n)})" if not _isna(q) else f"– (n = {n0(n)})"
            for q, n in zip(cb[c], cb[c.replace('Quote_', 'N_')])] for c in cohort_cols},
    })
    md_mode = D.modes[D.modes["Merkmal"] == "mode"].set_index("Ausprägung")
    check(md_mode.loc["Single + Multiplayer", "Erfolgsquote"] > md_mode.loc["Nur Singleplayer", "Erfolgsquote"],
          "Single + Multiplayer erfolgreicher als nur Singleplayer")
    check(cb.loc["W+L", "Erfolgsquote"] < cb.loc["W", "Erfolgsquote"]
          and cb.loc["W+L", "Erfolgsquote_standardisiert"] < cb.loc["W", "Erfolgsquote_standardisiert"],
          "W+L unter nur Windows (roh und standardisiert)")
    recent_cols = [c for c in cohort_cols
                   if c.split("_")[1].isdigit() and int(c.split("_")[1]) >= 2015]
    for c in recent_cols:
        check(cb.loc["W+M", c] > cb.loc["W", c] and cb.loc["W+M+L", c] > cb.loc["W", c],
              f"W+M und W+M+L über nur Windows in {c}")
    wl_below = [c.replace("Quote_", "").replace("_", "–") for c in recent_cols if cb.loc["W+L", c] < cb.loc["W", c]]
    wml_gap_raw = cb.loc["W+M+L", "Erfolgsquote"] - C.succ
    wml_gap_std = cb.loc["W+M+L", "Erfolgsquote_standardisiert"] - C.succ
    check(wml_gap_std < wml_gap_raw / 2, "Standardisierung verkleinert den Vorsprung von W+M+L deutlich")
    out.append(f"""### 2.4 Spielmodi und Plattformen

{md(m_tbl)}

Plattformkombinationen (`platform_combo_overview.csv`; Quoten auch innerhalb der deskriptiven Kohorten):

{md(cb_tbl)}

Single- plus Multiplayer-Spiele ({pct(md_mode.loc['Single + Multiplayer', 'Erfolgsquote'])}) und reine Multiplayer-Spiele ({pct(md_mode.loc['Nur Multiplayer', 'Erfolgsquote'])}) sind häufiger erfolgreich als reine Singleplayer-Spiele ({pct(md_mode.loc['Nur Singleplayer', 'Erfolgsquote'])}). Die Kategorie „Singleplayer“ ist mit großem Abstand der Normalfall. Bei den Plattformen unterscheidet sich das Bild nach Kombination: Windows + macOS ({pct(cb.loc['W+M', 'Erfolgsquote'])}) und Windows + macOS + Linux ({pct(cb.loc['W+M+L', 'Erfolgsquote'])}) liegen über „nur Windows“ ({pct(cb.loc['W', 'Erfolgsquote'])}), Windows + Linux ohne macOS liegt darunter ({pct(cb.loc['W+L', 'Erfolgsquote'])}; standardisiert {pct(cb.loc['W+L', 'Erfolgsquote_standardisiert'])} gegen {pct(cb.loc['W', 'Erfolgsquote_standardisiert'])}). Innerhalb der Kohorten ab 2015 liegen W+M und W+M+L durchgehend über „nur Windows“, W+L dagegen in {len(wl_below)} von {len(recent_cols)} Kohorten darunter ({', '.join(wl_below)}). Spiele mit drei Plattformen sind im Mittel deutlich älter ({de(cb.loc['W+M+L', 'Mittleres_Alter'], 1)} gegen {de(cb.loc['W', 'Mittleres_Alter'], 1)} Jahre): Die Altersstandardisierung verringert den Vorsprung von W+M+L vor der Gesamtquote von {de(wml_gap_raw * 100, 1)} auf {de(wml_gap_std * 100, 1)} Prozentpunkte. Für „ohne Windows“ (n = {n0(cb.loc['ohne Windows', 'N'])}) wird keine standardisierte Quote berechnet.

{fig('fig_04_modes_platforms', 'Erfolgsquote nach Spielmodus und Plattformkombination mit 95-%-KI, roh und altersstandardisiert.')}
""")

    # ---- 2.5 Preis, kostenlos, Rabatt
    pr = D.prices
    pr_tbl = pd.DataFrame({
        "Preisklasse (Listenpreis, USD)": pr["Preisklasse"],
        "N": pr["N"].map(n0),
        "Anteil": pr["Anteil_an_Analysestichprobe"].map(pct),
        "Erfolgsquote [95-%-KI]": [rate_cell(r, lo, hi) for r, lo, hi in zip(pr["Erfolgsquote"], pr["CI_unten"], pr["CI_oben"])],
        "altersstandardisiert": pr["Erfolgsquote_standardisiert"].map(pct),
        "≥ 100k": pr["Anteil_ab_100k"].map(pct),
        "≥ 1 Mio.": pr["Anteil_ab_1Mio"].map(pct),
        "mittl. Alter (J.)": pr["Mittleres_Alter"].map(lambda v: de(v, 1)),
        "N nach Tagespreis": pr["N_Tagespreisklasse"].map(n0),
        "Erfolgsquote nach Tagespreis": pr["Erfolgsquote_Tagespreisklasse"].map(pct),
        "Hinweis": pr["Hinweis"].map(de_text),
    })
    paid_keys = ["p0_5", "p5_10", "p10_20", "p20_40", "p40plus"]
    # Klasse 40+ ohne Preisausreisser und ohne 54,99-Cluster (beide eigene Zeilen).
    check((np.diff(pr.loc[paid_keys, "Erfolgsquote"].to_numpy()) > 0).all(), "Erfolgsquote steigt bis 40+")
    check(pr.loc["p40plus", "CI_unten"] > pr.loc["p20_40", "CI_oben"],
          "40+ (ohne Ausreißer und Cluster) gesichert über 20–39,99 (kein Plateau)")
    check(pr.loc["p40plus", "Erfolgsquote_standardisiert"] > pr.loc["p20_40", "Erfolgsquote_standardisiert"]
          and pr.loc["p40plus", "Anteil_ab_1Mio"] > pr.loc["p20_40", "Anteil_ab_1Mio"],
          "40+ auch standardisiert und bei ≥ 1 Mio. über 20–39,99")
    cur_rates = pr.loc[paid_keys, "Erfolgsquote_Tagespreisklasse"].to_numpy()
    check((np.diff(cur_rates) > 0).all() and cur_rates[-1] - cur_rates[0] < pr.loc["p40plus", "Erfolgsquote"] - pr.loc["p0_5", "Erfolgsquote"],
          "nach Tagespreis flacherer, aber monotoner Anstieg")
    comp = D.p40
    comp_tbl = pd.DataFrame({
        "Teilgruppe": comp["Teilgruppe"],
        "N": comp["N"].map(n0),
        "Anteil an Klasse": comp["Anteil_an_Klasse"].map(pct),
        "Erfolgsquote [95-%-KI]": [rate_cell(r, lo, hi) for r, lo, hi in zip(comp["Erfolgsquote"], comp["CI_unten"], comp["CI_oben"])],
        "≥ 1 Mio.": comp["Anteil_ab_1Mio"].map(pct),
        "Median Reviews": comp["Median_Reviews"].map(lambda v: de(v, 0)),
        "Release 2024/25": comp["Anteil_Release_2024_2025"].map(pct),
        "Publisher mit ≥ 3 Titeln": comp["Anteil_Publisher_ab_3_Titeln"].map(pct),
        "Anteil größter Publisher": comp["Anteil_groesster_Publisher"].map(pct),
    })
    check(comp.loc["rest", "Erfolgsquote"] > comp.loc["all", "Erfolgsquote"] > comp.loc["cluster_5499", "Erfolgsquote"],
          "Ausreißer und 54,99-Cluster drücken die Quote der Klasse 40+")
    check(int(comp.loc["rest", "N"]) == int(pr.loc["p40plus", "N"])
          and int(comp.loc["cluster_5499", "N"]) == int(pr.loc["cluster_5499", "N"]),
          "Zusammensetzung und Preisübersicht passen zusammen")
    with_cluster = comp.loc["no_outlier"]
    check(covers(with_cluster["CI_unten"], with_cluster["CI_oben"], pr.loc["p20_40", "Erfolgsquote"]),
          "mit Cluster läge 40+ auf dem Niveau von 20–39,99")
    rho_nc, rho_main = T(D, "h2_price_rho_nocluster_partial"), T(D, "h2_price_rho_partial")
    check(abs(rho_nc - rho_main) < 0.01, "Rangkorrelation ohne Cluster praktisch unverändert")
    cl_label = f"{de(D.cc['CLUSTER_PRICE'], 2)}-USD-Cluster"
    fr = D.free
    fr_tbl = pd.DataFrame({
        "Gruppe": fr["Gruppe"],
        "N": fr["N"].map(n0),
        "Anteil": fr["Anteil_an_Analysestichprobe"].map(pct),
        "Erfolgsquote [95-%-KI]": [rate_cell(r, lo, hi) for r, lo, hi in zip(fr["Erfolgsquote"], fr["CI_unten"], fr["CI_oben"])],
        "altersstandardisiert [95-%-KI]": [rate_cell(r, lo, hi) for r, lo, hi in zip(fr["Erfolgsquote_standardisiert"], fr["CI_unten_std"], fr["CI_oben_std"])],
        "≥ 100k": fr["Anteil_ab_100k"].map(pct),
        "≥ 1 Mio.": fr["Anteil_ab_1Mio"].map(pct),
        "mittl. Alter (J.)": fr["Mittleres_Alter"].map(lambda v: de(v, 1)),
    })
    check(fr.loc["free_f2p", "Erfolgsquote_standardisiert"] > fr.loc["free_no_f2p", "Erfolgsquote_standardisiert"],
          "kostenlos mit F2P-Tag erfolgreicher als ohne")
    di = D.disc
    di_tbl = pd.DataFrame({
        "Rabattgruppe": di["Rabattgruppe"],
        "N": di["N"].map(n0),
        "Anteil an Basis": di["Anteil_an_Basis"].map(pct),
        "Erfolgsquote [95-%-KI]": [rate_cell(r, lo, hi) for r, lo, hi in zip(di["Erfolgsquote"], di["CI_unten"], di["CI_oben"])],
        "altersstandardisiert [95-%-KI]": [rate_cell(r, lo, hi) for r, lo, hi in zip(di["Erfolgsquote_standardisiert"], di["CI_unten_std"], di["CI_oben_std"])],
        "≥ 100k": di["Anteil_ab_100k"].map(pct),
        "≥ 1 Mio.": di["Anteil_ab_1Mio"].map(pct),
        "mittl. Alter (J.)": di["Mittleres_Alter"].map(lambda v: de(v, 1)),
    })
    groups = ["d1_25", "d26_50", "d51_75", "d76_100"]
    raw_g = di.loc[groups, "Erfolgsquote"]
    std_g = di.loc[groups, "Erfolgsquote_standardisiert"]
    check((np.diff(raw_g.to_numpy()) > 0).all(), "roher Rabattgradient monoton")
    check(std_g.min() - di.loc["d0", "Erfolgsquote_standardisiert"] > 2 * (std_g.max() - std_g.min()),
          "standardisiert: Sprung bei Rabatt ja/nein, danach Plateau")
    ages = di.loc[groups, "Mittleres_Alter"]
    check((np.diff(ages.to_numpy()) > 0).all(), "Alter steigt mit der Rabattstufe")
    age_no, age_any = di.loc["d0", "Mittleres_Alter"], di.loc["d_any", "Mittleres_Alter"]
    check(abs(age_no - age_any) < 0.2, "rabattierte und nicht rabattierte Spiele im Mittel gleich alt")
    check(not (np.diff(std_g.to_numpy()) > 0).all(), "standardisiert kein monotoner Gradient")
    disc_labels = dict(zip(groups, ["1–25 %", "26–50 %", "51–75 %", "76–100 %"]))
    sep_pairs = [(a, b) for i, a in enumerate(groups) for b in groups[i + 1:]
                 if di.loc[a, "CI_oben_std"] < di.loc[b, "CI_unten_std"] or di.loc[b, "CI_oben_std"] < di.loc[a, "CI_unten_std"]]
    if sep_pairs:
        a, b = sep_pairs[0]
        sep_text = (f"Zwischen einzelnen Stufen bestehen kleine, gesicherte Unterschiede (z. B. {disc_labels[a]} "
                    f"{kip(di.loc[a, 'CI_unten_std'], di.loc[a, 'CI_oben_std'])} gegen {disc_labels[b]} "
                    f"{kip(di.loc[b, 'CI_unten_std'], di.loc[b, 'CI_oben_std'])}, nicht überlappende Konfidenzintervalle), "
                    "aber kein monotoner Gradient über die Rabatthöhe.")
    else:
        sep_text = "Die Stufen unterscheiden sich nicht gesichert; ein monotoner Gradient über die Rabatthöhe zeigt sich nicht."
    paid = di.loc["paid_all"]
    check(int(paid["N"]) == int(T(D, "h2_discount_or", "N")), "Rabatttabelle und Rabatttests haben dieselbe Basis")
    ph0 = (D.dq.loc["ph_disc0_le90"], D.dq.loc["ph_disc0_above"], D.dq.loc["ph_disc_list_eq_base"])
    check(ph0[0]["Anteil"] > 0.5 and ph0[2]["Anteil"] > 0.8, "Rabatt 0 meist unter Basispreis, rekonstruierter Listenpreis meist = Basispreis")
    out.append(f"""### 2.5 Preis, kostenlose Spiele und Rabatte

Preisklassen nach dem rekonstruierten **Listenpreis**. Ergänzend sind die Fallzahl und die Erfolgsquote angegeben, wenn man die Klassen nach dem Tagespreis bildet:

{md(pr_tbl)}

Preisausreißer über 100 USD und das {cl_label} (Listenpreis exakt {de(D.cc['CLUSTER_PRICE'], 2)} USD, Release ab {D.cc['CLUSTER_FROM_YEAR']}; Zusammensetzung unten) sind eigene Zeilen; die Klasse 40+ enthält beide nicht. Unter den kostenpflichtigen Spielen **steigt die Erfolgsquote mit dem Listenpreis durchgehend**: von {pct(pr.loc['p0_5', 'Erfolgsquote'])} (0,01–4,99 USD) über {pct(pr.loc['p20_40', 'Erfolgsquote'])} (20–39,99 USD) auf {pct(pr.loc['p40plus', 'Erfolgsquote'])} in der Klasse 40+ (Konfidenzintervalle überlappen nicht; altersstandardisiert {pct(pr.loc['p40plus', 'Erfolgsquote_standardisiert'])} gegen {pct(pr.loc['p20_40', 'Erfolgsquote_standardisiert'])}, mindestens 1 Mio. Besitzer {pct(pr.loc['p40plus', 'Anteil_ab_1Mio'])} gegen {pct(pr.loc['p20_40', 'Anteil_ab_1Mio'])}). Einen Einbruch oder ein Plateau bei teuren Titeln gibt es also nicht. Ein Plateau entstünde nur, wenn man das Cluster in der Klasse ließe: Dann läge sie bei {pct(with_cluster['Erfolgsquote'])} {kip(with_cluster['CI_unten'], with_cluster['CI_oben'])} und damit auf dem Niveau von 20–39,99. Nach dem Tagespreis gebildet steigt die Quote ebenfalls, aber flacher ({pct(cur_rates[0])} bis {pct(cur_rates[-1])}), weil rabattierte Vollpreistitel, die häufiger erfolgreich sind, dann in niedrigere Klassen fallen. Spiele mit Listenpreis über 100 USD sind fast nie erfolgreich ({pct(pr.loc['outlier', 'Erfolgsquote'])}, n = {n0(pr.loc['outlier', 'N'])}), die Titel des Clusters nie ({pct(pr.loc['cluster_5499', 'Erfolgsquote'])}, n = {n0(pr.loc['cluster_5499', 'N'])}). Die Tests in Abschnitt 3.2 rechnen auf allen kostenpflichtigen Spielen ohne Preisausreißer, also einschließlich des Clusters (dort Klasse 40+ mit Cluster); ohne Cluster ändert sich die partielle Rangkorrelation kaum ({de(rho_nc, 3)} statt {de(rho_main, 3)}).

Zusammensetzung der Klasse 40+ einschließlich Ausreißern (`price_40plus_composition.csv`):

{md(comp_tbl)}

Von den {n0(comp.loc['all', 'N'])} Spielen mit Listenpreis ab 40 USD sind {pct(comp.loc['outlier', 'Anteil_an_Klasse'])} Preisausreißer über 100 USD ({pct(comp.loc['outlier', 'Anteil_groesster_Publisher'])} davon von einem einzigen Publisher). Weitere {pct(comp.loc['cluster_5499', 'Anteil_an_Klasse'])} bilden ein Cluster mit Listenpreis exakt {de(D.cc['CLUSTER_PRICE'], 2)} USD und Release ab {D.cc['CLUSTER_FROM_YEAR']}: Kein Titel davon ist erfolgreich, der Median liegt bei {de(comp.loc['cluster_5499', 'Median_Reviews'], 0)} Review(s), und {pct(comp.loc['cluster_5499', 'Anteil_Publisher_ab_3_Titeln'])} stammen von Publishern mit mindestens drei Titeln. Die übrigen Titel zwischen 40 und 100 USD sind zu {pct(comp.loc['rest', 'Erfolgsquote'])} erfolgreich.

Kostenlose Spiele nach Steam-Tag „Free To Play“ (`free_overview.csv`):

{md(fr_tbl)}

Kostenlose Spiele mit F2P-Tag sind häufiger erfolgreich ({pct(fr.loc['free_f2p', 'Erfolgsquote'])}; standardisiert {pct(fr.loc['free_f2p', 'Erfolgsquote_standardisiert'])}) als kostenlose Spiele ohne Tag ({pct(fr.loc['free_no_f2p', 'Erfolgsquote'])}; standardisiert {pct(fr.loc['free_no_f2p', 'Erfolgsquote_standardisiert'])}). Letztere liegen standardisiert nahe bei den kostenpflichtigen Spielen ({pct(fr.loc['paid_all', 'Erfolgsquote_standardisiert'])}).

**Rabatte, nur kostenpflichtige Spiele** (Listenpreis > 0, ohne Preisausreißer; dieselbe Basis wie die Tests in Abschnitt 3.2; Erfolgsquote dieser Spiele insgesamt: {pct(paid['Erfolgsquote'])}; Analysestichprobe insgesamt: {pct(C.succ)}). Der Rabatt ist der Stand zum Store-Snapshot und teils nicht erfasst, also eine Untergrenze. Wie unvollständig er erfasst ist, zeigt die Preishistorie-Teilstichprobe: Von {n0(ph0[0]['N_Basis'])} kostenpflichtigen Spielen mit Rabatt 0 zum Snapshot kosten {n0(ph0[0]['Wert'])} ({pct(ph0[0]['Anteil'])}) höchstens 90 % des Basispreises der Preishistorie und {'keines' if int(ph0[1]['Wert']) == 0 else n0(ph0[1]['Wert'])} mehr als diesen. Bei den rabattierten Spielen stimmt der rekonstruierte Listenpreis dagegen in {n0(ph0[2]['Wert'])} von {n0(ph0[2]['N_Basis'])} Fällen ({pct(ph0[2]['Anteil'])}) auf ± 0,02 USD mit dem Basispreis überein. Fehlende Rabatte ordnen rabattierte Spiele der Gruppe „kein Rabatt“ zu und unterschätzen ihren Listenpreis; das verzerrt den Rabattzusammenhang eher zur Null. Die Teilstichprobe ist stark selektiert (fast nur erfolgreiche Spiele), der Anteil lässt sich deshalb nicht auf alle Spiele übertragen; zudem können dauerhafte Preissenkungen nach Oktober 2025 einen Teil erklären:

{md(di_tbl)}

Roh steigt die Erfolgsquote mit der Rabatthöhe von {pct(di.loc['d0', 'Erfolgsquote'])} (kein Rabatt) auf {pct(di.loc['d76_100', 'Erfolgsquote'])} (76–100 %). Die Rabattstufen sind aber sehr unterschiedlich alt: Niedrige Rabatte betreffen eher jüngere Titel (1–25 %: mittleres Alter {de(ages.iloc[0], 1)} Jahre), hohe eher ältere (76–100 %: {de(ages.iloc[-1], 1)} Jahre). Im Mittel sind rabattierte und nicht rabattierte Spiele dagegen gleich alt ({de(age_any, 1)} gegen {de(age_no, 1)} Jahre). Altersstandardisiert bleibt vor allem der Unterschied zwischen „kein Rabatt“ ({pct(di.loc['d0', 'Erfolgsquote_standardisiert'])}) und „Rabatt“: Ab 1 % Rabatt liegen die Stufen eng beieinander ({pct(std_g.min())} bis {pct(std_g.max())}). {sep_text} {discount_cohort_pattern(D).long} Die Richtung des Zusammenhangs ist offen.

{fig('fig_05_price', f"Anteile der Besitzerklassen nach Listenpreisklasse (Klasse 40–100 ohne das {de(D.cc['CLUSTER_PRICE'], 2)}-USD-Cluster, Cluster und Preisausreißer als eigene Balken) sowie Erfolgsquote kostenloser Spiele mit und ohne F2P-Tag gegenüber kostenpflichtigen Spielen.")}

{fig('fig_06_discount', 'Erfolgsquote nach Rabattgruppe (nur kostenpflichtige Spiele), roh mit 95-%-KI und altersstandardisiert (links) sowie innerhalb der Veröffentlichungskohorten (rechts).')}
""")

    # ---- 2.6 Bewertungen
    ra = D.ratings
    ra_tbl = pd.DataFrame({
        "Bewertungsmaß": ra["Bewertungsmaß"],
        "Basis": ra["Basis"],
        "Quartil": ra["Quartil"],
        "N": ra["N"].map(n0),
        "Wertebereich": [f"{de(a, 3)}–{de(b, 3)}" if v != "mc_metascore" else f"{de(a, 0)}–{de(b, 0)}"
                         for a, b, v in zip(ra["Wert_min"], ra["Wert_max"], ra["Variable"])],
        "Erfolgsquote [95-%-KI]": [rate_cell(r, lo, hi) for r, lo, hi in zip(ra["Erfolgsquote"], ra["CI_unten"], ra["CI_oben"])],
        "≥ 100k": ra["Anteil_ab_100k"].map(pct),
        "≥ 1 Mio.": ra["Anteil_ab_1Mio"].map(pct),
        "mittl. Alter (J.)": ra["Mittleres_Alter"].map(lambda v: de(v, 1)),
    })
    prq = ra[ra["Variable"] == "positive_ratio"].reset_index(drop=True)
    wq = ra[ra["Variable"] == "wilson"].reset_index(drop=True)
    mq = ra[ra["Variable"] == "mc_metascore"].reset_index(drop=True)
    check((np.diff(wq["Erfolgsquote"].to_numpy()) > 0).all(), "Wilson-Quartile monoton")
    check((np.diff(mq["Erfolgsquote"].to_numpy()) > 0).all(), "Metascore-Quartile monoton")
    check(prq.loc[3, "Mittleres_Alter"] < prq.loc[0, "Mittleres_Alter"], "hohe Positive Ratio = jüngere Spiele")
    me = D.mc_ext
    me_tbl = pd.DataFrame({
        "Bewertungsmaß": me["Bewertungsmaß"],
        "Basis": me["Basis"],
        "Quartil": me["Quartil"],
        "N": me["N"].map(n0),
        "Wertebereich": [f"{de(a, 0)}–{de(b, 0)}" if v == "mc_metascore" else f"{de(a, 1)}–{de(b, 1)}"
                         for a, b, v in zip(me["Wert_min"], me["Wert_max"], me["Variable"])],
        "Erfolgsquote [95-%-KI]": [rate_cell(r, lo, hi) for r, lo, hi in zip(me["Erfolgsquote"], me["CI_unten"], me["CI_oben"])],
        "≥ 100k": me["Anteil_ab_100k"].map(pct),
        "≥ 1 Mio.": me["Anteil_ab_1Mio"].map(pct),
    })
    mus = me[me["Variable"] == "mc_user_score"]["Erfolgsquote"].to_numpy()
    check((np.diff(mus) > 0).all(), "MC-Nutzerscore-Quartile monoton")
    mc_cov = D.mc_cov
    mc_cov_tbl = pd.DataFrame({
        "Kennzahl": mc_cov["Kennzahl"],
        "Wert": [n0(v) if "(N)" in str(k) else pct(v) for k, v in zip(mc_cov["Kennzahl"], mc_cov["Wert"])],
    })
    corr_pr_year = D.corr[(D.corr["Matrix"] == "Steam") & (D.corr["Variable_1"] == "release_year")
                          & (D.corr["Variable_2"] == "positive_ratio")]["rho"].iloc[0]
    n_fig10 = int((D.mc["mc_metascore"].notna() & D.mc["rating_sample"] & D.mc["positive_ratio"].notna()).sum())
    check(n_fig10 == int(D.ident.loc["ident_mc_metascore_vs_positive_ratio", "N"]), "n in fig_10 = identische Stichprobe")
    succ_mc = D.sel.loc["sel_has_mc_external_gesamt", "Erfolgsquote_verfuegbar"]
    out.append(f"""### 2.6 Steam-Bewertungen und Metacritic

Bewertungsquartile (bindungstreu: gleiche Werte liegen immer im selben Quartil, die Gruppen sind deshalb nicht exakt gleich groß; Wertebereiche je Quartil angegeben):

{md(ra_tbl)}

Die **Positive Ratio** zeigt **keinen monotonen Verlauf**: Die mittleren Quartile sind am häufigsten erfolgreich ({pct(prq.loc[1, 'Erfolgsquote'])} und {pct(prq.loc[2, 'Erfolgsquote'])}), die Ränder seltener ({pct(prq.loc[0, 'Erfolgsquote'])} im untersten, {pct(prq.loc[3, 'Erfolgsquote'])} im obersten Quartil). Spiele mit sehr hoher Positive Ratio sind im Mittel jünger ({de(prq.loc[3, 'Mittleres_Alter'], 1)} gegen {de(prq.loc[0, 'Mittleres_Alter'], 1)} Jahre im untersten Quartil); neuere Jahrgänge werden besser bewertet (ρ(Jahr, Positive Ratio) = {de(corr_pr_year, 2)} in der Steam-Korrelationsmatrix). Der Wilson-Score steigt monoton ({pct(wq.loc[0, 'Erfolgsquote'])} bis {pct(wq.loc[3, 'Erfolgsquote'])}), weil er das Review-Volumen enthält; er dient nur als Diagnose (Abschnitt 3.6). Der externe Metascore steigt ebenfalls monoton ({pct(mq.loc[0, 'Erfolgsquote'])} bis {pct(mq.loc[3, 'Erfolgsquote'])}; Anteil mit mindestens 1 Mio. Besitzern {pct(mq.loc[0, 'Anteil_ab_1Mio'])} bis {pct(mq.loc[3, 'Anteil_ab_1Mio'])}). Die Metacritic-Teilstichprobe ist stark selektiert (Erfolgsquote {pct(succ_mc)}).

Externe Metacritic-Teilstichprobe, Kritiker- und Nutzerscore (Nutzerscore nur, wo vorhanden):

{md(me_tbl)}

{md(mc_cov_tbl)}

{fig('fig_07_ratings', f'Besitzer (Klassenmittelpunkt, log-Skala) gegen die Positive Ratio (≥ {MIN_REVIEWS} Reviews; Hexbin mit Anzahl Spiele, links) und gegen den externen Metascore (Streudiagramm, rechts).')}

{fig('fig_10_rating_agreement', f'Externer Metascore gegen Positive Ratio (≥ {MIN_REVIEWS} Reviews, n = {n0(n_fig10)}).')}

{fig('fig_13_external_metacritic', 'Metacritic-Nutzer- gegen -Kritikerscore sowie Anteile mit mindestens 20.000 bzw. 1 Mio. Besitzern je Metascore-Quartil.')}
""")

    # ---- 2.7 IGDB
    dx = D.desc_ext
    dx_tbl = pd.DataFrame({
        "Variable": dx["Variable"].map(lambda s: f"`{s}`"),
        "Basis": dx["Basis"],
        "N": dx["N"].map(n0),
        "Mittelwert": dx["Mittelwert"].map(lambda v: de(v, 2)),
        "SD": dx["SD"].map(lambda v: de(v, 2)),
        "Median": dx["Median"].map(lambda v: de(v, 2)),
        "Q1": dx["Q1"].map(lambda v: de(v, 2)),
        "Q3": dx["Q3"].map(lambda v: de(v, 2)),
        "Min": dx["Min"].map(lambda v: de(v, 2)),
        "Max": dx["Max"].map(lambda v: de(v, 2)),
        "Hinweis": dx["Hinweis"].map(de_text),
    })
    ig = D.igdb_genres
    ig_tbl = pd.DataFrame({
        "IGDB-Genre": ig["IGDB_Genre"],
        "N": ig["N"].map(n0),
        "Anteil an IGDB mit Genreangabe": ig["Anteil_an_IGDB_mit_Genreangabe"].map(pct),
        "Anteil an der IGDB-Teilstichprobe": ig["Anteil_an_IGDB_Teilstichprobe"].map(pct),
        "Erfolgsquote [95-%-KI]": [rate_cell(r, lo, hi) for r, lo, hi in zip(ig["Erfolgsquote"], ig["CI_unten"], ig["CI_oben"])],
        "≥ 100k": ig["Anteil_ab_100k"].map(pct),
        "≥ 1 Mio.": ig["Anteil_ab_1Mio"].map(pct),
        "mittl. Alter (J.)": ig["Mittleres_Alter"].map(lambda v: de(v, 1)),
        "Median Listenpreis (USD)": ig["Median_Listenpreis_kostenpflichtig"].map(lambda v: de(v, 2)),
    })
    im = D.igdb_modes
    im_tbl = pd.DataFrame({
        "IGDB-Merkmal": im["IGDB_Merkmal"],
        "Basis": im["Basis"],
        "N": im["N"].map(n0),
        "Anteil an Basis": im["Anteil_an_Basis"].map(pct),
        "Erfolgsquote [95-%-KI]": [rate_cell(r, lo, hi) for r, lo, hi in zip(im["Erfolgsquote"], im["CI_unten"], im["CI_oben"])],
        "≥ 100k": im["Anteil_ab_100k"].map(pct),
        "≥ 1 Mio.": im["Anteil_ab_1Mio"].map(pct),
        "N ohne Merkmal": im["N_ohne_Merkmal"].map(n0),
        "Erfolgsquote ohne Merkmal": im["Erfolgsquote_ohne_Merkmal"].map(pct),
    })
    check(im.loc["igdb_singleplayer", "Erfolgsquote"] < im.loc["igdb_singleplayer", "Erfolgsquote_ohne_Merkmal"],
          "IGDB-Singleplayer seltener erfolgreich als ohne Singleplayer (wie Steam)")
    check(im.loc["without_modes", "Erfolgsquote"] < im.loc["with_modes", "Erfolgsquote"], "ohne Modusangabe seltener erfolgreich")
    ag = D.agree
    ag_tbl = pd.DataFrame({
        "Merkmal": ag["Merkmal"],
        "IGDB": ag["IGDB"].map(lambda s: f"`{s}`"),
        "Steam": ag["Steam"].map(lambda s: f"`{s}`"),
        "Stichprobe": ag["Stichprobe"],
        "N": ag["N"].map(n0),
        "Anteil IGDB": ag["Anteil_IGDB"].map(pct),
        "Anteil Steam": ag["Anteil_Steam"].map(pct),
        "identisch klassifiziert": ag["Anteil_identisch"].map(pct),
        "Cohens κ": ag["Kappa"].map(lambda v: de(v, 2)),
        "PABAK": ag["PABAK"].map(lambda v: de(v, 2)),
    })
    iv = D.igdb_values
    iv_tbl = pd.DataFrame({
        "Gruppe": iv["Gruppe"],
        "N": iv["N"].map(n0),
        "Anteil an Nutzerbewertungen": iv["Anteil_an_Nutzerbewertungen"].map(pct),
        "Erfolgsquote [95-%-KI]": [rate_cell(r, lo, hi) for r, lo, hi in zip(iv["Erfolgsquote"], iv["CI_unten"], iv["CI_oben"])],
        "≥ 1 Mio.": iv["Anteil_ab_1Mio"].map(pct),
        "Median Reviews (Steam)": iv["Median_Reviews"].map(lambda v: de(v, 0)),
        "Median Bewertung": iv["Median_Bewertung"].map(lambda v: de(v, 1)),
    })
    ir = D.igdb_ratings
    ir_tbl = pd.DataFrame({
        "Bewertungsmaß": ir["Bewertungsmaß"],
        "Quartil": ir["Quartil"],
        "N": ir["N"].map(n0),
        "Wertebereich": [f"{de(a, 1)}–{de(b, 1)}" for a, b in zip(ir["Wert_min"], ir["Wert_max"])],
        "Erfolgsquote [95-%-KI]": [rate_cell(r, lo, hi) for r, lo, hi in zip(ir["Erfolgsquote"], ir["CI_unten"], ir["CI_oben"])],
        "≥ 1 Mio.": ir["Anteil_ab_1Mio"].map(pct),
        "ganzzahlig": ir["Anteil_ganzzahlig"].map(pct),
        "exakt 70": ir["Anteil_Wert_70"].map(pct),
        "Median Reviews (Steam)": ir["Median_Reviews"].map(lambda v: de(v, 0)),
    })
    user = ir[ir["Bewertungsmaß"] == "IGDB-Nutzerbewertung"].reset_index(drop=True)
    nonint = ir[ir["Bewertungsmaß"] == "IGDB-Nutzerbewertung (nicht ganzzahlig)"].reset_index(drop=True)
    crit = ir[ir["Bewertungsmaß"] == "IGDB-Kritikerbewertung"].reset_index(drop=True)
    check(user.loc[2, "Anteil_Wert_70"] > 0.9, "Q3 der IGDB-Nutzerbewertung ist der Block um 70")
    check(nonint["Erfolgsquote"].max() - nonint["Erfolgsquote"].min() < 0.1, "nicht ganzzahlige Quartile nahezu flach")
    check((np.diff(crit["Erfolgsquote"].to_numpy()) > 0).all(), "IGDB-Kritikerquartile monoton")
    out.append(f"""### 2.7 IGDB-Merkmale

**Deskriptive Statistik der externen Variablen (IGDB, Metacritic, Preishistorie)**, jeweils auf der angegebenen Teilstichprobe:

{md(dx_tbl)}

Die Anzahl der IGDB-Genres fehlt, wenn keine Genres angegeben sind ({pct(1 - D.dq.loc['igdb_has_genres', 'Anteil'])} der IGDB-Treffer). Sie misst neben der Genrebreite auch die Dokumentationstiefe eines Eintrags.

Erfolgsquote nach IGDB-Genre (exakte IGDB-Bezeichnungen; Basis sind die {n0(D.dq.loc['igdb_has_genres', 'Wert'])} IGDB-Treffer mit Genreangabe; Genres unter 100 Spielen haben breite Konfidenzintervalle):

{md(ig_tbl)}

Erfolgsquote nach IGDB-Spielmodus. Die Modus-Anteile beziehen sich auf die {n0(D.dq.loc['igdb_has_modes', 'Wert'])} Treffer mit Modusangabe ({pct(D.dq.loc['igdb_has_modes', 'Anteil'])} der IGDB-Teilstichprobe); ohne Angabe gelten die Modi als unbekannt, nicht als „nein“:

{md(im_tbl)}

Ohne Modusangabe ist die Erfolgsquote mit {pct(im.loc['without_modes', 'Erfolgsquote'])} deutlich niedriger als mit Angabe ({pct(im.loc['with_modes', 'Erfolgsquote'])}): Die Dokumentationstiefe selbst hängt mit der Popularität zusammen. Unter Spielen mit Modusangabe sind Singleplayer-Spiele seltener erfolgreich als Spiele ohne Singleplayer ({pct(im.loc['igdb_singleplayer', 'Erfolgsquote'])} gegen {pct(im.loc['igdb_singleplayer', 'Erfolgsquote_ohne_Merkmal'])}), also in derselben Richtung wie bei Steam. Multiplayer ({pct(im.loc['igdb_multiplayer', 'Erfolgsquote'])}), Koop ({pct(im.loc['igdb_coop', 'Erfolgsquote'])}) und MMO ({pct(im.loc['igdb_mmo', 'Erfolgsquote'])}, n = {n0(im.loc['igdb_mmo', 'N'])}) liegen darüber.

**Übereinstimmung mit Steam.** IGDB-Genres und -Modi sind keine unabhängige Quelle. Mit den Steam-Kategorien stimmen sie weitgehend überein (Cohens κ {de(C.kappa.min(), 2)} bis {de(C.kappa.max(), 2)}). Nur die näherungsweise Zuordnung „Action“ weicht ab (κ = {de(C.kappa_action.iloc[0], 2)}); sie geht nicht in die H4-Entscheidung ein. Die IGDB-Merkmale sind deshalb eine Reliabilitäts- und Matchingprüfung, keine Bestätigung aus einer zweiten Quelle:

{md(ag_tbl)}

**IGDB-Nutzerbewertung.** {pct(iv.loc['integer', 'Anteil_an_Nutzerbewertungen'])} der Nutzerbewertungen sind ganzzahlig, {pct(iv.loc['value_70', 'Anteil_an_Nutzerbewertungen'])} liegen exakt bei 70. Ganzzahlige Werte deuten auf sehr wenige Stimmen hin; die IGDB-Datei enthält keine Stimmenzahl. Entsprechend haben Spiele mit ganzzahliger Bewertung im Median {de(iv.loc['integer', 'Median_Reviews'], 0)} Steam-Reviews und sind zu {pct(iv.loc['integer', 'Erfolgsquote'])} erfolgreich, Spiele mit nicht ganzzahliger Bewertung {de(iv.loc['non_integer', 'Median_Reviews'], 0)} Reviews und {pct(iv.loc['non_integer', 'Erfolgsquote'])}:

{md(iv_tbl)}

{md(ir_tbl)}

Das dritte Quartil der Nutzerbewertung ist praktisch der Block um 70 ({de(user.loc[2, 'Wert_min'], 1)}–{de(user.loc[2, 'Wert_max'], 1)}; n = {n0(user.loc[2, 'N'])}, davon {pct(user.loc[2, 'Anteil_Wert_70'])} exakt 70, Erfolgsquote {pct(user.loc[2, 'Erfolgsquote'])}). Die Quartilsquoten ({' / '.join(pct(v) for v in user['Erfolgsquote'])}) bilden deshalb vor allem den Anteil ganzzahliger Bewertungen ab. Innerhalb der nicht ganzzahligen Bewertungen ist der Verlauf nahezu flach ({pct(nonint['Erfolgsquote'].min())} bis {pct(nonint['Erfolgsquote'].max())}). Die IGDB-Kritikerbewertung (Platzhalter 0 als fehlend) steigt monoton ({' / '.join(pct(v) for v in crit['Erfolgsquote'])}).

{fig('fig_11_igdb_ratings', 'Verteilung der IGDB-Nutzerbewertung nach Ganzzahligkeit, Erfolgsquote je Quartil getrennt nach ganzzahligen und nicht ganzzahligen Bewertungen sowie Quartile der IGDB-Kritikerbewertung.')}
""")

    # ---- 2.8 Preishistorie
    pm = D.ph_months
    bins = [("6–12 Monate", pm["Monate"] <= 12), ("13–22 Monate", (pm["Monate"] > 12) & (pm["Monate"] < 23)),
            ("23 Monate", pm["Monate"] == 23), ("24 Monate", pm["Monate"] == 24)]
    check(int(pm["Monate"].min()) >= 6 and int(pm["Monate"].max()) == 24, "Fensterlängen 6–24 Monate")
    rows = []
    for label, mask in bins:
        part = pm[mask]
        n = part["N"].sum()
        rows.append({
            "Fensterlänge": label,
            "Apps": n0(n),
            "Anteil": pct(part["Anteil"].sum()),
            "davon volles Fenster": n0(part["N_volles_Fenster"].sum()),
            "Erfolgsquote": pct((part["N"] * part["Erfolgsquote"]).sum() / n),
            "≥ 1 Mio.": pct((part["N"] * part["Anteil_ab_1Mio"]).sum() / n),
            "Median-Release-Jahr (Spanne)": f"{int(part['Median_Release_Jahr'].min())}–{int(part['Median_Release_Jahr'].max())}"
            if part["Median_Release_Jahr"].nunique() > 1 else str(int(part["Median_Release_Jahr"].iloc[0])),
            "erster Monat": f"{part['Erster_Monat_min'].min()} bis {part['Erster_Monat_max'].max()}",
        })
    pm_tbl = md(pd.DataFrame(rows))
    po = D.ph_overview
    po_tbl = pd.DataFrame({
        "Preismerkmal": po["Preismerkmal"],
        "Quartil": po["Quartil"],
        "N": po["N"].map(n0),
        "Wertebereich": [f"{de(a, 2)}–{de(b, 2)}" for a, b in zip(po["Wert_min"], po["Wert_max"])],
        "Erfolgsquote": po["Erfolgsquote"].map(pct),
        "≥ 100k": po["Anteil_ab_100k"].map(pct),
        "≥ 1 Mio.": po["Anteil_ab_1Mio"].map(pct),
        "mittl. Alter (J.)": po["Mittleres_Alter"].map(lambda v: de(v, 1)),
    })
    check(po["Erfolgsquote"].min() > 0.95, "Erfolgsquote im vollen Fenster überall nahe 100 %")

    def mio(feature):
        return po[po["Variable"] == feature]["Anteil_ab_1Mio"].to_numpy()

    mm, mx, pc = mio("ph_mean_monthly_max_discount_pct"), mio("ph_max_discount_pct"), mio("ph_price_changes_per_month")
    check((np.diff(mm) > 0).all() and (np.diff(mx) > 0).all(), "≥ 1 Mio. steigt mit mittlerem und maximalem Rabatt")
    check(not (np.diff(pc) > 0).all(), "≥ 1 Mio. nicht monoton in den Preisänderungen")
    n_full = int(D.dq.loc["ph_full_window", "Wert"])
    short = pm[pm["N_volles_Fenster"] == 0]
    full = pm[pm["N_volles_Fenster"] > 0]
    short_rate = (short["N"] * short["Erfolgsquote"]).sum() / short["N"].sum()
    full_rate = (full["N"] * full["Erfolgsquote"]).sum() / full["N"].sum()
    check(int(full["N"].sum()) == n_full and short_rate < full_rate, "kurze Fenster seltener erfolgreich")
    mx_age = po[po["Variable"] == "ph_max_discount_pct"]["Mittleres_Alter"].to_numpy()
    check(mx_age[-1] > mx_age[0], "hoher Maximalrabatt = ältere Spiele")
    out.append(f"""### 2.8 Preishistorie

Die Preishistorie umfasst {n0(C.n_price_history_sample)} Spiele der Analysestichprobe (Datenstand: „{D.prov.loc['price_history', 'Datenstand']}“). Die Beobachtungsfenster sind unterschiedlich lang (`price_history_months.csv`); nur {n0(n_full)} Apps ({pct(D.dq.loc['ph_full_window', 'Anteil'])}) haben das volle Fenster:

{pm_tbl}

Weil Summen über ungleich lange Fenster vor allem die Fensterlänge messen, werden nur **normierte Merkmale** verwendet: Preisänderungen je Monat, Anteil der Monate mit Rabatt, Rabatttage je Monat und das Mittel der monatlichen Höchstrabatte (aus dem monatlichen Tiefstpreis). Der maximale Rabatt wird nur im vollen Fenster verglichen. Quartile bei vollem Fenster (n = {n0(n_full)}):

{md(po_tbl)}

Die Erfolgsquote liegt in allen Quartilen bei {pct(po['Erfolgsquote'].min())} bis {pct(po['Erfolgsquote'].max())}; Unterschiede zeigen sich daher nur im Anteil mit mindestens 1 Mio. Besitzern. Dieser steigt mit dem mittleren monatlichen Maximalrabatt ({pct(mm[0])} bis {pct(mm[-1])}) und mit dem maximalen Rabatt ({pct(mx[0])} bis {pct(mx[-1])}), aber nicht monoton mit den Preisänderungen je Monat ({' / '.join(pct(v) for v in pc)}). Die Beschränkung auf das volle Fenster vermeidet ein Expositionsartefakt: Apps mit kurzem Fenster sind junge Releases und seltener erfolgreich ({pct(short_rate)} gegenüber {pct(full_rate)} bei vollem Fenster). Spiele mit hohem Rabatt sind zudem im Mittel deutlich älter (mittleres Alter im obersten Quartil des Maximalrabatts {de(mx_age[-1], 1)} gegen {de(mx_age[0], 1)} Jahre im untersten); Abschnitt 3.4 prüft die Zusammenhänge deshalb partiell für Jahr und Fensterlänge.

{fig('fig_12_price_history', f'Verteilung der Fensterlängen (links) und normierte Preismerkmale gegen Besitzer bei vollem Fenster (n = {n0(n_full)}).')}
""")

    # ---- 2.9 Korrelationen und Ueberlappung
    cr = D.corr
    target = cr[(cr["Variable_1"] == "log_owners") | (cr["Variable_2"] == "log_owners")].copy()
    target["Variable"] = np.where(target["Variable_1"] == "log_owners", target["Variable_2"], target["Variable_1"])
    cr_tbl = pd.DataFrame({
        "Matrix": target["Matrix"],
        "Basis": target["Basis"],
        "Variable": target["Variable"].map(lambda s: f"`{s}`"),
        "ρ mit log_owners": target["rho"].map(lambda v: de(v, 2)),
        "N": target["N"].map(n0),
    })
    steam = cr[cr["Matrix"] == "Steam"]

    def srho(a, b):
        part = steam[((steam["Variable_1"] == a) & (steam["Variable_2"] == b)) | ((steam["Variable_1"] == b) & (steam["Variable_2"] == a))]
        check(len(part) == 1, f"Steam-Korrelation {a}/{b}")
        return part.iloc[0]

    r_pr, r_w, r_lp = srho("log_owners", "positive_ratio"), srho("log_owners", "wilson"), srho("log_owners", "list_price")
    check(abs(r_pr["rho"]) < 0.05 and r_w["rho"] > 0.2, "Positive Ratio ≈ 0, Wilson positiv")
    r_peak = srho("log_owners", "log_peak_ccu")
    peak_only = cr[cr["Matrix"] == "Steam, nur Peak > 0"]
    check(len(peak_only) == 1, "Zusatzzeile Steam, nur Peak > 0")
    peak_only = peak_only.iloc[0]
    peak_zero_share = 1 - peak_only["N"] / r_peak["N"]
    check(peak_only["rho"] > r_peak["rho"], "Peak-Korrelation bei Peak > 0 höher")
    n_ext = int(D.overlap.loc[D.overlap["Anzahl_Datenquellen"] >= 2, "N"].sum())
    out.append(f"""### 2.9 Korrelationen und Überlappung der Datenquellen

Statt einer gemischten Gesamtmatrix gibt es vier Matrizen je Teilstichprobe (`correlation_spearman.csv`, mit N je Paar). Die Steam-Matrix nutzt die vollständigen Fälle der kostenpflichtigen Spiele der Bewertungsstichprobe ohne Preisausreißer, damit Listenpreis und Rabatt definiert sind (N = {n0(r_pr['N'])}); die Matrizen für IGDB, Metacritic und Preishistorie sind paarweise innerhalb ihrer Teilstichprobe berechnet. Dort ist ρ(Positive Ratio, Besitzer) = {de(r_pr['rho'], 2)}, ρ(Wilson, Besitzer) = {de(r_w['rho'], 2)} und ρ(Listenpreis, Besitzer) = {de(r_lp['rho'], 2)}. Peak CCU geht in die Steam-Matrix als log10(1 + Peak) einschließlich der Nullwerte ein ({pct(peak_zero_share)} der Fälle) und mischt damit Aktivität am Messtag und Intensität: ρ(Peak, Besitzer) beträgt dort {de(r_peak['rho'], 2)}, bei Beschränkung auf Peak > 0 dagegen {de(peak_only['rho'], 2)} (Zeile „Steam, nur Peak > 0“). Korrelationen mit den Besitzern:

{md(cr_tbl)}

{fig('fig_08_correlation_heatmap', 'Spearman-Korrelationen in vier Matrizen je Teilstichprobe (Steam: vollständige Fälle; IGDB, Metacritic, Preishistorie: paarweise innerhalb der Teilstichprobe).')}

Alle Spiele stammen aus dem Basisdatensatz; {pct(n_ext / C.n)} haben zusätzlich mindestens eine externe Quelle, {n0(C.n_multi_source_sample)} alle vier. Die quellenübergreifenden Analysen nutzen jeweils die größtmögliche Teilstichprobe.

{fig('fig_14_source_overlap', 'Abdeckung der Analysestichprobe durch die externen Quellen.')}
""")
    return "\n".join(out)


def section3(D, C) -> str:
    out = ["## 3. Statistische Tests\n"]
    c3 = D.c03
    # Konvergenz aller Logit-Schaetzungen
    for name, table in [("tests_all", D.tests_all), ("robustness", D.rob), ("h1_joint_model", D.joint),
                        ("h2_free_sensitivity", D.free_sens), ("release_date_sensitivity", D.release),
                        ("release_cohort_sensitivity", D.rc)]:
        conv = table["konvergiert"].dropna()
        check(len(conv) > 0 and conv.map(is_true).all(), f"alle Logits in {name} konvergiert")
    comp = C.comp
    comp_tbl = pd.DataFrame({
        "Hypothese": comp["Hypothese"],
        "Komponente": comp["Komponente"],
        "Kernkomponente": comp["Kernkomponente"].map(yn),
        "Teiltests gestützt": [f"{n0(a)} von {n0(b)}" for a, b in zip(comp["n_gestützt"], comp["n_teiltests"])],
        "Komponente gestützt (Hauptregel)": comp["Komponente_gestützt"].map(yn),
        "Komponente gestützt (streng)": comp["Komponente_gestützt_streng"].map(yn),
        "Hypothese (Hauptregel)": comp["Entscheidung_Hypothese"],
        "Hypothese (streng)": comp["Entscheidung_streng"],
    })
    hyp_list = "\n".join(f"- **{h}:** {C.formulation[h]}" for h in sorted(C.formulation))
    cohorts = ", ".join(C.cohort_labels_03)
    ph_rows = C.ph.rows
    ph_comp = comp[comp["Komponente"].str.startswith("Preishistorie")].iloc[0]
    check(C.literal_changes == ["H4"] and not C.literal_changes_strict,
          "wörtliche Regel ändert nur H4 (Hauptregel), die strenge Regel nicht")
    check(not is_true(ph_comp["Komponente_gestützt"]) and is_true(ph_comp["Komponente_gestützt_wörtlich"]),
          "Preishistorie nur bei wörtlicher Regel gestützt")
    check((ph_rows["Effekt_adjustiert"] < c3["REL_RHO"]).all(), "adjustierte Preishistorien-ρ unter der Relevanzschwelle")
    out.append(f"""### 3.0 Methoden, Hypothesen und Entscheidungsregeln

**Verfahren** (`scripts/03_tests.py`):

- Binäre Merkmale: χ²-Test mit Odds Ratio (Woolf-Konfidenzintervall). Bei Quartilsvergleichen mit Zellen unter {c3['MIN_CELL']} Fällen wird ein bedingtes ML-Odds-Ratio mit exaktem Konfidenzintervall und Fisher-Test berechnet. Quartils-ORs gelten als **instabil**, wenn der Erfolgsanteil der Teilstichprobe über {pct(c3['UNSTABLE_SHARE'], 0)} liegt oder eine Zelle kleiner als {c3['MIN_CELL']} ist. Dort werden stetige Rangmaße bevorzugt.
- Ordinale bzw. metrische Merkmale: Spearman ρ mit log_owners (Konfidenzintervall über Fisher-z), Mann-Whitney-U mit Cliff's δ, Kruskal-Wallis mit ε² = H/(n − 1).
- **Alterskontrolle:** logistische Regression mit Jahr-Fixed-Effects (Hauptadjustierung). Aufeinanderfolgende seltene Jahrgänge werden zusammengefasst, bis jede Gruppe mindestens {c3['YEAR_FE_MIN_LOGIT']} Spiele hat. Eine quadratische Spezifikation mit zentriertem Jahr ((Jahr − {c3['YEAR_CENTER']})/{de(c3['YEAR_SCALE'], 0)}) dient nur als Sensitivität. Jede Logit-Schätzung wird auf Konvergenz geprüft; alle Schätzungen in allen Ergebnistabellen sind konvergiert. Partielle Rangkorrelationen residualisieren die Ränge beider Variablen auf Jahr-FE (Gruppen ab {c3['YEAR_FE_MIN_PARTIAL']} Spielen), bei der Preishistorie zusätzlich auf die Fensterlänge.
- Plattformen zusätzlich als Kombinationen gegen „nur Windows“ und macOS/Linux gegenseitig adjustiert; Genres, Modi und Plattformen außerdem in einem **gemeinsamen Modell** mit Jahr-FE.
- Peak CCU zweiteilig: P(Peak > 0) als Hürde (OR bzw. Cliff's δ) und die Intensität nur bei Peak > 0.
- **Kohorten** für Stratifizierung und Interaktionstests: {cohorts} (Jahrgang {STEAMSPY_CUTOFF.year} nur Januar bis Mai). Heterogenität der Korrelationen über Cochran-Q (Fisher-z), Interaktionen über Likelihood-Ratio-Tests.
- H3-Vergleich in **identischer Stichprobe**: Differenz der Spearman-Korrelationen per Bootstrap ({n0(c3['N_BOOT'])} Wiederholungen, Seed {RANDOM_STATE}).
- p-Werte werden je Ergebnistabelle über alle Zeilen per **Benjamini-Hochberg** korrigiert (q-Werte). Wegen der großen Stichproben sind viele p-Werte extrem klein; die Bewertung stützt sich deshalb auf Effektmaße und die folgenden Regeln. Alle Ergebnisse beschreiben **Zusammenhänge**, keine Wirkungen.

**Hypothesen** (ausformuliert, Quelle: `hypothesis_decisions.csv`):

{hyp_list}

**Entscheidungsregel:** {C.rule}

Kernkomponenten sind bei H1 Genres (mindestens ein Genre), Spielmodi und Plattformen; bei H2 kostenlos gegenüber kostenpflichtig, Preisniveau (Listenpreis) und Rabatt; bei H3 Kritikerbewertung (Metascore) und Nutzerbewertung (Positive Ratio); bei H4 die Richtungsgleichheit der IGDB-Modi und -Genres mit Steam (Konsistenzprüfung), die IGDB-Bewertungen und die normierten Preishistorien-Merkmale. DLC-Anzahl und die Positive Ratio in der identischen Metacritic-Stichprobe sind Zusatzzeilen ohne Einfluss auf die Entscheidung.

**Zeitpunkt der Regeln:** Relevanzschwellen, Kernkomponenten und Entscheidungsregel wurden nicht vor der Datenanalyse festgelegt oder präregistriert, sondern erst während der Auswertung, als erste Ergebnisse bereits bekannt waren. Die Schwellen folgen üblichen Konventionen für kleine Zusammenhänge (OR {de(c3['REL_OR'], 1)}, |ρ| {de(c3['REL_RHO'], 2)}, |δ| {de(c3['REL_DELTA'], 3)}, ε² {de(c3['REL_EPS2'], 2)}), sind aber nicht unabhängig von den Ergebnissen gewählt. Die Urteile sind deshalb eine regelgeleitete Zusammenfassung, kein konfirmatorischer Test.

**Abweichungen von der Planung (offengelegt):** (1) „Gleiche Richtung im adjustierten Modell“ wird als gleiches Vorzeichen mit statistisch gesicherter Richtung umgesetzt (adjustiertes q < 0,05). Ein Nullzusammenhang nach Alterskontrolle gilt also nicht als richtungskonsistent. Wörtlich angewandt (nur gleiches Vorzeichen), wären {n0(ph_comp['n_gestützt_wörtlich'])} von {n0(ph_comp['n_teiltests'])} Preishistorien-Teiltests gestützt. Damit wäre H4 nach der Hauptregel „{C.decision_literal['H4']}“ statt „{C.decision['H4']}“; nach der strengen Regel bliebe es bei „{C.decision_literal_strict['H4']}“, weil die adjustierten Korrelationen unter der Relevanzschwelle liegen ({de(ph_rows['Effekt_adjustiert'].min(), 2)} bis {de(ph_rows['Effekt_adjustiert'].max(), 2)}). Die übrigen Hypothesenurteile ändern sich durch die Auslegung nicht. (2) Statt der drei Kohorten aus dem Planungsbeispiel werden vier verwendet ({cohorts}). (3) Die näherungsweise IGDB-Zuordnung „Action“ (κ = {de(C.kappa_action.iloc[0], 2)}) wird berichtet, geht aber nicht in die H4-Entscheidung ein. (4) Die Rollen von Positive Ratio und Wilson-Score wurden nach der Volumendiagnose getauscht (Abschnitt 1.4).

**Übersicht der Entscheidungen:**

{md(comp_tbl)}
""")

    # ---- 3.1 H1
    jm = D.joint
    jrows = []
    for merkmal, part in jm.groupby("Merkmal", sort=False):
        row = {"Merkmal": merkmal}
        for outcome in ["success_top25", "success_50k", "success_100k"]:
            r = part[part["Erfolgsdefinition"] == outcome].iloc[0]
            mark = "" if r["p_adj"] < ALPHA else " (n. s.)"
            row[f"aOR {C.thr_label[outcome]}"] = ci(r["Effekt"], r["CI_unten"], r["CI_oben"]) + mark
        jrows.append(row)
    jm_tbl = md(pd.DataFrame(jrows))
    r2 = jm.drop_duplicates("Erfolgsdefinition").set_index("Erfolgsdefinition")["Pseudo_R2"]
    jtop = jm[jm["Erfolgsdefinition"] == "success_top25"].set_index("Variable")
    for v in ["genre_massively_multiplayer", "single_player", "linux"]:
        check(covers(jtop.loc[v, "CI_unten"], jtop.loc[v, "CI_oben"], 1.0), f"{v} im gemeinsamen Modell n. s.")
    plat_dec = D.dec[(D.dec["Hypothese"] == "H1") & (D.dec["Komponente"] == "Plattformen")]
    check(not plat_dec["Komponente_gestützt_streng"].map(is_true).any()
          and (plat_dec["Effekt_adjustiert"] < c3["REL_OR"]).all(), "Plattform-aORs unter der Relevanzschwelle")
    check(C.decision["H1"] == "gestützt" and C.decision_strict["H1"] == "teilweise gestützt",
          "H1 gestützt, streng teilweise")
    extra_h1 = ["h1_mode_categorical", "h1_n_platforms_categorical", "h1_platform_combo_categorical",
                "h1_platform_mac_aor_joint", "h1_platform_linux_aor_joint"]
    check(T(D, "h1_platform_mac_aor_joint") > 1 and sig(D, "h1_platform_mac_aor_joint"), "macOS gegenseitig adjustiert positiv")
    check(T(D, "h1_mode_pvp_or") < c3["REL_OR"], "PvP unter Relevanzschwelle")
    out.append(f"""### 3.1 H1 – Spieleigenschaften

Teiltests mit roher und altersadjustierter Odds Ratio (Jahr-FE), quadratischer Sensitivität und den Odds Ratios unter den alternativen Erfolgsdefinitionen:

{decision_table(D, 'H1')}

Kategoriale Gesamttests und gegenseitig adjustierte Plattformen:

{test_table(D, extra_h1)}

Gemeinsames Modell ({jm['Modell'].iloc[0].replace('Logit: ', '')}; Pseudo-R² {de(r2['success_top25'], 3)} / {de(r2['success_50k'], 3)} / {de(r2['success_100k'], 3)}):

{jm_tbl}

Kernergebnisse:

- **Genres:** Massively Multiplayer, RPG und Casual sind praktisch relevant und in allen Erfolgsdefinitionen richtungsgleich. Adventure, Strategy, Action, Indie, Sports und Racing sind signifikant, bleiben aber unter der Relevanzschwelle. Simulation ist nicht robust: roh leicht negativ, adjustiert nicht signifikant ({tci(D, 'h1_genre_simulation_aor')}). Im gemeinsamen Modell geht der Genre-Zusammenhang von Massively Multiplayer im MMO-Modus auf ({ci(jtop.loc['genre_massively_multiplayer', 'Effekt'], jtop.loc['genre_massively_multiplayer', 'CI_unten'], jtop.loc['genre_massively_multiplayer', 'CI_oben'])}, n. s.).
- **Spielmodi:** Multiplayer, Koop und MMO sind positiv und relevant assoziiert. PvP ist positiv, aber unter der Relevanzschwelle (OR {tci(D, 'h1_mode_pvp_or')}). Die Singleplayer-Kategorie hat eine OR unter 1 ({tci(D, 'h1_mode_single_player_aor')} adjustiert), weil Spiele ohne Singleplayer überwiegend reine Multiplayer-Titel sind. Im gemeinsamen Modell verschwindet dieser Zusammenhang ({ci(jtop.loc['single_player', 'Effekt'], jtop.loc['single_player', 'CI_unten'], jtop.loc['single_player', 'CI_oben'])}, n. s.); er ist also kein Beleg gegen Singleplayer-Spiele.
- **Plattformen:** macOS (roh {tci(D, 'h1_platform_mac_or')}, adjustiert {tci(D, 'h1_platform_mac_aor')}) und die Kombinationen Windows + macOS ({tci(D, 'h1_combo_wm_aor')}) bzw. Windows + macOS + Linux ({tci(D, 'h1_combo_wml_aor')}) sind positiv assoziiert. Linux allein ist roh positiv ({tci(D, 'h1_platform_linux_or')}), gegenseitig mit macOS adjustiert aber nicht mehr ({tci(D, 'h1_platform_linux_aor_joint')}); Windows + Linux ohne macOS unterscheidet sich nicht von „nur Windows“ ({tci(D, 'h1_combo_wl_aor')}). Alle adjustierten Plattform-ORs liegen unter {de(c3['REL_OR'], 1)}. Die Plattformangabe kann auch Portierungsaufwand oder Studiogröße widerspiegeln; beides ist in den Daten nicht erfasst und wurde nicht geprüft.

**H1 wird nach der Hauptregel als „{C.decision['H1']}“ eingestuft:** In allen drei Kernkomponenten gibt es mindestens einen gestützten Teiltest. Nach der strengen Regel ist H1 nur „{C.decision_strict['H1']}“, weil die adjustierten Plattform-ORs unter der Relevanzschwelle liegen.
""")

    # ---- 3.2 H2
    h2_extra = ["h2_free_owners_delta", "h2_f2ptag_or", "h2_f2ptag_aor", "h2_free_with_tag_aor", "h2_free_without_tag_aor",
                "h2_price_class_kw", "h2_price_rho_with_outliers", "h2_price_rho_nodiscount", "h2_price_rho_nodiscount_partial",
                "h2_price_rho_discount", "h2_price_rho_discount_partial", "h2_price_rho_discount_partial_depth",
                "h2_price_current_rho", "h2_price_current_rho_partial",
                "h2_price_class_current_kw", "h2_discount_rho", "h2_discount_rho_partial", "h2_discount_group_kw"]
    h2_peak = ["h2_free_peakpos_or", "h2_free_peakpos_aor", "h2_free_peak_delta", "h2_price_peakpos_delta",
               "h2_price_peak_rho", "h2_price_peak_rho_partial", "h2_price_rho_peaksample", "h2_discount_peakpos_or", "h2_discount_peakpos_aor",
               "h2_discount_peak_rho", "h2_discount_peak_rho_partial"]
    fs = D.free_sens
    fs_part = fs[fs["Merkmal"] == "kostenlos (Listenpreis 0)"]
    fs_tbl = pd.DataFrame({
        "Stichprobe": fs_part["Stichprobe"],
        "Kohorte": fs_part["Kohorte"],
        "N": fs_part["N"].map(n0),
        "N kostenlos": fs_part["N_Gruppe"].map(n0),
        "Erfolgsquote kostenlos / kostenpfl.": [f"{pct(a)} / {pct(b)}" if not _isna(a) else "–"
                                                 for a, b in zip(fs_part["Erfolgsquote_Gruppe"], fs_part["Erfolgsquote_Referenz"])],
        "OR [95-%-KI]": [ci(a, b, c) for a, b, c in zip(fs_part["OR"], fs_part["OR_CI_unten"], fs_part["OR_CI_oben"])],
        "aOR Jahr-FE [95-%-KI]": [ci(a, b, c) for a, b, c in zip(fs_part["aOR_JahrFE"], fs_part["aOR_CI_unten"], fs_part["aOR_CI_oben"])],
        "p (aOR, unkorr.)": fs_part["p_aOR"].map(psci),
        "Interaktion (LR χ², p)": [f"χ²({n0(df)}) = {de(chi, 1)}, p = {psci(p)}" if not _isna(chi) else ""
                                   for chi, df, p in zip(fs_part["LR_Chi2"], fs_part["df"], fs_part["p_Interaktion"])],
        "Hinweis": fs_part["Hinweis"].map(lambda v: de_text(v, english=True)),
    })
    inter = fs.loc["free_an_is_free_interaction"]
    check(inter["p_Interaktion"] < ALPHA, "Interaktion kostenlos × Kohorte signifikant")
    tag_inter = fs.loc["free_an_f2p_tag_interaction"]
    tag_last = fs.loc[f"free_an_f2p_tag_c{len(C.cohort_labels_03)}"]
    check(tag_last["Kohorte"] == C.cohort_labels_03[-1], "F2P-Tag: jüngste Kohorte")
    check(covers(tag_last["aOR_CI_unten"], tag_last["aOR_CI_oben"], 1.0), "F2P-Tag in jüngster Kohorte n. s.")
    zero_c4 = fs[(fs["Stichprobe"].str.contains("0 - 0")) & (fs["Merkmal"] == "kostenlos (Listenpreis 0)")
                 & (fs["Kohorte"] == C.cohort_labels_03[-1])].iloc[0]
    check(zero_c4["aOR_CI_oben"] < 1, "mit '0 - 0' in jüngster Kohorte negativ")
    le2020 = fs.loc["free_le2020_is_free_all"]
    check(T(D, "h2_price_current_rho_partial") < c3["REL_RHO"], "Tagespreis-Zusammenhang unter Relevanzschwelle")
    check(T(D, "h2_price_class_kw", "Zusatz_Effekt") >= c3["REL_EPS2"], "KW nach Listenpreisklasse relevant")
    check(C.decision["H2"] == "gestützt", "H2 gestützt")
    check(ci_covers(D, "h2_price_rho_nodiscount_partial", 0.0)
          and T(D, "h2_price_rho_discount_partial") > T(D, "h2_price_rho")
          and T(D, "h2_price_rho_discount_partial_depth") >= c3["REL_RHO"],
          "Listenpreis: Effektmodifikation durch Rabattstatus, auch bei Kontrolle der Rabatthöhe")
    check(T(D, "h2_price_peak_rho") > T(D, "h2_price_rho_peaksample"), "Peak > 0: Listenpreis stärker mit Intensität als mit Besitzern")
    dec_h2 = D.dec[D.dec["Hypothese"] == "H2"]
    check(not dec_h2[dec_h2["Kernkomponente"].map(is_true)]["Sensitivitaeten_bestaetigt"].map(is_true).all(),
          "mindestens eine H2-Sensitivität bestätigt die Richtung nicht")
    out.append(f"""### 3.2 H2 – Preis

Teiltests (Preisniveau als Spearman ρ mit log_owners; unter den alternativen Erfolgsdefinitionen Cliff's δ):

{decision_table(D, 'H2')}

Sensitivitäten je Teiltest (aus `hypothesis_decisions.csv`; sie gehen nicht in die Entscheidung ein, schränken die Aussage aber ein):

{sensitivity_table(D, 'H2')}

Kostenlos gegenüber kostenpflichtig nach Kohorte, mit „0 - 0“ als Nicht-Erfolg und für Releases bis 2020 (`h2_free_sensitivity.csv`):

{md(fs_tbl)}

Weitere Tests (Steam-Tag „Free To Play“, Preisniveau-Sensitivitäten, Rabatthöhe):

{test_table(D, h2_extra)}

Peak CCU als Hürde (P(Peak > 0)) und Intensität (nur Peak > 0), Momentaufnahme am Vortag des SteamSpy-Abrufs:

{test_table(D, h2_peak)}

Kernergebnisse:

- **Kostenlos (Listenpreis 0):** Gepoolt sind kostenlose Spiele häufiger erfolgreich (aOR {tci(D, 'h2_free_aor')}). Der Zusammenhang ist aber **kohortenabhängig** (Interaktion χ²({n0(inter['df'])}) = {de(inter['LR_Chi2'], 1)}, p = {psci(inter['p_Interaktion'])}): Er besteht in den Kohorten bis einschließlich {C.cohort_labels_03[-2]} und fehlt in {C.cohort_labels_03[-1]}. Er ist **nicht robust** gegenüber dem Ausschluss der „0 - 0“-Einträge, die überwiegend kostenlos sind: Zählen sie als Nicht-Erfolg, liegt die aOR bei {ci(fs.loc['free_zero_is_free_all', 'aOR_JahrFE'], fs.loc['free_zero_is_free_all', 'aOR_CI_unten'], fs.loc['free_zero_is_free_all', 'aOR_CI_oben'])}, in {C.cohort_labels_03[-1]} sogar bei {ci(zero_c4['aOR_JahrFE'], zero_c4['aOR_CI_unten'], zero_c4['aOR_CI_oben'])}. Für Releases bis 2020 ist er dagegen deutlich ({ci(le2020['aOR_JahrFE'], le2020['aOR_CI_unten'], le2020['aOR_CI_oben'])}). Getragen wird er von den Titeln mit Steam-Tag „Free To Play“ (mit Tag {tci(D, 'h2_free_with_tag_aor')}, ohne Tag {tci(D, 'h2_free_without_tag_aor')}); auch der Tag allein ist kohortenabhängig (Interaktion p = {psci(tag_inter['p_Interaktion'])}). Bei Peak CCU ist der Unterschied klein (Hürde aOR {tci(D, 'h2_free_peakpos_aor')}, Intensität δ = {de(T(D, 'h2_free_peak_delta'), 3)}).
- **Preisniveau (kostenpflichtig, ohne Ausreißer):** Der Listenpreis korreliert schwach positiv mit den Besitzern (ρ {tci(D, 'h2_price_rho')}, partiell {tci(D, 'h2_price_rho_partial')}; Kruskal-Wallis nach Preisklasse ε² = {de(T(D, 'h2_price_class_kw', 'Zusatz_Effekt'), 3)}, Klasse 40+ dort einschließlich {de(D.cc['CLUSTER_PRICE'], 2)}-USD-Cluster; ohne Cluster ε² = {de(T(D, 'h2_price_class_kw_nocluster', 'Zusatz_Effekt'), 3)}). Der Zusammenhang **hängt am aktuellen Rabatt** (Effektmodifikation): Unter Spielen ohne aktuellen Rabatt ist er praktisch null (partiell {tci(D, 'h2_price_rho_nodiscount_partial')}), unter rabattierten Spielen deutlich und größer als gepoolt ({tci(D, 'h2_price_rho_discount_partial')}; zusätzlich für die Rabatthöhe kontrolliert {tci(D, 'h2_price_rho_discount_partial_depth')}). Ob das inhaltlich ist oder aus der Rekonstruktion stammt (bei nicht erfasstem Rabatt ist der Listenpreis zu niedrig), lässt sich nicht trennen. Beim Tagespreis bleibt nur ein kleiner Zusammenhang (partiell {tci(D, 'h2_price_current_rho_partial')}). Unter Spielen mit Aktivität am Messtag (Peak > 0, n = {n0(T(D, 'h2_price_peak_rho', 'N'))}) korreliert der Listenpreis stärker mit der Intensität (ρ {tci(D, 'h2_price_peak_rho')}) als in derselben Teilstichprobe mit den Besitzern (ρ {tci(D, 'h2_price_rho_peaksample')}).
- **Rabatt (kostenpflichtig):** Ein aktiver Rabatt geht mit einer höheren Erfolgsquote einher (OR {tci(D, 'h2_discount_or')}, aOR {tci(D, 'h2_discount_aor')}) und deutlich häufiger mit Aktivität am Messtag (Hürde aOR {tci(D, 'h2_discount_peakpos_aor')}). Über die Rabatthöhe ergibt sich nach Altersstandardisierung gepoolt kein monotoner Gradient; {discount_cohort_pattern(D).short} (Abschnitt 2.5). Der Rabatt ist zum Store-Snapshot gemessen, also später als die Besitzer (SteamSpy-Stand), und eine Untergrenze, die den Zusammenhang eher abschwächt. Die Richtung ist offen: Bereits populäre Titel könnten häufiger rabattiert werden, Rabatte könnten aber auch die Verbreitung fördern.
- **DLC-Anzahl (Zusatz):** ρ {tci(D, 'h2_dlc_rho')}, partiell {tci(D, 'h2_dlc_rho_partial')}; auch hier ist die Richtung offen.

**H2 wird nach der Hauptregel als „{C.decision['H2']}“ eingestuft** (streng: „{C.decision_strict['H2']}“), weil alle drei Kernkomponenten einen gestützten Teiltest haben. Die Sensitivitäten schränken das Urteil deutlich ein: Der Vorteil kostenloser Spiele ist kohortenabhängig und hängt am Umgang mit „0 - 0“, der Listenpreis-Zusammenhang entsteht fast nur unter rabattierten Spielen.
""")

    # ---- 3.3 H3
    h3_extra = ["h3_pr_r10_q", "h3_pr_r10_qchi2", "h3_pr_r10_qtrend", "h3_pr_r10_qnonlin", "h3_pr_r50_rho", "h3_pr_r50_rho_partial",
                "h3_pr_r50_q", "h3_pr_r10_nomc_rho", "h3_wilson_r10_rho", "h3_wilson_r10_rho_partial", "h3_wilson_r10_q",
                "h3_mc_q", "h3_mc_qchi2", "h3_mc_qtrend", "h3_mc_qnonlin", "h3_mcuser_rho", "h3_mcuser_rho_partial",
                "h3_agree_pr_mc", "h3_agree_wilson_mc", "h3_agree_mc_fg"]
    h3_peak = ["h3_pr_r10_peakpos_delta", "h3_pr_r10_peak_rho", "h3_pr_r10_peak_rho_partial", "h3_wilson_r10_peak_rho",
               "h3_mc_peakpos_delta", "h3_mc_peak_rho", "h3_mc_peak_rho_partial", "h3_mcuser_peak_rho"]
    idn = D.ident
    id_tbl = pd.DataFrame({
        "Vergleich": idn["Vergleich"],
        "Modell": idn["Modell"],
        "Stichprobe": idn["Stichprobe"],
        "N": idn["N"].map(n0),
        "Erfolgsquote": idn["Erfolgsquote"].map(pct),
        "ρ A": idn["rho_A"].map(lambda v: de(v, 3)),
        "ρ B": idn["rho_B"].map(lambda v: de(v, 3)),
        "Δρ (A − B) [95-%-KI]": [ci(a, b, c, 3) for a, b, c in zip(idn["Differenz"], idn["CI_unten"], idn["CI_oben"])],
        "p (Bootstrap)": idn["p_bootstrap"].map(pval),
    })
    igdb_internal = idn.index.str.startswith("ident_igdb_aggregated_rating_vs_igdb_rating")
    check((idn.loc[~igdb_internal, "CI_unten"] > 0).all(),
          "Kritiker-/Nutzer-Differenzen (Metascore bzw. IGDB-Kritiker gegen Steam/Metacritic) positiv gesichert")
    ch = D.cohort_h3
    ch_tbl = pd.DataFrame({
        "Maß": ch["Merkmal"],
        "Stichprobe": ch["Stichprobe"],
        "Kohorte": ch["Kohorte"],
        "N": ch["N"].map(n0),
        "Erfolgsquote": ch["Erfolgsquote"].map(pct),
        "Wert [95-%-KI]": [ci(a, b, c, 3) if not str(t).startswith("Q") else de(a, 1)
                            for a, b, c, t in zip(ch["Effekt"], ch["CI_unten"], ch["CI_oben"], ch["Effekt_typ"])],
        "Maß des Werts": ch["Effekt_typ"].map(etype),
        "p (BH)": ch["p_adj"].map(psci),
    })
    mc_het = ch.loc["cohort_3_heterogeneity"]
    pr_het = ch.loc["cohort_0_heterogeneity"]
    check(pr_het["p_adj"] < ALPHA and mc_het["p_adj"] >= ALPHA, "Positive Ratio heterogen, Metascore nicht")
    mc_coh = [ch.loc[f"cohort_3_c{i}", "Effekt"] for i in range(len(C.cohort_labels_03))]
    check((np.diff(mc_coh) < 0).all(), "Metascore-ρ nimmt über die Kohorten ab")
    pr_coh = [ch.loc[f"cohort_0_c{i}", "Effekt"] for i in range(len(C.cohort_labels_03))]
    qg = D.qgroups[D.qgroups["Kennung"].isin(["h3_pr_r10", "h3_mc"])]
    qg_tbl = pd.DataFrame({
        "Maß": qg["Merkmal"],
        "Quartil": qg["Gruppe"],
        "n": qg["n"].map(n0),
        "Wertebereich": [f"{de(a, 3)}–{de(b, 3)}" if k == "h3_pr_r10" else f"{de(a, 0)}–{de(b, 0)}"
                         for a, b, k in zip(qg["min"], qg["max"], qg["Kennung"])],
        f"Erfolgsquote {C.thr_label['success_top25']}": qg["Erfolgsquote_success_top25"].map(pct),
        f"Erfolgsquote {C.thr_label['success_50k']}": qg["Erfolgsquote_success_50k"].map(pct),
        f"Erfolgsquote {C.thr_label['success_100k']}": qg["Erfolgsquote_success_100k"].map(pct),
    })
    q_or = {o: rob(D, f"rob_pr_r10_{o}_q") for o in ["success_top25", "success_50k", "success_100k"]}
    check(C.decision["H3"] == "teilweise gestützt", "H3 teilweise gestützt")
    dec_h3 = D.dec[D.dec["Hypothese"] == "H3"].set_index("Komponente")
    check(is_true(dec_h3.loc["Kritikerbewertung", "Komponente_gestützt"])
          and not is_true(dec_h3.loc["Nutzerbewertung (Positive Ratio)", "Komponente_gestützt"]), "Kritiker gestützt, PR nicht")
    check(T(D, "h3_pr_r10_qtrend") < 0 and T(D, "h3_pr_r10_qnonlin") > abs(T(D, "h3_pr_r10_qtrend")) ** 2,
          "Nichtlinearität dominiert den linearen Trend")
    check(T(D, "h3_mc_q", "instabil") in (True, "True"), "Metascore-Quartils-OR instabil markiert")
    out.append(f"""### 3.3 H3 – Bewertungen

Teiltests (Spearman ρ mit log_owners; unter den alternativen Erfolgsdefinitionen Cliff's δ):

{decision_table(D, 'H3')}

Sensitivitäten und Hinweise je Teiltest:

{sensitivity_table(D, 'H3')}

**Vergleich in identischer Stichprobe** (`h3_identical_sample.csv`; A = Kritiker- bzw. erstgenanntes Maß, B = Nutzermaß):

{md(id_tbl)}

**Kohortenstratifizierte Korrelationen** (`h3_cohort_stratified.csv`):

{md(ch_tbl)}

Quartile der Positive Ratio und des Metascores unter allen drei Erfolgsdefinitionen (`quartile_groups.csv`):

{md(qg_tbl)}

Weitere Tests (Quartile, Nichtlinearität, Wilson-Score als Diagnose, Metacritic-Nutzerscore, Übereinstimmung):

{test_table(D, h3_extra)}

Peak CCU als Hürde und Intensität:

{test_table(D, h3_peak)}

Kernergebnisse:

- **Kritikerbewertung:** Der externe Metascore korreliert moderat positiv mit den Besitzern (ρ {tci(D, 'h3_mc_rho')}, partiell {tci(D, 'h3_mc_rho_partial')}), über die Kohorten nicht signifikant heterogen (Q = {de(mc_het['Effekt'], 1)}, q = {psci(mc_het['p_adj'])}), aber tendenziell abnehmend ({' → '.join(de(v, 2) for v in mc_coh)}), und unter allen Erfolgsdefinitionen richtungsgleich. Die Quartils-OR ist wegen der Erfolgsquote von über {pct(c3['UNSTABLE_SHARE'], 0)} als instabil markiert; maßgeblich ist die Rangkorrelation.
- **Identische Stichprobe:** Auf denselben {n0(idn.loc['ident_mc_metascore_vs_positive_ratio', 'N'])} Spielen hängen beide Bewertungsarten positiv mit den Besitzern zusammen, der Metascore etwas stärker als die Positive Ratio (Δρ {ci(idn.loc['ident_mc_metascore_vs_positive_ratio', 'Differenz'], idn.loc['ident_mc_metascore_vs_positive_ratio', 'CI_unten'], idn.loc['ident_mc_metascore_vs_positive_ratio', 'CI_oben'])}; partiell {ci(idn.loc['ident_mc_metascore_vs_positive_ratio_partial', 'Differenz'], idn.loc['ident_mc_metascore_vs_positive_ratio_partial', 'CI_unten'], idn.loc['ident_mc_metascore_vs_positive_ratio_partial', 'CI_oben'])}). Dasselbe Muster zeigen Metascore gegen Metacritic-Nutzerscore sowie IGDB-Kritiker- gegen Steam-Nutzerbewertung. Innerhalb von IGDB sind Kritiker- und Nutzerbewertung auf denselben {n0(C.igdb_ident['N'])} Spielen dagegen etwa gleich stark mit den Besitzern verknüpft ({C.igdb_ident_text}); das gilt auch, wenn nur nicht ganzzahlige Nutzerbewertungen betrachtet werden (n = {n0(C.igdb_ident_nonint['N'])}, Δρ {ci(C.igdb_ident_nonint['Differenz'], C.igdb_ident_nonint['CI_unten'], C.igdb_ident_nonint['CI_oben'], 3)}).
- **Nutzerbewertung in der Breite:** In der Bewertungsstichprobe ist die Positive Ratio gepoolt unkorreliert (ρ {tci(D, 'h3_pr_r10_rho', 3)}; partiell mit Jahr-FE {tci(D, 'h3_pr_r10_rho_partial', 3)}, unter der Relevanzschwelle). Die Konfundierung über die Jahrgänge erklärt nur die Verschiebung von {de(T(D, 'h3_pr_r10_rho'), 3)} auf {de(T(D, 'h3_pr_r10_rho_partial'), 3)}: Neuere Jahrgänge werden besser bewertet (ρ(Jahr, Positive Ratio) = {de(C.rho_year_pr, 2)}) und haben weniger Besitzer (ρ(Jahr, Besitzer) = {de(C.rho_year_owners, 2)}; beides in der Steam-Korrelationsmatrix). Auch innerhalb der Jahrgänge bleibt der Zusammenhang im Mittel unter der Relevanzschwelle (gewichtetes Mittel der Kohorten-ρ {de(C.pr_cohort_mean, 3)}, partiell mit Jahr-FE {de(T(D, 'h3_pr_r10_rho_partial'), 3)}). Bestimmend ist die Heterogenität über die Kohorten: bei älteren Titeln deutlich positiv ({de(pr_coh[0], 2)}, {ch.loc['cohort_0_c0', 'Kohorte']}), bei Neuerscheinungen negativ ({de(pr_coh[-1], 2)}, {ch.loc[f'cohort_0_c{len(pr_coh) - 1}', 'Kohorte']}; Heterogenität Q = {de(pr_het['Effekt'], 1)}, q = {psci(pr_het['p_adj'])}). Der Verlauf über die Quartile ist umgekehrt U-förmig: Der lineare Trend ist schwach negativ (z = {de(T(D, 'h3_pr_r10_qtrend'), 1)}), die Abweichung davon dominiert (χ²(2) = {de(T(D, 'h3_pr_r10_qnonlin'), 1)}). Das Q4/Q1-Verhältnis wechselt mit der Erfolgsschwelle die Seite ({' / '.join(de(v, 2) for v in q_or.values())} für {', '.join(C.thr_label.values())}). Die Mindestzahl von {MIN_REVIEWS} Reviews selektiert zudem bereits sichtbare Spiele.
- **Wilson-Score (Diagnose):** ρ {tci(D, 'h3_wilson_r10_rho')} mit den Besitzern, aber ρ {ci(D.diag.loc['diag_2_wilson_vs_reviews', 'Effekt'], D.diag.loc['diag_2_wilson_vs_reviews', 'CI_unten'], D.diag.loc['diag_2_wilson_vs_reviews', 'CI_oben'], 3)} mit der Review-Anzahl (Positive Ratio: {ci(D.diag.loc['diag_0_positive_ratio_vs_reviews', 'Effekt'], D.diag.loc['diag_0_positive_ratio_vs_reviews', 'CI_unten'], D.diag.loc['diag_0_positive_ratio_vs_reviews', 'CI_oben'], 3)}). Sein Zusammenhang mit den Besitzern bildet vor allem das Review-Volumen ab.
- **Peak CCU:** Höhere Bewertungen gehen häufiger mit Aktivität am Messtag einher (Hürde Positive Ratio δ = {de(T(D, 'h3_pr_r10_peakpos_delta'), 3)}, Metascore δ = {de(T(D, 'h3_mc_peakpos_delta'), 3)}); die Intensität korreliert mit dem Metascore mit ρ {tci(D, 'h3_mc_peak_rho')} (Metacritic-Teilstichprobe mit Peak > 0, n = {n0(T(D, 'h3_mc_peak_rho', 'N'))}) und mit der Positive Ratio mit ρ {tci(D, 'h3_pr_r10_peak_rho')} (mind. {MIN_REVIEWS} Reviews und Peak > 0, n = {n0(T(D, 'h3_pr_r10_peak_rho', 'N'))}). Die beiden Werte stammen aus verschiedenen Teilstichproben und sind nicht direkt vergleichbar.
- **Metacritic-Quellen:** Externer und FronkonGames-Metascore stimmen auf der Schnittmenge nahezu überein (ρ {tci(D, 'h3_agree_mc_fg')}); das ist ein Matching-Check, kein unabhängiger Validitätsnachweis.

**H3 wird als „{C.decision['H3']}“ eingestuft** (streng: „{C.decision_strict['H3']}“): Die Kritikerbewertung ist gestützt, die Positive Ratio in der Breite nicht. In der dokumentierten Metacritic-Teilstichprobe hängen beide Bewertungsarten positiv mit den Besitzern zusammen, Kritikerbewertungen etwas stärker; in der Breite gibt es keinen monotonen Zusammenhang der Nutzerbewertung.
""")

    # ---- 3.4 H4
    modes = [("Singleplayer", "singleplayer"), ("Multiplayer", "multiplayer"), ("Koop", "coop"), ("MMO", "mmo")]
    genres = [("Adventure", "genre_adventure"), ("Indie", "genre_indie"), ("Racing", "genre_racing"),
              ("RPG", "genre_role_playing_rpg"), ("Simulation/Simulator", "genre_simulator"),
              ("Sports/Sport", "genre_sport"), ("Strategy (Strategy, RTS, TBS)", "strategy_any"),
              ("Action (näherungsweise)", "action_like")]
    rows = []
    raw_diff = []
    for label, key in modes + genres:
        ik, sk = f"h4_igdb_{key}", f"h4_steam_{key}"
        kappa_row = D.agree[D.agree["IGDB"] == f"igdb_{key}"]
        same_raw = (T(D, f"{ik}_or") - 1) * (T(D, f"{sk}_or") - 1) > 0
        same_adj = (T(D, f"{ik}_aor") - 1) * (T(D, f"{sk}_aor") - 1) > 0
        if not same_raw and key != "action_like":
            raw_diff.append((label, key))
        rows.append({
            "Merkmal": label,
            "IGDB OR": tci(D, f"{ik}_or"),
            "Steam OR (gleiche Teilstichprobe)": tci(D, f"{sk}_or") + ("" if sig(D, f"{sk}_or") else " (n. s.)"),
            "IGDB aOR (Jahr-FE)": tci(D, f"{ik}_aor"),
            "Steam aOR (gleiche Teilstichprobe)": tci(D, f"{sk}_aor") + ("" if sig(D, f"{sk}_aor") else " (n. s.)"),
            "gleiche Richtung (roh / adjustiert)": f"{yn(same_raw)} / {yn(same_adj)}",
            "Cohens κ": de(kappa_row["Kappa"].iloc[0], 2) if len(kappa_row) else "–",
            "IGDB aOR gegeben Steam-Merkmal": tci(D, f"{ik}_aor_given_steam") + ("" if sig(D, f"{ik}_aor_given_steam") else " (n. s.)"),
        })
    cons_tbl = md(pd.DataFrame(rows))
    for label, key in raw_diff:
        check(not sig(D, f"h4_steam_{key}_or") or not sig(D, f"h4_steam_{key}_aor"),
              f"abweichende Richtung bei {label} nur bei nicht gesichertem Steam-Merkmal")
    if raw_diff:
        diff_text = "; ".join(
            f"{label}: Dort hat das Steam-Merkmal selbst keine gesicherte Richtung (roh {tci(D, f'h4_steam_{key}_or')}, adjustiert {tci(D, f'h4_steam_{key}_aor')})"
            for label, key in raw_diff)
        diff_text = f" Ausnahme ist {diff_text}."
    else:
        diff_text = " Es gibt keine Ausnahme."
    sp_sign_same = (T(D, "h4_igdb_singleplayer_aor") < 1) and (T(D, "h4_steam_singleplayer_aor") < 1)
    check(sp_sign_same, "IGDB- und Steam-Singleplayer gleiche Richtung")
    h4_ratings = ["h4_igdbuser_all_rho", "h4_igdbuser_all_rho_partial", "h4_igdbuser_all_q", "h4_igdbuser_nonint_q",
                  "h4_igdbuser_int_rho", "h4_igdbuser_int_rho_partial", "h4_igdbuser_int_q", "h4_igdbuser_integerflag_rho",
                  "h4_igdbcrit_q", "h4_igdb_ngenres_rho", "h4_igdbuser_all_peak_rho", "h4_igdbuser_nonint_peak_rho",
                  "h4_igdbcrit_peak_rho", "h4_agree_igdbuser_pr", "h4_agree_igdbcrit_pr", "h4_agree_igdbcrit_mc"]
    check(T(D, "h4_igdbuser_integerflag_rho") < -0.3, "Ganzzahligkeit stark negativ mit Besitzern verbunden")
    check(abs(T(D, "h4_igdbuser_int_rho")) < c3["REL_RHO"], "ganzzahlige Bewertungen ohne relevanten Zusammenhang")
    check(T(D, "h4_igdbuser_int_rho") < 0 and sig(D, "h4_igdbuser_int_rho") and ci_covers(D, "h4_igdbuser_int_rho_partial", 0.0),
          "ganzzahlige Bewertungen roh schwach negativ, partiell null")
    q_all, q_nonint, q_int = (T(D, "h4_igdbuser_all_q"), T(D, "h4_igdbuser_nonint_q"), T(D, "h4_igdbuser_int_q"))
    q_keep = {"nonint": math.log(q_nonint) / math.log(q_all), "int": math.log(q_int) / math.log(q_all)}

    def qrange(kennung: str, idx: int) -> str:
        part = D.qgroups[D.qgroups["Kennung"] == kennung].reset_index(drop=True)
        check(len(part) == 4, f"{kennung}: vier Quartilsgruppen erwartet")
        d = 0 if kennung.endswith("_int") else 1
        return f"{de(part.loc[idx, 'min'], d)}–{de(part.loc[idx, 'max'], d)}"

    check(T(D, "h4_igdbuser_nonint_q", "CI_unten") > 1 and T(D, "h4_igdbuser_int_q", "CI_unten") > 1
          and max(q_nonint, q_int) < q_all / 1.5, "Quartils-OR innerhalb der Wertetypen > 1, aber deutlich kleiner")
    check(is_true(T(D, "h4_igdbuser_nonint_q", "instabil")), "nicht ganzzahlige Quartils-OR instabil markiert")
    int_values = D.igdb_values[D.igdb_values["Kennung"].str.match(r"value_\d+$")].copy()
    int_values["v"] = int_values["Kennung"].str.removeprefix("value_").astype(int)
    int_rates = int_values.sort_values("v")["Erfolgsquote"].to_numpy()
    check(not (np.diff(int_rates) > 0).all(), "ganzzahlige Werte: Erfolgsquote nicht monoton")
    ph = C.ph
    clean_sig = ph.clean_sig
    share_raw, share_full = T(D, "h4_ph_share_months_on_sale_rho"), T(D, "h4_ph_share_months_on_sale_full_rho")

    def rel_word(v: float) -> str:
        return "knapp unter" if abs(v) < c3["REL_RHO"] else "knapp über" if abs(v) < c3["REL_RHO"] + 0.02 else "über"
    ph_feats = ["price_changes_per_month", "share_months_on_sale", "days_on_sale_per_month", "mean_monthly_max_discount_pct"]
    prow = []
    for f in ph_feats + ["max_discount_pct"]:
        base = f"h4_ph_{f}"
        prow.append({
            "Merkmal": D.T.loc[f"{base}_rho", "Merkmal"].replace(" vs. Besitzer", ""),
            "N": n0(T(D, f"{base}_rho", "N")),
            "ρ roh": tci(D, f"{base}_rho") + ("" if sig(D, f"{base}_rho") else " (n. s.)"),
            "partiell (Jahr-FE + Fensterlänge)": tci(D, f"{base}_rho_partial") + ("" if sig(D, f"{base}_rho_partial") else " (n. s.)"),
            "volles Fenster roh": (tci(D, f"{base}_full_rho") + ("" if sig(D, f"{base}_full_rho") else " (n. s.)")) if f"{base}_full_rho" in D.T.index else "= Hauptzeile",
            "volles Fenster partiell": (tci(D, f"{base}_full_rho_partial") + ("" if sig(D, f"{base}_full_rho_partial") else " (n. s.)")) if f"{base}_full_rho_partial" in D.T.index else "= Hauptzeile",
            "ohne Tiefstpreis > Basispreis (partiell)": tci(D, f"{base}_clean_rho_partial") + ("" if sig(D, f"{base}_clean_rho_partial") else " (n. s.)"),
            "Peak CCU (nur > 0)": tci(D, f"{base}_peak_rho") + ("" if sig(D, f"{base}_peak_rho") else " (n. s.)"),
        })
    ph_tbl = md(pd.DataFrame(prow))
    for f in ph_feats + ["max_discount_pct"]:
        check(not sig(D, f"h4_ph_{f}_rho_partial"), f"Preishistorie {f} partiell n. s.")
    full_sig = [f for f in ph_feats if f"h4_ph_{f}_full_rho_partial" in D.T.index and sig(D, f"h4_ph_{f}_full_rho_partial")]
    check(set(full_sig) >= {"share_months_on_sale", "days_on_sale_per_month"}, "im vollen Fenster kleine positive Partialkorrelationen")
    check(math.isinf(T(D, "h4_ph_max_discount_pct_q")) and is_true(T(D, "h4_ph_max_discount_pct_q", "instabil")),
          "Q4/Q1 maximaler Rabatt unendlich und instabil")
    ph_q = ["h4_ph_max_discount_pct_q", "h4_ph_share_months_on_sale_q"]
    check(C.decision["H4"] == "teilweise gestützt", "H4 teilweise gestützt")
    dec_h4 = D.dec[D.dec["Hypothese"] == "H4"].drop_duplicates("Komponente").set_index("Komponente")
    check(not is_true(dec_h4.loc["Preishistorie (normierte Merkmale)", "Komponente_gestützt"]), "Preishistorie nicht gestützt")
    out.append(f"""### 3.4 H4 – Quellenübergreifende Analysen

Teiltests (IGDB-Modi und -Genres nur auf Spielen mit Modus- bzw. Genreangabe; Bewertungen und Preishistorie als Spearman ρ, unter den alternativen Erfolgsdefinitionen Cliff's δ):

{decision_table(D, 'H4')}

Sensitivitäten und Hinweise je Teiltest:

{sensitivity_table(D, 'H4')}

**Konsistenz von IGDB und Steam.** IGDB-Merkmal und Steam-Merkmal in derselben Teilstichprobe, Übereinstimmung (κ) und IGDB-Merkmal bei gegebenem Steam-Merkmal:

{cons_tbl}

**IGDB-Bewertungen** (weitere Tests):

{test_table(D, h4_ratings)}

**Preishistorie** (normierte Merkmale; Rangkorrelationen mit log_owners, weil die Erfolgsquote nahe 100 % liegt; „n. s.“: BH-q ≥ 0,05 innerhalb von `h4_external_tests.csv`, die Konfidenzintervalle sind unkorrigiert und können die 0 deshalb auch bei „n. s.“ knapp ausschließen):

{ph_tbl}

Quartils-ORs der Preishistorie (nur zur Dokumentation, wegen der Erfolgsquote nahe 100 % instabil):

{test_table(D, ph_q)}

Kernergebnisse:

- **IGDB-Modi und -Genres:** Die Merkmale mit sinnvoller Zuordnung zeigen dieselbe Richtung wie das Steam-Merkmal in derselben Teilstichprobe.{diff_text} Auch Singleplayer zeigt dieselbe Richtung wie Steam (IGDB aOR {tci(D, 'h4_igdb_singleplayer_aor')}, Steam {tci(D, 'h4_steam_singleplayer_aor')}); fehlende Modusangaben gelten dabei als unbekannt, nicht als „nein“. IGDB-MMO ist positiv assoziiert (OR {tci(D, 'h4_igdb_mmo_or')}, n = {n0(D.igdb_modes.loc['igdb_mmo', 'N'])}). Wegen der hohen Übereinstimmung (κ {de(C.kappa.min(), 2)}–{de(C.kappa.max(), 2)}) ist das eine Konsistenzprüfung. Dass einige IGDB-Merkmale bei gegebenem Steam-Merkmal zusätzlich assoziiert sind, kann auch die Dokumentationstiefe widerspiegeln: Gut dokumentierte IGDB-Einträge sind häufiger erfolgreich (Abschnitt 2.7; Anzahl IGDB-Genres ρ {tci(D, 'h4_igdb_ngenres_rho')}).
- **IGDB-Nutzerbewertung:** Nicht ganzzahlige Bewertungen (mehr Stimmen) korrelieren positiv mit den Besitzern (ρ {tci(D, 'h4_igdbuser_nonint_rho')}, partiell {tci(D, 'h4_igdbuser_nonint_rho_partial')}). Ganzzahlige Bewertungen korrelieren roh schwach negativ ({tci(D, 'h4_igdbuser_int_rho')}), partiell nicht ({tci(D, 'h4_igdbuser_int_rho_partial')}). Die Quartils-OR über alle Bewertungen ({tci(D, 'h4_igdbuser_all_q')}) ist zu einem großen Teil durch die Zusammensetzung nach Stimmenvolumen aufgebläht: Schon die Ganzzahligkeit allein korreliert mit ρ {tci(D, 'h4_igdbuser_integerflag_rho')} mit den Besitzern. Innerhalb der Wertetypen ist die Quartils-OR deutlich kleiner, aber größer als 1: nicht ganzzahlig {tci(D, 'h4_igdbuser_nonint_q')} (instabil, Erfolgsquote über {pct(c3['UNSTABLE_SHARE'], 0)}), ganzzahlig {tci(D, 'h4_igdbuser_int_q')}. Auf der Log-Skala bleiben damit {pct(q_keep['nonint'], 0)} bzw. {pct(q_keep['int'], 0)} der Gesamt-OR erhalten; die Zusammensetzung erklärt je nach Wertetyp also etwa {pct(1 - q_keep['int'], 0)} bis {pct(1 - q_keep['nonint'], 0)}, den überwiegenden Teil nur bei den nicht ganzzahligen Werten. Die Quartile haben allerdings andere Grenzen (alle Werte: Q1 {qrange('h4_igdbuser_all', 0)}, Q4 {qrange('h4_igdbuser_all', 3)}; ganzzahlig: Q1 {qrange('h4_igdbuser_int', 0)}, Q4 {qrange('h4_igdbuser_int', 3)}; nicht ganzzahlig: Q1 {qrange('h4_igdbuser_nonint', 0)}, Q4 {qrange('h4_igdbuser_nonint', 3)}), sodass die ORs nicht direkt vergleichbar sind; bei ganzzahligen Werten ist der Verlauf wegen des Blocks bei 70 nicht monoton (Abschnitt 2.7).
- **IGDB-Kritikerbewertung:** ρ {tci(D, 'h4_igdbcrit_rho')}, partiell {tci(D, 'h4_igdbcrit_rho_partial')}; sie stimmt eng mit dem Metascore überein (ρ {tci(D, 'h4_agree_igdbcrit_mc')}). Die Einzelwerte von Kritiker- und Nutzerbewertung stammen aus verschiedenen Teilstichproben; in identischer Stichprobe (n = {n0(C.igdb_ident['N'])}) sind beide etwa gleich stark ({C.igdb_ident_text}; Abschnitt 3.3). Beide IGDB-Bewertungsstichproben sind stark selektiert (Abschnitt 3.5).
- **Preishistorie:** Roh korrelieren mehrere normierte Merkmale positiv mit den Besitzern, am stärksten der mittlere monatliche Maximalrabatt ({tci(D, 'h4_ph_mean_monthly_max_discount_pct_rho')}). Nach Kontrolle von Veröffentlichungsjahr und Fensterlänge ist in der Hauptspezifikation **kein** Merkmal signifikant (partiell {de(ph_rows['Effekt_adjustiert'].min(), 2)} bis {de(ph_rows['Effekt_adjustiert'].max(), 2)}); auch die Anzahl der Preisänderungen je Monat nicht ({tci(D, 'h4_ph_price_changes_per_month_rho_partial')}). Im vollen Fenster (n = {n0(ph.full_n)}) zeigen dagegen {ph.full_text} kleine positive Partialkorrelationen{' oberhalb der Relevanzschwelle' if ph.full_rel else ''}. Ohne Apps mit Tiefstpreis über dem Basispreis {('ist' if len(clean_sig) == 1 else 'sind') + ' partiell ' + join_de([ph.label_art[f] + ' (' + tci(D, f'h4_ph_{f}_clean_rho_partial', 3) + ')' for f in clean_sig]) + ' signifikant' if clean_sig else 'ist partiell kein Merkmal signifikant'}. Der maximale Rabatt ist im vollen Fenster roh signifikant ({tci(D, 'h4_ph_max_discount_pct_rho')}), nach Alterskontrolle nicht mehr ({tci(D, 'h4_ph_max_discount_pct_rho_partial')}). Die Rabatthäufigkeit liegt roh in der gepoolten Teilstichprobe {rel_word(share_raw)} der Relevanzschwelle ({de(share_raw, 3)}), im vollen Fenster {rel_word(share_full)} der Schwelle ({de(share_full, 3)}). Die Quartils-OR des Maximalrabatts ist nicht interpretierbar (die bedingte OR ist bei einer Nullzelle unendlich, exaktes KI {de(T(D, 'h4_ph_max_discount_pct_q', 'CI_unten'), 2)} bis ∞). Bei n = {n0(C.n_price_history_sample)} und einer Erfolgsquote nahe 100 % heißt „nicht signifikant“ nur „nicht nachweisbar“: Kleine Zusammenhänge sind mit den Konfidenzintervallen vereinbar.{' Mit dem geschätzten Markteintritt (`market_entry_year`) statt des Store-Jahres als Jahreskontrolle wird ' + join_de([C.ph.label_art[f] + ' (ρ ' + de(D.rc.loc[f'rc_entry_ph_{f}', 'Effekt'], 3) + ')' for f in C.entry_ph_sig]) + ' signifikant (Abschnitt 3.7).' if C.entry_ph_sig else ''}
- **Quellenübereinstimmung:** IGDB-Nutzerbewertung und Positive Ratio ({tci(D, 'h4_agree_igdbuser_pr')}) sowie Positive Ratio und Metascore ({tci(D, 'h3_agree_pr_mc')}) stimmen moderat überein; die Quellen messen verwandte, aber verschiedene Konstrukte.

**H4 wird als „{C.decision['H4']}“ eingestuft** (streng: „{C.decision_strict['H4']}“): IGDB-Modi und -Genres (als Konsistenzprüfung) und IGDB-Bewertungen sind gestützt, die Preishistorie nach der festgelegten Regel nicht. Bei wörtlicher Anwendung der Richtungsregel wäre auch die Preishistorie gestützt und H4 damit „{C.decision_literal['H4']}“ (Abschnitt 3.0); das Urteil hängt also an dieser Auslegung. Wegen unterschiedlicher Abdeckung und Zeiträume sind die Quellen komplementär, nicht austauschbar.
""")

    # ---- 3.5 Selektion
    sel = D.sel
    main_sel = sel[(sel["Schicht"] == "gesamt") | sel["Kennung"].str.startswith("sel_igdb_stage")]

    def sel_table(part: pd.DataFrame) -> str:
        return md(drop_empty(pd.DataFrame({
            "Verfügbarkeit": part["Verfuegbarkeit"],
            "Schicht": part["Schicht"],
            "N": part["N"].map(n0),
            "verfügbar": [f"{n0(a)} ({pct(b)})" for a, b in zip(part["N_verfuegbar"], part["Anteil_verfuegbar"])],
            "Erfolgsquote verfügbar / nicht": [f"{pct(a)} / {pct(b)}" for a, b in zip(part["Erfolgsquote_verfuegbar"], part["Erfolgsquote_nicht_verfuegbar"])],
            "Cliff's δ (Besitzer)": part["Effekt"].map(lambda v: de(v, 3)),
            "p (BH, δ)": part["p_adj"].map(pval),
            "OR Erfolg [95-%-KI]": [ci(a, b, c) for a, b, c in zip(part["OR_Erfolg"], part["OR_CI_unten"], part["OR_CI_oben"])],
            "p (BH, OR)": part["p_adj_OR"].map(pval),
            "Hinweis": part["Hinweis"].map(lambda v: de_text(v, english=True)),
        })))

    igdb_strata = sel[sel["Verfuegbarkeit"] == "IGDB-Treffer"]
    check(abs(sel.loc["sel_has_igdb_gesamt", "Effekt"]) < c3["REL_DELTA"], "IGDB-Treffer kaum selektiert")
    igdb_coh = igdb_strata[igdb_strata["Schicht"].str.startswith("Kohorte")]
    coh_eff = igdb_coh["Effekt"]
    check(coh_eff.min() < 0 < coh_eff.max(), "IGDB-Selektion kohortenabhängig mit wechselndem Vorzeichen")
    check(sel.loc["sel_igdb_rating_gesamt", "OR_Erfolg"] > 5 and sel.loc["sel_igdb_aggregated_rating_gesamt", "OR_Erfolg"] > 5,
          "IGDB-Bewertungen stark selektiert")

    def or_range(verf: str) -> str:
        part = sel[(sel["Verfuegbarkeit"] == verf) & sel["Schicht"].str.startswith("Kohorte")]
        return f"{de(part['OR_Erfolg'].min(), 1)} bis {de(part['OR_Erfolg'].max(), 1)}"

    no_en = sel[sel["Schicht"] == "ohne Englisch (mit Sprachangabe)"].set_index("Verfuegbarkeit")
    en = sel[sel["Schicht"] == "Englisch unterstützt"].set_index("Verfuegbarkeit")
    check(sel.loc["sel_igdb_stage3", "OR_Erfolg"] < 1 < sel.loc["sel_igdb_stage2", "OR_Erfolg"], "Stufe 3 weniger, Stufe 2 mehr erfolgreich")
    for verf in ["IGDB-Nutzerbewertung", "IGDB-Kritikerbewertung", "Metascore extern"]:
        coh_or = sel[(sel["Verfuegbarkeit"] == verf) & sel["Schicht"].str.startswith("Kohorte")]["OR_Erfolg"].to_numpy()
        check((np.diff(coh_or) > 0).all(), f"Selektion ({verf}) nimmt über die Kohorten zu")
    strong_sel = ["sel_igdb_rating_gesamt", "sel_igdb_aggregated_rating_gesamt", "sel_has_mc_external_gesamt", "sel_has_price_history_gesamt"]
    check((sel.loc[strong_sel, "Effekt"] > c3["REL_DELTA"]).all(), "Bewertungs-, Metascore- und Preishistorien-Stichproben: höhere Besitzerklassen")
    igdb_all = sel.loc["sel_has_igdb_gesamt"]
    modes_rate = D.igdb_modes.loc["with_modes", "Erfolgsquote"]
    check(igdb_all["Erfolgsquote_verfuegbar"] - igdb_all["Erfolgsquote_nicht_verfuegbar"] < 0.05
          and modes_rate - igdb_all["Erfolgsquote_nicht_verfuegbar"] < 0.06, "IGDB-Treffer und Modusangabe kaum selektiert")
    out.append(f"""### 3.5 Selektion der externen Teilstichproben

Gematchte und nicht gematchte Spiele werden über die Besitzerverteilung (Cliff's δ) und die Erfolgsquote (OR) verglichen; zusätzlich je Kohorte, nach Sprachunterstützung und je IGDB-Stufe (`external_selection_tests.csv`):

{sel_table(main_sel)}

IGDB-Treffer nach Schicht:

{sel_table(igdb_strata)}

Der **IGDB-Treffer** selbst ist kaum selektiert (δ = {de(sel.loc['sel_has_igdb_gesamt', 'Effekt'], 3)}). Das gilt aber nicht in jeder Kohorte: In {igdb_coh.loc[coh_eff.idxmin(), 'Schicht'].replace('Kohorte ', '')} haben gematchte Spiele eher weniger Besitzer (δ = {de(coh_eff.min(), 3)}), in {igdb_coh.loc[coh_eff.idxmax(), 'Schicht'].replace('Kohorte ', '')} mehr (δ = {de(coh_eff.max(), 3)}). Stufe-2-Treffer sind häufiger erfolgreich (OR {de(sel.loc['sel_igdb_stage2', 'OR_Erfolg'], 2)}), Stufe-3-Treffer seltener ({de(sel.loc['sel_igdb_stage3', 'OR_Erfolg'], 2)}). **Stark selektiert** sind die Teilstichproben mit IGDB-Nutzerbewertung (OR {de(sel.loc['sel_igdb_rating_gesamt', 'OR_Erfolg'], 1)}; je Kohorte {or_range('IGDB-Nutzerbewertung')}), mit IGDB-Kritikerbewertung (OR {de(sel.loc['sel_igdb_aggregated_rating_gesamt', 'OR_Erfolg'], 1)}; {or_range('IGDB-Kritikerbewertung')}), mit externem Metascore (OR {de(sel.loc['sel_has_mc_external_gesamt', 'OR_Erfolg'], 1)}; {or_range('Metascore extern')}) und mit Preishistorie (OR {de(sel.loc['sel_has_price_history_gesamt', 'OR_Erfolg'], 1)}; in einzelnen Kohorten mit Nullzellen). Bei IGDB-Bewertungen und Metascore nimmt die Selektion von Kohorte zu Kohorte zu. Unter Spielen ohne Englisch (bei vorhandener Sprachangabe) liegt die IGDB-Abdeckung bei {pct(no_en.loc['IGDB-Treffer', 'Anteil_verfuegbar'])} gegenüber {pct(en.loc['IGDB-Treffer', 'Anteil_verfuegbar'])} bei englischsprachigen Spielen; externe Bewertungen und Preishistorie fehlen dort fast vollständig.

Die Analysen der IGDB-Bewertungen, des externen Metascores und der Preishistorie sind deshalb **Analysen der in diesen Quellen dokumentierten, überwiegend erfolgreichen und englischsprachigen Spiele**; die Spiele mit diesen Daten haben überwiegend höhere Besitzerklassen (Cliff's δ {de(sel.loc[strong_sel, 'Effekt'].min(), 2)} bis {de(sel.loc[strong_sel, 'Effekt'].max(), 2)}). Für die IGDB-Modus- und Genre-Analysen gilt das kaum: Der IGDB-Treffer selbst ist nur schwach selektiert (Erfolgsquote {pct(igdb_all['Erfolgsquote_verfuegbar'])} gegenüber {pct(igdb_all['Erfolgsquote_nicht_verfuegbar'])} ohne Treffer, mit Modusangabe {pct(modes_rate)}; δ = {de(igdb_all['Effekt'], 3)}), allerdings je nach Kohorte in unterschiedlicher Richtung. Die Ergebnisse der externen Teilstichproben lassen sich nicht ohne Weiteres auf alle Steam-Spiele verallgemeinern.
""")

    # ---- 3.6 Review-Volumen
    dg = D.diag
    drows = []
    prefixes = sorted({re.match(r"(diag_\d+)_", k).group(1) for k in dg.index}, key=lambda s: int(s.split("_")[1]))
    for p in prefixes:
        part = dg[dg.index.str.startswith(p + "_")]
        raw = part[part.index.str.endswith("_raw")].iloc[0]
        vsr = part[part.index.str.endswith("_vs_reviews")].iloc[0]
        rvo = part[part.index.str.endswith("_reviews_owners")].iloc[0]
        par = part[part.index.str.endswith("_partial_reviews")].iloc[0]
        check(par["Effekt"] < 0 or par["p_adj"] >= ALPHA, f"{p}: nach Partialisierung ≤ 0 oder n. s.")
        drows.append({
            "Bewertungsmaß": raw["Merkmal"].replace(" vs. Besitzer (roh)", ""),
            "Stichprobe": raw["Stichprobe"],
            "N": n0(raw["N"]),
            "ρ(Reviews, Besitzer)": de(rvo["Effekt"], 3),
            "ρ(Maß, Reviews)": de(vsr["Effekt"], 3),
            "ρ(Maß, Besitzer) roh": ci(raw["Effekt"], raw["CI_unten"], raw["CI_oben"], 3),
            "partiell (Reviews herausgerechnet)": ci(par["Effekt"], par["CI_unten"], par["CI_oben"], 3)
            + ("" if par["p_adj"] < ALPHA else " (n. s.)"),
        })
    out.append(f"""### 3.6 Bewertungsvolumen: Diagnose, keine Kontrolle

Die Review-Anzahl ist selbst ein Popularitätsmaß. Rechnet man sie aus dem Zusammenhang zwischen Bewertung und Besitzern heraus, entfernt man einen Teil der Zielgröße (**Überkontrolle**). Die folgende Tabelle dient deshalb nur als **Diagnose der Volumenabhängigkeit** und wird einheitlich für alle Bewertungsmaße berechnet:

{md(pd.DataFrame(drows))}

Nach Herauspartialisieren der Review-Anzahl liegen alle Bewertungsmaße bei null oder darunter, auch Metascore und IGDB-Bewertungen. Das ist ein Artefakt der Überkontrolle und kein inhaltlicher Befund; die Partialisierung wird für kein Maß als Prüfkriterium verwendet. Aussagekräftig ist die Spalte „ρ(Maß, Reviews)“: Der Wilson-Score hängt viel stärker vom Review-Volumen ab als die Positive Ratio. Er wird deshalb nur zur Diagnose berichtet. Auch die Kritiker- und IGDB-Bewertungen korrelieren deutlich mit der Review-Anzahl; ihr Zusammenhang mit den Besitzern ist daher ebenfalls nicht frei von Popularitätseffekten.
""")

    # ---- 3.7 Robustheit
    rl = D.release
    variants = list(dict.fromkeys(rl["Variante"]))
    check(len(variants) == 5 and variants[3].startswith("ohne EA-Absolventen-Proxy")
          and variants[4] == "Jahr-FE mit market_entry_year", "fünf Varianten der Release-Sensitivität")
    rrows = []
    # Relative Aenderung bezogen auf das Hauptmodell: |aOR_Variante / aOR_Haupt - 1|.
    changes = {v: (0.0, "") for v in variants[1:]}
    # Richtungswechsel ueber ALLE Merkmale erfassen (auch nicht signifikante), damit
    # der Text keine Stabilitaet behauptet, die nur fuer signifikante Merkmale gilt.
    rel_flips = []
    for merkmal, part in rl.groupby("Merkmal", sort=False):
        part = part.set_index("Variante")
        main = part.loc[variants[0]]
        row = {"Merkmal": merkmal, "N (Haupt / ohne release_shifted / ohne EA-Proxy)":
               f"{n0(main['N'])} / {n0(part.loc[variants[1], 'N'])} / {n0(part.loc[variants[3], 'N'])}"}
        for v in variants:
            r = part.loc[v]
            row[v] = ci(r["Effekt"], r["CI_unten"], r["CI_oben"])
            if v != variants[0]:
                rel = abs(r["Effekt"] / main["Effekt"] - 1)
                if rel > changes[v][0]:
                    changes[v] = (rel, merkmal)
                if (r["Effekt"] - 1) * (main["Effekt"] - 1) <= 0:
                    rel_flips.append((merkmal, v, main["Effekt"], main["p"], r["Effekt"], r["p"]))
        rrows.append(row)
    rows_rel = md(pd.DataFrame(rrows))
    check(all(f[3] >= ALPHA and f[5] >= ALPHA for f in rel_flips),
          "Release-Sensitivität: Richtungswechsel nur bei Merkmalen, die in Haupt- und Variante n. s. sind")
    short = dict(zip(variants[1:], ["ohne `release_shifted`", "mit `release_year_alt`", "ohne Proxy-Fälle",
                                    "mit `market_entry_year`"]))
    if rel_flips:
        flip_items = []
        for merkmal, part in pd.DataFrame(rel_flips, columns=["m", "v", "e0", "p0", "e1", "p1"]).groupby("m", sort=False):
            e0, p0 = part["e0"].iloc[0], part["p0"].iloc[0]
            moves = join_de([f"{short[r.v]} {de(r.e1, 3)} (p = {psci(r.p1)})" for r in part.itertuples()])
            flip_items.append(f"„{merkmal}“: im Hauptmodell nicht signifikant (aOR {de(e0, 3)}, p = {psci(p0)}), {moves}")
        hidden = all(round(f[4], 2) == 1 for f in rel_flips)
        rel_dir_text = ("Alle im Hauptmodell signifikanten adjustierten ORs bleiben in allen Varianten richtungsstabil. "
                        "Die Richtung wechselt nur bei " + "; ".join(flip_items) + "."
                        + (" In der Tabelle ist das wegen der Rundung auf zwei Stellen nicht sichtbar." if hidden else ""))
    else:
        rel_dir_text = "Alle adjustierten ORs bleiben in allen Varianten in der Richtung stabil."
    ig_change = max((changes[v] + (short[v],) for v in variants[1:3]), key=lambda t: t[0])
    ea_change = max((changes[v] + (short[v],) for v in variants[3:]), key=lambda t: t[0])
    check(max(ig_change[0], ea_change[0]) < 0.2, "Release-Sensitivität ändert die aORs um weniger als 20 %")
    dq = D.dq
    ph_early = (int(dq.loc["ph_before_release", "Wert"]), int(dq.loc["ph_before_release_shifted", "Wert"]),
                int(dq.loc["ph_before_release_igdb_same", "Wert"]), int(dq.loc["ph_before_release_igdb_same", "N_Basis"]),
                int(dq.loc["ph_before_release_appid", "Wert"]))
    check(ph_early[1] <= 1 and ph_early[2] >= ph_early[3] - 1,
          "IGDB datiert Spiele mit Preisdaten vor dem Store-Datum fast immer wie Steam; release_shifted erfasst sie kaum")
    rs_lookup = (int(dq.loc["an_release_shifted_lookup", "Wert"]), int(dq.loc["an_release_shifted_lookup", "N_Basis"]))
    check(rs_lookup[0] == rs_lookup[1] == int(dq.loc["an_release_shifted", "Wert"]),
          "release_shifted stammt vollständig aus dem Namens-Lookup")
    ea_igdb = (int(dq.loc["an_entry_proxy_igdb_year", "Wert"]), int(dq.loc["an_entry_proxy_igdb_same", "Wert"]),
               int(dq.loc["an_entry_proxy_shifted", "Wert"]))
    check(ea_igdb[1] > ea_igdb[0] / 2 and 0 < ea_igdb[2] < ea_igdb[0] - ea_igdb[1],
          "EA-Proxy: IGDB meist wie Steam, ein Teil release_shifted")
    gap23 = dq.loc["an_release_shifted_gap23"]
    ph_e = D.analytic[D.analytic["entry_proxy_ph"]]
    check(len(ph_e) == ph_early[0], "Preishistorie-Proxy: Parquet und Datenqualitätstabelle gleich")
    ph_e_match = int(ph_e["has_igdb"].eq(True).sum())
    ea_n = int(dq.loc["an_ea_graduate_proxy", "Wert"])
    ea_succ = D.analytic.loc[D.analytic["ea_graduate_proxy"], "success_top25"].mean()
    rest_succ = D.analytic.loc[~D.analytic["ea_graduate_proxy"], "success_top25"].mean()
    check(int(D.analytic["ea_graduate_proxy"].sum()) == ea_n, "EA-Proxy: Parquet und Datenqualitätstabelle gleich")
    # Richtung und Staerke der rohen ORs ueber die drei Erfolgsdefinitionen
    rob_or = D.rob[D.rob["Kennung"].str.endswith("_or") & D.rob["Kennung"].str.contains("_success_")].copy()
    rob_or["feature"] = rob_or["Kennung"].str.replace(r"_success_(top25|50k|100k)_or$", "", regex=True)
    # Staerker heisst: gleiche Richtung unter allen drei Schwellen und |log OR| bei
    # der strengsten Schwelle groesser; Vorzeichenwechsel zaehlen nicht als staerker.
    stronger, flips, not_stronger = 0, [], []
    feats = list(dict.fromkeys(rob_or["feature"]))
    for f in feats:
        part = rob_or[rob_or["feature"] == f].set_index("Erfolgsdefinition")["Effekt"]
        logs = np.log(part[["success_top25", "success_50k", "success_100k"]].to_numpy(dtype=float))
        label = rob_or[rob_or["feature"] == f]["Merkmal"].iloc[0]
        same_sign = (logs > 0).all() or (logs < 0).all()
        if not same_sign:
            flips.append(label)
        elif abs(logs[2]) > abs(logs[0]):
            stronger += 1
        else:
            not_stronger.append(label)
    check(stronger > len(feats) / 2, "Mehrheit der ORs wird mit strengerer Schwelle in gleicher Richtung stärker")
    ph_keys = {f: f"rob_ph_{f}_success_top25_delta" for f in ["share_months_on_sale", "days_on_sale_per_month",
                                                                 "max_discount_pct"]}
    ph_delta = {f: rob(D, k) for f, k in ph_keys.items()}
    ph_p = {f: rob(D, k, "p") for f, k in ph_keys.items()}

    def n_fail(key: str) -> int:
        match = re.search(r"nein \(n=(\d+)\)", str(rob(D, key, "Gruppe")))
        check(match is not None, f"{key}: Gruppengröße nicht lesbar")
        return int(match.group(1))

    n_fail_ph, n_fail_full = n_fail(ph_keys["share_months_on_sale"]), n_fail(ph_keys["max_discount_pct"])
    check(ph_delta["share_months_on_sale"] < 0 and ph_delta["days_on_sale_per_month"] < 0
          and ph_p["share_months_on_sale"] >= ALPHA and ph_p["days_on_sale_per_month"] >= ALPHA
          and _isna(ph_delta["max_discount_pct"]),
          "Preishistorie: Rabattmerkmale unter Hauptdefinition negativ, aber nicht signifikant bzw. nicht schätzbar")
    ph_rate = D.sel.loc["sel_has_price_history_gesamt", "Erfolgsquote_verfuegbar"]

    # Kohortenanalysen mit alternativem Veroeffentlichungsjahr
    rc = D.rc
    rc_vars = [("main", "Steam-Jahr (Hauptanalyse)"), ("noshift", "ohne release_shifted"), ("alt", "release_year_alt"),
               ("noea", "ohne EA-Absolventen-Proxy"), ("entry", "market_entry_year")]
    n_coh = len(C.cohort_labels_03)
    rc_rows = []

    def rc_cell(key: str, kind: str) -> str:
        r = rc.loc[key]
        if kind == "rate":
            return f"{pct(r['Effekt'])} (n = {n0(r['N'])})"
        if kind == "q":
            return f"Q = {de(r['Effekt'], 1)}, p unkorr. = {psci(r['p'])}"
        return ci(r["Effekt"], r["CI_unten"], r["CI_oben"], 3 if kind == "rho" else 2)

    blocks = [("success", "Erfolgsquote", "rate"), ("pr", "Positive Ratio vs. Besitzer (ρ, mind. 10 Reviews)", "rho"),
              ("free", "kostenlos vs. kostenpflichtig (aOR Jahr-FE)", "or")]
    for b_key, b_label, kind in blocks:
        for i, c_label in enumerate(C.cohort_labels_03):
            rc_rows.append({"Analyse": b_label, "Kohorte": c_label,
                            **{v_label: rc_cell(f"rc_{v}_{b_key}_c{i}", kind) for v, v_label in rc_vars}})
        if b_key == "pr":
            rc_rows.append({"Analyse": b_label, "Kohorte": "Heterogenität (Cochran-Q)",
                            **{v_label: rc_cell(f"rc_{v}_pr_heterogeneity", "q") for v, v_label in rc_vars}})
    for f in C.ph.feats + ["max_discount_pct"]:
        lab = D.T.loc[f"h4_ph_{f}_rho", "Merkmal"].replace(" vs. Besitzer", "")
        rc_rows.append({"Analyse": "Preishistorie, partielles ρ (Jahr-FE + Fensterlänge; n. s.: p unkorr. ≥ 0,05)",
                        "Kohorte": lab,
                        **{v_label: rc_cell(f"rc_{v}_ph_{f}", "rho") + ("" if rc.loc[f"rc_{v}_ph_{f}", "p"] < ALPHA else " (n. s.)")
                           for v, v_label in rc_vars}})
    rc_tbl = md(pd.DataFrame(rc_rows))
    for v, _ in rc_vars:
        check(abs(rc.loc[f"rc_{v}_ph_{C.ph.feats[0]}", "Effekt"] - T(D, f"h4_ph_{C.ph.feats[0]}_rho_partial")) < 0.05,
              "Preishistorie-Sensitivität nahe an der Hauptspezifikation")
    check(all(abs(rc.loc[f"rc_main_ph_{f}", "Effekt"] - T(D, f"h4_ph_{f}_rho_partial")) < 1e-9
              for f in C.ph.feats + ["max_discount_pct"]), "Hauptvariante = Hauptspezifikation")
    succ_diff_noshift = max(abs(rc.loc[f"rc_noshift_success_c{i}", "Effekt"] - rc.loc[f"rc_main_success_c{i}", "Effekt"])
                            for i in range(n_coh))
    succ_diff_alt = max(abs(rc.loc[f"rc_alt_success_c{i}", "Effekt"] - rc.loc[f"rc_main_success_c{i}", "Effekt"])
                        for i in range(n_coh))
    alt_first = (rc.loc["rc_main_success_c0", "Effekt"], rc.loc["rc_alt_success_c0", "Effekt"],
                 rc.loc["rc_main_success_c0", "N"], rc.loc["rc_alt_success_c0", "N"])
    check(succ_diff_alt == abs(alt_first[1] - alt_first[0]), "größte Verschiebung bei release_year_alt in der ältesten Kohorte")
    for v, _ in rc_vars:
        check(rc.loc[f"rc_{v}_pr_c0", "Effekt"] > 0 and rc.loc[f"rc_{v}_pr_c{n_coh - 1}", "Effekt"] < 0
              and rc.loc[f"rc_{v}_pr_heterogeneity", "p"] < ALPHA, f"Positive-Ratio-Muster bleibt ({v})")
        check(rc.loc[f"rc_{v}_free_c{n_coh - 1}", "CI_unten"] < 1, f"kostenlos in jüngster Kohorte nicht positiv ({v})")
    free_last_vals = [rc.loc[f"rc_{v}_free_c{n_coh - 1}"] for v, _ in rc_vars]
    free_last_sig_neg = [lab for (v, lab), r in zip(rc_vars, free_last_vals) if r["CI_oben"] < 1]
    ph_shift = max(abs(rc.loc[f"rc_{v}_ph_{f}", "Effekt"] - rc.loc[f"rc_main_ph_{f}", "Effekt"])
                   for v, _ in rc_vars for f in C.ph.feats + ["max_discount_pct"])
    var_text = {"noshift": "ohne `release_shifted`", "alt": "mit `release_year_alt`",
                "noea": "ohne EA-Absolventen-Proxy", "entry": "mit `market_entry_year`"}
    last = n_coh - 1
    ea_last = {v: (rc.loc[f"rc_{v}_success_c{last}", "Effekt"], rc.loc[f"rc_{v}_success_c{last}", "N"])
               for v in ("main", "noea", "entry")}
    succ_diff_ea = max(abs(rc.loc[f"rc_{v}_success_c{i}", "Effekt"] - rc.loc[f"rc_main_success_c{i}", "Effekt"])
                       for v in ("noea", "entry") for i in range(n_coh))
    check(succ_diff_ea == max(abs(ea_last[v][0] - ea_last["main"][0]) for v in ("noea", "entry")),
          "EA-Proxy verschiebt vor allem die jüngste Kohorte")
    ph_sig_alt = [(v, var_text[v], f) for v, lab in rc_vars[1:] for f in C.ph.feats + ["max_discount_pct"]
                  if rc.loc[f"rc_{v}_ph_{f}", "p"] < ALPHA]
    def ph_item(v: str, f: str) -> str:
        r = rc.loc[f"rc_{v}_ph_{f}"]
        rel = ", über der Relevanzschwelle" if abs(r["Effekt"]) >= D.c03["REL_RHO"] else ""
        return f"{C.ph.label_art[f]} (ρ {de(r['Effekt'], 3)}, p = {psci(r['p'])}{rel})"

    by_variant = {}
    for v, lab, f in ph_sig_alt:
        by_variant.setdefault((v, lab), []).append(f)
    ph_sig_text = ("unkorrigiert signifikant werden " + "; ".join(
        f"{lab} {join_de([ph_item(v, f) for f in feats])}" for (v, lab), feats in by_variant.items())
                   if ph_sig_alt else "in keiner Variante wird ein Merkmal signifikant")
    ph_entry_rel = [f for v, _, f in ph_sig_alt if v == "entry" and abs(rc.loc[f"rc_entry_ph_{f}", "Effekt"]) >= D.c03["REL_RHO"]]
    out.append(f"""### 3.7 Robustheit: Veröffentlichungsdatum und Erfolgsdefinitionen

**Veröffentlichungsdatum.** Das Store-Datum ist bei Early-Access-Absolventen das Datum der Vollversion. **IGDB datiert diese Spiele teils ebenfalls auf die Vollversion, teils auf den Early-Access-Start.** `release_shifted` kann zudem nur Spiele ohne IGDB-Treffer und Abstände ab zwei Jahren erfassen: Alle {n0(rs_lookup[0])} Fälle stammen aus dem Namens-Lookup, {pct(gap23['Anteil'])} davon mit einem Abstand von zwei bis drei Jahren (der Größenordnung typischer Early-Access-Phasen, aber auch vieler Portierungen). Von den Spielen des Markteintritts-Proxys (siehe unten) mit IGDB-Jahr ({n0(ea_igdb[0])}) hat IGDB in {n0(ea_igdb[1])} Fällen dasselbe Jahr wie Steam, {n0(ea_igdb[2])} sind `release_shifted`. Ähnlich bei den {n0(ph_early[0])} Spielen der Preishistorie, deren Preisdaten mehr als einen Monat vor dem Store-Datum beginnen (das Preisfenster beginnt erst im Nov. 2023, die frühere Phase kann also deutlich länger sein): IGDB hat in {n0(ph_early[2])} von {n0(ph_early[3])} Fällen mit IGDB-Jahr dasselbe Jahr wie Steam, {n0(ph_early[1])} ist `release_shifted`; {n0(ph_e_match)} von ihnen haben einen IGDB-Treffer und können schon deshalb nicht `release_shifted` sein. `release_shifted` und `release_year_alt` erfassen daher vor allem Neuauflagen, Portierungen und frühere Veröffentlichungen außerhalb von Steam, Early-Access-Phasen nur teilweise. Ergänzend dient ein grober **Markteintritts-Proxy** (`ea_graduate_proxy`, nur Spiele ohne aktuelle Early-Access-Markierung): AppID unter dem Median der AppIDs des Jahrgangs {D.c01['APPID_PROXY_YEARS']} Jahre vor dem Store-Jahr (Referenzjahrgänge mit mindestens {n0(D.c01['APPID_REF_MIN_N'])} Spielen) oder erster Preismonat der Preishistorie mehr als einen Monat vor dem Store-Datum. Das trifft auf {n0(ea_n)} Spiele zu ({n0(dq.loc['an_entry_proxy_appid', 'Wert'])} über die AppID, {n0(dq.loc['an_entry_proxy_ph', 'Wert'])} über die Preishistorie, davon {n0(ph_early[4])} auch über die AppID); nur {pct(dq.loc['an_entry_proxy_shifted', 'Anteil'])} davon sind zugleich `release_shifted`. Sie sind häufiger erfolgreich ({pct(ea_succ)} gegen {pct(rest_succ)}). Der Proxy mischt Early-Access-Absolventen mit lange vorab angekündigten Titeln und erkennt kürzere Early-Access-Phasen nicht; `market_entry_year` (geschätztes Jahr des Markteintritts: AppID-Jahr bzw. Jahr des ersten Preismonats) ist entsprechend grob.

Adjustierte ORs (`release_date_sensitivity.csv`) im Hauptmodell, ohne die {n0(D.dq.loc['an_release_shifted', 'Wert'])} Spiele mit früherer IGDB-Erstveröffentlichung (`release_shifted`), mit dem früheren Jahr als Jahr-FE (`release_year_alt`), ohne die {n0(ea_n)} Spiele des Markteintritts-Proxys und mit `market_entry_year` als Jahr-FE:

{rows_rel}

{rel_dir_text} Bezogen auf das Hauptmodell ändern sich die adjustierten ORs mit den IGDB-Varianten relativ um höchstens {de(ig_change[0] * 100, 1)} % ({ig_change[1]}, {ig_change[2]}), mit den Varianten des Markteintritts-Proxys um höchstens {de(ea_change[0] * 100, 1)} % ({ea_change[1]}, {ea_change[2]}). Weil der Proxy unvollständig ist, belegt das nur, dass die erkennbaren Fälle die Ergebnisse wenig verschieben, nicht, dass Early-Access-Phasen insgesamt ohne Einfluss sind.

**Kohortenanalysen mit alternativem Veröffentlichungsjahr** (`release_cohort_sensitivity.csv`): Kohortenzuordnung nach dem Steam-Jahr, ohne die Spiele mit `release_shifted`, nach `release_year_alt` (Jahre vor 1997 zählen zur ältesten Kohorte), ohne die Spiele des Markteintritts-Proxys und nach `market_entry_year`; die Jahr-FE folgen jeweils derselben Jahresvariable:

{rc_tbl}

Ohne `release_shifted` ändern sich die Erfolgsquoten der Kohorten um höchstens {de(succ_diff_noshift * 100, 1)} Prozentpunkte. Mit `release_year_alt` wandern Neuauflagen und Portierungen in ältere Kohorten; die Quote der ältesten Kohorte sinkt deshalb von {pct(alt_first[0])} (n = {n0(alt_first[2])}) auf {pct(alt_first[1])} (n = {n0(alt_first[3])}). Der Markteintritts-Proxy wirkt vor allem auf die jüngste Kohorte: Ihre Erfolgsquote sinkt ohne die Proxy-Fälle von {pct(ea_last['main'][0])} (n = {n0(ea_last['main'][1])}) auf {pct(ea_last['noea'][0])} (n = {n0(ea_last['noea'][1])}), mit `market_entry_year` auf {pct(ea_last['entry'][0])} (n = {n0(ea_last['entry'][1])}); jüngere Jahrgänge sind also etwas weniger erfolgreich, als das Store-Datum nahelegt. Das Muster der Positive Ratio bleibt in allen Varianten erhalten: in der ältesten Kohorte positiv, in der jüngsten negativ, signifikant heterogen. Kostenlose Spiele sind in der jüngsten Kohorte in keiner Variante positiv assoziiert (aOR {' / '.join(de(r['Effekt'], 2) for r in free_last_vals)}){'; mit ' + ', '.join(f'`{x}`' for x in free_last_sig_neg) + ' sogar signifikant negativ' if free_last_sig_neg else ''}. Die partiellen Preishistorien-Korrelationen ändern sich um höchstens {de(ph_shift, 3)}; {ph_sig_text}.{' Die Preishistorien-Befunde hängen damit auch an der Alterskontrolle: Mit dem geschätzten Markteintritt statt des Store-Jahres ist ' + join_de([C.ph.label_art[f] for f in ph_entry_rel]) + ' signifikant und praktisch relevant, in der Hauptspezifikation dagegen kein Merkmal.' if ph_entry_rel else ''}

**Erfolgsdefinitionen** (`robustness_success_definitions.csv`): Die Entscheidungstabellen in 3.1 bis 3.4 zeigen jeden Teiltest zusätzlich unter {C.thr_label['success_50k']} und {C.thr_label['success_100k']} Besitzern; ein Teiltest gilt nur als gestützt, wenn die Richtung dort erhalten bleibt. Bei {stronger} von {len(feats)} binären Merkmalen wird die rohe Odds Ratio mit der strengsten Schwelle in gleicher Richtung stärker; nicht stärker werden {', '.join(not_stronger) if not_stronger else 'keine'}. Das Vorzeichen wechselt bei {', '.join(flips) if flips else 'keinem Merkmal'}. Bei den stetigen Maßen wechselt das Q4/Q1-Verhältnis der Positive Ratio die Seite (Abschnitt 3.3). In der Preishistorie sind die Cliff's δ von Rabattmonaten und Rabatttagen je Monat unter {C.thr_label['success_top25']} negativ ({de(ph_delta['share_months_on_sale'], 3)} bzw. {de(ph_delta['days_on_sale_per_month'], 3)}), aber nicht signifikant (p = {psci(ph_p['share_months_on_sale'])} bzw. {psci(ph_p['days_on_sale_per_month'])}). Sie beruhen auf nur {n0(n_fail_ph)} nicht erfolgreichen von {n0(rob(D, ph_keys['share_months_on_sale'], 'N'))} Spielen und sind wegen der Erfolgsquote von {pct(ph_rate)} nicht interpretierbar. Der maximale Rabatt ist dort nicht schätzbar ({n0(n_fail_full)} nicht erfolgreiche Spiele im vollen Fenster).
""")
    return "\n".join(out)


def section4(D, C) -> str:
    jtop = D.joint[D.joint["Erfolgsdefinition"] == "success_top25"].set_index("Variable")
    ident = D.ident.loc["ident_mc_metascore_vs_positive_ratio"]
    c_last = C.cohort_labels_03[-1]
    free_last = C.free_last
    ch = D.cohort_h3
    n_coh = len(C.cohort_labels_03)
    check(T(D, "h1_genre_action_or") > 1 and sig(D, "h1_genre_action_or") and T(D, "h1_genre_action_aor") > 1
          and sig(D, "h1_genre_action_aor") and covers(jtop.loc["genre_action", "CI_unten"], jtop.loc["genre_action", "CI_oben"], 1.0)
          and not covers(jtop.loc["genre_adventure", "CI_unten"], jtop.loc["genre_adventure", "CI_oben"], 1.0)
          and not covers(jtop.loc["genre_strategy", "CI_unten"], jtop.loc["genre_strategy", "CI_oben"], 1.0),
          "Action positiv, im gemeinsamen Modell n. s.; Strategy und Adventure dort signifikant")
    mc_coh = [ch.loc[f"cohort_3_c{i}", "Effekt"] for i in range(n_coh)]
    ph = C.ph
    # Welche der in F1 genannten Merkmale sind im gemeinsamen Modell nicht mehr
    # signifikant (bei >= 20.000 bzw. unter allen drei Erfolgsdefinitionen)?
    f1_feats = {"genre_rpg": "RPG", "genre_strategy": "Strategy", "genre_adventure": "Adventure",
                "genre_action": "Action", "genre_casual": "Casual", "genre_sports": "Sports", "genre_indie": "Indie",
                "genre_racing": "Racing", "multi_player": "Multiplayer", "coop": "Koop", "mmo": "MMO",
                "pvp": "PvP", "mac": "macOS"}
    jall = D.joint.set_index(["Variable", "Erfolgsdefinition"])
    defs = list(dict.fromkeys(D.joint["Erfolgsdefinition"]))

    def joint_ns(var: str, outcome: str) -> bool:
        return covers(jall.loc[(var, outcome), "CI_unten"], jall.loc[(var, outcome), "CI_oben"], 1.0)

    ns_joint = [lab for v, lab in f1_feats.items() if joint_ns(v, "success_top25")]
    ns_all = [lab for v, lab in f1_feats.items() if all(joint_ns(v, o) for o in defs)]
    check("Action" in ns_joint and len(defs) == 3, "Action im gemeinsamen Modell n. s.; drei Erfolgsdefinitionen")
    check(T(D, "h4_igdbcrit_rho") > 0 and T(D, "h4_igdbuser_nonint_rho") > 0 and T(D, "h4_igdbuser_all_rho") > 0
          and T(D, "h4_igdbcrit_rho", "p_adj") < ALPHA and T(D, "h4_igdbuser_nonint_rho", "p_adj") < ALPHA,
          "IGDB: Kritiker- und Nutzerbewertung signifikant positiv")
    mc_user_ident = D.ident.loc["ident_mc_metascore_vs_mc_user_score"]
    check(mc_user_ident["CI_unten"] > 0, "Metacritic: Kritikerscore in identischer Stichprobe stärker als Nutzerscore")
    check(T(D, "h4_agree_igdbuser_pr") >= 0.5, "IGDB-Nutzerbewertung eng mit der Positive Ratio verknüpft")
    ns_text = (f"Im gemeinsamen Modell (Genres, Modi und Plattformen zusammen, mit Jahr-FE) sind bei "
               f"{C.thr_label['success_top25']} Besitzern {join_de(ns_joint)} nicht mehr signifikant"
               + (f" ({join_de(ns_all)} unter keiner der drei Erfolgsschwellen)" if ns_all else "")
               + "; das Genre Massively Multiplayer geht dort im MMO-Modus auf "
               f"(aOR {de(jtop.loc['genre_massively_multiplayer', 'Effekt'], 2)}, n. s.).")
    return f"""## 4. Beantwortung der Forschungsfragen

### F1 – Spieleigenschaften

**Welche Genres, Spielmodi und Plattformen sind bei erfolgreichen Spielen überdurchschnittlich häufig vertreten?**

Überdurchschnittlich vertreten sind Massively Multiplayer, RPG und, schwächer, Strategy, Adventure und Action; unterdurchschnittlich Casual sowie, schwächer, Sports, Indie und Racing. Mehrspieler-, Koop- und MMO-Spiele sind deutlich häufiger erfolgreich, PvP schwächer. Bei den Plattformen gilt das für macOS bzw. Windows + macOS (+ Linux), nicht für Linux ohne macOS. Diese Zusammenhänge bleiben in den Einzelmodellen nach Kontrolle des Veröffentlichungsjahres bestehen; Simulation ist nicht robust. {ns_text} Die adjustierten Plattform-ORs sind klein (unter {de(D.c03['REL_OR'], 1)}). IGDB zeigt, wo das Steam-Merkmal eine gesicherte Richtung hat, dieselben Richtungen, ist wegen der hohen Übereinstimmung mit den Steam-Kategorien aber nur eine Konsistenzprüfung. H1 wird nach der Hauptregel als „{C.decision['H1']}“, nach der strengen Regel als „{C.decision_strict['H1']}“ eingestuft.

### F2 – Preis

**Welcher Zusammenhang besteht zwischen Preisstrategie und Popularität?**

Kostenlose Spiele (Listenpreis 0) sind gepoolt häufiger erfolgreich. Dieser Zusammenhang gilt aber nur für ältere Kohorten (in {c_last} aOR {de(free_last['aOR_JahrFE'], 2)}, n. s.), hängt am Ausschluss der Einträge mit „0 - 0“ und an den als „Free To Play“ markierten Titeln. Unter kostenpflichtigen Spielen hängt ein höherer Listenpreis schwach positiv mit den Besitzern zusammen; das gilt fast nur für Spiele, die zum Snapshot rabattiert sind. Auch über 40 USD steigt die Erfolgsquote weiter ({pct(D.prices.loc['p40plus', 'Erfolgsquote'])} gegen {pct(D.prices.loc['p20_40', 'Erfolgsquote'])} bei 20–39,99 USD), wenn Preisausreißer über 100 USD und das {de(D.cc['CLUSTER_PRICE'], 2)}-USD-Cluster getrennt betrachtet werden; einen Einbruch bei teuren Titeln gibt es nicht. Ein aktiver Rabatt geht mit einer deutlich höheren Erfolgsquote einher, nach Altersstandardisierung gepoolt aber ohne monotonen Gradienten über die Rabatthöhe; {discount_cohort_pattern(D).short}. Die Richtung ist offen: Bereits populäre Titel könnten häufiger rabattiert werden, Rabatte könnten aber auch die Verbreitung fördern; zudem ist der Rabatt später gemessen als die Besitzer und unvollständig erfasst. In der Preishistorie (normierte Merkmale, n = {n0(C.n_price_history_sample)}) korrelieren Rabattmerkmale roh positiv mit den Besitzern. Nach Kontrolle von Jahr und Fensterlänge ist in der Hauptspezifikation keines signifikant; im vollen Beobachtungsfenster (n = {n0(ph.full_n)}) bleiben kleine positive Zusammenhänge ({ph.full_text_short}). Wegen anderer Konstrukte (Rabattverlauf statt Rabatt zum Snapshot) und einer fast vollständig erfolgreichen Teilstichprobe ist das weder eine Bestätigung noch ein Widerspruch zum Rabattzusammenhang der Vollstichprobe. H2 wird als „{C.decision['H2']}“ eingestuft, mit den genannten Einschränkungen.

### F3 – Bewertungen

**Wie stark hängen Nutzer- und Kritikerbewertungen mit Popularität zusammen?**

Kritikerbewertungen (Metascore) hängen moderat positiv mit den Besitzern zusammen; über die Kohorten ist der Zusammenhang nicht signifikant verschieden, nimmt aber tendenziell ab (ρ {de(mc_coh[0], 2)} bis {de(mc_coh[-1], 2)}). In derselben Teilstichprobe gilt das auch für die Steam-Positive-Ratio, aber etwas schwächer (ρ {de(ident['rho_A'], 2)} gegen {de(ident['rho_B'], 2)}). In der Breite der Bewertungsstichprobe gibt es keinen monotonen Zusammenhang der Nutzerbewertung: Die mittleren Bewertungsbereiche sind am häufigsten erfolgreich. Die Konfundierung über die Jahrgänge (neuere Jahrgänge werden besser bewertet und haben weniger Besitzer) drückt die gepoolte Rangkorrelation von {de(T(D, 'h3_pr_r10_rho_partial'), 3)} (partiell) auf {de(T(D, 'h3_pr_r10_rho'), 3)}; auch innerhalb der Jahrgänge bleibt sie im Mittel unter der Relevanzschwelle. Bestimmend ist die Heterogenität über die Kohorten: bei älteren Titeln deutlich positiv (ρ {de(ch.loc['cohort_0_c0', 'Effekt'], 2)}) und bei Neuerscheinungen negativ ({de(ch.loc[f'cohort_0_c{n_coh - 1}', 'Effekt'], 2)}). Der Wilson-Score ist wegen seiner Volumenabhängigkeit nur eine Diagnose; die Kontrolle auf das Review-Volumen ist Überkontrolle und wird nicht als Befund gewertet. Die IGDB-Bewertungen hängen in ihren stark selektierten Teilstichproben wie die Metacritic-Bewertungen und anders als die Positive Ratio in der Breite positiv mit den Besitzern zusammen (Kritikerbewertung ρ {de(T(D, 'h4_igdbcrit_rho'), 3)}, n = {n0(T(D, 'h4_igdbcrit_rho', 'N'))}; nicht ganzzahlige Nutzerbewertungen {de(T(D, 'h4_igdbuser_nonint_rho'), 3)}, n = {n0(T(D, 'h4_igdbuser_nonint_rho', 'N'))}; alle Nutzerbewertungen {de(T(D, 'h4_igdbuser_all_rho'), 3)}, n = {n0(T(D, 'h4_igdbuser_all_rho', 'N'))}; verschiedene Teilstichproben). Anders als bei Metacritic ist die Kritikerbewertung aber nicht stärker: In identischer Stichprobe (n = {n0(C.igdb_ident['N'])}) sind beide etwa gleich stark ({C.igdb_ident_text}). Eine unabhängige Bestätigung ist das nicht, denn die IGDB-Nutzerbewertung hängt eng mit der Positive Ratio zusammen (ρ {de(T(D, 'h4_agree_igdbuser_pr'), 3)}). H3 wird als „{C.decision['H3']}“ eingestuft: gestützt für Kritikerbewertungen, nicht für die Nutzerbewertung in der Breite.

### Quellenübergreifende Einordnung

Die vier Quellen messen Unterschiedliches: Steam liefert Reichweite und Nutzerfeedback, IGDB zusätzliche Klassifikationen und Bewertungen, Metacritic das Kritikerurteil, die Preishistorie die Rabattdynamik einer kleinen Teilmenge. Wo sie dasselbe messen und die Richtung gesichert ist, stimmen sie überein. Scheinbare Widersprüche gehen auf die Behandlung fehlender Angaben (fehlende IGDB-Modi sind unbekannt, nicht „nein“) oder auf verschiedene Stichproben zurück (Positive Ratio in der Breite gegenüber der Metacritic-Teilstichprobe). Die externen Teilstichproben mit Bewertungen und Preishistorie sind stark selektiert; ihre Ergebnisse ergänzen den Steam-Querschnitt, ersetzen ihn aber nicht. H4 wird als „{C.decision['H4']}“ eingestuft; bei wörtlicher Anwendung der Richtungsregel wäre H4 „{C.decision_literal['H4']}“ (Abschnitt 3.0). Die Analyse ist eine Multi-Source-Beschreibung von Zusammenhängen, keine Prognose und keine kausale Identifikation.
"""


def section5(D, C) -> str:
    fr = C.fr
    m = C.match
    ph0 = D.dq.loc["ph_disc0_le90"]
    ea_max = C.ea_max_change
    n_last = len(C.cohort_labels_03) - 1
    entry_ph = C.entry_ph_sig
    entry_ph_text = ("; in der Preishistorie wird mit `market_entry_year` als Jahreskontrolle "
                     + join_de([f"{C.ph.label_art[f]} (ρ {de(D.rc.loc[f'rc_entry_ph_{f}', 'Effekt'], 3)})" for f in entry_ph])
                     + " signifikant, die Preishistorien-Befunde hängen also an der Alterskontrolle") if entry_ph else ""

    def share(pair: tuple[int, int]) -> str:
        return pct(pair[0] / pair[1])

    return f"""## 5. Grenzen der Umsetzung

- **Grobe, kumulierte Zielgröße:** Die Besitzer liegen nur als Klassen vor; {pct(D.owners.sort_values('owners_class').iloc[0]['Anteil'])} der Analysestichprobe liegen in der untersten Klasse. „Erfolg“ heißt praktisch, diese Klasse zu verlassen. Die Besitzerzahl ist kumuliert und hängt stark vom Alter ab. Jahr-Fixed-Effects und Altersstandardisierung mildern das, Restkonfundierung bleibt möglich.
- **Zeitliche Inkonsistenz der Quellen:** {C.store_stand}, {C.steamspy_stand}, IGDB „{D.prov.loc['igdb', 'Datenstand']}“, Metacritic „{D.prov.loc['metacritic_raw', 'Datenstand']}“, Preishistorie „{D.prov.loc['price_history', 'Datenstand']}“. Preis und Rabatt stammen damit aus einem deutlich späteren Stand als Besitzer und Reviews. Der Jahrgang {STEAMSPY_CUTOFF.year} ist nur bis Mai beobachtet, und die jüngsten Spiele hatten wenig Zeit, Besitzer zu sammeln.
- **Stichtag:** Spiele ab dem {C.cutoff} sind ausgeschlossen, weil ihre SteamSpy-Felder strukturell fehlen; darunter Early-Access-Absolventen mit späterem Store-Datum ({n0(D.flow.loc['removed_after_cutoff_with_steamspy', 'N'])} mit SteamSpy-Daten).
- **Fehlende Besitzerangaben „0 - 0“:** Der Ausfall ist nicht zufällig (Anteil steigt ab 2021 stark, überwiegend kostenlose Einträge). Erfolgsquote, Kohortenverlauf und der Befund zu kostenlosen Spielen hängen an diesem Ausschluss (Abschnitt 3.2).
- **SteamSpy-Lücke:** {n0(D.dq.loc['steamspy_missing', 'Wert'])} Einträge des Rohdatensatzes haben trotz Store-Recommendations keine SteamSpy-Daten, darunter bekannte Titel; sie sind ausgeschlossen. Recommendations wären ein möglicher Ersatz, sind aber nicht mit den Reviews vergleichbar.
- **Veröffentlichungsdatum:** `release_year` ist das aktuelle Store-Datum; bei Early-Access-Absolventen ist es das Datum der Vollversion, bei Neuauflagen und Portierungen das Steam-Datum. IGDB datiert Early-Access-Absolventen teils ebenfalls auf die Vollversion, und `release_shifted` greift nur bei Spielen ohne IGDB-Treffer (Namens-Lookup) und ab zwei Jahren Abstand: `release_shifted` und `release_year_alt` erfassen deshalb vor allem Neuauflagen und Portierungen, Early-Access-Phasen nur teilweise (von {n0(D.dq.loc['an_entry_proxy_igdb_year', 'Wert'])} Spielen des Markteintritts-Proxys mit IGDB-Jahr haben {n0(D.dq.loc['an_entry_proxy_igdb_same', 'Wert'])} dasselbe Jahr wie Steam, {n0(D.dq.loc['an_entry_proxy_shifted', 'Wert'])} sind `release_shifted`). Ihre Robustheit (die adjustierten ORs ändern sich kaum; `release_year_alt` verschiebt vor allem die älteste Kohorte, Erfolgsquote {pct(D.rc.loc['rc_alt_success_c0', 'Effekt'])} statt {pct(D.rc.loc['rc_main_success_c0', 'Effekt'])}) sagt daher nichts über Early Access. Dafür gibt es nur einen groben Markteintritts-Proxy über AppID und ersten Preismonat ({n0(D.dq.loc['an_ea_graduate_proxy', 'Wert'])} Spiele, gemischt mit lange angekündigten Titeln, kürzere Early-Access-Phasen unerkannt). Mit ihm ändern sich die adjustierten ORs um höchstens {de(ea_max * 100, 1)} %, die Erfolgsquote der jüngsten Kohorte sinkt von {pct(D.rc.loc[f'rc_main_success_c{n_last}', 'Effekt'])} auf {pct(D.rc.loc[f'rc_noea_success_c{n_last}', 'Effekt'])} (ohne Proxy-Fälle){entry_ph_text} (Abschnitt 3.7). Restkonfundierung durch nicht erkannte Early-Access-Phasen bleibt möglich.
- **Preis und Rabatt:** Der Listenpreis ist aus Tagespreis und Rabatt rekonstruiert und wegen teils fehlender Rabatte eine Untergrenze; Rabattbefunde sind entsprechend verunreinigt. In der (stark selektierten) Preishistorie-Teilstichprobe kosten {n0(ph0['Wert'])} von {n0(ph0['N_Basis'])} Spielen ohne erfassten Rabatt höchstens 90 % des Basispreises; fehlende Rabatte verzerren den Rabattzusammenhang eher zur Null. Bei {n0(D.dq.loc['an_list_price_undefined', 'Wert'])} Spielen ist der Listenpreis nicht bestimmbar (100 % Rabatt bzw. Tagespreis 0 trotz Rabatt); sie fehlen in den Preisvergleichen. Preisausreißer über 100 USD und ein Cluster von {n0(D.dq.loc['an_price_cluster', 'Wert'])} Titeln zu {de(D.cc['CLUSTER_PRICE'], 2)} USD (Release ab {D.cc['CLUSTER_FROM_YEAR']}) verzerren die oberste Preisklasse; in der Preisübersicht (Abschnitt 2.5) und in `fig_05` werden beide getrennt ausgewiesen. Die Rangtests in Abschnitt 3.2 schließen nur die Ausreißer aus, das Cluster bleibt enthalten (ohne Cluster partiell ρ {de(T(D, 'h2_price_rho_nocluster_partial'), 3)} statt {de(T(D, 'h2_price_rho_partial'), 3)}). Bei {n0(D.dq.loc['an_list_price_conflict', 'Wert'])} rabattierten Spielen widerspricht die Rabattangabe dem Tagespreis (z. B. 39,99 USD bei 85 % Rabatt ergäbe 266,60 USD); dort gilt der Tagespreis als Listenpreis, die Rabattgruppe folgt weiter der Quelle. Weitere {n0(D.dq.loc['an_list_price_off_point', 'Wert'])} rabattierte Spiele ({pct(D.dq.loc['an_list_price_off_point', 'Anteil'])}) haben ungewöhnliche, aber mit Tagespreis und Rabatt vereinbare Listenpreise (nicht x,99 oder ganzer Dollar); sie bleiben unverändert.
- **„Kostenlos“:** Listenpreis 0 umfasst neben echten Free-to-Play-Titeln auch delistete, werbefinanzierte und sonstige Gratisangebote. „Free Editions“ und Testserver gelten als Nicht-Spiele, obwohl einzelne spielbare F2P-Produkte sind.
- **Peak CCU:** Momentaufnahme eines Tages (Vortag des SteamSpy-Abrufs), kein Allzeit-Peak; bei {pct(D.dq.loc['an_peak_zero', 'Anteil'])} der Spiele gleich 0.
- **Matching:** Namensbasiertes Matching kann Editionen, Remakes und gleichnamige Spiele verwechseln oder verfehlen. Die Recall-Referenz liegt bei {pct(sc(D, 'IGDB', 'Recall-Referenz', 'N_analytisch') / D.flow.loc['meta_sample', 'N'])}. Namen mit nicht-lateinischem Schlüssel werden kaum gematcht ({n0(m.nl['key'].matched)} von {n0(m.nl['key'].total_an)}); gemischtschriftliche Namen werden auf ihren lateinischen Teil reduziert, was generische Schlüssel erzeugen kann. Stufe-3-Treffer sind spärlich dokumentiert und werden nur ohne Firmenwiderspruch übernommen. Widersprüchliche Firmen bei IGDB-Treffern der Stufen 1 und 2 ({share(m.firm[('IGDB', 1)])} bzw. {share(m.firm[('IGDB', 2)])}) und bei Metacritic ({share(m.firm[('Metacritic', 1)])} bzw. {share(m.firm[('Metacritic', 2)])}) sowie seltener gleiche Metascores in Stufe 2 ({share(m.score_eq[2])} gegenüber {share(m.score_eq[1])} in Stufe 1) zeigen einzelne Fehlzuordnungen und Versionsverwechslungen. Eine vollständige manuelle Prüfung fehlt.
- **IGDB:** Keine unabhängige Quelle für Genres und Modi (κ {de(C.kappa.min(), 2)}–{de(C.kappa.max(), 2)}). Ohne Stimmenzahlen ist die Nutzerbewertung nur über die Ganzzahligkeit nach Volumen zu stratifizieren; es gibt eine starke Häufung bei 70. Momentaufnahme vom Juli 2025.
- **Metacritic extern:** Nur PC-Einträge; das Metacritic-Datum ist das spielweite Erstveröffentlichungsdatum, späte PC-Portierungen werden nur über die Stufe 2 erfasst. Der Nutzerscore fehlt teils ({pct(1 - D.dq.loc['mc_user_score', 'Anteil'])} der Teilstichprobe). Die Werte decken sich mit der FronkonGames-Spalte, die Abdeckung nicht.
- **Preishistorie:** Nur {n0(C.n_price_history_sample)} Apps, fast alle erfolgreich; unterschiedlich lange Fenster (nur {pct(D.dq.loc['ph_full_window', 'Anteil'])} mit vollem Fenster); monatliche Aggregation; statischer Basispreis der Quelle (Rabatte und Rabatttage beziehen sich auf einen festen Referenzpreis; {n0(D.dq.loc['ph_lowest_above_base', 'Wert'])} Apps mit Tiefstpreis über dem Basispreis); unvollständiger Monat 2025-11 verworfen; doppelte App-Monate in der Quelle; keine kostenlosen Titel; einzelne falsche Namensangaben in der Quelle.
- **Selektion:** Review-Schwelle, externe Bewertungen und Preishistorie verschieben die Teilstichproben zu populären, englischsprachigen und älteren Spielen; der IGDB-Treffer selbst ist nur schwach selektiert (Abschnitt 3.5).
- **Bewertungsmaße:** Der Wilson-Score enthält das Review-Volumen; aber auch die übrigen Bewertungen korrelieren mit der Review-Anzahl. Eine Kontrolle auf das Volumen wäre Überkontrolle. Die Wahl der Positive Ratio als Hauptmaß fiel nach der Volumendiagnose.
- **Multiples Testen und Entscheidungsregeln:** BH-Korrektur je Tabelle; bei sehr großen N sind auch kleine Zusammenhänge signifikant, deshalb die Relevanzschwellen. Schwellen und Entscheidungsregel wurden erst während der Auswertung festgelegt (nicht präregistriert). Die Auslegung „Richtung im adjustierten Modell gesichert“ und die vier Kohorten sind Umsetzungsentscheidungen (Abschnitt 3.0); vom ersten hängt das H4-Urteil ab (wörtlich angewandt: „{C.decision_literal['H4']}“).
- **Keine Prognose, keine Kausalität:** Es werden bewusst keine Vorhersagemodelle geschätzt; alle Aussagen sind deskriptiv und assoziativ.
"""


def section6(D, C) -> str:
    fr = C.fr
    csv_from_01 = ["data_provenance", "steamspy_coverage_by_month", "sample_flow", "exclusion_samples",
                   "data_quality_summary", "success_thresholds", "missing_values", "variable_overview",
                   "source_coverage", "source_counts", "metacritic_overlap", "source_overlap"]
    for name in csv_from_01:
        check((TAB / f"{name}.csv").exists(), f"{name}.csv fehlt")
    # Die dokumentierte Determinismuspruefung gilt nur fuer die damalige Menge der Ausgaben.
    check(len(list(TAB.glob("*.csv"))) == DETERMINISM_CHECK["csv"]
          and len(list(DATA.glob("*.parquet"))) == DETERMINISM_CHECK["parquet"]
          and len(list(FIG.glob("*.png"))) == DETERMINISM_CHECK["png"],
          "Menge der Ausgaben weicht von der dokumentierten Determinismusprüfung ab")
    req = (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    py_version = (ROOT / ".python-version").read_text(encoding="utf-8").strip()
    verified = re.search(r"verifiziert mit (\d+\.\d+\.\d+)", req[0])
    check(verified is not None and verified.group(1).startswith(py_version), "Python-Version in requirements.txt")
    pins = ", ".join(f"`{line.strip()}`" for line in req if line.strip() and not line.startswith("#"))
    return f"""## 6. Reproduzierbarkeit

**Umgebung:** Python {py_version} (festgehalten in `.python-version` und im Kopf von `requirements.txt`; verifiziert mit {verified.group(1)}). Die gepinnten Versionen von pandas und numpy bieten für neuere Python-Versionen keine fertigen Pakete. Alle direkten Abhängigkeiten sind in `requirements.txt` exakt gepinnt ({pins}); transitive Abhängigkeiten sind nicht gelockt.

**Aufbau:**

- `scripts/common.py` – importiertes Modul: Pfade, Seed ({RANDOM_STATE}), SteamSpy-Stichtag, Quellentabelle `SOURCES` (Version, Datei, Bytes, SHA-256, Zeilen), Download- und Prüffunktionen (`fetch_source`, `record_rows`, `write_provenance`), Einlesen und Strukturprüfung der Rohdatei, Namensnormalisierung und statistische Hilfsfunktionen. Es wird nicht separat ausgeführt.
- `scripts/external_data.py` – importiertes Modul (von 01 aufgerufen): Laden, Bereinigen und stufenweises Matchen von IGDB, Metacritic und Preishistorie, inklusive `source_counts.csv` und `metacritic_overlap.csv`.
- `scripts/01_prepare_data.py` – Aufbereitung, Stichproben, Erfolgsdefinitionen, Parquets in `data/` und die Zwischenergebnisse {', '.join(f'`{n}.csv`' for n in csv_from_01)}.
- `scripts/02_eda.py` – deskriptive Tabellen und Abbildungen; liest die Parquets.
- `scripts/03_tests.py` – χ²/Odds Ratio, Mann-Whitney-U/Cliff's δ, Kruskal-Wallis/ε², Spearman und partielle Rangkorrelationen (Jahr-FE), logistische Regressionen mit Jahr-FE inkl. Konvergenzprüfung und gemeinsamem Modell, bedingte ORs mit exaktem KI, Bootstrap, BH-Korrektur und die regelbasierten Hypothesenentscheidungen (`hypothesis_decisions.csv`); liest nur die Parquets.
- `scripts/05_report.py` – erzeugt diesen Bericht ausschließlich aus den Tabellen und Parquets; qualitative Aussagen werden gegen die Tabellen geprüft. Die Nummer 04 ist nicht vergeben; ein Skript 04 gab es auch in der Projekthistorie nicht.

Die CSV-Zwischenergebnisse von 01 sind Teil der versionierten Ausgaben; 05 liest sie zusammen mit den Ausgaben von 02 und 03.

**Ausführung** aus dem Projektordner (Reihenfolge 01 → 02 → 03 → 05):

```bash
pip install -r requirements.txt
python scripts/01_prepare_data.py
python scripts/02_eda.py
python scripts/03_tests.py
python scripts/05_report.py
```

**Rohdaten:** `common.fetch_source` lädt je Quelle genau die benötigte Datei der gepinnten Kaggle-Version über `kagglehub` (Cache-Ort über `KAGGLEHUB_CACHE` steuerbar) und prüft Größe und SHA-256; bei einer Abweichung oder einer nicht mehr abrufbaren Version bricht der Lauf mit einer Meldung ab. Liefert `kagglehub` einen Pfad ohne `versions/N` (in Kaggle- und Colab-Notebooks über deren Cache-Resolver), entscheidet allein die Hashprüfung; `data_provenance.csv` vermerkt in `Version_Herkunft`, woher die Versionsangabe stammt. Eine lokale Kopie unter `data/raw/<Datei>` hat Vorrang, wird aber nur nach erfolgreicher Hashprüfung verwendet; `data/raw/` ist von der Versionierung ausgenommen. Der Basisdatensatz ist `{fr['Datei']}` in {C.version_text}: {n0(fr['Bytes'])} Bytes, SHA-256 `{fr['SHA256']}` (Prüfung gegen die ursprüngliche Auswertung: Abschnitt 1.1). **Empfehlung:** `{fr['Datei']}` mit diesem Hash dauerhaft archivieren (z. B. Zenodo/OSF, Release-Asset oder Git LFS), weil der Eigentümer ältere Kaggle-Versionen löscht. Hashes aller Rohdateien: Abschnitt 1.1 und `data_provenance.csv`.

**Determinismus:** Bei einem Kontrolllauf am {DETERMINISM_CHECK['date']} mit dem Codestand dieses Berichts wurde die Folge {DETERMINISM_CHECK['steps']} zweimal hintereinander vollständig ausgeführt; alle {DETERMINISM_CHECK['csv']} CSV-Tabellen, {DETERMINISM_CHECK['parquet']} Parquet-Dateien und {DETERMINISM_CHECK['png']} Abbildungen sowie dieser Bericht waren bytegleich (MD5-Vergleich, Python 3.12.0 mit den gepinnten Versionen). Dieser Vergleich ist nicht Teil der Pipeline; sein Ergebnis ist als Konstante in `scripts/05_report.py` dokumentiert und muss nach Codeänderungen wiederholt werden. Gleitkommazahlen werden mit 10 signifikanten Stellen geschrieben; das unterdrückt plattformabhängiges Rauschen der Logit-Schätzungen in der letzten Stelle. Auf anderen Plattformen oder mit anderen Bibliotheksversionen sind Abweichungen in den hinteren Nachkommastellen möglich, aber ohne inhaltliche Folgen.
"""


TABLE_DESCRIPTIONS = {
    "data_provenance": "Provenienz der Rohdateien (Version, Bytes, SHA-256, Zeilen, Datenstand)",
    "steamspy_coverage_by_month": "Abdeckung der SteamSpy-Felder je Release-Monat (Beleg des Stichtags)",
    "sample_flow": "Stichprobenfluss und Teilstichproben",
    "exclusion_samples": "Zufällige Beispiele je Ausschlussgrund",
    "data_quality_summary": "Datenqualitätsprüfungen mit Bezugsbasis",
    "success_thresholds": "Erfolgsschwellen, Basis und Prävalenz",
    "missing_values": "Fehlanteile je Variable (Analysestichprobe)",
    "variable_overview": "Variablen, Definitionen und Rollen",
    "source_coverage": "Matching-Ergebnisse je Datenquelle",
    "source_counts": "Zählungen aller Lade-, Filter- und Matching-Schritte",
    "metacritic_overlap": "Abgleich FronkonGames- und externer Metascore",
    "source_overlap": "Überlappung der Datenquellen",
    "descriptives_overall": "Deskriptive Statistik (mit Basis)",
    "descriptives_by_success": "Deskriptive Statistik nach Erfolg",
    "owners_distribution": "Besitzerklassen mit CCDF",
    "cohort_by_year": "Veröffentlichungsjahrgänge inkl. „0 - 0“-Anteil",
    "age_standardized_rates": "Rohe, altersstandardisierte und kohortenstratifizierte Erfolgsquoten",
    "genre_overview": "Steam-Genres",
    "genre_cohort_overview": "Steam-Genres × Kohorte (Daten zu fig_09)",
    "mode_platform_overview": "Spielmodi, Plattformanzahl und -kombinationen",
    "platform_combo_overview": "Plattformkombinationen inkl. Kohorten",
    "price_overview": "Preisklassen nach Listenpreis (Tagespreis als Zusatz)",
    "price_40plus_composition": "Zusammensetzung der Preisklasse 40+",
    "discount_overview": "Rabattgruppen (nur kostenpflichtige Spiele)",
    "free_overview": "Kostenlose Spiele mit/ohne F2P-Tag",
    "free_share_owners_zero": "Anteil kostenloser Spiele mit und ohne „0 - 0“",
    "ratings_overview": "Bewertungsquartile mit Wertebereichen",
    "metacritic_coverage": "Abdeckung der Metacritic-Quellen",
    "descriptives_external": "Deskriptive Statistik der externen Variablen",
    "igdb_genre_overview": "IGDB-Genres",
    "igdb_mode_overview": "IGDB-Spielmodi (Basis: Modusangabe)",
    "igdb_rating_overview": "IGDB-Bewertungsquartile nach Ganzzahligkeit",
    "igdb_rating_values": "Struktur der IGDB-Nutzerbewertung (Ganzzahligkeit, Häufung bei 70)",
    "price_history_months": "Fensterlängen der Preishistorie",
    "price_history_overview": "Normierte Preishistorien-Merkmale (volles Fenster)",
    "external_metacritic_overview": "Quartile von Metascore und Metacritic-Nutzerscore",
    "correlation_spearman": "Korrelationsmatrizen (Langformat mit N)",
    "h1_tests": "H1-Tests (Genres, Modi, Plattformen)",
    "h1_joint_model": "H1: gemeinsames Modell je Erfolgsdefinition",
    "h2_tests": "H2-Tests (kostenlos, Preisniveau, Rabatt, DLC)",
    "h2_free_sensitivity": "H2: kostenlos nach Kohorte, „0 - 0“, bis 2020, F2P-Tag",
    "h3_tests": "H3-Tests (Positive Ratio, Wilson, Metascore)",
    "h3_identical_sample": "H3: Bootstrap-Vergleich in identischer Stichprobe",
    "h3_cohort_stratified": "H3: kohortenstratifizierte Korrelationen",
    "h4_external_tests": "H4-Tests (IGDB, Preishistorie, Übereinstimmung)",
    "igdb_steam_agreement": "Übereinstimmung IGDB/Steam (Anteil identisch, κ)",
    "tests_all": "Alle Hypothesentests (H1–H4) mit Kennung",
    "robustness_success_definitions": "Zentrale Tests unter allen drei Erfolgsdefinitionen",
    "release_date_sensitivity": "Sensitivität der adjustierten ORs zum Veröffentlichungsdatum",
    "release_cohort_sensitivity": "Kohortenanalysen mit alternativem Veröffentlichungsjahr",
    "quartile_groups": "Quartilsgruppen: Wertebereiche, n und Erfolgsquoten",
    "external_selection_tests": "Selektion der externen Teilstichproben",
    "review_volume_diagnostic": "Diagnose der Volumenabhängigkeit der Bewertungsmaße",
    "hypothesis_decisions": "Regelbasierte Hypothesenentscheidungen",
}


def appendix(D, C) -> str:
    files = sorted(p.stem for p in TAB.glob("*.csv"))
    undocumented = [f for f in files if f not in TABLE_DESCRIPTIONS]
    check(not undocumented, f"Tabellen ohne Beschreibung im Anhang: {undocumented}")
    rows = [{"Datei": f"`outputs/tables/{f}.csv`", "Inhalt": TABLE_DESCRIPTIONS[f]} for f in files]
    rows += [{"Datei": f"`data/{p.name}`", "Inhalt": "Parquet (aufbereitete Daten bzw. Teilstichprobe)"}
             for p in sorted(DATA.glob("*.parquet"))]
    return f"""## Anhang: Ausgabedateien

{md(pd.DataFrame(rows))}

*Hinweis: Alle Ergebniszahlen in diesem Bericht werden von `scripts/05_report.py` aus den oben genannten Tabellen und Parquets gelesen, Methodenparameter aus den Quelltexten von 02 und 03; qualitative Aussagen sind gegen die Tabellen geprüft. Ausgenommen sind die als Konstanten dokumentierten Ergebnisse des Versionsabgleichs (Abschnitt 1.1) und der Determinismusprüfung (Abschnitt 6), die außerhalb der Pipeline ermittelt wurden. Auswertungsstand: {REPORT_DATE}.*
"""


def main() -> None:
    D = load()
    C = context(D)
    parts = [header(D, C), section0(D, C), section1(D, C), section2(D, C), section3(D, C),
             section4(D, C), section5(D, C), section6(D, C), appendix(D, C)]
    text = "\n".join(parts)
    # Sicherheitsnetz: keine unformatierten Fehlwerte oder Python-Literale im Bericht.
    for token in ["nan", "NaN", "inf", "True", "False"]:
        hit = re.search(rf"(?<![\w`]){token}(?![\w`])", text)
        check(hit is None, f"Token '{token}' im Bericht: …{text[max(0, hit.start() - 80):hit.end() + 40] if hit else ''}…")
    (OUT / "Ergebnisse.md").write_text(text, encoding="utf-8")
    print(f"Bericht geschrieben: {OUT / 'Ergebnisse.md'}")
    print(f"Länge: {len(text):,} Zeichen")


if __name__ == "__main__":
    main()

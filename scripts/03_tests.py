"""Statistische Tests fuer H1 (Spieleigenschaften), H2 (Preis) und H3 (Bewertungen).

Alle p-Werte werden innerhalb jeder Hypothese per Benjamini-Hochberg korrigiert.
Neben p-Werten werden Effektstaerken und Konfidenzintervalle berichtet.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import chi2_contingency, kruskal, mannwhitneyu, spearmanr

from common import (
    DATA,
    STANDARD_GENRES,
    bh_correction,
    cliffs_delta_from_mannwhitney,
    odds_ratio_ci,
    partial_spearman,
    save_table,
)


def spearman_ci(rho: float, n: int, controls: int = 0) -> tuple[float, float]:
    if not np.isfinite(rho) or n <= 3 + controls:
        return np.nan, np.nan
    z = np.arctanh(rho)
    se = 1.0 / math.sqrt(n - 3 - controls)
    return float(np.tanh(z - 1.96 * se)), float(np.tanh(z + 1.96 * se))


def add_bh(df: pd.DataFrame, family: str) -> pd.DataFrame:
    out = df.copy()
    out["p_adj"] = bh_correction(out["p"])
    out["signifikant"] = out["p_adj"] < 0.05
    out.insert(0, "Familie", family)
    return out


def chi2_indicator(df: pd.DataFrame, flag: str, label: str) -> dict:
    tab = pd.crosstab(df[flag].astype(bool), df["success_top25"].astype(bool))
    tab = tab.reindex(index=[False, True], columns=[False, True], fill_value=0)
    chi2, p, dof, _ = chi2_contingency(tab.to_numpy(), correction=False)
    odds_ratio, low, high, _ = odds_ratio_ci(tab.to_numpy())
    n = int(tab.to_numpy().sum())
    cramers_v = math.sqrt(chi2 / n) if n else np.nan
    rate_true = tab.loc[True, True] / tab.loc[True].sum() if tab.loc[True].sum() else np.nan
    rate_false = tab.loc[False, True] / tab.loc[False].sum() if tab.loc[False].sum() else np.nan
    return {
        "Test": "Chi2 / Odds Ratio",
        "Merkmal": label,
        "Gruppe": f"{flag} = ja vs. nein",
        "N": n,
        "Erfolgsquote_Gruppe": rate_true,
        "Erfolgsquote_Referenz": rate_false,
        "Effekt": odds_ratio,
        "Effekt_typ": "Odds Ratio",
        "CI_unten": low,
        "CI_oben": high,
        "p": p,
        "Zusatz_Effekt": cramers_v,
        "Zusatz_Effekt_Typ": "Cramers V",
    }


def chi2_categorical(df: pd.DataFrame, var: str, label: str) -> dict:
    tab = pd.crosstab(df[var], df["success_top25"].astype(bool))
    chi2, p, dof, _ = chi2_contingency(tab.to_numpy(), correction=False)
    n = int(tab.to_numpy().sum())
    v = math.sqrt(chi2 / (n * (min(tab.shape) - 1))) if n and min(tab.shape) > 1 else np.nan
    return {
        "Test": "Chi2 (kategorial)",
        "Merkmal": label,
        "Gruppe": var,
        "N": n,
        "Erfolgsquote_Gruppe": np.nan,
        "Erfolgsquote_Referenz": np.nan,
        "Effekt": chi2,
        "Effekt_typ": "Chi2",
        "CI_unten": np.nan,
        "CI_oben": np.nan,
        "p": p,
        "Zusatz_Effekt": v,
        "Zusatz_Effekt_Typ": "Cramers V",
    }


def adjusted_or(df: pd.DataFrame, flag: str, label: str) -> dict:
    d = df[[flag, "success_top25", "release_year"]].dropna().copy()
    d[flag] = d[flag].astype(int)
    d["success_top25"] = d["success_top25"].astype(int)
    formula = f"success_top25 ~ {flag} + release_year + I(release_year**2)"
    model = smf.logit(formula, data=d).fit(disp=0, maxiter=2000)
    coef = model.params[flag]
    ci = model.conf_int().loc[flag]
    return {
        "Test": "Logistische Regression (adjustiert)",
        "Merkmal": label,
        "Gruppe": f"{flag} (kontrolliert fuer Jahr und Jahr^2)",
        "N": int(len(d)),
        "Erfolgsquote_Gruppe": np.nan,
        "Erfolgsquote_Referenz": np.nan,
        "Effekt": math.exp(coef),
        "Effekt_typ": "adjustierte Odds Ratio",
        "CI_unten": math.exp(ci[0]),
        "CI_oben": math.exp(ci[1]),
        "p": model.pvalues[flag],
        "Zusatz_Effekt": np.nan,
        "Zusatz_Effekt_Typ": "",
    }


def mannwhitney_row(df: pd.DataFrame, value: str, group: str, label: str) -> dict:
    x = df.loc[df[group].astype(bool), value].dropna()
    y = df.loc[~df[group].astype(bool), value].dropna()
    u, p = mannwhitneyu(x, y, alternative="two-sided")
    delta = cliffs_delta_from_mannwhitney(u, len(x), len(y))
    return {
        "Test": "Mann-Whitney-U",
        "Merkmal": label,
        "Gruppe": f"{group} = ja (n={len(x)}) vs. nein (n={len(y)})",
        "N": len(x) + len(y),
        "Effekt": delta,
        "Effekt_typ": "Cliff's Delta",
        "CI_unten": np.nan,
        "CI_oben": np.nan,
        "p": p,
        "Median_Gruppe": x.median(),
        "Median_Referenz": y.median(),
        "Mittelwert_Gruppe": x.mean(),
        "Mittelwert_Referenz": y.mean(),
    }


def kruskal_row(df: pd.DataFrame, value: str, group: str, label: str) -> dict:
    groups = [g[value].dropna().to_numpy() for _, g in df.groupby(group, observed=True)]
    groups = [g for g in groups if len(g) > 0]
    h, p = kruskal(*groups)
    n = sum(len(g) for g in groups)
    k = len(groups)
    epsilon = (h - k + 1) / (n - k) if n > k else np.nan
    medians = df.groupby(group, observed=True)[value].median()
    return {
        "Test": "Kruskal-Wallis",
        "Merkmal": label,
        "Gruppe": group,
        "N": n,
        "Effekt": h,
        "Effekt_typ": "Kruskal-Wallis H",
        "CI_unten": np.nan,
        "CI_oben": np.nan,
        "p": p,
        "Zusatz_Effekt": epsilon,
        "Zusatz_Effekt_Typ": "Epsilon-Quadrat",
        "Median_Gruppe": "; ".join(f"{k}: {v:,.0f}" for k, v in medians.items()),
    }


def spearman_row(df: pd.DataFrame, x: str, y: str, label: str, partial: bool = False) -> dict:
    d = df[[x, y, "release_year"]].dropna()
    rho, p = spearmanr(d[x], d[y])
    low, high = spearman_ci(rho, len(d))
    row = {
        "Test": "Spearman" + (" (partiell, Jahr)" if partial else ""),
        "Merkmal": label,
        "Gruppe": f"{x} vs. {y}",
        "N": len(d),
        "Effekt": rho,
        "Effekt_typ": "Spearman rho",
        "CI_unten": low,
        "CI_oben": high,
        "p": p,
    }
    if partial:
        prho, pp, pn = partial_spearman(d[x], d[y], d["release_year"])
        row["Effekt"] = prho
        row["p"] = pp
        row["N"] = pn
        row["CI_unten"], row["CI_oben"] = spearman_ci(prho, pn, controls=1)
    return row


def quartile_or(df: pd.DataFrame, value: str, label: str) -> dict:
    d = df[[value, "success_top25"]].dropna().copy()
    d["quartil"] = pd.qcut(d[value], 4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
    labels = list(d["quartil"].cat.categories)
    if len(labels) < 2:
        return {"Test": "Chi2 / Odds Ratio (Q4 vs. Q1)", "Merkmal": label,
                "Gruppe": f"{value}: zu wenige Quartile", "N": 0, "Effekt": np.nan,
                "Effekt_typ": "Odds Ratio Q4/Q1", "p": np.nan}
    top = d[d["quartil"] == labels[-1]]
    bottom = d[d["quartil"] == labels[0]]
    table = np.array(
        [
            [top["success_top25"].sum(), (~top["success_top25"]).sum()],
            [bottom["success_top25"].sum(), (~bottom["success_top25"]).sum()],
        ]
    )
    odds_ratio, low, high, p = odds_ratio_ci(table)
    return {
        "Test": "Chi2 / Odds Ratio (Q4 vs. Q1)",
        "Merkmal": label,
        "Gruppe": f"{value}: oberstes vs. unterstes Quartil",
        "N": len(top) + len(bottom),
        "Erfolgsquote_Gruppe": top["success_top25"].mean(),
        "Erfolgsquote_Referenz": bottom["success_top25"].mean(),
        "Effekt": odds_ratio,
        "Effekt_typ": "Odds Ratio Q4/Q1",
        "CI_unten": low,
        "CI_oben": high,
        "p": p,
    }


def main() -> None:
    df = pd.read_parquet(DATA / "analytic_sample.parquet")
    rating = pd.read_parquet(DATA / "rating_sample.parquet")
    rating50 = pd.read_parquet(DATA / "rating_sample_50.parquet")
    meta = pd.read_parquet(DATA / "mc_external_sample.parquet")
    meta_fronk = pd.read_parquet(DATA / "meta_sample.parquet")
    igdb = pd.read_parquet(DATA / "igdb_sample.parquet")
    price_hist = pd.read_parquet(DATA / "price_history_sample.parquet")

    # ---------------------------------------------------------------- H1
    h1_rows = []
    genre_cols = {
        genre: "genre_" + genre.lower().replace(" ", "_").replace("&", "").replace("__", "_")
        for genre in STANDARD_GENRES
    }
    for genre, col in genre_cols.items():
        if col in df.columns:
            h1_rows.append(chi2_indicator(df, col, f"Genre: {genre}"))
            h1_rows.append(adjusted_or(df, col, f"Genre: {genre}"))
    for col, label in [
        ("single_player", "Spielmodus: Singleplayer"),
        ("multi_player", "Spielmodus: Multiplayer"),
        ("coop", "Spielmodus: Koop"),
        ("pvp", "Spielmodus: PvP"),
        ("mmo", "Spielmodus: MMO"),
    ]:
        h1_rows.append(chi2_indicator(df, col, label))
        h1_rows.append(adjusted_or(df, col, label))

    h1_rows.append(chi2_categorical(df, "mode", "Spielmodus (4 Kategorien)"))
    h1_rows.append(chi2_categorical(df, "n_platforms", "Anzahl Plattformen (1/2/3)"))

    df["multi_platform"] = df["n_platforms"] >= 2
    h1_rows.append(chi2_indicator(df, "multi_platform", "Mehr als eine Plattform"))
    h1_rows.append(adjusted_or(df, "multi_platform", "Mehr als eine Plattform"))
    h1_rows.append(chi2_indicator(df, "mac", "Plattform: macOS"))
    h1_rows.append(chi2_indicator(df, "linux", "Plattform: Linux"))

    h1 = add_bh(pd.DataFrame(h1_rows), "H1 - Spieleigenschaften")
    h1 = h1.sort_values(["Merkmal", "Test"])
    save_table(h1, "h1_tests")

    # ---------------------------------------------------------------- H2
    df["paid"] = ~df["is_free"]
    df["multi_platform"] = df["n_platforms"] >= 2
    df["discount_group"] = pd.cut(
        df["discount"], bins=[-0.1, 0, 25, 50, 75, 100],
        labels=["kein Rabatt", "1-25 %", "26-50 %", "51-75 %", "76-100 %"],
    )
    h2_rows = []
    h2_rows.append(mannwhitney_row(df, "owners_mid", "is_free", "Besitzer: Free-to-Play vs. kostenpflichtig"))
    h2_rows.append(mannwhitney_row(df, "peak_ccu", "is_free", "Peak CCU: Free-to-Play vs. kostenpflichtig"))
    h2_rows.append(chi2_indicator(df, "is_free", "Erfolgsquote: Free-to-Play vs. kostenpflichtig"))
    h2_rows.append(spearman_row(df[df["paid"]], "price", "log_owners", "Preis vs. Besitzer (nur kostenpflichtig)"))
    h2_rows.append(spearman_row(df[df["paid"]], "price", "log_owners", "Preis vs. Besitzer (nur kostenpflichtig)", partial=True))
    h2_rows.append(spearman_row(df[df["paid"]], "price", "log_peak_ccu", "Preis vs. Peak CCU (nur kostenpflichtig)"))
    h2_rows.append(spearman_row(df[df["paid"]], "discount", "log_owners", "Rabatt vs. Besitzer (nur kostenpflichtig)"))
    h2_rows.append(spearman_row(df[df["paid"]], "discount", "log_owners", "Rabatt vs. Besitzer (nur kostenpflichtig)", partial=True))
    h2_rows.append(spearman_row(df[df["paid"]], "discount", "log_peak_ccu", "Rabatt vs. Peak CCU (nur kostenpflichtig)"))
    h2_rows.append(chi2_indicator(df[df["paid"]], "has_discount", "Erfolgsquote: Rabatt aktiv vs. kein Rabatt"))
    h2_rows.append(kruskal_row(df[df["paid"]], "owners_mid", "discount_group", "Besitzer nach Rabattgruppe"))
    h2_rows.append(kruskal_row(df[df["paid"]], "owners_mid", "price_class", "Besitzer nach Preisklasse"))
    h2_rows.append(spearman_row(df, "dlc_count", "log_owners", "DLC-Anzahl vs. Besitzer (alle Spiele)"))

    h2 = add_bh(pd.DataFrame(h2_rows), "H2 - Preis")
    save_table(h2, "h2_tests")

    # ---------------------------------------------------------------- H3
    h3_rows = []
    for data, suffix in [(rating, "mind. 10 Reviews"), (rating50, "mind. 50 Reviews")]:
        h3_rows.append(spearman_row(data, "wilson", "log_owners", f"Wilson-Score vs. Besitzer ({suffix})"))
        h3_rows.append(spearman_row(data, "wilson", "log_owners", f"Wilson-Score vs. Besitzer ({suffix})", partial=True))
        h3_rows.append(spearman_row(data, "wilson", "log_peak_ccu", f"Wilson-Score vs. Peak CCU ({suffix})"))
        h3_rows.append(spearman_row(data, "positive_ratio", "log_owners", f"Positive Ratio vs. Besitzer ({suffix})"))
        h3_rows.append(spearman_row(data, "positive_ratio", "log_owners", f"Positive Ratio vs. Besitzer ({suffix})", partial=True))
        h3_rows.append(spearman_row(data, "positive_ratio", "log_peak_ccu", f"Positive Ratio vs. Peak CCU ({suffix})"))
        h3_rows.append(quartile_or(data, "wilson", f"Erfolgsquote Wilson Q4 vs. Q1 ({suffix})"))
        h3_rows.append(quartile_or(data, "positive_ratio", f"Erfolgsquote Positive Ratio Q4 vs. Q1 ({suffix})"))

    h3_rows.append(spearman_row(meta, "mc_metascore", "log_owners", "Metacritic (extern) vs. Besitzer"))
    h3_rows.append(spearman_row(meta, "mc_metascore", "log_owners", "Metacritic (extern) vs. Besitzer", partial=True))
    h3_rows.append(spearman_row(meta, "mc_metascore", "log_peak_ccu", "Metacritic (extern) vs. Peak CCU"))
    h3_rows.append(spearman_row(meta, "mc_user_score", "log_owners", "Metacritic-Nutzerscore (extern) vs. Besitzer"))
    h3_rows.append(spearman_row(meta, "mc_user_score", "log_peak_ccu", "Metacritic-Nutzerscore (extern) vs. Peak CCU"))
    h3_rows.append(spearman_row(meta, "positive_ratio", "log_owners", "Positive Ratio vs. Besitzer (externe Metacritic-Teilstichprobe)"))
    h3_rows.append(spearman_row(meta, "wilson", "log_owners", "Wilson vs. Besitzer (externe Metacritic-Teilstichprobe)"))
    h3_rows.append(spearman_row(meta, "wilson", "mc_metascore", "Wilson vs. Metacritic (externe Quelle, Übereinstimmung)"))
    h3_rows.append(spearman_row(meta, "mc_metascore", "metacritic", "Externes vs. FronkonGames-Metacritic (Übereinstimmung)"))
    h3_rows.append(quartile_or(meta, "mc_metascore", "Erfolgsquote Metacritic (extern) Q4 vs. Q1"))

    h3 = add_bh(pd.DataFrame(h3_rows), "H3 - Bewertungen")
    save_table(h3, "h3_tests")

    # ---------------------------------------------------------------- H4
    h4_rows = []
    h4_rows.append(spearman_row(igdb, "igdb_rating", "log_owners", "IGDB-Nutzerbewertung vs. Besitzer"))
    h4_rows.append(spearman_row(igdb, "igdb_rating", "log_owners", "IGDB-Nutzerbewertung vs. Besitzer", partial=True))
    h4_rows.append(spearman_row(igdb, "igdb_rating", "log_peak_ccu", "IGDB-Nutzerbewertung vs. Peak CCU"))
    h4_rows.append(spearman_row(igdb, "igdb_aggregated_rating", "log_owners", "IGDB-Kritikerbewertung vs. Besitzer"))
    h4_rows.append(spearman_row(igdb, "igdb_aggregated_rating", "log_owners", "IGDB-Kritikerbewertung vs. Besitzer", partial=True))
    h4_rows.append(spearman_row(igdb, "igdb_aggregated_rating", "log_peak_ccu", "IGDB-Kritikerbewertung vs. Peak CCU"))
    h4_rows.append(spearman_row(igdb, "igdb_n_genres", "log_owners", "Anzahl IGDB-Genres vs. Besitzer"))
    h4_rows.append(quartile_or(igdb, "igdb_rating", "Erfolgsquote IGDB-Nutzerbewertung Q4 vs. Q1"))
    h4_rows.append(quartile_or(igdb, "igdb_aggregated_rating", "Erfolgsquote IGDB-Kritikerbewertung Q4 vs. Q1"))
    h4_rows.append(chi2_indicator(igdb, "igdb_multiplayer", "IGDB: Multiplayer vs. kein Multiplayer"))
    h4_rows.append(adjusted_or(igdb, "igdb_multiplayer", "IGDB: Multiplayer vs. kein Multiplayer"))
    h4_rows.append(chi2_indicator(igdb, "igdb_singleplayer", "IGDB: Singleplayer vs. kein Singleplayer"))
    h4_rows.append(adjusted_or(igdb, "igdb_singleplayer", "IGDB: Singleplayer vs. kein Singleplayer"))
    h4_rows.append(chi2_indicator(igdb, "igdb_coop", "IGDB: Kooperation vs. keine Kooperation"))

    h4_rows.append(spearman_row(price_hist, "ph_max_discount_pct", "log_owners", "Maximaler Rabatt (Preishistorie) vs. Besitzer"))
    h4_rows.append(spearman_row(price_hist, "ph_max_discount_pct", "log_owners", "Maximaler Rabatt (Preishistorie) vs. Besitzer", partial=True))
    h4_rows.append(spearman_row(price_hist, "ph_max_discount_pct", "log_peak_ccu", "Maximaler Rabatt (Preishistorie) vs. Peak CCU"))
    h4_rows.append(spearman_row(price_hist, "ph_share_months_on_sale", "log_owners", "Anteil Monate mit Rabatt vs. Besitzer"))
    h4_rows.append(spearman_row(price_hist, "ph_share_months_on_sale", "log_owners", "Anteil Monate mit Rabatt vs. Besitzer", partial=True))
    h4_rows.append(spearman_row(price_hist, "ph_share_months_on_sale", "log_peak_ccu", "Anteil Monate mit Rabatt vs. Peak CCU"))
    h4_rows.append(spearman_row(price_hist, "ph_price_changes_total", "log_owners", "Preisänderungen (Preishistorie) vs. Besitzer"))
    h4_rows.append(spearman_row(price_hist, "ph_price_changes_total", "log_owners", "Preisänderungen (Preishistorie) vs. Besitzer", partial=True))
    h4_rows.append(quartile_or(price_hist, "ph_max_discount_pct", "Erfolgsquote maximaler Rabatt Q4 vs. Q1"))
    h4_rows.append(quartile_or(price_hist, "ph_share_months_on_sale", "Erfolgsquote Rabatthäufigkeit Q4 vs. Q1"))

    h4_rows.append(spearman_row(igdb, "igdb_rating", "wilson", "IGDB-Nutzerbewertung vs. Steam-Wilson"))
    h4_rows.append(spearman_row(igdb, "igdb_aggregated_rating", "wilson", "IGDB-Kritikerbewertung vs. Steam-Wilson"))
    both_mc = igdb.dropna(subset=["igdb_aggregated_rating", "mc_metascore"])
    h4_rows.append(spearman_row(both_mc, "igdb_aggregated_rating", "mc_metascore", "IGDB-Kritikerbewertung vs. Metacritic (extern)"))

    h4 = add_bh(pd.DataFrame(h4_rows), "H4 - Externe Datenquellen")
    save_table(h4, "h4_external_tests")

    all_tests = pd.concat([h1, h2, h3, h4], ignore_index=True, sort=False)
    save_table(all_tests, "tests_all")

    # Diagnose: Bewertungsvolumen vs. Bewertungshoehe.
    # Zeigt, ob die Wilson-Korrelation mit den Besitzern nur das Review-Volumen abbildet.
    diag_rows = []
    for data, suffix in [(rating, "mind. 10 Reviews"), (rating50, "mind. 50 Reviews")]:
        d = data.copy()
        d["log_review_count"] = np.log10(1 + d["review_count"])
        for x, y, label in [
            ("review_count", "log_owners", f"Review-Anzahl vs. Besitzer ({suffix})"),
            ("review_count", "log_peak_ccu", f"Review-Anzahl vs. Peak CCU ({suffix})"),
            ("wilson", "review_count", f"Wilson-Score vs. Review-Anzahl ({suffix})"),
            ("positive_ratio", "review_count", f"Positive Ratio vs. Review-Anzahl ({suffix})"),
        ]:
            dd = d[[x, y]].dropna()
            rho, p = spearmanr(dd[x], dd[y])
            low, high = spearman_ci(rho, len(dd))
            diag_rows.append({"Test": "Spearman (Diagnose)", "Merkmal": label,
                              "Gruppe": f"{x} vs. {y}", "N": len(dd), "Effekt": rho,
                              "Effekt_typ": "Spearman rho", "CI_unten": low,
                              "CI_oben": high, "p": p})
        for x, label in [("wilson", f"Wilson vs. Besitzer, kontrolliert Review-Anzahl ({suffix})"),
                         ("positive_ratio", f"Positive Ratio vs. Besitzer, kontrolliert Review-Anzahl ({suffix})")]:
            rho, p, n = partial_spearman(d[x], d["log_owners"], d["log_review_count"])
            low, high = spearman_ci(rho, n, controls=1)
            diag_rows.append({"Test": "Spearman (partiell, Review-Anzahl)", "Merkmal": label,
                              "Gruppe": f"{x} vs. log_owners | log_review_count", "N": n,
                              "Effekt": rho, "Effekt_typ": "Spearman rho", "CI_unten": low,
                              "CI_oben": high, "p": p})
    diag = add_bh(pd.DataFrame(diag_rows), "Diagnose - Bewertungsvolumen")
    save_table(diag, "review_volume_diagnostic")

    # Selektionsverzerrung der externen Teilstichproben dokumentieren.
    selection_rows = [
        mannwhitney_row(df, "owners_mid", "has_igdb", "Besitzer: IGDB-Match vorhanden vs. nicht"),
        mannwhitney_row(df, "owners_mid", "has_mc_external", "Besitzer: externer Metacritic-Match vorhanden vs. nicht"),
        mannwhitney_row(df, "owners_mid", "has_price_history", "Besitzer: Preishistorie-Match vorhanden vs. nicht"),
        mannwhitney_row(df, "owners_mid", "has_metacritic", "Besitzer: FronkonGames-Metacritic vorhanden vs. nicht"),
    ]
    selection = pd.DataFrame(selection_rows)
    selection["p_adj"] = bh_correction(selection["p"])
    save_table(selection, "external_selection_tests")

    print("H1:")
    print(h1[["Merkmal", "Test", "N", "Effekt", "CI_unten", "CI_oben", "p", "p_adj"]].to_string(index=False))
    print("\nH2:")
    print(h2[["Merkmal", "Test", "N", "Effekt", "CI_unten", "CI_oben", "p", "p_adj"]].to_string(index=False))
    print("\nH3:")
    print(h3[["Merkmal", "Test", "N", "Effekt", "CI_unten", "CI_oben", "p", "p_adj"]].to_string(index=False))
    print("\nH4:")
    print(h4[["Merkmal", "Test", "N", "Effekt", "CI_unten", "CI_oben", "p", "p_adj"]].to_string(index=False))


if __name__ == "__main__":
    main()

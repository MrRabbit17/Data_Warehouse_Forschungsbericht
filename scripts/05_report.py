"""Erzeugt den Ergebnisbericht outputs/Ergebnisse.md aus den Tabellen/CSVs."""

from __future__ import annotations

import pandas as pd

from common import DATA, OUT, TAB


def de(value: object, decimals: int = 1) -> str:
    if pd.isna(value):
        return "–"
    text = f"{float(value):,.{decimals}f}"
    return text.replace(",", "X").replace(".", ",").replace("X", ".")


def pct(value: object, decimals: int = 1) -> str:
    if pd.isna(value):
        return "–"
    return de(float(value) * 100, decimals) + " %"


def pval(value: object) -> str:
    if pd.isna(value):
        return "–"
    if float(value) < 0.001:
        return "< 0,001"
    return de(value, 3)


def eff_ci(value: object, low: object, high: object, decimals: int = 2) -> str:
    if pd.isna(value):
        return "–"
    if abs(float(value)) < 0.5 * 10 ** (-decimals):
        value = 0.0
    if pd.isna(low) or pd.isna(high):
        return de(value, decimals)
    return f"{de(value, decimals)} [{de(low, decimals)}; {de(high, decimals)}]"


def read(name: str) -> pd.DataFrame:
    return pd.read_csv(TAB / f"{name}.csv")


def md(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False, disable_numparse=True)


def find_row(df: pd.DataFrame, label: str, test: str | None = None) -> pd.Series | None:
    part = df[df["Merkmal"] == label]
    if test is not None:
        part = part[part["Test"] == test]
    return None if part.empty else part.iloc[0]


def cite(df: pd.DataFrame, label: str, test: str | None = None, decimals: int = 2) -> str:
    row = find_row(df, label, test)
    if row is None:
        return "–"
    return eff_ci(row["Effekt"], row["CI_unten"], row["CI_oben"], decimals)


def p_cite(df: pd.DataFrame, label: str, test: str | None = None) -> str:
    row = find_row(df, label, test)
    return "–" if row is None else pval(row["p_adj"])


def main() -> None:
    flow = read("sample_flow")
    quality = read("data_quality_summary")
    thresholds = read("success_thresholds")
    desc = read("descriptives_overall")
    owners = read("owners_distribution")
    genres = read("genre_overview")
    modes = read("mode_platform_overview")
    prices = read("price_overview")
    discounts = read("discount_overview")
    ratings = read("ratings_overview")
    meta_cov = read("metacritic_coverage")
    desc_ext = read("descriptives_external")
    igdb_genres = read("igdb_genre_overview")
    igdb_modes = read("igdb_mode_overview")
    igdb_ratings = read("igdb_rating_overview")
    price_hist = read("price_history_overview")
    mc_ext = read("external_metacritic_overview")
    coverage = read("source_coverage")
    overlap = read("source_overlap")
    h1 = read("h1_tests")
    h2 = read("h2_tests")
    h3 = read("h3_tests")
    h4 = read("h4_external_tests")
    diag = read("review_volume_diagnostic")
    selection = read("external_selection_tests")

    analytic = pd.read_parquet(DATA / "analytic_sample.parquet")
    n_analytic = len(analytic)
    n_rating = int(analytic["rating_sample"].sum())
    n_rating50 = int(analytic["rating_sample_50"].sum())
    n_igdb = int(analytic["igdb_sample"].sum())
    n_mc_ext = int(analytic["mc_external_sample"].sum())
    n_price = int(analytic["price_history_sample"].sum())
    n_all_sources = int(analytic["multi_source_sample"].sum())
    n_multi_source = int((analytic["n_sources"] >= 2).sum())
    n_peak = int(analytic["peak_sample"].sum())
    n_meta_fronk = int(analytic["meta_sample"].sum())

    n_100_200 = int(
        ((analytic["owners_mid"] >= 100_000_000) & (analytic["owners_mid"] < 200_000_000)).sum()
    )
    share_100_200 = n_100_200 / n_analytic if n_analytic else float("nan")

    mc_both = analytic.dropna(subset=["mc_metascore", "metacritic"])
    n_mc_overlap = len(mc_both)
    n_mc_exact = int((mc_both["mc_metascore"] == mc_both["metacritic"]).sum())
    mc_agreement_rho = (
        mc_both["mc_metascore"].corr(mc_both["metacritic"], method="spearman")
        if n_mc_overlap > 2
        else float("nan")
    )

    igdb_rating_values = analytic.loc[analytic["igdb_sample"], "igdb_rating"].dropna()
    igdb_tie_share = (
        float((igdb_rating_values == 70.0).mean()) if len(igdb_rating_values) else float("nan")
    )

    igdb_df = analytic.loc[analytic["igdb_sample"]].dropna(subset=["igdb_rating"])
    igdb_integer_mask = igdb_df["igdb_rating"].mod(1) == 0
    n_igdb_integer = int(igdb_integer_mask.sum())
    n_igdb_decimal = int((~igdb_integer_mask).sum())
    succ_igdb_integer = igdb_df.loc[igdb_integer_mask, "success_top25"].mean()
    succ_igdb_decimal = igdb_df.loc[~igdb_integer_mask, "success_top25"].mean()

    top25 = thresholds.loc[thresholds["Kriterium"].str.startswith("Top 25"), "Anteil_der_analysestichprobe"].iloc[0]
    top10 = thresholds.loc[thresholds["Kriterium"].str.startswith("Top 10"), "Anteil_der_analysestichprobe"].iloc[0]
    top100k = thresholds.loc[thresholds["Kriterium"].str.startswith("Mindestens 100.000"), "Anteil_der_analysestichprobe"].iloc[0]
    q75 = thresholds.loc[thresholds["Kriterium"].str.startswith("Top 25"), "Schwelle_owners_low"].iloc[0]
    q90 = thresholds.loc[thresholds["Kriterium"].str.startswith("Top 10"), "Schwelle_owners_low"].iloc[0]

    # Tabellen fuer den Bericht kompakt aufbereiten.
    flow_tbl = flow.copy()
    flow_tbl["Anteil_am_Rohdatensatz"] = flow_tbl["Anteil_am_Rohdatensatz"].map(pct)
    flow_tbl["Entfernt_gegenueber_Vorzeile"] = flow_tbl["Entfernt_gegenueber_Vorzeile"].map(lambda x: de(x, 0))
    flow_tbl = flow_tbl.rename(columns={"Basis": "Basis", "Entfernt_gegenueber_Vorzeile": "Entfernt"})

    quality_sel = quality[quality["Pruefung"].isin(
        [
            "Zeilen im Rohdatensatz", "Eindeutige AppIDs", "Besitzer '0 - 0' (fehlend)",
            "Preis = 0", "Keine Reviews", "Kein Metacritic-Score", "Peak CCU = 0",
            "Keine Genre-Angabe", "Software/Utilities", "Nicht-Spiel-Name (Demo/Playtest usw.)",
            "Namens-/Editionsduplikate (entfernt)", "Analysestichprobe",
            "IGDB-Match in Analysestichprobe", "Externer Metacritic-Match in Analysestichprobe",
            "Preishistorie-Match in Analysestichprobe", "Mindestens zwei Datenquellen",
            "Alle vier Datenquellen",
        ]
    )].copy()
    quality_sel["Anteil"] = quality_sel["Anteil"].map(pct)
    quality_sel = quality_sel.rename(columns={"Pruefung": "Prüfung"})

    th_tbl = thresholds.copy()
    th_tbl["Schwelle_owners_low"] = th_tbl["Schwelle_owners_low"].map(lambda x: de(x, 0))
    th_tbl["Anteil_der_analysestichprobe"] = th_tbl["Anteil_der_analysestichprobe"].map(pct)
    th_tbl = th_tbl.rename(
        columns={
            "Schwelle_owners_low": "Schwelle (untere Besitzergrenze)",
            "Anteil_der_analysestichprobe": "Anteil in der Analysestichprobe",
        }
    )

    desc_sel = desc[desc["Variable"].isin(
        ["owners_mid", "peak_ccu", "price", "discount", "dlc_count", "release_year",
         "n_platforms", "n_genres", "review_count", "positive_ratio", "wilson", "metacritic"]
    )].copy()
    for col in ["Mittelwert", "SD", "Median", "Q1", "Q3", "Min", "Max"]:
        desc_sel[col] = desc_sel[col].map(lambda x: de(x, 2))
    desc_sel = desc_sel.rename(columns={"Variable": "Variable"})

    owners_tbl = owners.copy()
    owners_tbl["Anteil"] = owners_tbl["Anteil"].map(pct)
    owners_tbl = owners_tbl[["owners_label", "N", "Anteil"]].rename(
        columns={"owners_label": "Besitzerklasse", "N": "N", "Anteil": "Anteil"}
    )

    genre_tbl = genres.copy()
    genre_tbl["Anteil_Stichprobe"] = genre_tbl["Anteil_Stichprobe"].map(pct)
    genre_tbl["Erfolgsquote"] = genre_tbl["Erfolgsquote"].map(pct)
    genre_tbl["CI"] = genre_tbl.apply(lambda r: f"[{pct(r['CI_unten'])}; {pct(r['CI_oben'])}]", axis=1)
    genre_tbl["Anteil_F2P"] = genre_tbl["Anteil_F2P"].map(pct)
    genre_tbl = genre_tbl[["Genre", "N", "Anteil_Stichprobe", "Erfolgsquote", "CI", "Median_Preis", "Anteil_F2P"]]
    genre_tbl["Median_Preis"] = genre_tbl["Median_Preis"].map(lambda x: de(x, 2))
    genre_tbl = genre_tbl.rename(
        columns={
            "Anteil_Stichprobe": "Anteil", "Erfolgsquote": "Erfolgsquote",
            "CI": "95-%-KI", "Median_Preis": "Median-Preis (USD)", "Anteil_F2P": "F2P-Anteil",
        }
    )

    mode_tbl = modes[modes["Merkmal"].isin(["mode", "n_platforms"])].copy()
    mode_tbl["Erfolgsquote"] = mode_tbl["Erfolgsquote"].map(pct)
    mode_tbl["CI"] = mode_tbl.apply(lambda r: f"[{pct(r['CI_unten'])}; {pct(r['CI_oben'])}]", axis=1)
    mode_tbl = mode_tbl[["Merkmal", "Ausprägung", "N", "Erfolgsquote", "CI"]].rename(
        columns={"Ausprägung": "Ausprägung", "Erfolgsquote": "Erfolgsquote", "CI": "95-%-KI"}
    )

    price_order = ["Free", "0,01 - 4,99", "5 - 9,99", "10 - 19,99", "20 - 39,99", "40+"]
    prices["_order"] = prices["Preisklasse"].map({k: i for i, k in enumerate(price_order)})
    prices = prices.sort_values("_order").drop(columns="_order")
    price_tbl = prices.copy()
    price_tbl["Erfolgsquote"] = price_tbl["Erfolgsquote"].map(pct)
    price_tbl["CI"] = price_tbl.apply(lambda r: f"[{pct(r['CI_unten'])}; {pct(r['CI_oben'])}]", axis=1)
    price_tbl["Median_Besitzer"] = price_tbl["Median_Besitzer"].map(lambda x: de(x, 0))
    price_tbl["Mittelwert_Besitzer"] = price_tbl["Mittelwert_Besitzer"].map(lambda x: de(x, 0))
    price_tbl = price_tbl[["Preisklasse", "N", "Erfolgsquote", "CI", "Median_Besitzer", "Mittelwert_Besitzer"]].rename(
        columns={"Erfolgsquote": "Erfolgsquote", "CI": "95-%-KI", "Median_Besitzer": "Median Besitzer",
                 "Mittelwert_Besitzer": "Mittelwert Besitzer"}
    )

    disc_tbl = discounts.copy()
    disc_tbl["Erfolgsquote"] = disc_tbl["Erfolgsquote"].map(pct)
    disc_tbl["CI"] = disc_tbl.apply(lambda r: f"[{pct(r['CI_unten'])}; {pct(r['CI_oben'])}]", axis=1)
    disc_tbl = disc_tbl[["Rabattgruppe", "N", "Erfolgsquote", "CI"]].rename(
        columns={"Erfolgsquote": "Erfolgsquote", "CI": "95-%-KI"}
    )

    ratings_tbl = ratings.copy()
    ratings_tbl["Erfolgsquote"] = ratings_tbl["Erfolgsquote"].map(pct)
    ratings_tbl["CI"] = ratings_tbl.apply(lambda r: f"[{pct(r['CI_unten'])}; {pct(r['CI_oben'])}]", axis=1)
    ratings_tbl["N"] = ratings_tbl["N"].map(lambda x: de(x, 0))
    ratings_tbl = ratings_tbl[["Bewertungsmaß", "Quartil", "N", "Erfolgsquote", "CI"]].rename(
        columns={"Erfolgsquote": "Erfolgsquote", "CI": "95-%-KI"}
    )

    meta_cov_tbl = meta_cov.copy()
    meta_cov_tbl["Wert"] = meta_cov_tbl.apply(
        lambda r: de(r["Wert"], 0) if "N)" in str(r["Kennzahl"]) else pct(r["Wert"]), axis=1
    )

    cov_tbl = coverage.copy()
    cov_tbl["N_Quelle"] = cov_tbl["N_Quelle"].map(lambda x: de(x, 0))
    cov_tbl["N_gematcht_analytisch"] = cov_tbl["N_gematcht_analytisch"].map(lambda x: de(x, 0))
    cov_tbl["Match_Rate_analytisch"] = cov_tbl["Match_Rate_analytisch"].map(pct)
    cov_tbl = cov_tbl[["Quelle", "Datensatz", "N_Quelle", "N_gematcht_analytisch",
                       "Match_Rate_analytisch", "Schluessel"]].rename(
        columns={
            "N_gematcht_analytisch": "N in Analysestichprobe",
            "Match_Rate_analytisch": "Match-Rate",
            "Schluessel": "Matching-Schlüssel",
        }
    )

    overlap_tbl = overlap.copy()
    overlap_tbl["N"] = overlap_tbl["N"].map(lambda x: de(x, 0))
    overlap_tbl["Anteil"] = overlap_tbl["Anteil"].map(pct)
    overlap_tbl = overlap_tbl.rename(columns={"Anzahl_Datenquellen": "Anzahl Datenquellen"})

    desc_ext_tbl = desc_ext[desc_ext["Variable"].isin(
        ["igdb_rating", "igdb_aggregated_rating", "igdb_n_genres", "mc_metascore",
         "mc_user_score", "ph_max_discount_pct", "ph_share_months_on_sale",
         "ph_price_changes_total", "ph_base_price", "ph_lowest_price"]
    )].copy()
    desc_ext_tbl["N"] = desc_ext_tbl["N"].map(lambda x: de(x, 0))
    for col in ["Mittelwert", "SD", "Median", "Q1", "Q3", "Min", "Max"]:
        desc_ext_tbl[col] = desc_ext_tbl[col].map(lambda x: de(x, 2))

    igdb_genre_tbl = igdb_genres[igdb_genres["N"] >= 100].copy()
    igdb_genre_tbl["Anteil_Stichprobe"] = igdb_genre_tbl["Anteil_Stichprobe"].map(pct)
    igdb_genre_tbl["Erfolgsquote"] = igdb_genre_tbl["Erfolgsquote"].map(pct)
    igdb_genre_tbl["CI"] = igdb_genre_tbl.apply(lambda r: f"[{pct(r['CI_unten'])}; {pct(r['CI_oben'])}]", axis=1)
    igdb_genre_tbl["Median_Besitzer"] = igdb_genre_tbl["Median_Besitzer"].map(lambda x: de(x, 0))
    igdb_genre_tbl["Median_Preis"] = igdb_genre_tbl["Median_Preis"].map(lambda x: de(x, 2))
    igdb_genre_tbl = igdb_genre_tbl[["IGDB_Genre", "N", "Anteil_Stichprobe", "Erfolgsquote", "CI",
                                     "Median_Besitzer", "Median_Preis"]].rename(
        columns={
            "IGDB_Genre": "IGDB-Genre", "Anteil_Stichprobe": "Anteil", "Erfolgsquote": "Erfolgsquote",
            "CI": "95-%-KI", "Median_Besitzer": "Median Besitzer", "Median_Preis": "Median-Preis (USD)",
        }
    )

    igdb_mode_tbl = igdb_modes.copy()
    igdb_mode_tbl["Erfolgsquote"] = igdb_mode_tbl["Erfolgsquote"].map(pct)
    igdb_mode_tbl["CI"] = igdb_mode_tbl.apply(lambda r: f"[{pct(r['CI_unten'])}; {pct(r['CI_oben'])}]", axis=1)
    igdb_mode_tbl["Median_Besitzer"] = igdb_mode_tbl["Median_Besitzer"].map(lambda x: de(x, 0))
    igdb_mode_tbl = igdb_mode_tbl[["IGDB_Merkmal", "N", "Erfolgsquote", "CI", "Median_Besitzer"]].rename(
        columns={"IGDB_Merkmal": "IGDB-Merkmal", "Erfolgsquote": "Erfolgsquote",
                 "CI": "95-%-KI", "Median_Besitzer": "Median Besitzer"}
    )

    def format_rating_table(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["Erfolgsquote"] = out["Erfolgsquote"].map(pct)
        out["CI"] = out.apply(lambda r: f"[{pct(r['CI_unten'])}; {pct(r['CI_oben'])}]", axis=1)
        out["Median_Besitzer"] = out["Median_Besitzer"].map(lambda x: de(x, 0))
        out["N"] = out["N"].map(lambda x: de(x, 0))
        return out[["Bewertungsmaß", "Quartil", "N", "Erfolgsquote", "CI", "Median_Besitzer"]].rename(
            columns={"Erfolgsquote": "Erfolgsquote", "CI": "95-%-KI", "Median_Besitzer": "Median Besitzer"}
        )

    igdb_rating_tbl = format_rating_table(igdb_ratings)
    mc_ext_tbl = format_rating_table(mc_ext)

    price_hist_tbl = price_hist.copy()
    price_hist_tbl["Preismerkmal"] = price_hist_tbl["Preismerkmal"].replace({"Preisänderungen": "Preisänderungen"})
    price_hist_tbl["Erfolgsquote"] = price_hist_tbl["Erfolgsquote"].map(pct)
    price_hist_tbl["CI"] = price_hist_tbl.apply(lambda r: f"[{pct(r['CI_unten'])}; {pct(r['CI_oben'])}]", axis=1)
    price_hist_tbl["Median_Besitzer"] = price_hist_tbl["Median_Besitzer"].map(lambda x: de(x, 0))
    price_hist_tbl = price_hist_tbl[["Preismerkmal", "Quartil", "N", "Erfolgsquote", "CI", "Median_Besitzer"]].rename(
        columns={"Preismerkmal": "Preismerkmal", "Erfolgsquote": "Erfolgsquote",
                 "CI": "95-%-KI", "Median_Besitzer": "Median Besitzer"}
    )

    def format_test_table(df: pd.DataFrame, family: str) -> str:
        part = df[df["Familie"] == family].copy()

        def effect_cell(row: pd.Series) -> str:
            test = str(row["Test"])
            if test.startswith("Chi2 (kategorial)") and pd.notna(row.get("Zusatz_Effekt")):
                return de(row["Zusatz_Effekt"], 2)
            if test.startswith("Kruskal") and pd.notna(row.get("Zusatz_Effekt")):
                return de(row["Zusatz_Effekt"], 3)
            return eff_ci(row["Effekt"], row["CI_unten"], row["CI_oben"], 2)

        def effect_label(row: pd.Series) -> str:
            test = str(row["Test"])
            if test.startswith("Chi2 (kategorial)") and pd.notna(row.get("Zusatz_Effekt")):
                return "Cramér's V"
            if test.startswith("Kruskal") and pd.notna(row.get("Zusatz_Effekt")):
                return "ε²"
            return str(row["Effekt_typ"])

        part["Effekt"] = part.apply(effect_cell, axis=1)
        part["Effekt_typ"] = part.apply(effect_label, axis=1)
        part["p_adj"] = part["p_adj"].map(pval)
        part["N"] = part["N"].map(lambda x: de(x, 0))
        cols = ["Merkmal", "Test", "Effekt_typ", "N", "Effekt", "p_adj"]
        part = part[cols].rename(
            columns={"Effekt_typ": "Effektmaß", "Effekt": "Effekt [95-%-KI]",
                     "p_adj": "p (BH-korrigiert)"}
        )
        return md(part)

    diag_tbl = diag.copy()
    diag_tbl["Effekt"] = diag_tbl.apply(lambda r: eff_ci(r["Effekt"], r["CI_unten"], r["CI_oben"], 3), axis=1)
    diag_tbl["p_adj"] = diag_tbl["p_adj"].map(pval)
    diag_tbl = diag_tbl[["Merkmal", "N", "Effekt", "p_adj"]].rename(
        columns={"Effekt": "Spearman rho [95-%-KI]", "p_adj": "p (BH-korrigiert)"}
    )

    selection_tbl = selection.copy()
    selection_tbl["Effekt"] = selection_tbl.apply(
        lambda r: eff_ci(r["Effekt"], r["CI_unten"], r["CI_oben"], 2), axis=1
    )
    selection_tbl["p_adj"] = selection_tbl["p_adj"].map(pval)
    selection_tbl = selection_tbl[["Merkmal", "N", "Effekt", "p_adj"]].rename(
        columns={"Effekt": "Cliff's Delta", "p_adj": "p (BH-korrigiert)"}
    )

    # Zentrale Effekte fuer den Fliesstext.
    mmo_or = cite(h1, "Genre: Massively Multiplayer", "Logistische Regression (adjustiert)")
    rpg_or = cite(h1, "Genre: RPG", "Logistische Regression (adjustiert)")
    casual_or = cite(h1, "Genre: Casual", "Logistische Regression (adjustiert)")
    f2p_or = cite(h2, "Erfolgsquote: Free-to-Play vs. kostenpflichtig", "Chi2 / Odds Ratio")
    positive_ratio_rho = cite(h3, "Positive Ratio vs. Besitzer (mind. 10 Reviews)", "Spearman", 3)
    wilson_rho = cite(h3, "Wilson-Score vs. Besitzer (mind. 10 Reviews)", "Spearman", 3)
    mc_ext_rho = cite(h3, "Metacritic (extern) vs. Besitzer", "Spearman", 3)
    igdb_rating_rho = cite(h4, "IGDB-Nutzerbewertung vs. Besitzer", "Spearman", 3)
    igdb_agg_rho = cite(h4, "IGDB-Kritikerbewertung vs. Besitzer", "Spearman", 3)
    ph_discount_rho = cite(h4, "Maximaler Rabatt (Preishistorie) vs. Besitzer", "Spearman", 3)
    ph_discount_partial = cite(h4, "Maximaler Rabatt (Preishistorie) vs. Besitzer", "Spearman (partiell, Jahr)", 3)
    ph_share_rho = cite(h4, "Anteil Monate mit Rabatt vs. Besitzer", "Spearman", 3)
    ph_share_partial = cite(h4, "Anteil Monate mit Rabatt vs. Besitzer", "Spearman (partiell, Jahr)", 3)
    ph_changes_rho = cite(h4, "Preisänderungen (Preishistorie) vs. Besitzer", "Spearman", 3)
    ph_changes_partial = cite(h4, "Preisänderungen (Preishistorie) vs. Besitzer", "Spearman (partiell, Jahr)", 3)
    igdb_mp_or = cite(h4, "IGDB: Multiplayer vs. kein Multiplayer", "Logistische Regression (adjustiert)")

    text = f"""# Umsetzung der Analyse: Erfolgsfaktoren von Videospielen auf Steam

*Ergebnisbericht zur empirischen Umsetzung des Forschungsvorhabens.*  
*Datengrundlage: FronkonGames Steam Games Dataset (Version 57) sowie externe IGDB-, Metacritic- und Preishistorien-Daten. Auswertungsstand: 10. November 2026.*

## 0. Kurzfassung

Die Analyse verbindet **vier Datenquellen**: den FronkonGames-Steam-Datensatz, IGDB-Spielemetadaten, einen externen Metacritic-Datensatz und eine Steam-Preishistorie. Die Basis-Analysestichprobe umfasst **{de(n_analytic, 0)} Spiele**. Davon lassen sich **{de(n_igdb, 0)}** mit IGDB ({pct(n_igdb / n_analytic)}), **{de(n_mc_ext, 0)}** mit externen Metacritic-Daten ({pct(n_mc_ext / n_analytic)}) und **{de(n_price, 0)}** mit der Preishistorie ({pct(n_price / n_analytic)}) verknüpfen. Für **{de(n_multi_source, 0)}** Spiele liegen mindestens zwei Quellen vor, für **{de(n_all_sources, 0)}** alle vier.

Als Erfolg wird primär das **obere Quartil der Besitzerverteilung** definiert (untere Besitzergrenze ≥ {de(q75, 0)}; {pct(top25)} der Analysestichprobe). Ergänzend werden die oberen 10 % (≥ {de(q90, 0)}; {pct(top10)}) und „mindestens 100.000 Besitzer“ ({pct(top100k)}) untersucht. Es werden **keine Prognose- oder Klassifikationsmodelle** geschätzt; berichtet werden deskriptive Muster, Effektstärken und statistische Tests.

Zentrale Befunde:

- **Spieleigenschaften (F1):** Massively Multiplayer, RPG und Strategy sind unter erfolgreichen Spielen überdurchschnittlich häufig vertreten, Casual und Indie seltener. Die IGDB-Daten bestätigen die Bedeutung von Mehrspieler- und Koop-Merkmalen; die IGDB-Bewertungen korrelieren ebenfalls mit der Reichweite.
- **Preis (F2):** Free-to-Play-Spiele erreichen häufiger hohe Besitzerzahlen. Unter kostenpflichtigen Spielen ist der Zusammenhang zwischen aktuellem Preisniveau und Besitzern schwach. Die **Preishistorie** zeigt für die 645 gematchten Spiele eine starke Rohkorrelation zwischen maximalem Rabatt und Besitzern; nach Kontrolle des Veröffentlichungsjahres verschwindet sie jedoch. Das spricht eher für Alters- und Selektionskonfounder als für einen eigenständigen Rabatteffekt.
- **Bewertungen (F3):** Die Höhe der Steam-Nutzerbewertung (Positive Ratio) hängt praktisch nicht mit der Reichweite zusammen; der Wilson-Score tut dies nur, weil er das Review-Volumen enthält. Externe Metacritic-Scores zeigen in der selektiven Teilstichprobe einen moderaten positiven Zusammenhang mit Besitzern und Peak CCU.
- **Quellenübergreifend:** IGDB-Nutzer- und Kritikerbewertungen sowie externe Metacritic-Scores liefern konsistente, aber unterschiedlich starke Signale. Die Matching-Quote und die Selektionsverzerrung werden transparent ausgewiesen.

## 1. Datengrundlage, Datenqualität und Stichprobe

### 1.1 Datenquellen und Matching

Verwendet werden:

1. **FronkonGames Steam Games Dataset** (Kaggle, Version 57): 125.855 Zeilen mit AppID, Name, Release-Datum, geschätzten Besitzern, Peak CCU, Preis, Rabatt, DLC-Anzahl, Plattformen, Metacritic-Score, Reviews, Kategorien und Genres. Dies ist der Basisdatensatz.
2. **IGDB-Spielemetadaten** (Kaggle: `emirshn/igdb-dataset-for-data-mining-projects`, Version 1): 232.595 Rohzeilen (230.201 nach Name/Jahr-Deduplizierung) mit IGDB-ID, Name, Release-Datum, Nutzer- und Kritikerbewertung, Genres, Spielmodi, Plattformen, Entwicklern und Publishern.
3. **Externer Metacritic-Datensatz** (Kaggle: `stadynromn/metacritic-video-games-reviews-up-to-sep-2025`, Version 1): 22.224 Plattform-Einträge, davon 6.387 PC-Einträge (6.277 nach Name/Jahr-Deduplizierung) mit Metascore, Nutzerscore, Genre, Entwickler und Publisher.
4. **Steam-Preishistorie** (Kaggle: `andrii222/steam-price-history-nov-2023-oct-2025`, Version 1): 651 Apps mit monatlichen Preisinformationen von November 2023 bis November 2025 (Basispreis, niedrigster Preis, Preisänderungen, Rabatttage).

Hinweis: Der externe Metacritic-Datensatz und die Metacritic-Spalte von FronkonGames sind weitgehend deckungsgleich. Der externe Datensatz dient daher dem Quellenabgleich und der PC-spezifischen Ergänzung, nicht als unabhängige Validierung.

Die Verknüpfung erfolgt über:

- **IGDB:** normalisierter Spielname plus Veröffentlichungsjahr; pro Name/Jahr wird ein IGDB-Treffer ausgewählt.
- **Metacritic:** normalisierter Spielname plus Veröffentlichungsjahr, ausschließlich PC-Einträge; pro Name/Jahr ein Treffer.
- **Preishistorie:** Steam-AppID (exakter Join).

Die Matching-Ergebnisse:

{md(cov_tbl)}

Die **IGDB-Abdeckung** ist mit {pct(n_igdb / n_analytic)} der Analysestichprobe gut. **Externes Metacritic** ist mit {pct(n_mc_ext / n_analytic)} und die **Preishistorie** mit {pct(n_price / n_analytic)} deutlich selektiver. Die Preishistorie enthält nur 651 Apps; die Analysen zu Preisentwicklung sind daher als **Fallstudie für populäre Titel** zu lesen, nicht als repräsentative Stichprobe aller Steam-Spiele.

Die Überlappung der Quellen in der Analysestichprobe:

{md(overlap_tbl)}

### 1.2 Datenqualität

Ein technisches Problem des FronkonGames-Rohdatensatzes: Die Kopfzeile enthält 39 Spaltennamen, die Datenzeilen enthalten 40 Felder. Die Spalte `DiscountDLC count` ist in den Daten in `Discount` und `DLC count` aufgeteilt. Die Analyse liest die Datei daher mit einer expliziten 40er-Spaltenliste ein. Weitere Auffälligkeiten:

{md(quality_sel)}

Die wichtigsten Einschränkungen sind die **große Zahl fehlender Besitzerangaben** (`0 - 0`: 19,5 %), die **grobe Besitzerklassierung** (74,5 % der Analysestichprobe liegen in der untersten Klasse 0–20.000), die **hohe Zahl fehlender Metacritic-Scores** im Basisdatensatz und der hohe Anteil an Spielen ohne Reviews (34,1 %). Peak CCU ist bei 84,4 % der Spiele null; diese Nullwerte werden als „keine ausgewiesene Aktivität“ behandelt, nicht als fehlend.

Die externen Quellen haben eigene Einschränkungen: IGDB enthält viele Editionen und Mehrfacheinträge, die über Name/Jahr dedupliziert werden; Metacritic deckt nur PC-Versionen ab; die Preishistorie umfasst nur 651 Apps und monatliche Aggregationen. Diese Unterschiede werden in den Grenzen ausführlich diskutiert.

### 1.3 Bereinigung und Analyse-Sample

Die Bereinigung erfolgt sequenziell: fehlende Besitzerangaben, Veröffentlichungen nach 2025 (unvollständiges Jahr), Demo-/Playtest-/Soundtrack-/DLC-Einträge, Software-/Utilities-Einträge und Namens-/Editionsduplikate werden entfernt. Die Duplikaterkennung normalisiert Spielnamen (u. a. Entfernung von Editionszusätzen) und gruppiert zusätzlich nach Veröffentlichungsjahr; pro Gruppe bleibt der Eintrag mit den meisten Besitzern erhalten.

{md(flow_tbl)}

Die **Analysestichprobe** umfasst **{de(n_analytic, 0)} Spiele**. Für Bewertungsanalysen wird zusätzlich eine Mindestzahl von **10 Reviews** verlangt ({de(n_rating, 0)} Spiele; Robustheit: 50 Reviews, {de(n_rating50, 0)} Spiele). Für IGDB-Analysen stehen {de(n_igdb, 0)} Spiele zur Verfügung, für externe Metacritic-Analysen {de(n_mc_ext, 0)}, für Preishistorien-Analysen {de(n_price, 0)}. Für Peak-CCU-Analysen werden nur Spiele mit Peak CCU > 0 betrachtet ({de(n_peak, 0)} Spiele). Die FronkonGames-eigene Metacritic-Spalte deckt {de(n_meta_fronk, 0)} Spiele ab und dient nur als Vergleichswert.

**Selektionshinweis:** Die Review-Schwelle verschiebt die Stichprobe deutlich in Richtung populärer Spiele. Die Erfolgsprävalenz (oberes Quartil) beträgt in der Analysestichprobe {pct(top25)}, in der Bewertungsstichprobe dagegen {pct(analytic.loc[analytic['rating_sample'], 'success_top25'].mean())}. Die externen Teilstichproben sind noch stärker selektiert; die Selektions-Tests werden in Abschnitt 3.5 dokumentiert.

### 1.4 Operationalisierung von Erfolg

Die Besitzerangabe liegt nur als Spanne vor. Verwendet werden die untere Grenze (`owners_low`) für Schwellenwerte und der Mittelpunkt (`owners_mid`) für Rangkorrelationen und die log-transformierte Zielgröße. Die Erfolgsschwellen und ihre tatsächliche Prävalenz:

{md(th_tbl)}

Die starke Konzentration in der untersten Klasse (0–20.000) ist eine zentrale Messgrenze: Das obere Quartil entspricht praktisch der Menge der Spiele, die die unterste Klasse verlassen. Feine Unterschiede innerhalb der 0–20.000-Klasse sind nicht beobachtbar.

## 2. Deskriptive Ergebnisse

### 2.1 Verteilung der Besitzer

{md(owners_tbl)}

Die Verteilung ist stark rechtsschief: 74,5 % der Spiele liegen in der untersten Klasse, während nur {pct(share_100_200, 3)} ({de(n_100_200, 0)} Spiele) die Klasse 100–200 Mio. erreichen. Für Korrelationen und Gruppenvergleiche wird daher `log10(owners_mid)` verwendet.

![Verteilung der geschätzten Besitzer](../figures/fig_01_owners_distribution.png)

### 2.2 Steam-Genres

{md(genre_tbl)}

Massively Multiplayer, RPG und Strategy weisen die höchsten Erfolgsquoten auf; Casual, Sports und Racing liegen unter dem Gesamtwert. Der Median-Preis unterscheidet sich zwischen den Genres nur moderat; der F2P-Anteil ist bei Massively Multiplayer deutlich erhöht.

![Erfolgsquote nach Genre](../figures/fig_03_genre_success.png)

### 2.3 Spielmodi und Plattformen

{md(mode_tbl)}

Nur-Singleplayer-Spiele haben eine niedrigere Erfolgsquote als Single- plus Multiplayer-Spiele; reine Multiplayer-Titel liegen dazwischen. Spiele mit mehreren Plattformen sind erfolgreicher. Der Anteil an Spielen mit Singleplayer-Kategorie ist sehr hoch; die Kategorie „Singleplayer“ ist daher eher der Normalfall als ein Differenzierungsmerkmal.

![Spielmodi und Plattformen](../figures/fig_04_modes_platforms.png)

### 2.4 Aktueller Preis und aktuelle Rabatte

{md(price_tbl)}

Free-to-Play-Spiele haben die höchste Erfolgsquote. Unter den kostenpflichtigen Spielen ist der Zusammenhang nicht monoton; die 40+-Klasse hat eine niedrige Erfolgsquote, aber einen hohen Mittelwert der Besitzer, was auf einzelne sehr große Titel hindeutet.

{md(disc_tbl)}

Mit steigender aktuellem Rabattstufe steigt die Erfolgsquote deutlich. Dieser Zusammenhang ist **nicht kausal** zu lesen: Rabatte werden häufig gerade für ältere oder bereits populäre Titel gesetzt, und der Basisdatensatz enthält nur den **aktuellen** Rabatt, nicht die Rabattgeschichte.

![Preis und Besitzer](../figures/fig_05_price.png)

![Rabatt und Erfolg](../figures/fig_06_discount.png)

### 2.5 Steam-Bewertungen und externes Metacritic

{md(ratings_tbl)}

Der Wilson-Score zeigt einen monotonen Anstieg der Erfolgsquote über die Quartile. Die Positive Ratio zeigt dagegen ein nicht-monotones Muster. Der scheinbare Wilson-Effekt ist teilweise auf das Bewertungsvolumen zurückzuführen (siehe Abschnitt 3.6).

Die externe Metacritic-Teilstichprobe ist stark selektiert; die Erfolgsquoten und Besitzer-Mediane nach Metacritic-Quartilen:

{md(mc_ext_tbl)}

**Wichtig:** Der externe Metacritic-Datensatz ist keine unabhängige zweite Quelle. Von den {de(n_mc_overlap, 0)} Spielen mit beiden Scores stimmen {de(n_mc_exact, 0)} exakt mit der FronkonGames-Spalte überein (Spearman ρ = {de(mc_agreement_rho, 4)}). Der Abgleich prüft daher das Matching, nicht die externe Validität.

{md(meta_cov_tbl)}

![Bewertungen und Besitzer](../figures/fig_07_ratings.png)

![Kritiker- vs. Nutzerbewertung](../figures/fig_10_rating_agreement.png)

### 2.6 IGDB-Merkmale

Die IGDB-Daten liefern eine unabhängige Quelle für Genres, Spielmodi und Bewertungen. Deskriptive Statistik der IGDB-Variablen:

{md(desc_ext_tbl)}

Erfolgsquote nach IGDB-Genre (nur Genres mit mindestens 100 Spielen):

{md(igdb_genre_tbl)}

Erfolgsquote nach IGDB-Spielmodus:

{md(igdb_mode_tbl)}

Erfolgsquote nach IGDB-Bewertungsquartilen:

Die IGDB-Nutzerbewertung ist stark gebunden: {pct(igdb_tie_share)} der Spiele mit IGDB-Bewertung haben exakt 70/100. Die Quartile werden daher rangbasiert gebildet; bei Gleichständen entscheidet eine feste Zufallsreihenfolge (Seed 42), welchem Quartil ein Spiel zugeordnet wird. Dadurch sind die vier Gruppen gleich groß und die Erfolgsquoten vergleichbar, die Quartilsgrenzen selbst sind wegen der Bindungen aber nicht inhaltlich interpretierbar. Zusätzlich hängt die Nachkommastelle der IGDB-Bewertung vom Bewertungsvolumen ab: Spiele mit ganzzahliger Bewertung ({de(n_igdb_integer, 0)}; Erfolgsquote {pct(succ_igdb_integer)}) unterscheiden sich deutlich von Spielen mit nicht-ganzzahliger Bewertung ({de(n_igdb_decimal, 0)}; {pct(succ_igdb_decimal)}). Die IGDB-Nutzerbewertung ist daher als ordinale Erfolgskennzahl nur eingeschränkt interpretierbar; die Quartilstabelle zeigt entsprechend kein monotones Muster.

{md(igdb_rating_tbl)}

![IGDB-Bewertungen und Besitzer](../figures/fig_11_igdb_ratings.png)

### 2.7 Preishistorie

Für die 645 gematchten Spiele stehen monatliche Preisinformationen von November 2023 bis November 2025 zur Verfügung. Analysiert werden der maximale Rabatt, die Rabatthäufigkeit (Anteil Monate mit Rabatt) und die Anzahl der Preisänderungen.

{md(price_hist_tbl)}

Roh betrachtet weisen Spiele mit höheren maximalen Rabatten und mehr Preisänderungen höhere Besitzerzahlen auf; die Rabatthäufigkeit allein zeigt dagegen kaum einen Zusammenhang. Nach Kontrolle des Veröffentlichungsjahres bleibt nur der Zusammenhang der Preisänderungen bestehen, während der maximale Rabatt und die Rabatthäufigkeit statistisch nicht mehr signifikant sind. Wegen der kleinen, selektiven Stichprobe und der gemeinsamen Zeitperiode sind dies **Assoziationen, keine Kausaleffekte**.

![Preishistorie und Erfolg](../figures/fig_12_price_history.png)

### 2.8 Datenquellen-Überlappung

{md(overlap_tbl)}

Die meisten Spiele stammen aus dem Basisdatensatz; 66.354 Spiele haben zusätzlich IGDB-Daten. Nur 290 Spiele haben alle vier Quellen. DieMulti-Source-Analysen nutzen daher jeweils die größtmögliche Teilstichprobe und dokumentieren die unterschiedliche Abdeckung.

![Datenquellen-Overlap](../figures/fig_14_source_overlap.png)

## 3. Statistische Tests

Die Tests berichten Effektstärken. 95-%-Konfidenzintervalle werden für Odds Ratios und Spearman-Korrelationen ausgewiesen; für Chi²-, Kruskal-Wallis- und Cliff's-Delta-Maße werden die Effektstärken (Cramér's V, ε², Cliff's Delta) ohne Konfidenzintervall angegeben. p-Werte werden innerhalb jeder Hypothese per **Benjamini-Hochberg**-Verfahren korrigiert. Wegen der großen Stichprobe sind viele p-Werte extrem klein; die inhaltliche Bewertung stützt sich daher auf Effektgrößen.

### 3.1 H1 – Spieleigenschaften

**Genres, Spielmodi und Plattformen** (unadjustierte OR sowie adjustierte OR mit Kontrolle für Veröffentlichungsjahr und Jahr²):

{format_test_table(h1, "H1 - Spieleigenschaften")}

Kernergebnisse:

- **Genres:** Massively Multiplayer (adjustierte OR {mmo_or}), RPG ({rpg_or}) und Strategy sind nach Alterskontrolle überrepräsentiert. Casual ({casual_or}) und Indie sind unterrepräsentiert.
- **Spielmodi:** Multiplayer, Koop, MMO und PvP sind positiv mit Erfolg assoziiert. Die Singleplayer-Kennzahl ist kleiner als 1, weil Spiele **ohne** Singleplayer (also reine Multiplayer-Titel) erfolgreicher sind; das ist kein Beleg gegen Singleplayer-Spiele als solche.
- **Plattformen:** Mehr als eine Plattform, macOS und Linux sind positiv assoziiert. Die Plattformangabe ist auch ein Indikator für Portierungsaufwand und Zielgruppe; eine kausale Interpretation ist nicht möglich.

**H1 wird insgesamt gestützt:** Bestimmte Genres, Spielmodi und Plattformen sind unter erfolgreichen Spielen überrepräsentiert, und die Befunde bleiben nach Alterskontrolle bestehen.

### 3.2 H2 – Preis

{format_test_table(h2, "H2 - Preis")}

Kernergebnisse:

- **Free-to-Play vs. kostenpflichtig:** F2P-Spiele haben eine höhere Erfolgsquote (OR {f2p_or}) und tendenziell mehr Besitzer. Der Unterschied bei Peak CCU ist dagegen vernachlässigbar. F2P-Spiele erreichen also eher eine große Besitzerbasis, aber nicht zwingend eine höhere gleichzeitige Spieleraktivität.
- **Preisniveau (nur kostenpflichtige Spiele):** Die unadjustierte Spearman-Korrelation zwischen Preis und Besitzern ist praktisch null; nach Kontrolle des Veröffentlichungsjahres ergibt sich eine kleine positive Korrelation. Mit Peak CCU ist der Zusammenhang etwas stärker.
- **Rabatte:** Der aktuelle Rabatt korreliert positiv mit Besitzern und Peak CCU; Spiele mit aktivem Rabatt haben eine höhere Erfolgsquote. Die Rabattgruppen zeigen einen monotonen Gradienten. Wegen der Querschnittsdaten und der momentanen Rabattmessung sind diese Zusammenhänge nicht kausal interpretierbar.
- **DLC-Anzahl:** Die DLC-Anzahl korreliert positiv mit Besitzern. Auch hier ist die Richtung offen: Erfolgreiche Spiele erhalten mehr DLCs, oder DLCs erhöhen die Sichtbarkeit.

**H2 wird teilweise gestützt:** Der Unterschied zwischen kostenlosen und kostenpflichtigen Spielen ist deutlich. Unter den kostenpflichtigen Spielen ist der Preisniveau-Effekt auf die Besitzerzahl jedoch schwach; Rabatte und DLC-Anzahl zeigen stärkere, aber nicht kausal interpretierbare Zusammenhänge.

### 3.3 H3 – Bewertungen

{format_test_table(h3, "H3 - Bewertungen")}

Kernergebnisse:

- **Steam-Nutzerbewertungen:** Der Wilson-Score korreliert positiv mit Besitzern ({wilson_rho}) und Peak CCU. Die **Positive Ratio** – der reine Anteil positiver Bewertungen – korreliert dagegen praktisch nicht mit Besitzern ({positive_ratio_rho}). Mit Peak CCU besteht ein schwacher positiver Zusammenhang. Nach Kontrolle des Veröffentlichungsjahres wird die Positive-Ratio-Korrelation mit Besitzern leicht positiv, bleibt aber klein.
- **Externes Metacritic:** In der Teilstichprobe korreliert der externe Metacritic-Score positiv mit Besitzern ({mc_ext_rho}) und Peak CCU. Die Übereinstimmung zwischen Steam-Wilson und externem Metacritic ist moderat. Wegen der starken Selektion (nur {pct(n_mc_ext / n_analytic)} der Analysestichprobe haben externe Metacritic-Daten) sind diese Werte nicht auf alle Spiele verallgemeinerbar.
- **Vergleich der Metacritic-Quellen:** Der externe Metacritic-Score und die FronkonGames-eigene Metacritic-Spalte werden direkt gegeneinander geprüft. Da {de(n_mc_exact, 0)} von {de(n_mc_overlap, 0)} Werten exakt übereinstimmen (ρ = {de(mc_agreement_rho, 4)}), ist das faktisch dieselbe Quelle; der Abgleich ist ein Matching-Check, kein unabhängiger Validitätsnachweis.

**H3 wird für Kritikerbewertungen gestützt, für Steam-Nutzerbewertungen nur eingeschränkt:** Die Bewertungshöhe der Steam-Nutzer hängt nicht mit der Reichweite zusammen; der positive Wilson-Besitzer-Zusammenhang ist größtenteils ein Volumenartefakt.

### 3.4 H4 – Quellenübergreifende Analysen

Die folgende Tabelle prüft, ob die externen Quellen die Befunde stützen. IGDB-Bewertungen und -Modi, Preishistorien-Merkmale und die Übereinstimmung zwischen den Bewertungsquellen werden getrennt ausgewiesen:

{format_test_table(h4, "H4 - Externe Datenquellen")}

Kernergebnisse:

- **IGDB-Bewertungen:** IGDB-Nutzerbewertungen korrelieren mit Besitzern ({igdb_rating_rho}), IGDB-Kritikerbewertungen ebenfalls ({igdb_agg_rho}). Die Rangkorrelationen sind kleiner als beim externen Metacritic, aber konsistent in der Richtung. Bei der Nutzerbewertung ist zu beachten, dass die Bewertungspräzision mit dem Bewertungsvolumen zusammenhängt (Abschnitt 2.6); der Zusammenhang ist daher teilweise ein Popularitätsartefakt.
- **IGDB-Spielmodi:** IGDB-Multiplayer ist positiv mit Erfolg assoziiert (adjustierte OR {igdb_mp_or}); die übrigen Modi zeigen ebenfalls positive, aber unterschiedlich starke Assoziationen.
- **Preishistorie:** Der maximale Rabatt korreliert roh mit Besitzern ({ph_discount_rho}), nach Alterskontrolle jedoch nicht mehr ({ph_discount_partial}). Die Rabatthäufigkeit ist roh ({ph_share_rho}) und partiell ({ph_share_partial}) schwach. Die Anzahl der Preisänderungen korreliert roh ({ph_changes_rho}) und auch nach Alterskontrolle ({ph_changes_partial}) positiv. Die bivariaten Rabattmuster sind damit stark durch die selektive Stichprobe und das Alter der Spiele geprägt.
- **Quellenübereinstimmung:** IGDB-Kritikerbewertungen, externes Metacritic und Steam-Wilson zeigen positive Übereinstimmungen. Die externen Quellen widersprechen den Steam-Befunden nicht, messen aber unterschiedliche Konstrukte (Kritikerurteil, Nutzerurteil, Reichweite).

**H4 wird teilweise gestützt:** IGDB-Bewertungen und -Modi liefern konsistente Zusatzinformationen; die Preishistorie zeigt dagegen nur für Preisänderungen einen robusten Zusammenhang. Wegen der unterschiedlichen Abdeckung und Zeiträume sind die Ergebnisse **komplementär und nicht austauschbar**.

### 3.5 Selektionsverzerrung der externen Teilstichproben

Spiele mit externen Daten sind im Mittel größer und populärer als Spiele ohne externe Daten; der Unterschied ist für Metacritic (Cliff's Delta 0,77) und Preishistorie (0,81) stark, für IGDB nur schwach (0,05). Die folgende Tabelle vergleicht die Besitzerverteilung zwischen gematchten und nicht gematchten Spielen:

{md(selection_tbl)}

Die Matching-Quote ist daher kein neutraler Datenzusatz: IGDB-, Metacritic- und Preishistorien-Analysen sind **Selektionsanalysen** für Spiele, die in den jeweiligen Quellen dokumentiert sind. Das ist besonders für die Preishistorie (651 Apps) und das externe Metacritic (PC-only) relevant.

### 3.6 Bewertungsvolumen als Störfaktor

Die folgende Diagnose trennt Bewertungshöhe und Bewertungsvolumen:

{md(diag_tbl)}

Die Review-Anzahl korreliert sehr stark mit Besitzern und mit dem Wilson-Score. Wird das Review-Volumen herauspartialisiert, wird die Wilson-Besitzer-Korrelation **negativ** (−0,087 bei mind. 10, −0,197 bei mind. 50 Reviews). Der Wilson-Score misst daher nicht nur Zufriedenheit, sondern auch, wie viele Reviews ein Spiel hat – und die Review-Anzahl ist selbst ein Popularitätsmaß. Die **Positive Ratio** ist davon weniger betroffen, aber nicht frei davon: Nach Kontrolle des Review-Volumens ist auch ihr Zusammenhang mit Besitzern negativ (−0,057 bzw. −0,193). Sie wird daher als näher an der reinen Bewertungshöhe verwendet; der Wilson-Score dient nur als Sensitivitätsanalyse.

## 4. Beantwortung der Forschungsfragen

### F1 – Spieleigenschaften

**Welche Genres, Spielmodi und Plattformen sind bei erfolgreichen Spielen überdurchschnittlich häufig vertreten?**

Massively Multiplayer, RPG und Strategy sind überdurchschnittlich, Casual und Indie unterdurchschnittlich vertreten. Mehrspieler-, Koop-, MMO- und PvP-Spiele sowie Spiele mit mehreren Plattformen (insbesondere macOS und Linux) sind häufiger erfolgreich. Die IGDB-Daten bestätigen die Bedeutung von Multiplayer- und Koop-Merkmalen. Die Befunde bleiben nach Kontrolle des Veröffentlichungsjahres bestehen. H1 wird gestützt.

### F2 – Preis

**Welcher Zusammenhang besteht zwischen Preisstrategie und Popularität?**

Free-to-Play-Spiele erreichen häufiger hohe Besitzerzahlen. Unter kostenpflichtigen Spielen ist der Preisniveau-Effekt auf die Besitzerzahl schwach; mit der Spieleraktivität (Peak CCU) ist er etwas stärker. Rabatte und DLC-Anzahl sind deutlich stärker mit Popularität assoziiert, aber die Richtung ist offen (Reverse Causality). Die **Preishistorie** ergänzt den Querschnitt: Maximale Rabatte korrelieren roh, aber nicht nach Alterskontrolle; Preisänderungen bleiben auch nach Alterskontrolle positiv assoziiert. Die Stichprobe umfasst nur 645 Apps. H2 wird teilweise gestützt.

### F3 – Bewertungen

**Wie stark hängen Nutzer- und Kritikerbewertungen mit Popularität zusammen?**

Die Steam-Bewertungshöhe (Positive Ratio) hängt roh praktisch nicht mit der Reichweite zusammen; nach Kontrolle des Review-Volumens ist der Zusammenhang sogar leicht negativ. Mit der Spieleraktivität besteht ein schwacher positiver Zusammenhang. Der Wilson-Score zeigt nur deshalb einen positiven Besitzer-Zusammenhang, weil er das Review-Volumen enthält. Externe Metacritic-Scores und IGDB-Bewertungen korrelieren in den jeweiligen Teilstichproben moderat positiv mit Besitzern und Peak CCU; die Übereinstimmung zwischen den Bewertungsquellen ist moderat. H3 wird für Kritikerbewertungen gestützt, für Steam-Nutzerbewertungen nur eingeschränkt.

### Quellenübergreifende Einordnung

Die vier Quellen messen unterschiedliche Dinge: Steam liefert Reichweite und Nutzerfeedback, IGDB zusätzliche Genres/Modi und Bewertungen, Metacritic das Kritikerurteil, die Preishistorie die Rabattdynamik. Die Ergebnisse sind **konsistent in der Richtung**, aber **nicht deckungsgleich**. Die Analyse ist damit eine Multi-Source-Beschreibung, keine Prognose und keine kausale Identifikation.

## 5. Grenzen der Umsetzung

- **Grobe Zielgröße:** `estimated_owners` liegt nur als Spanne vor; 74,5 % der Analysestichprobe liegen in der untersten Klasse 0–20.000. Das obere Quartil entspricht daher im Wesentlichen „verlässt die unterste Klasse“. Feine Unterschiede innerhalb dieser Klasse sind nicht beobachtbar.
- **Querschnitt und Snapshot:** Preis, Rabatt, Reviews, Besitzer, Peak CCU und der FronkonGames-Metacritic-Score beziehen sich auf unterschiedliche Erhebungszeitpunkte. Die Preishistorie deckt 2023–2025 ab, der externe Metacritic-Datensatz bis September 2025. Zeitliche Entwicklungen und Kausalität sind nicht sauber identifizierbar.
- **Matching-Fehler:** IGDB und Metacritic werden über normalisierten Namen und Jahr verknüpft. Editionsvarianten, Remakes, abweichende Schreibweisen und fehlende Jahresangaben können zu Fehltreffern oder Nicht-Treffern führen. Die IGDB-Abdeckung ist gut, die Metacritic- und Preishistorien-Abdeckung klein.
- **Selektionsverzerrung:** Die Review-Schwelle verschiebt die Stichprobe zu populäreren Spielen. Externe Metacritic-Daten liegen nur für 3,1 % der Analysestichprobe vor, Preishistorien nur für 0,6 %. Die Selektions-Tests zeigen systematische Unterschiede bei den Besitzern.
- **Preishistorie:** Nur 651 Apps, monatliche Aggregation, keine täglichen Preisänderungen und keine Repräsentation des gesamten Steam-Katalogs. Die Ergebnisse sind eine Fallstudie für populäre Titel.
- **IGDB-Datenqualität:** Der Kaggle-Dump enthält viele Editionen und Mehrfacheinträge; die Deduplizierung ist konservativ, kann aber falsche Einträge auswählen. Die IGDB-Bewertungen haben eine andere Skala und Nutzerbasis als Steam; zudem korreliert die Bewertungspräzision mit dem Bewertungsvolumen.
- **Externes Metacritic:** Nur PC-Einträge, stark selektiv, andere Plattform- und Zeitabdeckung. Die hohe Übereinstimmung mit der FronkonGames-Spalte ({de(n_mc_exact, 0)} von {de(n_mc_overlap, 0)} exakt) zeigt vor allem, dass beide Felder aus derselben Quelle stammen; sie ist kein unabhängiger Beleg für Messäquivalenz.
- **Keine Prognose:** Es werden bewusst **keine Klassifikations- oder Vorhersagemodelle** geschätzt. Die Aussagen sind deskriptiv und assoziativ; sie sind keine Erfolgsprognose für neue Spiele.
- **Wilson-Score:** Der Wilson-Score enthält das Review-Volumen und damit ein Popularitätsmaß. Er wird nur als Sensitivitätsanalyse berichtet; die Positive Ratio liegt näher an der reinen Bewertungshöhe, ist aber ebenfalls nicht vollständig vom Review-Volumen unabhängig.

## 6. Reproduzierbarkeit

Die Analyse ist in vier Auswertungsskripten plus einem Modul für die externen Daten organisiert (Python 3.11):

1. `scripts/external_data.py` – Laden, Bereinigen und Matchen von IGDB, Metacritic und Preishistorie.
2. `scripts/01_prepare_data.py` – Laden, Bereinigen, Feature Engineering, externe Quellen, Stichprobenziehung.
3. `scripts/02_eda.py` – deskriptive Tabellen und Abbildungen.
4. `scripts/03_tests.py` – Chi²-/Odds-Ratio-, Mann-Whitney-, Kruskal-Wallis- und Spearman-Tests inkl. BH-Korrektur.
5. `scripts/05_report.py` – Erzeugung dieses Berichts.

Ausführung aus dem Projektordner:

```bash
pip install -r requirements.txt
python scripts/01_prepare_data.py
python scripts/02_eda.py
python scripts/03_tests.py
python scripts/05_report.py
```

Falls die Rohdaten nicht vorhanden sind, laden `01_prepare_data.py` und `external_data.py` die Datensätze automatisch über `kagglehub`. Die Downloads sind auf feste Versionen gepinnt (FronkonGames Version 57; IGDB, Metacritic und Preishistorie jeweils Version 1), damit ein erneuter Lauf dieselben Rohdaten verwendet. Die bereinigten Daten werden als Parquet in `data/` gespeichert; Tabellen liegen in `outputs/tables/`, Abbildungen in `figures/`.

## Anhang: Zentrale Output-Dateien

| Datei | Inhalt |
|---|---|
| `outputs/tables/source_coverage.csv` | Matching-Quote je Datenquelle |
| `outputs/tables/source_overlap.csv` | Überlappung der Datenquellen |
| `outputs/tables/sample_flow.csv` | Stichprobenfluss und Teilstichproben |
| `outputs/tables/data_quality_summary.csv` | Datenqualitätsprüfungen |
| `outputs/tables/success_thresholds.csv` | Erfolgsschwellen und Prävalenz |
| `outputs/tables/descriptives_overall.csv` | Deskriptive Statistik |
| `outputs/tables/descriptives_external.csv` | Deskriptive Statistik der externen Variablen |
| `outputs/tables/genre_overview.csv` | Steam-Genres: N, Erfolgsquote, Preis, F2P |
| `outputs/tables/igdb_genre_overview.csv` | IGDB-Genres: N, Erfolgsquote |
| `outputs/tables/igdb_mode_overview.csv` | IGDB-Spielmodi |
| `outputs/tables/igdb_rating_overview.csv` | IGDB-Bewertungsquartile |
| `outputs/tables/price_history_overview.csv` | Preishistorie: Rabatte und Erfolg |
| `outputs/tables/external_metacritic_overview.csv` | Externe Metacritic-Quartile |
| `outputs/tables/mode_platform_overview.csv` | Modi und Plattformen |
| `outputs/tables/price_overview.csv` | Preisklassen |
| `outputs/tables/discount_overview.csv` | Rabattgruppen |
| `outputs/tables/ratings_overview.csv` | Bewertungsquartile |
| `outputs/tables/h1_tests.csv` | H1-Tests (Genres, Modi, Plattformen) |
| `outputs/tables/h2_tests.csv` | H2-Tests (Preis, Rabatt, DLC) |
| `outputs/tables/h3_tests.csv` | H3-Tests (Steam, externes Metacritic) |
| `outputs/tables/h4_external_tests.csv` | Quellenübergreifende Tests |
| `outputs/tables/external_selection_tests.csv` | Selektionsverzerrung der externen Teilstichproben |
| `outputs/tables/review_volume_diagnostic.csv` | Diagnose Bewertungsvolumen |
| `data/external_features.parquet` | Externe Merkmale je AppID |

*Hinweis: Die im Text genannten Zahlen stammen aus den zum Auswertungszeitpunkt erzeugten Tabellen. Die Kaggle-Rohdaten sind auf feste Versionen gepinnt; bei erneuter Ausführung können sich Werte durch abweichende Bibliotheksversionen geringfügig ändern.*
"""

    text = text.replace("Veroeffentlichung 1997-2025", "Veröffentlichung 1997–2025")
    text = text.replace("ergaenzende Zielgroesse", "ergänzende Zielgröße")
    text = text.replace("Übereinstimmung", "Übereinstimmung")
    text = text.replace("Veroeffentlichungsjahr", "Veröffentlichungsjahr")
    text = text.replace("Geschaetzte", "Geschätzte")
    text = text.replace("geschaetzte", "geschätzte")
    text = text.replace("Zielgroesse", "Zielgröße")
    text = text.replace("Auspraegung", "Ausprägung")
    text = text.replace("Pruefung", "Prüfung")
    text = text.replace("DieMulti-Source", "Die Multi-Source")
    (OUT / "Ergebnisse.md").write_text(text, encoding="utf-8")
    print(f"Bericht geschrieben: {OUT / 'Ergebnisse.md'}")
    print(f"Länge: {len(text):,} Zeichen")


if __name__ == "__main__":
    main()

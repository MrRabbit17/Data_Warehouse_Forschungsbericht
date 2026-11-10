"""Explorative Datenanalyse und deskriptive Tabellen/ABBILDUNGEN."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from statsmodels.stats.proportion import proportion_confint

from common import DATA, FIG, REFERENCE_YEAR, STANDARD_GENRES, ranked_quartiles, save_table
from common import TAB

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
    }
)
COLORS = {"main": "#1f4e79", "accent": "#c55a11", "grey": "#7f7f7f", "green": "#548235"}


def proportion_ci(success: pd.Series) -> tuple[float, float, float, int]:
    n = int(success.notna().sum())
    k = int(success.sum())
    if n == 0:
        return np.nan, np.nan, np.nan, 0
    rate = k / n
    low, high = proportion_confint(k, n, method="wilson")
    return rate, low, high, n


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIG / f"{name}.png")
    plt.close(fig)


def describe_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = []
    for col in columns:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        rows.append(
            {
                "Variable": col,
                "N": len(s),
                "Mittelwert": s.mean(),
                "SD": s.std(),
                "Median": s.median(),
                "Q1": s.quantile(0.25),
                "Q3": s.quantile(0.75),
                "Min": s.min(),
                "Max": s.max(),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    df = pd.read_parquet(DATA / "analytic_sample.parquet")
    rating = pd.read_parquet(DATA / "rating_sample.parquet") if (DATA / "rating_sample.parquet").exists() else df[df["rating_sample"]]
    igdb = pd.read_parquet(DATA / "igdb_sample.parquet") if (DATA / "igdb_sample.parquet").exists() else df[df["igdb_sample"]]
    meta = pd.read_parquet(DATA / "mc_external_sample.parquet") if (DATA / "mc_external_sample.parquet").exists() else df[df["mc_external_sample"]]
    meta_fronk = df[df["meta_sample"]]
    price_hist = pd.read_parquet(DATA / "price_history_sample.parquet") if (DATA / "price_history_sample.parquet").exists() else df[df["price_history_sample"]]
    coverage = pd.read_csv(TAB / "source_coverage.csv")
    overlap = pd.read_csv(TAB / "source_overlap.csv")

    # ------------------------------------------------------------------ Tabellen
    numeric = [
        "owners_mid", "log_owners", "peak_ccu", "log_peak_ccu", "price",
        "discount", "dlc_count", "release_year", "age_years", "n_platforms",
        "n_genres", "review_count", "positive_ratio", "wilson", "metacritic",
    ]
    desc = describe_numeric(df, numeric)
    save_table(desc, "descriptives_overall")

    owners_dist = (
        df.groupby(["owners_class", "owners_label"], as_index=False)
        .agg(N=("AppID", "size"), Anteil=("AppID", lambda s: len(s) / len(df)))
        .sort_values("owners_class")
    )
    save_table(owners_dist, "owners_distribution")

    # Erfolg vs. Nicht-Erfolg: zentrale deskriptive Gegenüberstellung.
    comparison = []
    for col in ["owners_mid", "peak_ccu", "price", "discount", "dlc_count",
                "release_year", "n_platforms", "n_genres", "wilson",
                "positive_ratio", "review_count"]:
        for group, label in [(True, "Erfolgreich (Top 25 %)"), (False, "Nicht erfolgreich")]:
            s = pd.to_numeric(df.loc[df["success_top25"] == group, col], errors="coerce").dropna()
            comparison.append(
                {
                    "Variable": col,
                    "Gruppe": label,
                    "N": len(s),
                    "Median": s.median(),
                    "Mittelwert": s.mean(),
                    "Q1": s.quantile(0.25),
                    "Q3": s.quantile(0.75),
                }
            )
    save_table(pd.DataFrame(comparison), "descriptives_by_success")

    # Genres
    genre_rows = []
    for genre in STANDARD_GENRES:
        col = "genre_" + genre.lower().replace(" ", "_").replace("&", "").replace("__", "_")
        if col not in df.columns:
            continue
        mask = df[col].astype(bool)
        rate, low, high, n = proportion_ci(df.loc[mask, "success_top25"])
        genre_rows.append(
            {
                "Genre": genre,
                "N": n,
                "Anteil_Stichprobe": n / len(df),
                "Erfolgsquote": rate,
                "CI_unten": low,
                "CI_oben": high,
                "Median_Besitzer": df.loc[mask, "owners_mid"].median(),
                "Median_Preis": df.loc[mask, "price"].median(),
                "Anteil_F2P": df.loc[mask, "is_free"].mean(),
            }
        )
    genres = pd.DataFrame(genre_rows).sort_values("Erfolgsquote", ascending=False)
    save_table(genres, "genre_overview")

    # Spielmodi / Plattformen
    mode_rows = []
    for col in ["mode", "n_platforms", "platform_combo"]:
        for value, group in df.groupby(col, dropna=False):
            rate, low, high, n = proportion_ci(group["success_top25"])
            mode_rows.append(
                {
                    "Merkmal": col,
                    "Ausprägung": str(value),
                    "N": n,
                    "Erfolgsquote": rate,
                    "CI_unten": low,
                    "CI_oben": high,
                    "Median_Besitzer": group["owners_mid"].median(),
                }
            )
    modes = pd.DataFrame(mode_rows)
    save_table(modes, "mode_platform_overview")

    # Preis
    price_rows = []
    for value, group in df.groupby("price_class", dropna=False):
        rate, low, high, n = proportion_ci(group["success_top25"])
        price_rows.append(
            {
                "Preisklasse": str(value),
                "N": n,
                "Erfolgsquote": rate,
                "CI_unten": low,
                "CI_oben": high,
                "Median_Besitzer": group["owners_mid"].median(),
                "Mittelwert_Besitzer": group["owners_mid"].mean(),
            }
        )
    save_table(pd.DataFrame(price_rows).sort_values("Preisklasse"), "price_overview")

    paid = df[~df["is_free"]].copy()
    paid["discount_group"] = pd.cut(
        paid["discount"], bins=[-0.1, 0, 25, 50, 75, 100],
        labels=["kein Rabatt", "1-25 %", "26-50 %", "51-75 %", "76-100 %"],
    )
    discount_rows = []
    for value, group in paid.groupby("discount_group", observed=True):
        rate, low, high, n = proportion_ci(group["success_top25"])
        discount_rows.append(
            {
                "Rabattgruppe": str(value),
                "N": n,
                "Erfolgsquote": rate,
                "CI_unten": low,
                "CI_oben": high,
                "Median_Besitzer": group["owners_mid"].median(),
            }
        )
    save_table(pd.DataFrame(discount_rows), "discount_overview")

    # Bewertungen
    rating_rows = []
    for col, label in [("wilson", "Wilson-Score"), ("positive_ratio", "Positive Ratio")]:
        bins = ranked_quartiles(rating[col])
        for value, group in rating.groupby(bins, observed=True):
            rate, low, high, n = proportion_ci(group["success_top25"])
            rating_rows.append(
                {
                    "Bewertungsmaß": label,
                    "Quartil": str(value),
                    "N": n,
                    "Erfolgsquote": rate,
                    "CI_unten": low,
                    "CI_oben": high,
                    "Median_Besitzer": group["owners_mid"].median(),
                }
            )
    meta_q = ranked_quartiles(meta["mc_metascore"])
    for value, group in meta.groupby(meta_q, observed=True):
        rate, low, high, n = proportion_ci(group["success_top25"])
        rating_rows.append(
            {
                "Bewertungsmaß": "Metacritic (extern)",
                "Quartil": str(value),
                "N": n,
                "Erfolgsquote": rate,
                "CI_unten": low,
                "CI_oben": high,
                "Median_Besitzer": group["owners_mid"].median(),
            }
        )
    ratings = pd.DataFrame(rating_rows)
    save_table(ratings, "ratings_overview")

    meta_coverage = pd.DataFrame(
        [
            {"Kennzahl": "Analysestichprobe (N)", "Wert": len(df)},
            {"Kennzahl": "mit externem Metacritic-Score (N)", "Wert": int(df["has_mc_external"].sum())},
            {"Kennzahl": "externe Metacritic-Coverage", "Wert": df["has_mc_external"].mean()},
            {"Kennzahl": "externe Metacritic-Coverage unter erfolgreichen Spielen", "Wert": df.loc[df["success_top25"], "has_mc_external"].mean()},
            {"Kennzahl": "externe Metacritic-Coverage unter nicht erfolgreichen Spielen", "Wert": df.loc[~df["success_top25"], "has_mc_external"].mean()},
            {"Kennzahl": "mit FronkonGames-Metacritic-Score (N)", "Wert": int(df["has_metacritic"].sum())},
            {"Kennzahl": "FronkonGames-Metacritic-Coverage", "Wert": df["has_metacritic"].mean()},
        ]
    )
    save_table(meta_coverage, "metacritic_coverage")

    # Externe Quellen: Deskription und Erfolgsquoten
    external_numeric = [
        "igdb_rating", "igdb_aggregated_rating", "igdb_n_genres",
        "mc_metascore", "mc_user_score", "ph_max_discount_pct",
        "ph_mean_discount_pct", "ph_share_months_on_sale",
        "ph_price_changes_total", "ph_days_on_sale_total",
        "ph_base_price", "ph_lowest_price",
    ]
    save_table(describe_numeric(df, external_numeric), "descriptives_external")

    igdb_genre_rows = []
    for col in [c for c in df.columns if c.startswith("igdb_genre_")]:
        mask = df[col].astype(bool)
        rate, low, high, n = proportion_ci(df.loc[mask, "success_top25"])
        label = col.removeprefix("igdb_genre_").replace("_", " ").title()
        igdb_genre_rows.append(
            {
                "IGDB_Genre": label,
                "N": n,
                "Anteil_Stichprobe": n / len(igdb) if len(igdb) else np.nan,
                "Erfolgsquote": rate,
                "CI_unten": low,
                "CI_oben": high,
                "Median_Besitzer": df.loc[mask, "owners_mid"].median(),
                "Median_Preis": df.loc[mask, "price"].median(),
            }
        )
    save_table(pd.DataFrame(igdb_genre_rows).sort_values("Erfolgsquote", ascending=False), "igdb_genre_overview")

    igdb_mode_rows = []
    for col, label in [
        ("igdb_singleplayer", "Singleplayer"),
        ("igdb_multiplayer", "Multiplayer"),
        ("igdb_coop", "Kooperation"),
        ("igdb_mmo", "MMO"),
    ]:
        mask = df[col].astype(bool)
        rate, low, high, n = proportion_ci(df.loc[mask, "success_top25"])
        if n == 0:
            continue
        igdb_mode_rows.append(
            {
                "IGDB_Merkmal": label,
                "N": n,
                "Erfolgsquote": rate,
                "CI_unten": low,
                "CI_oben": high,
                "Median_Besitzer": df.loc[mask, "owners_mid"].median(),
            }
        )
    save_table(pd.DataFrame(igdb_mode_rows), "igdb_mode_overview")

    igdb_rating_rows = []
    for var, label in [
        ("igdb_rating", "IGDB-Nutzerbewertung"),
        ("igdb_aggregated_rating", "IGDB-Kritikerbewertung"),
    ]:
        d = igdb.dropna(subset=[var])
        if d.empty:
            continue
        quartile = ranked_quartiles(d[var])
        for value, group in d.groupby(quartile, observed=True):
            rate, low, high, n = proportion_ci(group["success_top25"])
            igdb_rating_rows.append(
                {
                    "Bewertungsmaß": label,
                    "Quartil": str(value),
                    "N": n,
                    "Erfolgsquote": rate,
                    "CI_unten": low,
                    "CI_oben": high,
                    "Median_Besitzer": group["owners_mid"].median(),
                }
            )
    save_table(pd.DataFrame(igdb_rating_rows), "igdb_rating_overview")

    price_hist_rows = []
    for var, label in [
        ("ph_max_discount_pct", "Maximaler Rabatt"),
        ("ph_share_months_on_sale", "Anteil Monate mit Rabatt"),
        ("ph_price_changes_total", "Preisänderungen"),
    ]:
        d = price_hist.dropna(subset=[var])
        if d.empty:
            continue
        quartile = ranked_quartiles(d[var])
        for value, group in d.groupby(quartile, observed=True):
            rate, low, high, n = proportion_ci(group["success_top25"])
            price_hist_rows.append(
                {
                    "Preismerkmal": label,
                    "Quartil": str(value),
                    "N": n,
                    "Erfolgsquote": rate,
                    "CI_unten": low,
                    "CI_oben": high,
                    "Median_Besitzer": group["owners_mid"].median(),
                }
            )
    save_table(pd.DataFrame(price_hist_rows), "price_history_overview")

    mc_rows = []
    for var, label in [
        ("mc_metascore", "Metacritic-Kritikerscore"),
        ("mc_user_score", "Metacritic-Nutzerscore"),
    ]:
        d = meta.dropna(subset=[var])
        if d.empty:
            continue
        quartile = ranked_quartiles(d[var])
        for value, group in d.groupby(quartile, observed=True):
            rate, low, high, n = proportion_ci(group["success_top25"])
            mc_rows.append(
                {
                    "Bewertungsmaß": label,
                    "Quartil": str(value),
                    "N": n,
                    "Erfolgsquote": rate,
                    "CI_unten": low,
                    "CI_oben": high,
                    "Median_Besitzer": group["owners_mid"].median(),
                }
            )
    save_table(pd.DataFrame(mc_rows), "external_metacritic_overview")

    # Kohorten nach Veröffentlichungsjahr
    cohort = (
        df.groupby("release_year")
        .agg(
            N=("AppID", "size"),
            Erfolgsquote=("success_top25", "mean"),
            Median_Besitzer=("owners_mid", "median"),
            Anteil_F2P=("is_free", "mean"),
            Median_Preis=("price", "median"),
            Anteil_Multiplayer=("multi_player", "mean"),
        )
        .reset_index()
    )
    cohort = cohort[cohort["N"] >= 100]
    save_table(cohort, "cohort_by_year")

    # Korrelationsmatrix (paarweise, Spearman)
    corr_vars = [
        "log_owners", "log_peak_ccu", "release_year", "n_platforms", "n_genres",
        "price", "discount", "dlc_count", "wilson", "positive_ratio",
        "review_count", "metacritic", "mc_metascore", "igdb_rating",
        "igdb_aggregated_rating", "ph_max_discount_pct", "ph_share_months_on_sale",
    ]
    corr = df[corr_vars].corr(method="spearman")
    save_table(corr.reset_index().rename(columns={"index": "Variable"}), "correlation_spearman")

    # ------------------------------------------------------------------ Grafiken
    # 1) Verteilung der Besitzer
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].barh(owners_dist["owners_label"], owners_dist["N"], color=COLORS["main"])
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Anzahl Spiele (log-Skala)")
    axes[0].set_ylabel("")
    axes[0].set_title("Verteilung der geschätzten Besitzer")
    for y, n in enumerate(owners_dist["N"]):
        axes[0].text(n * 1.1, y, f"{n:,}".replace(",", "."), va="center", fontsize=7)
    axes[1].hist(df["log_owners"], bins=40, color=COLORS["main"], edgecolor="white")
    axes[1].set_xlabel("log10(Besitzer-Mittelpunkt)")
    axes[1].set_ylabel("Anzahl Spiele")
    axes[1].set_title("Log-transformierte Besitzerverteilung")
    save(fig, "fig_01_owners_distribution")

    # 2) Veröffentlichungskohorten
    fig, ax1 = plt.subplots(figsize=(9, 4.2))
    ax2 = ax1.twinx()
    ax1.bar(cohort["release_year"], cohort["N"], color="#d9e2f3", label="Spiele pro Jahr")
    ax1.set_ylabel("Anzahl Spiele")
    ax1.set_xlabel("Veröffentlichungsjahr")
    ax2.plot(cohort["release_year"], cohort["Erfolgsquote"] * 100, color=COLORS["accent"], marker="o", ms=3, label="Erfolgsquote")
    ax2.set_ylabel("Erfolgsquote (Top 25 %) in %")
    ax1.set_title("Spiele und Erfolgsquote nach Veröffentlichungsjahr")
    lines = [ax1.patches[0], ax2.lines[0]]
    ax1.legend(lines, ["Spiele pro Jahr", "Erfolgsquote"], loc="upper left")
    save(fig, "fig_02_release_cohorts")

    # 3) Genres
    fig, ax = plt.subplots(figsize=(8, 4.8))
    g = genres.sort_values("Erfolgsquote")
    ax.barh(g["Genre"], g["Erfolgsquote"] * 100, color=COLORS["main"])
    ax.errorbar(
        g["Erfolgsquote"] * 100, np.arange(len(g)),
        xerr=[(g["Erfolgsquote"] - g["CI_unten"]) * 100, (g["CI_oben"] - g["Erfolgsquote"]) * 100],
        fmt="none", ecolor="black", elinewidth=1, capsize=3,
    )
    overall = df["success_top25"].mean() * 100
    ax.axvline(overall, color=COLORS["accent"], ls="--", lw=1, label=f"Gesamt: {overall:.1f} %")
    ax.set_xlabel("Anteil erfolgreicher Spiele (Top 25 %) in %")
    ax.set_ylabel("")
    ax.set_title("Erfolgsquote nach Genre (95-%-Konfidenzintervalle)")
    ax.legend(loc="lower right")
    save(fig, "fig_03_genre_success")

    # 4) Modi und Plattformen
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    mode_data = modes[modes["Merkmal"] == "mode"].sort_values("Erfolgsquote")
    axes[0].barh(mode_data["Ausprägung"], mode_data["Erfolgsquote"] * 100, color=COLORS["main"])
    axes[0].set_xlabel("Erfolgsquote (Top 25 %) in %")
    axes[0].set_title("Spielmodi")
    plat_data = modes[modes["Merkmal"] == "n_platforms"].sort_values("Ausprägung")
    axes[1].bar(plat_data["Ausprägung"], plat_data["Erfolgsquote"] * 100, color=COLORS["green"])
    axes[1].set_xlabel("Anzahl unterstuetzter Plattformen")
    axes[1].set_ylabel("Erfolgsquote (Top 25 %) in %")
    axes[1].set_title("Plattformabdeckung")
    save(fig, "fig_04_modes_platforms")

    # 5) Preis
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    order = [c for c in ["Free", "0,01 - 4,99", "5 - 9,99", "10 - 19,99", "20 - 39,99", "40+"] if c in set(df["price_class"])]
    sns.boxplot(data=df, x="price_class", y="owners_mid", order=order, ax=axes[0], color="#bdd7ee", fliersize=0)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Preisklasse (USD)")
    axes[0].set_ylabel("Geschätzte Besitzer (log-Skala)")
    axes[0].set_title("Besitzer nach Preisklasse")
    axes[0].tick_params(axis="x", rotation=30)
    sns.boxplot(data=df, x="is_free", y="owners_mid", ax=axes[1], color="#f4b183", fliersize=0)
    axes[1].set_yscale("log")
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(["Kostenpflichtig", "Free-to-Play"])
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Geschätzte Besitzer (log-Skala)")
    axes[1].set_title("Free-to-Play vs. kostenpflichtig")
    save(fig, "fig_05_price")

    # 6) Rabatte
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    disc = pd.DataFrame(discount_rows)
    axes[0].bar(disc["Rabattgruppe"], disc["Erfolgsquote"] * 100, color=COLORS["accent"])
    axes[0].set_ylabel("Erfolgsquote (Top 25 %) in %")
    axes[0].set_xlabel("Aktueller Rabatt")
    axes[0].set_title("Erfolgsquote nach Rabattgruppe (nur kostenpflichtige Spiele)")
    axes[0].tick_params(axis="x", rotation=25)
    sns.boxplot(data=paid, x="has_discount", y="owners_mid", ax=axes[1], color="#f4b183", fliersize=0)
    axes[1].set_yscale("log")
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(["Kein Rabatt", "Rabatt aktiv"])
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Geschätzte Besitzer (log-Skala)")
    axes[1].set_title("Besitzer nach Rabattstatus")
    save(fig, "fig_06_discount")

    # 7) Bewertungen
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    hb = axes[0].hexbin(rating["wilson"], rating["log_owners"], gridsize=45, cmap="Blues", mincnt=1)
    axes[0].set_xlabel("Wilson-Score (Steam-Nutzerbewertungen)")
    axes[0].set_ylabel("log10(Besitzer-Mittelpunkt)")
    axes[0].set_title("Steam-Bewertung und Besitzer")
    fig.colorbar(hb, ax=axes[0], label="Anzahl Spiele")
    axes[1].scatter(meta["mc_metascore"], meta["log_owners"], s=8, alpha=0.25, color=COLORS["accent"])
    axes[1].set_xlabel("Metacritic-Score (extern)")
    axes[1].set_ylabel("log10(Besitzer-Mittelpunkt)")
    axes[1].set_title(f"Externes Metacritic und Besitzer (n = {len(meta):,})".replace(",", "."))
    save(fig, "fig_07_ratings")

    # 8) Korrelationsheatmap
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                vmin=-1, vmax=1, square=True, linewidths=0.4, ax=ax,
                cbar_kws={"shrink": 0.75})
    ax.set_title("Spearman-Korrelationen (paarweise Verfuegbarkeit)")
    save(fig, "fig_08_correlation_heatmap")

    # 9) Genre x Kohorte
    top_genres = genres.head(8)["Genre"].tolist()
    cohort_bins = pd.cut(df["release_year"], bins=[1996, 2009, 2014, 2019, 2022, 2025],
                         labels=["bis 2009", "2010-2014", "2015-2019", "2020-2022", "2023-2025"])
    heat_rows = []
    for genre in top_genres:
        col = "genre_" + genre.lower().replace(" ", "_").replace("&", "").replace("__", "_")
        for cohort_label, group in df.groupby(cohort_bins, observed=True):
            heat_rows.append(
                {
                    "Genre": genre,
                    "Kohorte": str(cohort_label),
                    "Erfolgsquote": group[col].astype(bool).pipe(lambda m: group.loc[m, "success_top25"].mean()),
                    "N": int(group[col].sum()),
                }
            )
    heat = pd.DataFrame(heat_rows)
    pivot = heat.pivot(index="Genre", columns="Kohorte", values="Erfolgsquote")
    pivot = pivot[[c for c in ["bis 2009", "2010-2014", "2015-2019", "2020-2022", "2023-2025"] if c in pivot.columns]]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    sns.heatmap(pivot * 100, annot=True, fmt=".1f", cmap="YlGnBu", ax=ax,
                cbar_kws={"label": "Erfolgsquote in %"})
    ax.set_title("Erfolgsquote nach Genre und Veröffentlichungskohorte")
    ax.set_xlabel("Veröffentlichungskohorte")
    ax.set_ylabel("")
    save(fig, "fig_09_genre_cohort")

    # 10) Übereinstimmung Steam vs. Metacritic
    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    ax.scatter(meta["mc_metascore"], meta["wilson"], s=8, alpha=0.25, color=COLORS["main"])
    ax.set_xlabel("Metacritic-Score (extern)")
    ax.set_ylabel("Wilson-Score (Steam)")
    ax.set_title(f"Kritiker- vs. Nutzerbewertung (n = {len(meta):,})".replace(",", "."))
    save(fig, "fig_10_rating_agreement")

    # 11) IGDB-Bewertungen und Besitzer
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, var, label in [
        (axes[0], "igdb_rating", "IGDB-Nutzerbewertung"),
        (axes[1], "igdb_aggregated_rating", "IGDB-Kritikerbewertung"),
    ]:
        d = igdb.dropna(subset=[var, "log_owners"])
        hb = ax.hexbin(d[var], d["log_owners"], gridsize=42, cmap="Blues", mincnt=1)
        ax.set_xlabel(label)
        ax.set_ylabel("log10(Besitzer-Mittelpunkt)")
        ax.set_title(f"{label} und Besitzer (n = {len(d):,})".replace(",", "."))
        fig.colorbar(hb, ax=ax, label="Anzahl Spiele")
    save(fig, "fig_11_igdb_ratings")

    # 12) Preishistorie und Erfolg
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    ph_plot = pd.DataFrame(price_hist_rows)
    ph_plot = ph_plot[(ph_plot["Preismerkmal"] == "Maximaler Rabatt") & ph_plot["Erfolgsquote"].notna()]
    axes[0].bar(ph_plot["Quartil"], ph_plot["Erfolgsquote"] * 100, color=COLORS["accent"])
    axes[0].set_xlabel("Maximaler Rabatt in der Preishistorie")
    axes[0].set_ylabel("Erfolgsquote (Top 25 %) in %")
    axes[0].set_title("Erfolgsquote nach maximalem Rabatt")
    axes[0].tick_params(axis="x", rotation=25)
    d = price_hist.dropna(subset=["ph_share_months_on_sale", "log_owners"])
    axes[1].scatter(d["ph_share_months_on_sale"], d["log_owners"], s=14, alpha=0.45, color=COLORS["main"])
    axes[1].set_xlabel("Anteil Monate mit Rabatt")
    axes[1].set_ylabel("log10(Besitzer-Mittelpunkt)")
    axes[1].set_title(f"Rabatthäufigkeit und Besitzer (n = {len(d):,})".replace(",", "."))
    save(fig, "fig_12_price_history")

    # 13) Externe Metacritic-Daten
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    axes[0].scatter(meta["mc_metascore"], meta["log_owners"], s=10, alpha=0.3, color=COLORS["accent"])
    axes[0].set_xlabel("Metacritic-Kritikerscore (extern)")
    axes[0].set_ylabel("log10(Besitzer-Mittelpunkt)")
    axes[0].set_title(f"Externes Metacritic und Besitzer (n = {len(meta):,})".replace(",", "."))
    axes[1].scatter(meta["mc_metascore"], meta["wilson"], s=10, alpha=0.3, color=COLORS["main"])
    axes[1].set_xlabel("Metacritic-Kritikerscore (extern)")
    axes[1].set_ylabel("Wilson-Score (Steam)")
    axes[1].set_title(f"Externes Metacritic und Steam-Bewertung (n = {len(meta):,})".replace(",", "."))
    save(fig, "fig_13_external_metacritic")

    # 14) Datenquellen-Overlap
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.bar(overlap["Anzahl_Datenquellen"].astype(str), overlap["N"], color=COLORS["main"])
    for x, n in zip(overlap["Anzahl_Datenquellen"].astype(str), overlap["N"]):
        ax.text(x, n, f"{n:,}".replace(",", "."), ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("Anzahl verfügbarer Datenquellen")
    ax.set_ylabel("Anzahl Spiele")
    ax.set_title("Abdeckung der Analysestichprobe nach Datenquellen")
    save(fig, "fig_14_source_overlap")

    print("EDA abgeschlossen.")
    print(f"Analysestichprobe: {len(df):,}")
    print(genres[["Genre", "N", "Erfolgsquote"]].to_string(index=False))
    print(coverage[["Quelle", "N_gematcht_analytisch", "Match_Rate_analytisch"]].to_string(index=False))


if __name__ == "__main__":
    main()

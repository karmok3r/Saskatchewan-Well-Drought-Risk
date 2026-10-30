"""Result figures (2-6) for the manuscript. Run after the analysis scripts.

Fig. 2 study area: domestic wells by available drawdown, observation wells, reserves, rivers, cities
Fig. 3 exposure: cumulative AD by construction and share exposed by depth / decade
Fig. 4 validation: simulated vs observed drought declines at observation wells
Fig. 5 share of wells dry through time, slow (fitted) vs fast response
Fig. 6 exposure and failure on / near reserves; adjusted odds ratios
"""
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

plt.rcParams.update({"font.family": "Arial", "font.size": 8, "axes.linewidth": 0.6, "pdf.fonttype": 42,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5,
                     "ytick.major.size": 2.5})
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e6e3"
BLUE, ORANGE, GREEN, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
SEQ = ["#d7e6f8", "#86b4ea", "#2a78d6", "#123f78"]  # light to dark blue
HERE = Path(__file__).resolve().parent
GW = HERE.parent / "Saskatoon-Groundwater-Climate-Drivers"
RES, FIG = HERE / "results", HERE / "figures"


def clean(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=INK)


def label(ax, s, x=-0.12):
    ax.text(x, 1.02, s, transform=ax.transAxes, weight="bold", fontsize=10, va="bottom")


def fig2(w):
    base = GW / "data/basemap"
    big = (-125, 40, -88, 61)
    box = (-110.4, 48.75, -101.0, 56.1)
    prov = gpd.read_file(base / "ne_10m_admin_1_states_provinces.zip", bbox=big)
    sask = prov[prov.name == "Saskatchewan"]
    na = gpd.read_file(base / "ne_10m_rivers_north_america.zip", bbox=box)
    rivers = pd.concat([gpd.read_file(base / "ne_10m_rivers_lake_centerlines.zip", bbox=box),
                        na[na.name.isin(["Souris", "Qu'Appelle", "Assiniboine", "Battle", "Red Deer", "Frenchman",
                                         "Carrot", "Saskatchewan"])]])
    lakes = gpd.read_file(base / "ne_10m_lakes.zip", bbox=box)
    res = gpd.read_file(next((HERE / "data/raw/reserves").glob("*.shp"))).to_crs(4326)
    obs = pd.read_csv(GW / "wells_province.csv").set_index("name").loc[pd.read_csv(RES / "obs_well_response.csv").name]

    fig = plt.figure(figsize=(7.2, 6.3))
    ax = fig.add_axes([0.07, 0.07, 0.62, 0.90])
    prov.plot(ax=ax, color="#f4f3ef", edgecolor="#a8a7a2", lw=0.5)
    sask.plot(ax=ax, color="white", edgecolor=INK, lw=0.8)
    res.plot(ax=ax, color="#c9c8c3", lw=0)
    lakes.plot(ax=ax, color="#cfe3f5", edgecolor="#8db8de", lw=0.3)
    rivers.plot(ax=ax, color="#6fa3d6", lw=0.6)
    x = w.dropna(subset=["ad_intake_m"]).sort_values("ad_intake_m", ascending=False)
    bins = [(10, 1e9, "> 10 m", SEQ[0]), (5, 10, "5-10 m", SEQ[1]), (2, 5, "2-5 m", SEQ[2]),
            (-1e9, 2, "<= 2 m", SEQ[3])]
    for lo, hi, lab, c in bins:
        s = x[(x.ad_intake_m > lo) & (x.ad_intake_m <= hi)]
        ax.scatter(s.lon, s.lat, s=0.5, color=c, lw=0, rasterized=True, zorder=3)
    ax.scatter(obs.lon, obs.lat, s=24, marker="^", color=ORANGE, edgecolor="white", lw=0.6, zorder=5)
    cities = {"Saskatoon": (52.13, -106.67), "Regina": (50.45, -104.61), "Prince Albert": (53.20, -105.75),
              "Moose Jaw": (50.39, -105.53), "Swift Current": (50.29, -107.80), "Yorkton": (51.21, -102.46),
              "North Battleford": (52.76, -108.29), "Estevan": (49.14, -102.99), "Lloydminster": (53.28, -110.0)}
    for city, (lat, lon) in cities.items():
        ax.plot(lon, lat, marker="s", ms=3, color=INK, zorder=6)
        left = city in ("Moose Jaw", "North Battleford")
        ax.annotate(city, (lon, lat), xytext=(-3, 3) if left else (3, 3), ha="right" if left else "left",
                    textcoords="offset points", fontsize=6.5, zorder=6,
                    bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.75))
    for name, (lon, lat, rot) in {"ALBERTA": (-110.15, 51.6, 90), "MANITOBA": (-101.25, 52.6, 90),
                                  "MONTANA (USA)": (-108.6, 48.85, 0), "NORTH DAKOTA (USA)": (-102.6, 48.85, 0)}.items():
        ax.text(lon, lat, name, fontsize=6.5, ha="center", va="center", color=MUTED, style="italic", rotation=rot)
    # scale bar (100 km at 52 N) and north arrow
    km_per_deg = 111.32 * np.cos(np.radians(52))
    x0, y0 = -109.9, 49.15
    ax.plot([x0, x0 + 100 / km_per_deg], [y0, y0], color=INK, lw=2, solid_capstyle="butt", zorder=7)
    ax.text(x0 + 50 / km_per_deg, y0 + 0.12, "100 km", ha="center", fontsize=6.5, zorder=7)
    ax.annotate("N", xy=(-101.55, 55.85), xytext=(-101.55, 55.25), ha="center", fontsize=8, weight="bold",
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1))
    ax.set_xlim(box[0], box[2])
    ax.set_ylim(box[1], box[3])
    ax.set_aspect(1 / np.cos(np.radians(52)))
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.grid(color="#e3e2de", lw=0.4, zorder=0)
    ax.tick_params(colors=INK)

    handles = [Line2D([], [], marker="o", ls="", color=c, ms=4) for *_, c in bins]
    labels = [f"{lab} (n = {len(x[(x.ad_intake_m > lo) & (x.ad_intake_m <= hi)]):,})" for lo, hi, lab, _ in bins]
    handles += [Line2D([], [], marker="^", ls="", color=ORANGE, mec="white", ms=6),
                Line2D([], [], marker="s", ls="", color="#c9c8c3", ms=6),
                Line2D([], [], color="#6fa3d6", lw=1), Line2D([], [], marker="s", ls="", color=INK, ms=3)]
    labels += ["Observation well (n = 39)", "First Nations reserve", "River", "City"]
    fig.legend(handles, labels, title="Domestic wells: water\nabove pump intake", loc="lower left",
               bbox_to_anchor=(0.705, 0.08), fontsize=6.5, title_fontsize=7, frameon=True, edgecolor="#cfcfcb",
               alignment="left", handletextpad=0.4)

    ins = fig.add_axes([0.715, 0.60, 0.27, 0.36])
    prov.plot(ax=ins, color="#f4f3ef", edgecolor="#b8b7b2", lw=0.3)
    countries = gpd.read_file(base / "ne_10m_admin_0_countries.zip", bbox=big)
    countries.boundary.plot(ax=ins, color=INK, lw=0.5)
    sask.plot(ax=ins, color="#d7e6f8", edgecolor=INK, lw=0.6)
    ins.text(-112, 58.2, "CANADA", fontsize=6.5, weight="bold")
    ins.text(-112, 44.5, "USA", fontsize=6.5, weight="bold")
    ins.set_xlim(big[0], big[2])
    ins.set_ylim(big[1], big[3])
    ins.set_aspect(1 / np.cos(np.radians(50)))
    ins.set_xticks([])
    ins.set_yticks([])
    ins.set_xlabel("")
    ins.set_ylabel("")
    for s in ins.spines.values():
        s.set_linewidth(0.6)
    fig.savefig(FIG / "fig2_map.png", dpi=600)
    fig.savefig(FIG / "fig2_map.pdf", dpi=600)


def fig3(w):
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.4), gridspec_kw={"width_ratios": [1.3, 1, 1]})
    ax = axs[0]
    for m, c in [("Bored", ORANGE), ("Drilled", BLUE)]:
        x = np.sort(w.loc[w.cinstmethd == m, "ad_intake_m"].dropna())
        ax.plot(x, np.arange(1, len(x) + 1) / len(x), color=c, lw=1.6, label=f"{m} (n = {len(x):,})")
    for t in [2, 5, 10]:
        ax.axvline(t, color=GRID, lw=0.6, zorder=0)
    ax.set_xlim(0, 40)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Water above pump intake (m)")
    ax.set_ylabel("Cumulative share of wells")
    ax.legend(frameon=False, loc="lower right")
    label(ax, "a", -0.2)
    ex = pd.read_csv(RES / "exposure_summary.csv", index_col=0)
    for ax, key, lab in [(axs[1], "depth_class", "Well depth"), (axs[2], "decade", "Decade drilled")]:
        e = ex[ex.index.str.startswith(key) & (ex.n >= 500)]
        names = [i.split(": ")[1] for i in e.index]
        names = [n + "s" if key == "decade" else n for n in names]
        ax.bar(range(len(e)), e.le_5m * 100, color=BLUE, width=0.7)
        ax.bar(range(len(e)), e.le_2m * 100, color=SEQ[3], width=0.7)
        ax.set_xticks(range(len(e)), names, rotation=45)
        ax.set_xlabel(lab)
        ax.set_ylabel("Wells exposed (%)")
        clean(ax)
    axs[1].legend([Line2D([], [], marker="s", ls="", color=c, ms=6) for c in [BLUE, SEQ[3]]],
                  ["<= 5 m above intake", "<= 2 m"], frameon=False, loc="upper right")
    label(axs[1], "b", -0.25)
    label(axs[2], "c", -0.25)
    clean(axs[0])
    fig.tight_layout(w_pad=1.2)
    fig.savefig(FIG / "fig3_exposure.png", dpi=600)
    fig.savefig(FIG / "fig3_exposure.pdf")


def fig4():
    v = pd.read_csv(RES / "decline_validation.csv")
    fig, ax = plt.subplots(figsize=(3.3, 3.1))
    for (d, c, mk) in [("1984-88", YELLOW, "o"), ("2001-03", BLUE, "s"), ("2021-23", ORANGE, "^")]:
        shown = {"1984-88": "1983-88", "2001-03": "2000-03", "2021-23": "2020-23"}[d]  # water years, as in the text
        s = v[v.drought == d]
        ax.scatter(s.obs_decline_m, s.sim_decline_m, s=16, color=c, marker=mk, edgecolor="white", lw=0.4,
                   label=f"{shown} (n = {len(s)})")
    lim = [-0.3, 4]
    ax.plot(lim, lim, color=MUTED, lw=0.6, ls="--")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal")
    ax.set_xlabel("Observed drought decline (m)")
    ax.set_ylabel("Simulated drought decline (m)")
    r = np.corrcoef(v.obs_decline_m, v.sim_decline_m)[0, 1]
    ratio = (v.sim_decline_m / v.obs_decline_m).median()
    ax.text(0.97, 0.30, f"r = {r:.2f}\nmedian ratio = {ratio:.2f}", transform=ax.transAxes, va="top", ha="right",
            fontsize=7)
    ax.legend(frameon=False, loc="upper left", fontsize=6.5)
    clean(ax)
    fig.savefig(FIG / "fig4_validation.png", dpi=600, bbox_inches="tight")
    fig.savefig(FIG / "fig4_validation.pdf", bbox_inches="tight")


def fig5():
    ts = pd.read_csv(RES / "failure_timeseries.csv")
    ts = ts[ts.basis == "intake"]
    scen = [("2001-03", "a  2000-03 drought (3 yr)"), ("2001-03 x10yr", "b  2000-03 repeated for 10 yr"),
            ("2021-23", "c  2020-23 drought (3 yr)")]
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.4), sharey=False)
    for ax, (s, title) in zip(axs, scen):
        g = ts[ts.scenario == s]
        dur = g.drought_days.iloc[0] / 365.25
        ax.axvspan(0, dur, color=GRID, lw=0)
        for m, c, lab in [("fast", ORANGE, "Fast response, same peak"), ("memory", BLUE, "Fitted Prairie response")]:
            x = g[g.model == m]
            ax.plot(x.day / 365.25, x.frac_dry * 100, color=c, lw=1.4, label=lab)
        ax.set_xlim(-1, 32)
        ax.set_ylim(0, None)
        ax.set_xlabel("Years since drought began")
        ax.set_title(title, loc="left", fontsize=8, weight="bold")
        clean(ax)
    axs[0].set_ylabel("Domestic wells dry (%)")
    axs[0].legend(frameon=False, loc="upper right", fontsize=6.5)
    fig.tight_layout(w_pad=1.0)
    fig.savefig(FIG / "fig5_timing.png", dpi=600)
    fig.savefig(FIG / "fig5_timing.pdf")


def fig6():
    z = pd.read_csv(RES / "rq3_reserves.csv", index_col=0).loc[["elsewhere", "within 5 km", "on reserve"]]
    o = pd.read_csv(RES / "rq3_logit.csv")
    fig, axs = plt.subplots(1, 2, figsize=(6.2, 2.4), gridspec_kw={"width_ratios": [1, 1.2]})
    ax = axs[0]
    x = np.arange(3)
    ax.bar(x - 0.18, z.le_5m * 100, width=0.34, color=BLUE, label="Exposed (<= 5 m above intake)")
    ax.bar(x + 0.18, z.dry_share * 100, width=0.34, color=ORANGE, label="Dry in 10-yr drought")
    names = ["Elsewhere", "Within 5 km\nof reserve", "On reserve"]
    ax.set_xticks(x, [f"{a}\n(n = {int(n):,})" for a, n in zip(names, z.n)], fontsize=7)
    ax.set_ylabel("Wells (%)")
    ax.set_ylim(0, 30)
    ax.legend(frameon=False, loc="upper left", fontsize=6.5)
    clean(ax)
    label(ax, "a", -0.2)
    ax = axs[1]
    models = ["zone only", "+ method", "+ method, depth, decade"]
    for j, (term, c, lab) in enumerate([("b_near", YELLOW, "Within 5 km"), ("c_on", ORANGE, "On reserve")]):
        s = o[o.term == term].set_index("model").loc[models]
        y = np.arange(3) + (0.15 if j else -0.15)
        ax.errorbar(s.odds_ratio, y, xerr=[s.odds_ratio - s.ci_lo, s.ci_hi - s.odds_ratio], fmt="o", color=c,
                    ms=4, lw=1, capsize=2, label=lab)
    ax.axvline(1, color=MUTED, lw=0.6, ls="--")
    ax.set_yticks(range(3), ["Location only", "+ construction", "+ construction,\ndepth, decade"])
    ax.invert_yaxis()
    ax.set_xlabel("Odds ratio of exposure (95% CI)")
    ax.legend(frameon=False, loc="lower right", fontsize=6.5)
    clean(ax)
    label(ax, "b", -0.35)
    fig.tight_layout(w_pad=1.5)
    fig.savefig(FIG / "fig6_reserves.png", dpi=600)
    fig.savefig(FIG / "fig6_reserves.pdf")


def fig7():
    o = pd.read_csv(RES / "drilling_panel.csv")
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.6), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axs[0]
    s = o[o.model == "weather, lags 0-10"].copy()
    s["lag"] = s.term.str.extract(r"met_l(\d+)")[0].fillna(0).astype(int)
    s = s.sort_values("lag")
    ax.axhline(1, color=MUTED, lw=0.6, ls="--")
    ax.errorbar(s.lag, s.rate_ratio_per_sd, yerr=[s.rate_ratio_per_sd - s.ci_lo, s.ci_hi - s.rate_ratio_per_sd],
                fmt="o-", color=BLUE, ms=3.5, lw=1.2, capsize=2)
    ax.set_xticks(range(0, 11))
    ax.set_xlabel("Years since dry weather")
    ax.set_ylabel("Replacement wells, rate ratio\nper SD drier year (95% CI)")
    clean(ax)
    label(ax, "a", -0.22)
    ax = axs[1]
    rows = [("groundwater (simulated)", "gw_sim", "Simulated level"),
            ("groundwater (observed)", "gw_obs", "Observed level"),
            ("weather lags 0-3 + groundwater (simulated)", "gw_sim", "Simulated, + weather 0-3 yr"),
            ("weather lags 0-3 + groundwater (observed)", "gw_obs", "Observed, + weather 0-3 yr"),
            ("weather lags 0-10 + groundwater (observed)", "gw_obs", "Observed, + weather 0-10 yr"),
            ("placebo: + groundwater 3 yr later", "gw_lead3", "Placebo: level 3 yr later"),
            ("placebo: + groundwater 5 yr later", "gw_lead5", "Placebo: level 5 yr later")]
    for i, (m, term, lab) in enumerate(rows):
        r = o[(o.model == m) & (o.term == term)].iloc[0]
        c = MUTED if term.startswith("gw_lead") else ORANGE
        ax.errorbar(r.rate_ratio_per_sd, i, xerr=[[r.rate_ratio_per_sd - r.ci_lo], [r.ci_hi - r.rate_ratio_per_sd]],
                    fmt="o", color=c, ms=4, lw=1.1, capsize=2)
    ax.axvline(1, color=MUTED, lw=0.6, ls="--")
    ax.set_yticks(range(len(rows)), [r[2] for r in rows], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Rate ratio per SD lower groundwater")
    clean(ax)
    label(ax, "b", -0.62)
    fig.tight_layout(w_pad=1.5)
    fig.savefig(FIG / "fig7_drilling.png", dpi=600)
    fig.savefig(FIG / "fig7_drilling.pdf")


def main():
    w = pd.read_csv(RES / "domestic_wells.csv", low_memory=False)
    fig2(w)
    fig3(w)
    fig4()
    fig5()
    fig6()
    fig7()


if __name__ == "__main__":
    main()

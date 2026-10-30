"""Graphical abstract (Elsevier: landscape, >= 1328 x 531 px, readable at 13 x 5 cm).

Three panels from the results: exposure of domestic wells, failure timing with
the fitted (slow) and fast responses, and observed replacement drilling by lag.
Writes figures/graphical_abstract.png and .pdf.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({"font.family": "Arial", "font.size": 7, "axes.linewidth": 0.6, "pdf.fonttype": 42})
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e6e3"
BLUE, ORANGE = "#2a78d6", "#eb6834"
HERE = Path(__file__).resolve().parent
RES, FIG = HERE / "results", HERE / "figures"


def main():
    fig = plt.figure(figsize=(13 / 2.54 * 1.6, 5 / 2.54 * 1.6))  # 13 x 5 cm, scaled 1.6x for legibility
    fig.text(0.5, 0.955, "Drought reaches Prairie wells late, and they stay dry long after the rain returns",
             ha="center", va="top", fontsize=10, weight="bold", color=INK)
    axs = [fig.add_axes([0.095, 0.17, 0.22, 0.62]), fig.add_axes([0.41, 0.17, 0.25, 0.62]),
           fig.add_axes([0.75, 0.17, 0.225, 0.62])]

    # 1 exposure
    ex = pd.read_csv(RES / "exposure_summary.csv", index_col=0)
    groups = [("All wells", "all"), ("Drilled", "method: Drilled"), ("Bored", "method: Bored"),
              ("< 15 m deep", "depth_class: <15 m")]
    vals = [ex.loc[k, "le_5m"] * 100 for _, k in groups]
    ax = axs[0]
    ax.barh(range(len(groups)), vals, color=[BLUE, BLUE, ORANGE, ORANGE], height=0.6)
    for i, v in enumerate(vals):
        ax.text(v + 1, i, f"{v:.0f}%", va="center", fontsize=8, weight="bold")
    ax.set_yticks(range(len(groups)), [g for g, _ in groups], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 48)
    ax.set_xlabel("Wells with <= 5 m of water above the pump")
    ax.set_title("1  Many wells sit near the water table", loc="left", fontsize=8.5, weight="bold")

    # 2 timing
    ts = pd.read_csv(RES / "failure_timeseries.csv")
    g = ts[(ts.basis == "intake") & (ts.scenario == "2001-03 x10yr")]
    ax = axs[1]
    ax.axvspan(0, 10, color=GRID, lw=0)
    ax.text(5, 1.22, "10-yr drought", ha="center", fontsize=7, color=MUTED)
    for m, c, lab in [("fast", ORANGE, "Fast aquifer, same peak"), ("memory", BLUE, "Prairie aquifer (fitted)")]:
        x = g[g.model == m]
        ax.plot(x.day / 365.25, x.frac_dry * 100, color=c, lw=1.8, label=lab)
    ax.set_xlim(-1, 32)
    ax.set_ylim(0, 1.35)
    ax.set_xlabel("Years since drought began")
    ax.set_ylabel("Wells dry (%)")
    ax.legend(frameon=False, loc="center right", bbox_to_anchor=(1.0, 0.3), fontsize=7)
    ax.set_title("2  Failures arrive late and last for years", loc="left", fontsize=8.5, weight="bold")

    # 3 drilling
    o = pd.read_csv(RES / "drilling_panel.csv")
    s = o[o.model == "weather, lags 0-10"].copy()
    s["lag"] = s.term.str.extract(r"met_l(\d+)")[0].fillna(0).astype(int)
    s = s.sort_values("lag")
    ax = axs[2]
    pct = (s.rate_ratio_per_sd - 1) * 100
    sig = s.ci_lo > 1
    ax.bar(s.lag, pct, color=np.where(sig, BLUE, "#c9c8c3"), width=0.7)
    ax.axhline(0, color=MUTED, lw=0.6)
    ax.set_xticks(range(0, 11, 2))
    ax.set_xlabel("Years since a dry year")
    ax.set_ylabel("Extra replacement wells (%)")
    ax.set_title("3  Observed: re-drilling lasts 5 yr", loc="left", fontsize=8.5, weight="bold")
    ax.text(0.98, 0.95, "Also tracks low groundwater\nbeyond recent weather", transform=ax.transAxes,
            ha="right", va="top", fontsize=7, color=INK)

    for ax in axs:
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(width=0.6, length=2.5, labelsize=7)
    fig.text(0.5, 0.015, "40,262 domestic wells with recorded pump depth  |  39 observation wells  |  "
             "27,005 replacement wells, Saskatchewan, Canada", ha="center", fontsize=7, color=MUTED)
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / "graphical_abstract.png", dpi=300)
    fig.savefig(FIG / "graphical_abstract.pdf")


if __name__ == "__main__":
    main()

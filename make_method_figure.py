"""Methodology figure: (a) workflow, (b) available drawdown at a domestic well,
(c) how aquifer memory changes when a well goes dry (schematic curves).

Writes figures/fig_method.png and figures/fig_method.pdf.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle

plt.rcParams.update({"font.family": "Arial", "font.size": 8, "axes.linewidth": 0.6,
                     "pdf.fonttype": 42, "svg.fonttype": "none"})
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e6e3"
BLUE, ORANGE = "#2a78d6", "#eb6834"
FILLS = {"data": "#eaf1fb", "proc": "#f4f3ef", "link": "#fdf0e9", "out": "#e8f6f0"}
FIG = Path(__file__).parent / "figures"


def box(ax, x, y, w, h, title, lines, fill):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.012",
                                fc=fill, ec="#bdbcb6", lw=0.6))
    ax.text(x + 0.012, y + h - 0.03, title, weight="bold", va="top", color=INK, fontsize=7.5)
    ax.text(x + 0.012, y + h - 0.085, "\n".join(lines), va="top", color=MUTED, fontsize=6.6, linespacing=1.35)


def arrow(ax, x0, y0, x1, y1):
    ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.8,
                                                         shrinkA=0, shrinkB=0, mutation_scale=8))


def workflow(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    cols = [0.0, 0.255, 0.51, 0.765]
    w = 0.225
    heads = ["Data", "Processing", "Linking", "Outputs"]
    for x, hd in zip(cols, heads):
        ax.text(x + w / 2, 0.985, hd.upper(), ha="center", va="top", fontsize=7, color=MUTED, weight="bold")
    # top row: domestic wells
    box(ax, cols[0], 0.52, w, 0.40, "WSA driller reports",
        ["148,598 well records", "depth, static level,", "pump intake, date, use"], FILLS["data"])
    box(ax, cols[1], 0.52, w, 0.40, "Domestic-well exposure",
        ["screen active domestic wells", "available drawdown (AD) =", "intake depth - static level",
         "era-correct static level"], FILLS["proc"])
    # bottom row: observation wells
    box(ax, cols[0], 0.03, w, 0.40, "Obs. wells + climate",
        ["39 core WSA hydrographs", "ERA5-Land P, snowmelt, E", "1960-2025"], FILLS["data"])
    box(ax, cols[1], 0.03, w, 0.40, "Drought response",
        ["Pastas TFN fit per well", "recharge anomaly of droughts", "1984-88, 2001-03, 2021-23,",
         "2001-03 tiled to 5-15 yr", "dh(t) = a(t) * b(t)"], FILLS["proc"])
    box(ax, cols[2], 0.27, w, 0.40, "Assign declines",
        ["3 nearest obs. wells", "within 150 km,", "inverse-distance weights", "-> dh(t) at each", "domestic well"],
        FILLS["link"])
    box(ax, cols[3], 0.52, w, 0.40, "Failure",
        ["dry while decline > AD - 1 m", "memory vs fast response", "scaled to the same", "peak decline"], FILLS["out"])
    box(ax, cols[3], 0.03, w, 0.40, "Research questions",
        ["RQ1 share exposed", "RQ2 when wells fail and", "how long after rain returns",
         "RQ3 which RMs, reserves,", "income groups"], FILLS["out"])
    arrow(ax, cols[0] + w, 0.72, cols[1], 0.72)
    arrow(ax, cols[0] + w, 0.23, cols[1], 0.23)
    arrow(ax, cols[1] + w, 0.62, cols[2], 0.55)
    arrow(ax, cols[1] + w, 0.33, cols[2], 0.40)
    arrow(ax, cols[2] + w, 0.55, cols[3], 0.68)
    arrow(ax, cols[3] + w / 2, 0.52, cols[3] + w / 2, 0.43)


def well_sketch(ax):
    ax.set_xlim(0, 10)
    ax.set_ylim(-26, 2.5)
    ax.axis("off")
    ax.add_patch(Rectangle((0, -26), 10, 26, fc="#f4f3ef", ec="none"))
    ax.plot([0, 10], [0, 0], color=INK, lw=0.8)
    ax.text(0.2, 0.5, "Ground", color=MUTED, fontsize=6.6)
    ax.add_patch(Rectangle((4.6, -24), 0.8, 24, fc="white", ec=INK, lw=0.6))  # casing
    swl, low, intake = -6, -12.5, -17
    ax.plot([0, 10], [swl, swl], color=BLUE, lw=1.4)
    ax.text(9.8, swl + 0.6, "Static level at drilling\n(era-corrected)", ha="right", va="bottom", fontsize=6.4, color=INK)
    ax.plot([0, 10], [low, low], color=BLUE, lw=1.2, ls=(0, (4, 2)))
    ax.text(9.8, low + 0.5, "Level during drought", ha="right", va="bottom", fontsize=6.4, color=INK)
    ax.add_patch(Rectangle((4.75, intake - 1.2), 0.5, 1.2, fc=MUTED, ec="none"))
    ax.text(5.6, intake - 0.6, "Pump intake", va="center", fontsize=6.4, color=INK)
    ax.plot([0, 10], [intake + 1, intake + 1], color=ORANGE, lw=0.8, ls=":")
    ax.text(9.8, intake + 1.3, "Failure (intake + 1 m)", ha="right", va="bottom", fontsize=6.2, color=INK)
    for x0, y0, y1, lab in [(1.4, swl, intake - 0.6, "AD"), (3.2, swl, low, "dh")]:
        ax.annotate("", (x0, y1), (x0, y0), arrowprops=dict(arrowstyle="<->", color=INK, lw=0.7))
        ax.text(x0 - 0.25, y0 + (y1 - y0) * 0.35, lab, ha="right", va="center", fontsize=7, weight="bold")


def timeline(ax):
    t = np.arange(-2, 30, 0.05)
    dt = t[1] - t[0]
    dur = 5.0
    a = np.where((t >= 0) & (t < dur), -1.0, 0.0)
    tau = np.arange(0, 80, dt) + dt / 2

    def conv(b):
        return np.convolve(a, b * dt)[:len(t)]

    slow = conv(tau * np.exp(-tau / 4.0))          # Gamma-type response, half-response ~ 7 yr
    fast = conv(np.exp(-tau / 0.7))                # exponential, half-response ~ 6 months
    slow, fast = -2.8 * slow / slow.min(), -2.8 * fast / fast.min()  # same peak decline
    thr = -2.2
    ax.axvspan(0, dur, color=GRID, lw=0)
    ax.text(dur / 2, 0.25, "Drought", ha="center", va="bottom", fontsize=6.6, color=MUTED)
    ax.axhline(0, color=MUTED, lw=0.5)
    ax.axhline(thr, color=ORANGE, lw=0.8, ls=":")
    ax.text(29.5, thr + 0.1, "Failure threshold of one well", ha="right", va="bottom", fontsize=6.4, color=INK)
    ax.plot(t, fast, color=ORANGE, lw=1.6, label="Fast response, same peak")
    ax.plot(t, slow, color=BLUE, lw=1.6, label="Slow Prairie aquifer")
    for y, c in [(fast, ORANGE), (slow, BLUE)]:
        ax.fill_between(t, thr, y, where=y < thr, color=c, alpha=0.18, lw=0)
    ax.set_xlim(-2, 30)
    ax.set_ylim(-3.3, 0.9)
    ax.set_xlabel("Years since drought began")
    ax.set_ylabel("Water-table change (m)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(width=0.6, length=2.5, colors=INK)
    ax.legend(frameon=False, loc="lower right", fontsize=6.6)
    ax.text(0.99, 0.97, "Schematic curves", transform=ax.transAxes, ha="right", va="top",
            fontsize=6, color=MUTED, style="italic")


def main():
    FIG.mkdir(exist_ok=True)
    fig = plt.figure(figsize=(7.2, 6.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1], width_ratios=[0.8, 1.2], hspace=0.28, wspace=0.18)
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])
    workflow(ax_a)
    well_sketch(ax_b)
    timeline(ax_c)
    for ax, lab in [(ax_a, "a"), (ax_b, "b"), (ax_c, "c")]:
        ax.text(-0.01 if ax is ax_a else -0.04, 1.02, lab, transform=ax.transAxes, weight="bold", fontsize=10, va="bottom")
    fig.savefig(FIG / "fig_method.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / "fig_method.pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()

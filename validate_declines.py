"""Do the fitted responses reproduce observed drought declines?

For each core observation well and each observed drought, the decline is the
mean level of the 12 months before the drought minus the lowest monthly level
from the start of the drought to 5 years after its end. The same measure is
taken on the full-forcing Pastas simulation (results/obs_sim.csv). A model that
damps droughts gives modelled declines well below the observed ones.

Writes results/decline_validation.csv and prints a summary.
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
GW = HERE.parent / "Saskatoon-Groundwater-Climate-Drivers"
RES = HERE / "results"
DROUGHTS = {"1984-88": ("1983-10", "1988-09"), "2001-03": ("2000-10", "2003-09"), "2021-23": ("2020-10", "2023-09")}
AFTER_YEARS = 5


def decline(s, start, end):
    s0, e0 = pd.Period(start, "M"), pd.Period(end, "M")
    before = s[(s.index >= s0 - 12) & (s.index < s0)]
    window = s[(s.index >= s0) & (s.index <= e0 + 12 * AFTER_YEARS)]
    if len(before) < 6 or len(window) < 24:
        return np.nan, np.nan
    low = window.idxmin()
    return before.mean() - window.min(), (low - s0).n / 12


def main():
    resp = pd.read_csv(RES / "obs_well_response.csv")
    obs = pd.read_csv(GW / "data" / "wsa_levels_monthly.csv").groupby(["name", "date"]).level_m.mean()
    sim = pd.read_csv(RES / "obs_sim.csv")
    sim["month"] = pd.PeriodIndex(sim.date, freq="M")
    sim = sim.groupby(["name", "month"]).sim_m.mean()
    rows = []
    for name in resp.name:
        so = obs[name]
        so.index = pd.PeriodIndex(so.index, freq="M")
        ss = sim[name]
        for d, (a, b) in DROUGHTS.items():
            do, to = decline(so, a, b)
            ds, ts = decline(ss, a, b)
            rows.append(dict(name=name, drought=d, obs_decline_m=do, sim_decline_m=ds,
                             obs_years_to_low=to, sim_years_to_low=ts))
    v = pd.DataFrame(rows).dropna(subset=["obs_decline_m", "sim_decline_m"])
    v.to_csv(RES / "decline_validation.csv", index=False, float_format="%.3f")
    ratio = v.sim_decline_m / v.obs_decline_m
    print(v.groupby("drought")[["obs_decline_m", "sim_decline_m", "obs_years_to_low", "sim_years_to_low"]]
          .median().round(2).to_string())
    print(f"\nn = {len(v)}; median sim/obs decline ratio {ratio.median():.2f} "
          f"(IQR {ratio.quantile(.25):.2f}-{ratio.quantile(.75):.2f}); "
          f"r = {np.corrcoef(v.obs_decline_m, v.sim_decline_m)[0, 1]:.2f}")
    print(f"median observed decline {v.obs_decline_m.median():.2f} m, 90th pct {v.obs_decline_m.quantile(.9):.2f} m")


if __name__ == "__main__":
    main()

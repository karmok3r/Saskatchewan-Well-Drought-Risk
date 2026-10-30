"""Regional panel test of replacement-drilling timing.

Province-wide counts mix drought with farm economics and drought-assistance
programmes that affect every region in a year. Here each domestic well is
assigned to its nearest core observation well (39 regions), and replacement
wells per region and water year are modelled with Poisson regression with
region and year fixed effects, so that only within-year differences between
regions inform the estimates. Predictors are the region's standardised
meteorological anomaly (P - E, ERA5-Land) at lags 0-3 years, or its standardised
groundwater level (simulated or observed) in the same year. Standard errors are
clustered by region. If wells fail late, replacement drilling should rise where
groundwater is low, beyond what recent dry weather explains.

Writes results/drilling_panel.csv.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from replacement_drilling import drilling, standardised, water_year

HERE = Path(__file__).resolve().parent
GW = HERE.parent / "Saskatoon-Groundwater-Climate-Drivers"
RES = HERE / "results"
YEARS = (1965, 2025)


def main():
    names = pd.read_csv(RES / "obs_well_response.csv").name
    loc = pd.read_csv(GW / "wells_province.csv").set_index("name").loc[names]
    w = drilling()
    w = w[w.replacement & w.lat.notna()]
    lat, lon = np.radians(w.lat.values)[:, None], np.radians(w.lon.values)[:, None]
    la2, lo2 = np.radians(loc.lat.values)[None, :], np.radians(loc.lon.values)[None, :]
    h = np.sin((la2 - lat) / 2) ** 2 + np.cos(lat) * np.cos(la2) * np.sin((lo2 - lon) / 2) ** 2
    d = 6371 * 2 * np.arcsin(np.sqrt(h))
    w["region"] = names.values[d.argmin(1)]
    w = w[d.min(1) <= 150]
    counts = w.groupby(["region", "wy"]).size().rename("n")

    dr = pd.read_csv(GW / "data" / "drivers_monthly_province.csv")
    dr = dr[dr.name.isin(names)].copy()
    dr["wy"] = water_year(dr.date + "-01")
    pme = dr.groupby(["name", "wy"]).p_minus_e_mm.sum().reset_index()
    pme["met"] = -standardised(pme, "p_minus_e_mm")  # positive = drier than usual
    sim = pd.read_csv(RES / "obs_sim.csv")
    sim["wy"] = water_year(sim.date)
    s = sim.groupby(["name", "wy"]).sim_m.mean().reset_index()
    s["gw_sim"] = -standardised(s, "sim_m")  # positive = lower than usual
    lv = pd.read_csv(GW / "data" / "wsa_levels_monthly.csv")
    lv = lv[lv.name.isin(names)].groupby(["name", "date"]).level_m.mean().reset_index()
    lv["wy"] = water_year(lv.date + "-01")
    o = lv.groupby(["name", "wy"]).level_m.mean().reset_index()
    o["gw_obs"] = -standardised(o, "level_m")

    p = pme[["name", "wy", "met"]].merge(s[["name", "wy", "gw_sim"]], on=["name", "wy"], how="left") \
        .merge(o[["name", "wy", "gw_obs"]], on=["name", "wy"], how="left").rename(columns={"name": "region"})
    p = p.sort_values(["region", "wy"])
    for L in range(1, 11):
        p[f"met_l{L}"] = p.groupby("region").met.shift(L)
    for k in [3, 5]:  # placebo: future groundwater should not explain today's drilling
        p[f"gw_lead{k}"] = p.groupby("region").gw_obs.shift(-k)
    p = p[(p.wy >= YEARS[0]) & (p.wy <= YEARS[1])]
    p = p.merge(counts.reset_index(), on=["region", "wy"], how="left").fillna({"n": 0})

    w3 = "met + met_l1 + met_l2 + met_l3"
    w10 = " + ".join(["met"] + [f"met_l{k}" for k in range(1, 11)])
    specs = {"weather, lags 0-3": f"n ~ {w3}",
             "weather, lags 0-10": f"n ~ {w10}",
             "groundwater (simulated)": "n ~ gw_sim",
             "groundwater (observed)": "n ~ gw_obs",
             "weather lags 0-3 + groundwater (simulated)": f"n ~ {w3} + gw_sim",
             "weather lags 0-3 + groundwater (observed)": f"n ~ {w3} + gw_obs",
             "weather lags 0-10 + groundwater (observed)": f"n ~ {w10} + gw_obs",
             "placebo: + groundwater 3 yr later": f"n ~ {w3} + gw_obs + gw_lead3",
             "placebo: + groundwater 5 yr later": f"n ~ {w3} + gw_obs + gw_lead5"}
    rows = []
    for label, f in specs.items():
        cols = [c.strip() for c in f.split("~")[1].split("+")]
        q = p.dropna(subset=cols).reset_index(drop=True)
        m = smf.poisson(f + " + C(region) + C(wy)", q).fit(disp=0, maxiter=200, cov_type="cluster",
                                                             cov_kwds={"groups": pd.factorize(q.region)[0]})
        for c in cols:
            lo, hi = m.conf_int().loc[c]
            rows.append(dict(model=label, term=c, rate_ratio_per_sd=np.exp(m.params[c]), ci_lo=np.exp(lo),
                             ci_hi=np.exp(hi), p=m.pvalues[c], n_obs=int(m.nobs), wells=int(q.n.sum())))
    out = pd.DataFrame(rows)
    out.to_csv(RES / "drilling_panel.csv", index=False, float_format="%.4f")
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

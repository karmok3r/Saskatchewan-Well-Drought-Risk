"""Independent check of failure timing: when do households drill replacement wells?

If droughts make domestic wells fail years later, replacement drilling should
follow the groundwater state (which lags drought) rather than the
meteorological drought itself, and should peak years after dry spells.

Series, by water year (October-September), 1962-2025:
- drilling: all new domestic supply wells, and replacement wells (a new domestic
  well on a quarter-section that already had one), from the driller reports;
  log counts detrended with a LOWESS trend (frac 0.5) to remove the long-term
  changes in drilling activity
- meteorological drought: precipitation minus evaporation (ERA5-Land), mean
  standardised anomaly over the 39 observation-well locations
- groundwater state: mean standardised anomaly of the simulated levels (fitted
  Pastas responses driven by observed weather) and of the observed levels

Cross-correlations at lags 0-15 years (driver leading drilling); significance
from 2000 phase-randomised surrogates of the driver (preserving its
autocorrelation), with the maximum over lags taken for the family-wise test.
Also: depth of replacement wells relative to the well they replace.

Writes results/drilling_annual.csv, results/drilling_xcorr.csv,
results/replacement_depth.csv.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess

HERE = Path(__file__).resolve().parent
GW = HERE.parent / "Saskatoon-Groundwater-Climate-Drivers"
RES = HERE / "results"
YEARS = (1962, 2025)
LAGS = range(0, 16)
N_SURR = 2000
RNG = np.random.default_rng(7)
FT = 0.3048


def water_year(dates):
    d = pd.to_datetime(dates)
    return d.dt.year + (d.dt.month >= 10).astype(int)


def drilling():
    d = pd.read_csv(HERE / "data" / "raw" / "wsa_water_wells.csv", low_memory=False)
    w = d[(d.cwateruse == "Domestic") & (d.cwelluse == "Withdrawal")].copy()
    w["date"] = pd.to_datetime(w.dCompleted, format="%Y.%m.%d", errors="coerce")
    w = w.dropna(subset=["date", "cWellsQuarter", "cWellsSection"]).sort_values("date")
    w["wy"] = water_year(w.date)
    w["parcel"] = w.cLandLocationDisplay.str.strip()
    w["order"] = w.groupby("parcel").cumcount()
    w["replacement"] = w.order > 0
    w["depth_m"] = w.nBoreHoleDepthInFeet.where(w.nBoreHoleDepthInFeet > 0) * FT
    # depth change relative to the previous domestic well on the same parcel
    w["prev_depth_m"] = w.groupby("parcel").depth_m.shift()
    w["prev_wy"] = w.groupby("parcel").wy.shift()
    return w


def standardised(df, col, by="name"):
    g = df.groupby(by)[col]
    return (df[col] - g.transform("mean")) / g.transform("std")


def drivers():
    names = pd.read_csv(RES / "obs_well_response.csv").name
    dr = pd.read_csv(GW / "data" / "drivers_monthly_province.csv")
    dr = dr[dr.name.isin(names)].copy()
    dr["wy"] = water_year(dr.date + "-01")
    pme = dr.groupby(["name", "wy"]).p_minus_e_mm.sum().reset_index()
    pme["z"] = standardised(pme, "p_minus_e_mm")
    met = pme.groupby("wy").z.mean().rename("pme")

    sim = pd.read_csv(RES / "obs_sim.csv")
    sim["wy"] = water_year(sim.date)
    s = sim.groupby(["name", "wy"]).sim_m.mean().reset_index()
    s["z"] = standardised(s, "sim_m")
    gw_sim = s.groupby("wy").z.mean().rename("gw_sim")

    lv = pd.read_csv(GW / "data" / "wsa_levels_monthly.csv")
    lv = lv[lv.name.isin(names)].groupby(["name", "date"]).level_m.mean().reset_index()
    lv["wy"] = water_year(lv.date + "-01")
    o = lv.groupby(["name", "wy"]).level_m.mean().reset_index()
    o["z"] = standardised(o, "level_m")
    gw_obs = o.groupby("wy").z.mean().rename("gw_obs")
    return pd.concat([met, gw_sim, gw_obs], axis=1)


def detrend_log(counts):
    y = np.log(counts.values.astype(float))
    trend = lowess(y, counts.index.values, frac=0.5, return_sorted=False)
    return pd.Series(y - trend, index=counts.index)


def surrogate(x):
    f = np.fft.rfft(x - x.mean())
    ph = RNG.uniform(0, 2 * np.pi, len(f))
    ph[0] = 0
    if len(x) % 2 == 0:
        ph[-1] = 0
    return np.fft.irfft(np.abs(f) * np.exp(1j * ph), n=len(x)) + x.mean()


def xcorr(driver, target, lags):
    out = []
    for L in lags:
        a, b = driver.values[:len(driver) - L], target.values[L:]
        ok = ~(np.isnan(a) | np.isnan(b))
        out.append(np.corrcoef(a[ok], b[ok])[0, 1] if ok.sum() > 10 else np.nan)
    return np.array(out)


def main():
    w = drilling()
    yrs = pd.RangeIndex(YEARS[0], YEARS[1] + 1, name="wy")
    allc = w.groupby("wy").size().reindex(yrs).fillna(0)
    rep = w[w.replacement].groupby("wy").size().reindex(yrs).fillna(0)
    ann = pd.DataFrame({"all_wells": allc, "replacement_wells": rep})
    ann["all_detr"] = detrend_log(allc.clip(lower=1))
    ann["rep_detr"] = detrend_log(rep.clip(lower=1))
    ann = ann.join(drivers())
    ann.to_csv(RES / "drilling_annual.csv", float_format="%.4f")

    rows = []
    for tcol in ["all_detr", "rep_detr"]:
        for dcol, sign in [("pme", -1), ("gw_sim", -1), ("gw_obs", -1)]:
            # sign: drilling should rise when the driver is low (dry / low water table)
            sub = ann[[dcol, tcol]].dropna()
            drv = sign * sub[dcol]
            r = xcorr(drv, sub[tcol], LAGS)
            surr = np.array([xcorr(pd.Series(surrogate(drv.values), index=drv.index), sub[tcol], LAGS)
                             for _ in range(N_SURR)])
            p_each = (np.abs(surr) >= np.abs(r)).mean(0)
            p_max = (np.nanmax(np.abs(surr), 1) >= np.nanmax(np.abs(r))).mean()
            for L, rr, pp in zip(LAGS, r, p_each):
                rows.append(dict(target=tcol, driver=dcol, lag_yr=L, r=rr, p=pp, p_familywise=p_max,
                                 years=f"{sub.index.min()}-{sub.index.max()}"))
    x = pd.DataFrame(rows)
    x.to_csv(RES / "drilling_xcorr.csv", index=False, float_format="%.4f")
    best = x.loc[x.groupby(["target", "driver"]).r.idxmax()]
    print(best.round(3).to_string(index=False))
    print("\nlag-0 correlations:")
    print(x[x.lag_yr == 0].round(3).to_string(index=False))

    rp = w[w.replacement & w.depth_m.notna() & w.prev_depth_m.notna()].copy()
    rp["deeper_m"] = rp.depth_m - rp.prev_depth_m
    rp["gap_yr"] = rp.wy - rp.prev_wy
    rp[["parcel", "wy", "prev_wy", "gap_yr", "depth_m", "prev_depth_m", "deeper_m"]].to_csv(
        RES / "replacement_depth.csv", index=False, float_format="%.2f")
    print(f"\nreplacement wells with both depths: {len(rp)}; median deepening {rp.deeper_m.median():.1f} m; "
          f"share deeper {(rp.deeper_m > 0).mean():.2f}")


if __name__ == "__main__":
    main()

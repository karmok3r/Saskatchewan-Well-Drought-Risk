"""Domestic-well exposure and failure under the drought scenarios.

1. Screen the WSA driller reports to active domestic supply wells with a static
   water level and a pump intake (or well bottom) below it; convert feet to m.
2. Available drawdown (AD): pump intake (or bottom) depth minus static water level.
3. Static levels were measured on the drilling date. Where the nearest
   observation well was recorded then, the level is moved to the 1991-2020 mean
   of that well (depth to water + level at drilling - mean level).
4. Each domestic well takes the scenario decline dh(t) of its K nearest core
   observation wells (inverse-distance weights) within MAX_KM.
5. A well fails while the decline exceeds AD minus a pump-submergence margin.

Writes results/domestic_wells.csv, results/failure_timeseries.csv and
results/failure_summary.csv and
results/failure_by_well.csv (intake basis).
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
GW = HERE.parent / "Saskatoon-Groundwater-Climate-Drivers"
RES = HERE / "results"
FT = 0.3048
K = int(os.environ.get("K_DONORS", 3))
MAX_KM = 150
MARGIN_M = float(os.environ.get("MARGIN_M", 1.0))
SCALE = float(os.environ.get("DECLINE_SCALE", 1.0))  # sensitivity: multiply modelled declines
TAG = os.environ.get("TAG", "")
# further sensitivity switches
PUMPING = os.environ.get("PUMPING", "0") == "1"  # subtract the pumping-test drawdown from AD (upper bound)
MIN_YEAR = int(os.environ.get("MIN_YEAR", 0))  # keep only wells drilled since this year
NO_BOUND = os.environ.get("NO_BOUND", "0") == "1"  # drop donors whose response time hit the upper bound
REF = ("1991-01", "2020-12")


def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    h = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(h))


def domestic_wells():
    d = pd.read_csv(HERE / "data" / "raw" / "wsa_water_wells.csv", low_memory=False)
    w = d[(d.cwateruse == "Domestic") & (d.cwelluse == "Withdrawal") & (d.cDecommisioned != 1)].copy()
    w["year"] = pd.to_datetime(w.dCompleted, format="%Y.%m.%d", errors="coerce").dt.year
    w["month"] = pd.to_datetime(w.dCompleted, format="%Y.%m.%d", errors="coerce").dt.strftime("%Y-%m")
    w["swl_m"] = w.nWaterLevel.where(w.nWaterLevel > 0) * FT
    w["depth_m"] = w.nBoreHoleDepthInFeet.where(w.nBoreHoleDepthInFeet > 0) * FT
    w["intake_m"] = w.nintake.where(w.nintake > 0) * FT
    w = w[w.swl_m.notna() & w.depth_m.between(1, 300) & (w.swl_m < w.depth_m)]
    w = w[w.intake_m.isna() | ((w.intake_m > w.swl_m) & (w.intake_m <= w.depth_m + 1))]
    w = w[w.lat.between(49, 60.1) & w.lon.between(-110.1, -101.3)]
    w["test_drawdown_m"] = w.ndrawdown.where(w.ndrawdown > 0) * FT
    keep = ["cWWDRId", "lat", "lon", "crm", "year", "month", "cinstmethd", "swl_m", "depth_m", "intake_m",
            "test_drawdown_m"]
    return w[keep].reset_index(drop=True)


def main():
    w = domestic_wells()
    w = w[w.year.fillna(0) >= MIN_YEAR].reset_index(drop=True)
    resp = pd.read_csv(RES / "obs_well_response.csv")
    if NO_BOUND:
        resp = resp[~resp.a_at_bound]
    loc = pd.read_csv(GW / "wells_province.csv").set_index("name")
    obs = loc.loc[resp.name, ["lat", "lon"]]

    # nearest observation wells
    D = haversine(w.lat.values[:, None], w.lon.values[:, None], obs.lat.values[None, :], obs.lon.values[None, :])
    order = np.argsort(D, axis=1)[:, :K]
    dk = np.take_along_axis(D, order, axis=1)
    w["donors"] = [";".join(obs.index[o]) for o in order]
    w["nearest_km"] = dk[:, 0]
    wts = np.where(dk <= MAX_KM, 1 / np.maximum(dk, 1.0), 0.0)
    w["covered"] = wts.sum(1) > 0

    # move static levels to the 1991-2020 mean of the nearest observation well
    lv = pd.read_csv(GW / "data" / "wsa_levels_monthly.csv").groupby(["name", "date"]).level_m.mean()
    ref = {n: lv[n][REF[0]:REF[1]].mean() for n in obs.index}
    near = obs.index[order[:, 0]]
    shift = [lv[n].get(m, np.nan) - ref[n] if isinstance(m, str) else np.nan for n, m in zip(near, w.month)]
    w["era_shift_m"] = shift
    w["swl_ref_m"] = (w.swl_m + w.era_shift_m.fillna(0)).clip(lower=0)
    w["ad_intake_m"] = w.intake_m - w.swl_ref_m
    w["ad_bottom_m"] = w.depth_m - w.swl_ref_m
    if PUMPING:
        w["ad_intake_m"] = w.ad_intake_m - w.test_drawdown_m.fillna(0)
        w["ad_bottom_m"] = w.ad_bottom_m - w.test_drawdown_m.fillna(0)
    if not TAG:
        w.to_csv(RES / "domestic_wells.csv", index=False, float_format="%.3f")
    print(f"{len(w)} screened domestic wells; {w.intake_m.notna().sum()} with pump intake; "
          f"{w.covered.mean():.0%} within {MAX_KM} km of an observation well; "
          f"{w.era_shift_m.notna().mean():.0%} with an era correction")

    # failures through time
    sc = pd.read_csv(RES / "scenario_dh.csv")
    idx = {n: i for i, n in enumerate(obs.index)}
    cov = w[w.covered].reset_index(drop=True)
    W = wts[w.covered.values]
    W = W / W.sum(1, keepdims=True)
    O = order[w.covered.values]
    ts, summ, per_well = [], [], []
    for (scen, model), g in sc.groupby(["scenario", "model"]):
        piv = g.pivot(index="day", columns="name", values="dh_m")[obs.index]
        days = piv.index.values
        dur = int(g.drought_days.iloc[0])
        DH = piv.values  # day x obs well
        dh = (DH[:, O] * W[None, :, :]).sum(2).T  # well x day
        for basis in ["intake", "bottom"]:
            ad = cov[f"ad_{basis}_m"].values
            ok = ~np.isnan(ad) & (ad > MARGIN_M)  # wells already at the margin are reported separately
            dry = (-dh[ok] * SCALE > (ad[ok, None] - MARGIN_M))
            frac = dry.mean(0)
            for d, fr in zip(days, frac):
                ts.append((scen, model, basis, dur, int(d), fr))
            ever = dry.any(1)
            if basis == "intake":
                step = np.median(np.diff(days))
                per_well.append(pd.DataFrame({
                    "cWWDRId": cov.cWWDRId.values[ok], "scenario": scen, "model": model, "dry": ever,
                    "first_dry_yr": np.where(ever, days[dry.argmax(1)] / 365.25, np.nan),
                    "dry_years_after_end": (dry & (days > dur)[None, :]).sum(1) * step / 365.25,
                    "max_decline_m": -dh[ok].min(1) * SCALE}))
            peak = days[frac.argmax()]
            after = days[(frac > 0.5 * frac.max()) & (days > dur)] if frac.max() > 0 else np.array([])
            summ.append(dict(scenario=scen, model=model, basis=basis, drought_years=dur / 365.25,
                             n_wells=int(ok.sum()), ever_dry=ever.mean(), peak_dry=frac.max(),
                             peak_year=peak / 365.25,
                             years_above_half_peak_after_end=(after.max() - dur) / 365.25 if after.size else 0.0,
                             at_margin_share=float(np.mean(ad[~np.isnan(ad)] <= MARGIN_M)),
                             first_dry_median_yr=float(np.median(days[dry.argmax(1)][ever]) / 365.25) if ever.any() else np.nan,
                             dry_years_after_end_median=float(np.median(
                                 (dry & (days > dur)[None, :]).sum(1)[ever] * np.median(np.diff(days)) / 365.25))
                             if ever.any() else np.nan,
                             max_decline_median_m=float(np.median(-dh[ok].min(1)) * SCALE)))
    pd.DataFrame(ts, columns=["scenario", "model", "basis", "drought_days", "day", "frac_dry"]).to_csv(
        RES / f"failure_timeseries{TAG}.csv", index=False, float_format="%.5f")
    pd.concat(per_well).to_csv(RES / f"failure_by_well{TAG}.csv", index=False, float_format="%.3f")
    s = pd.DataFrame(summ)
    s.to_csv(RES / f"failure_summary{TAG}.csv", index=False, float_format="%.4f")
    print(s[s.basis == "intake"].round(3).to_string(index=False))


if __name__ == "__main__":
    main()

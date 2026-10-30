"""RQ1 exposure of domestic wells and RQ3 where the exposed and failing wells are.

RQ1: share of wells with available drawdown (AD, pump intake minus static level)
below 2, 5 and 10 m, overall and by construction method, decade drilled and
well depth; with and without the era correction of static levels.

RQ3: exposure and modelled failure by rural municipality (RM number in the
driller reports) and for wells on or within 5 km of First Nations reserves
(Canada Lands Survey System boundaries, NRCan).

Writes results/exposure_*.csv and results/rq3_*.csv.
"""
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
THRESH = [2, 5, 10]
MAIN = ("2001-03 x10yr", "memory")  # headline scenario for RQ3
NEAR_KM = 5


def shares(g, col="ad_intake_m"):
    x = g[col].dropna()
    return pd.Series({"n": len(x), "median_ad_m": x.median(), **{f"le_{t}m": (x <= t).mean() for t in THRESH}})


def main():
    w = pd.read_csv(RES / "domestic_wells.csv", low_memory=False)
    w["decade"] = (w.year // 10 * 10).astype("Int64")
    w["method"] = w.cinstmethd.where(w.cinstmethd.isin(["Bored", "Drilled"]), "Other")
    w["depth_class"] = pd.cut(w.depth_m, [0, 15, 30, 60, 300], labels=["<15 m", "15-30 m", "30-60 m", ">60 m"])
    w["ad_intake_raw_m"] = w.intake_m - w.swl_m

    tabs = [shares(w).rename("all").to_frame().T,
            shares(w, "ad_intake_raw_m").rename("all, no era correction").to_frame().T,
            shares(w, "ad_bottom_m").rename("all, to well bottom").to_frame().T]
    for col in ["method", "depth_class", "decade"]:
        t = w.groupby(col, observed=True).apply(shares)
        t.index = [f"{col}: {i}" for i in t.index]
        tabs.append(t)
    ex = pd.concat(tabs)
    ex.to_csv(RES / "exposure_summary.csv", float_format="%.4f")
    print(ex.round(3).to_string())

    # RQ3: reserves
    res = gpd.read_file(next((HERE / "data" / "raw" / "reserves").glob("*.shp")))
    res = res[res.ALTYPE == "Indian Reserve"].to_crs(3347)
    pts = gpd.GeoDataFrame(w, geometry=gpd.points_from_xy(w.lon, w.lat), crs=4326).to_crs(3347)
    on = gpd.sjoin(pts, res[["NAME1", "geometry"]], predicate="within", how="left")
    w["on_reserve"] = on.groupby(level=0).NAME1.first().notna().values
    near = gpd.sjoin_nearest(pts, res[["geometry"]], how="left", distance_col="d_res")
    w["reserve_km"] = near.groupby(level=0).d_res.min().values / 1000
    w["reserve_zone"] = np.select([w.on_reserve, w.reserve_km <= NEAR_KM], ["on reserve", f"within {NEAR_KM} km"],
                                  "elsewhere")

    f = pd.read_csv(RES / "failure_by_well.csv")
    f = f[(f.scenario == MAIN[0]) & (f.model == MAIN[1])][["cWWDRId", "dry", "dry_years_after_end"]]
    w = w.merge(f, on="cWWDRId", how="left")
    z = w.groupby("reserve_zone").apply(lambda g: pd.concat([shares(g), pd.Series(
        {"dry_share": g.dry.mean(), "dry_years_after_end_if_dry": g.loc[g.dry == True, "dry_years_after_end"].median()})]))  # noqa: E712
    z.to_csv(RES / "rq3_reserves.csv", float_format="%.4f")
    print("\n", z.round(3).to_string())

    rm = w.dropna(subset=["crm"]).groupby("crm").apply(lambda g: pd.concat([shares(g), pd.Series(
        {"dry_share": g.dry.mean(), "lat": g.lat.median(), "lon": g.lon.median()})]))
    rm = rm[rm.n >= 30].sort_values("le_5m", ascending=False)
    rm.to_csv(RES / "rq3_rm.csv", float_format="%.4f")
    print(f"\n{len(rm)} RMs with >= 30 wells; share with AD <= 5 m: median {rm.le_5m.median():.3f}, "
          f"90th pct {rm.le_5m.quantile(.9):.3f}, max {rm.le_5m.max():.3f}")
    w[["cWWDRId", "on_reserve", "reserve_km", "reserve_zone"]].to_csv(RES / "domestic_wells_reserve.csv", index=False,
                                                                      float_format="%.3f")


if __name__ == "__main__":
    main()

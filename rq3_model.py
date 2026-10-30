"""Is the higher exposure of on-reserve wells explained by how and when they were built?

Logistic regression of exposure (AD <= 5 m at the pump intake) on reserve zone,
construction method, depth class and decade drilled, with standard errors
clustered by rural municipality (wells in one RM share aquifers and drillers);
wells outside RMs (mostly on reserves) are clustered by 0.5-degree cell.

Writes results/rq3_logit.csv.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

RES = Path(__file__).resolve().parent / "results"


def main():
    w = pd.read_csv(RES / "domestic_wells.csv", low_memory=False)
    z = pd.read_csv(RES / "domestic_wells_reserve.csv")
    w = w.merge(z, on="cWWDRId").dropna(subset=["ad_intake_m", "year"])
    # wells on reserves have no RM number: cluster them by 0.5-degree cell instead
    cell = "cell" + (w.lat * 2).round().astype(str) + "_" + (w.lon * 2).round().astype(str)
    w["cluster"] = w.crm.astype(str).where(w.crm.notna(), cell)
    w["exposed"] = (w.ad_intake_m <= 5).astype(int)
    w["method"] = w.cinstmethd.where(w.cinstmethd.isin(["Bored", "Drilled"]), "Other")
    w["depth_class"] = pd.cut(w.depth_m, [0, 15, 30, 60, 300], labels=["a<15", "b15-30", "c30-60", "d>60"])
    w["decade"] = (w.year.clip(1960, 2020) // 10 * 10).astype(int).astype(str)
    w["zone"] = w.reserve_zone.map({"elsewhere": "a_elsewhere", "within 5 km": "b_near", "on reserve": "c_on"})
    rows = []
    for label, f in [("zone only", "exposed ~ C(zone)"),
                     ("+ method", "exposed ~ C(zone) + C(method)"),
                     ("+ method, depth, decade", "exposed ~ C(zone) + C(method) + C(depth_class) + C(decade)")]:
        m = smf.logit(f, w).fit(disp=0, cov_type="cluster", cov_kwds={"groups": pd.factorize(w.cluster)[0]})
        for term in ["C(zone)[T.b_near]", "C(zone)[T.c_on]"]:
            lo, hi = m.conf_int().loc[term]
            rows.append(dict(model=label, term=term.split(".")[-1].rstrip("]"), odds_ratio=np.exp(m.params[term]),
                             ci_lo=np.exp(lo), ci_hi=np.exp(hi), p=m.pvalues[term], n=int(m.nobs)))
    out = pd.DataFrame(rows)
    out.to_csv(RES / "rq3_logit.csv", index=False, float_format="%.4f")
    print(out.round(3).to_string(index=False))
    print(pd.crosstab(w.zone, w.method, normalize="index").round(3))
    print(w.zone.value_counts())


if __name__ == "__main__":
    main()

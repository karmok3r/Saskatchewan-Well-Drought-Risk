"""Is well exposure related to household income?

Median total household income in 2020 for each rural municipality (2021 Census
Profile, census subdivisions, Statistics Canada 98-401-X2021023) is joined to the
domestic wells by RM number. Tests: Spearman correlation between RM income and
the RM share of exposed wells (AD <= 5 m), and a logistic regression of
exposure on income tercile with the construction, depth and decade controls of
rq3_model.py and RM-clustered standard errors.

Census rows are extracted once from the large profile file into
data/raw/statcan/sk_csd_income.csv. Writes results/rq3_income.csv.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
RAW = HERE / "data" / "raw" / "statcan"


def rm_income():
    out = RAW / "sk_csd_income.csv"
    if not out.exists():
        keep = []
        for c in pd.read_csv(RAW / "98-401-X2021023_English_CSV_data.csv", encoding="latin-1",
                             usecols=["ALT_GEO_CODE", "GEO_NAME", "CHARACTERISTIC_NAME", "C1_COUNT_TOTAL"],
                             chunksize=500_000):
            c = c[c.ALT_GEO_CODE.astype(str).str.startswith("47")
                  & c.CHARACTERISTIC_NAME.str.strip().eq("Median total income of household in 2020 ($)")]
            keep.append(c)
        pd.concat(keep).to_csv(out, index=False)
    d = pd.read_csv(out)
    rm = d[d.GEO_NAME.str.contains(r"Rural municipality \(RM\)")].copy()
    rm["crm"] = rm.GEO_NAME.str.extract(r"No\. (\d+)")[0].astype(float)
    return rm.rename(columns={"C1_COUNT_TOTAL": "income"})[["crm", "GEO_NAME", "income"]].dropna()


def main():
    inc = rm_income()
    w = pd.read_csv(RES / "domestic_wells.csv", low_memory=False).dropna(subset=["ad_intake_m", "year", "crm"])
    w = w.merge(inc, on="crm", how="inner")
    w["exposed"] = (w.ad_intake_m <= 5).astype(int)
    w["method"] = w.cinstmethd.where(w.cinstmethd.isin(["Bored", "Drilled"]), "Other")
    w["depth_class"] = pd.cut(w.depth_m, [0, 15, 30, 60, 300], labels=["a<15", "b15-30", "c30-60", "d>60"])
    w["decade"] = (w.year.clip(1960, 2020) // 10 * 10).astype(int).astype(str)
    w["income_k"] = w.income / 1000

    rm = w.groupby("crm").agg(n=("exposed", "size"), share=("exposed", "mean"), income=("income", "first"))
    rm = rm[rm.n >= 30]
    rho, p = spearmanr(rm.income, rm.share)
    terc = pd.qcut(rm.income, 3, labels=["low", "middle", "high"])
    w["tercile"] = w.crm.map(terc)
    w = w.dropna(subset=["tercile", "depth_class"]).reset_index(drop=True)
    w["tercile"] = w.tercile.astype(str)
    rows = [dict(test="RM Spearman income vs exposed share", estimate=rho, ci_lo=np.nan, ci_hi=np.nan, p=p,
                 n=len(rm))]
    for t, g in w.groupby("tercile"):
        lo, hi = rm.income[terc == t].min(), rm.income[terc == t].max()
        rows.append(dict(test=f"exposed share, {t} income RMs (${lo:,.0f}-{hi:,.0f})", estimate=g.exposed.mean(),
                         ci_lo=np.nan, ci_hi=np.nan, p=np.nan, n=len(g)))
    for label, f in [("tercile only", "exposed ~ C(tercile, Treatment('high'))"),
                     ("+ method, depth, decade",
                      "exposed ~ C(tercile, Treatment('high')) + C(method) + C(depth_class) + C(decade)")]:
        m = smf.logit(f, w).fit(disp=0, cov_type="cluster", cov_kwds={"groups": pd.factorize(w.crm)[0]})
        for term in [k for k in m.params.index if "tercile" in k]:
            lo, hi = m.conf_int().loc[term]
            rows.append(dict(test=f"OR {term.split('.')[-1].rstrip(']')} vs high, {label}",
                             estimate=np.exp(m.params[term]), ci_lo=np.exp(lo), ci_hi=np.exp(hi),
                             p=m.pvalues[term], n=int(m.nobs)))
    out = pd.DataFrame(rows)
    out.to_csv(RES / "rq3_income.csv", index=False, float_format="%.4f")
    print(f"{len(inc)} RMs with income; {w.crm.nunique()} RMs and {len(w)} wells matched")
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

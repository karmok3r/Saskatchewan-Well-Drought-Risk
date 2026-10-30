"""Water-table decline under drought scenarios at the WSA observation wells.

For each core observation well (the 39 wells of the groundwater-memory study,
without the pumping-affected and flagged wells) a deterministic Pastas model is
fitted on the whole record: recharge R = P_liquid + f E routed to the well by a
Gamma response (same set-up as pastas_models.py in Saskatoon-Groundwater-Climate-Drivers).

A drought scenario is a recharge anomaly a(t) = R(t) - Rclim(t): zero, then the
observed anomalies of a drought period (or that period tiled to a longer
length), then zero again. The level change is the convolution of a(t) with the
well's block response, dh(t) = sum a(tau) b(t - tau), so it is relative to the
level the well would have under average weather.

A "fast" counterfactual uses an exponential response with a half-response time
of 6 months, scaled so that its largest decline in each scenario equals that of
the fitted (memory) response: the two differ only in timing.

The full-forcing simulation of each well is also saved, to compare modelled and
observed drought declines (validate_declines.py).

Needs Pastas (pandas < 3); run with the groundwater project's environment:
  ../Saskatoon-Groundwater-Climate-Drivers/.venv_pastas/Scripts/python drought_response.py
Writes results/obs_well_response.csv, results/scenario_dh.csv and results/obs_sim.csv.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import fftconvolve

HERE = Path(__file__).resolve().parent
GW = HERE.parent / "Saskatoon-Groundwater-Climate-Drivers"
os.environ["GW_SCOPE"] = "province"
os.chdir(GW)  # common.py reads its inputs relative to the groundwater project
sys.path.insert(0, str(GW))
from common import EXCLUDED, FLAGGED, load_drivers, load_levels, WELLS  # noqa: E402
from pastas_models import daily_stresses, fit, head_series, response_times  # noqa: E402

OUT = HERE / "results"
CLIM = ("1961-01-01", "2020-12-31")  # reference period for average recharge
LEAD_YEARS, TAIL_YEARS = 2, 30  # years of average weather before and after the drought
FAST_T50_DAYS = 182.5

# Observed droughts as water years (October to September), and longer synthetic
# droughts made by repeating the 2001-2003 sequence.
OBSERVED = {"1984-88": ("1983-10-01", "1988-09-30"),
            "2001-03": ("2000-10-01", "2003-09-30"),
            "2021-23": ("2020-10-01", "2023-09-30")}
SYNTHETIC = {"2001-03 x5yr": 5, "2001-03 x10yr": 10, "2001-03 x15yr": 15}


def anomaly(st, f):
    R = st.prec + f * st.evap
    ref = R[CLIM[0]:CLIM[1]]
    clim = ref.groupby(ref.index.month).mean()
    return R - clim.reindex(R.index.month).values


def scenarios(a):
    """Daily anomaly sequence of each drought (without lead and tail)."""
    out = {k: a[s:e].values for k, (s, e) in OBSERVED.items()}
    base = out["2001-03"]
    for k, years in SYNTHETIC.items():
        n = int(round(years * 365.25))
        out[k] = np.resize(base, n)
    return out


def fast_response(gain, n_days):
    k = np.log(2) / FAST_T50_DAYS
    t = np.arange(n_days)
    b = np.exp(-k * t) - np.exp(-k * (t + 1))  # block response of an exponential reservoir
    return gain * b / b.sum()


def run(a_drought, b):
    lead, tail = int(LEAD_YEARS * 365.25), int(TAIL_YEARS * 365.25)
    a = np.concatenate([np.zeros(lead), a_drought, np.zeros(tail)])
    dh = fftconvolve(a, b)[:len(a)]
    days = np.arange(len(a)) - lead  # day 0 = start of the drought
    return days, dh


def main():
    OUT.mkdir(exist_ok=True)
    levels = load_levels()
    drivers = load_drivers()
    core = [w for w in levels.name.unique() if w not in FLAGGED and w not in EXCLUDED]
    resp, rows, sims = [], [], []
    for name in sorted(core):
        g = levels[levels.name == name]
        st = daily_stresses(drivers[drivers.driver_name == WELLS[name]])
        ml = fit(head_series(g), st, noise=False)
        f = ml.parameters.optimal["rch_f"]
        b = ml.get_block_response("rch").values
        gain = float(b.sum())
        info = response_times(ml)
        resp.append(dict(name=name, gain_m_per_mm_day=gain, **info))
        sim = ml.simulate()
        sims.append(pd.DataFrame({"name": name, "date": sim.index.strftime("%Y-%m-%d"), "sim_m": sim.values}))
        a = anomaly(st, f)
        for scen, ad in scenarios(a).items():
            dur = len(ad)
            days, slow = run(ad, b)
            _, fast = run(ad, fast_response(1.0, len(b)))
            fast = fast * slow.min() / fast.min()  # same peak decline as the fitted response
            keep = (days % 30 == 0) & (days >= -365)  # roughly monthly output
            for model, dh in [("memory", slow), ("fast", fast)]:
                for d, v in zip(days[keep], dh[keep]):
                    rows.append((name, scen, model, dur, int(d), float(v)))
        print(f"{name}: t50 {info['t50_days'] / 365.25:.1f} yr, gain {gain:.3f}", flush=True)
    pd.DataFrame(resp).to_csv(OUT / "obs_well_response.csv", index=False, float_format="%.4f")
    pd.DataFrame(rows, columns=["name", "scenario", "model", "drought_days", "day", "dh_m"]).to_csv(
        OUT / "scenario_dh.csv", index=False, float_format="%.4f")
    pd.concat(sims).to_csv(OUT / "obs_sim.csv", index=False, float_format="%.4f")


if __name__ == "__main__":
    main()

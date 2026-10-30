# Domestic Well Drought Risk — Saskatchewan

Code and data for studying how drought affects domestic water wells in Saskatchewan, Canada. The analysis combines the Water Security Agency (WSA) driller reports with transfer-function models of the provincial observation wells to estimate how much water each domestic well has above its pump, how far drought lowers the water table, and when and for how long wells could lose water at the pump. It also looks at where exposed wells are, relative to rural municipalities, household income and First Nations reserves.

## Pipeline
Run the scripts from the repository root, in this order. Outputs go to `results/` and `figures/`.

| Step | Script | Output |
| --- | --- | --- |
| Download the WSA driller reports (public GIS layer) | `fetch_wells.py` | `data/raw/wsa_water_wells.csv` |
| Fit Pastas models at the observation wells and run drought scenarios | `drought_response.py` | `results/obs_well_response.csv`, `scenario_dh.csv`, `obs_sim.csv` |
| Compare simulated and observed drought declines | `validate_declines.py` | `results/decline_validation.csv` |
| Screen domestic wells, available drawdown, failures through time | `well_failure.py` | `results/domestic_wells.csv`, `failure_*.csv` |
| Exposure by construction, depth and decade; reserves and rural municipalities | `exposure.py` | `results/exposure_summary.csv`, `rq3_*.csv` |
| Logistic models of exposure (reserves; household income) | `rq3_model.py`, `rq3_income.py` | `results/rq3_logit.csv`, `rq3_income.csv` |
| Figures | `make_method_figure.py`, `make_figures.py` | `figures/` |

`well_failure.py` reads environment variables for the sensitivity analyses: `MARGIN_M` (pump submergence margin, default 1 m), `DECLINE_SCALE`, `K_DONORS` (default 3), `PUMPING=1`, `MIN_YEAR` and `NO_BOUND=1`, with `TAG` as an output-file suffix. Example: `MARGIN_M=2 TAG=_m2 python well_failure.py`.

## Requirements
- Python with pandas, numpy, scipy, matplotlib, geopandas and statsmodels (`requirements.txt`).
- `drought_response.py` uses the observation-well hydrographs, climate drivers and Pastas set-up of the companion repository [Saskatoon-Groundwater-Climate-Drivers](https://github.com/karmok3r/Saskatoon-Groundwater-Climate-Drivers), which must be cloned next to this one. Pastas 2.0 needs pandas < 3, so run it in that repository's Pastas environment:
  ```
  ../Saskatoon-Groundwater-Climate-Drivers/.venv_pastas/Scripts/python drought_response.py
  ```
- `make_figures.py` uses the Natural Earth admin-1 layer from the companion repository (`data/basemap/`).

## Data
- **Driller reports** (`data/raw/wsa_water_wells.csv`): Water Security Agency, public water-well layer, downloaded 30 October 2026 with `fetch_wells.py`. Depths and levels are in feet as reported. The agency notes that the reports are as submitted by drillers and have not been verified. Please credit the Water Security Agency.
- **First Nations reserve boundaries** (`data/raw/reserves/`): Natural Resources Canada, Aboriginal Lands of Canada Legislative Boundaries (Canada Lands Survey System), Saskatchewan, revision of 31 October 2025. Open Government Licence – Canada.
- **Household income** (`data/raw/statcan/sk_csd_income.csv`): median total household income in 2020 for Saskatchewan census subdivisions, extracted from the Statistics Canada 2021 Census Profile (table 98-401-X2021023). Statistics Canada Open Licence. The full profile file is not included; download it from Statistics Canada and place the zip in `data/raw/statcan/` to rebuild the extract.
- **Climate** (ERA5-Land) and **observation-well hydrographs**: see the companion repository.

## Status
Work in progress; a manuscript is in preparation.

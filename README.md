# Germany 2018: Wind–Solar Complementarity and Low-Output Events

A reproducible climate-data prototype that uses the DWD COSMO-REA6 regional reanalysis
to study German wind and solar resource variability, complementarity, and sustained
low-output events during 2018.

**Status: analysis complete.** The repository contains seven executed notebooks: a reader-first
summary (`00`), a July 2019 domain/source-QC pilot (`01`), the full-year 2018 daily analysis
(`02`–`05`), and a November 2018 hourly case study (`06`), plus a small Streamlit demo in `app/`
that reads only the committed compact assets.

> **What this is:** weather-to-capacity-factor proxy modelling and validation against
> observed national generation.  
> **What this is not:** a bankable yield forecast, an operational power model, or an
> electricity-system adequacy study.

## Why it matters

Wind and solar are variable and partly anti-correlated. A resource proxy that tracks
both together can reveal how often the combined resource falls to low levels, and how
sensitive that result is to spatial and temporal representation.

This project asks one focused question:

> **How do German wind and solar resources complement each other, and how sensitive
> are low-output events to (a) area vs. installed-capacity weighting and (b) daily
> versus hourly resolution?**

## The conceptual pipeline

```text
DWD COSMO-REA6 (reanalysis weather)
          │
          ▼
   10 m wind / 2 m T / GHI
          │
          ▼
  weather → capacity factor (CF)
          │
   ┌──────┴───────┐
   ▼              ▼
area-weighted    OPSD MW-weighted
resource proxy   modelled fleet proxy
   │              │
   └──────┬───────┘
          ▼
   national CF proxy series
          │
          ▼
   SMARD observed generation
          │
          ▼
     validation & events
```

Three representations are kept explicitly separate:

| Layer | Meaning |
|-------|---------|
| **Area-weighted** | German land-area resource signal. |
| **Capacity-weighted** | Weather/resource at mapped 2018 installed-capacity locations. |
| **SMARD observed** | National grid-fed generation, normalised by year-end capacity. |

## Key results (2018)

- **Wind and solar CF proxies are anti-correlated on daily time scales**, so a 50/50
  capacity portfolio is smoother than either source alone.
- **Capacity weighting strongly improves wind mean bias** relative to SMARD
  (daily: −0.039 → +0.011; hourly: −0.040 → +0.016) but does not uniformly improve
  MAE/RMSE or hourly wind ramps.
- **Hourly processing reveals diurnal solar structure and wind ramps** that daily
  averages hide. Solar ramp timing is captured well; hourly wind ramps remain weak
  (r ≈ 0.18–0.24).
- **Low-output screening depends on threshold and weighting.** Area weighting finds
  more low-output hours; capacity weighting is more selective. Neither is an
  electricity-shortage assessment.

| Validation | Wind (r / bias / RMSE) | Solar (r / bias / RMSE) |
|---|---|---|
| Daily, area | 0.945 / −0.039 / 0.064 | 0.980 / +0.024 / 0.034 |
| Daily, capacity | 0.940 / +0.011 / 0.067 | 0.983 / +0.024 / 0.033 |
| Hourly (Nov), area | 0.877 / −0.040 / 0.078 | 0.968 / +0.005 / 0.018 |
| Hourly (Nov), capacity | 0.886 / +0.016 / 0.082 | 0.969 / +0.006 / 0.019 |

## Data

- **COSMO-REA6:** DWD regional atmospheric reanalysis (daily and hourly 2D fields).
- **SMARD:** Bundesnetzagentur filters 4067 (onshore wind) and 4068 (PV), CC BY 4.0,
  underlying data from ENTSO-E.
- **OPSD plant register:** German renewable-power-plant locations and MW (used for
  spatial capacity weighting, not as an independent generation record).

## Repository structure

```text
notebooks/00_story_germany_2018.ipynb         <- start here
notebooks/01_cosmo_rea6_domain_reference.ipynb     <- July 2019 pilot/QC reference
notebooks/02_germany_2018_eda.ipynb
notebooks/03_germany_2018_complementarity.ipynb
notebooks/04_germany_2018_validation.ipynb
notebooks/05_germany_2018_capacity_weighting.ipynb
notebooks/06_germany_2018_hourly_case_study.ipynb
app/streamlit_app.py                           <- five-page interactive demo (assets only)
app_assets/2018/                               <- compact pre-computed products
scripts/                                       <- live pipelines and generators
src/cosmo_rea6_pipeline.py                     <- shared pipeline module
```

## Live demo (Streamlit)

The app mirrors Notebook 00: overview, daily proxies vs SMARD, complementarity and low-output
screening with threshold/duration sliders, area vs capacity weighting, and the November 2018 hourly
case study. It performs no downloads and no GRIB decoding.

```bash
pip install -r app/requirements.txt
streamlit run app/streamlit_app.py
```

To deploy on Streamlit Community Cloud, point it at `app/streamlit_app.py` and set
`app/requirements.txt` as the dependency file.

## Reproduce

With the conda environment or `requirements.txt`:

```bash
conda env create -f environment.yml
conda activate renewable-diagnostics
python scripts/run_2018.py
python scripts/smard_download.py
python scripts/run_capacity_weighted_2018.py
python scripts/run_hourly_november_2018.py
python scripts/smard_hourly_download.py
python scripts/validate_hourly_november_2018.py
```

The notebooks can also be opened as pre-executed evidence without rebuilding:

```bash
jupyter lab notebooks/00_story_germany_2018.ipynb
```

## Reproducible notebooks

All executed notebooks are generated from `scripts/create_*.py` so prose and code
stay consistent. To regenerate:

```bash
python scripts/create_story_notebook.py
python scripts/create_hourly_case_study_notebook.py
python -c "from scripts.create_executed_notebooks import domain_notebook; domain_notebook()"
python scripts/create_germany_2018_notebooks.py
python scripts/create_capacity_weighting_notebook.py
```

## Main limitations

- 10 m wind is extrapolated to 100 m with a generic power-law profile and a generic
  turbine curve; actual plant hub heights, wakes, availability, and curtailment are
  not modelled.
- Solar conversion uses a simplified horizontal-panel, temperature-aware model without
  tilt, orientation, snow, or inverter/clipping losses.
- Validation is national and annual (daily) or one-month (hourly), not plant-level or
  multi-year.
- A 50/50 hybrid is a hypothetical equal-rated-capacity portfolio, not Germany's
  actual historical mix.

## Roadmap

- **Later:** extend the hourly pipeline to additional years if a use-case justifies it.

## License

Code is released under the MIT License.  
SMARD data is attributed to **Bundesnetzagentur | SMARD.de, CC BY 4.0**.  
COSMO-REA6 is © Deutscher Wetterdienst (DWD), open-data terms.  
OPSD data is published under the Open Data Commons Open Database License (ODbL).

# Germany 2018: Wind–Solar Complementarity and Low-Output Analysis

A reproducible climate-data analysis of German wind and solar resource variability in 2018 using DWD COSMO-REA6 regional reanalysis, technology-specific capacity-factor proxies, installed-capacity weighting, and SMARD generation observations.
**Live demo:** [Germany 2018 Wind–Solar Complementarity Analysis](https://germany-wind-solar-complementarity-2018.streamlit.app/)

**Status: analysis complete.** The repository contains seven executed notebooks: a reader-first summary (`00`), a July 2019 domain/source-QC pilot (`01`), the full-year 2018 daily analysis (`02`–`05`), and a November 2018 hourly case study (`06`). A five-page Streamlit application in `app/` provides an interactive view of the main results using only compact committed assets.

> **What this is:** weather-to-capacity-factor proxy modelling with comparison against observed national generation.
> **What this is not:** a bankable yield forecast, an operational power model, or an electricity-system adequacy study.

## Why it matters

Wind and solar vary across different temporal and spatial patterns and can exhibit partial anti-correlation. Combining their resource signals can provide a useful diagnostic of complementarity and sustained low-output periods.

This project asks:

> **How do German wind and solar resources complement each other, and how sensitive are low-output diagnostics to (a) area versus installed-capacity weighting and (b) daily versus hourly resolution?**

## Conceptual pipeline

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
     comparison & diagnostics
```

Three representations are kept explicitly separate:

| Representation        | Meaning                                                        |
| --------------------- | -------------------------------------------------------------- |
| **Area-weighted**     | German land-area resource signal.                              |
| **Capacity-weighted** | Weather/resource at mapped 2018 installed-capacity locations.  |
| **SMARD observed**    | National grid-fed generation, normalised by year-end capacity. |

## Key results

* Wind and solar CF proxies show partial anti-correlation on daily time scales, so a hypothetical 50/50 equal-capacity portfolio is smoother than either source alone.
* Capacity weighting substantially improves the wind mean bias relative to SMARD, from **−0.039 to +0.011** in the daily analysis. It does not, however, improve every validation metric.
* Solar shows stronger agreement with observed generation than wind in the analysed daily and hourly data.
* Hourly analysis reveals diurnal solar behaviour and wind ramps that are less visible in daily averages. Solar ramp timing is captured well, while hourly wind-ramp correlation remains weak.
* Low-output diagnostics depend on the threshold, duration definition, and spatial weighting. These are resource-screening diagnostics, not electricity-shortage or adequacy results.

### Validation summary

| Validation             | Wind (r / bias / RMSE) | Solar (r / bias / RMSE) |
| ---------------------- | ---------------------- | ----------------------- |
| Daily, area            | 0.945 / −0.039 / 0.064 | 0.980 / +0.024 / 0.034  |
| Daily, capacity        | 0.940 / +0.011 / 0.067 | 0.983 / +0.024 / 0.033  |
| Hourly (Nov), area     | 0.877 / −0.040 / 0.078 | 0.968 / +0.005 / 0.018  |
| Hourly (Nov), capacity | 0.886 / +0.016 / 0.082 | 0.969 / +0.006 / 0.019  |

The main result is that **moving from area weighting to installed-capacity weighting changes the representation of the renewable fleet and substantially reduces wind mean bias, but does not automatically improve all measures of agreement.**

## Data

* **COSMO-REA6:** DWD regional atmospheric reanalysis used for the weather fields.
* **SMARD:** German electricity-market data from the Bundesnetzagentur, including onshore wind and PV generation. Relevant filters are 4067 (onshore wind) and 4068 (PV). SMARD data are provided under CC BY 4.0, with underlying data from ENTSO-E.
* **OPSD plant register:** German renewable-power-plant locations and installed MW, used for spatial capacity weighting rather than as an independent generation record.

## Repository structure

```text
notebooks/
├── 00_story_germany_2018.ipynb
├── 01_cosmo_rea6_domain_reference.ipynb
├── 02_germany_2018_eda.ipynb
├── 03_germany_2018_complementarity.ipynb
├── 04_germany_2018_validation.ipynb
├── 05_germany_2018_capacity_weighting.ipynb
└── 06_germany_2018_hourly_case_study.ipynb

app/
└── streamlit_app.py

app_assets/2018/
└── compact pre-computed products

scripts/
└── analysis and notebook-generation scripts

src/
└── cosmo_rea6_pipeline.py
```

**Start here:** `notebooks/00_story_germany_2018.ipynb`

The `01` notebook is a July 2019 domain/source-QC pilot. The main analysis covers 2018 in notebooks `02`–`05`, followed by the November 2018 hourly case study in `06`.

## Streamlit app

The Streamlit application presents the main analysis through five pages:

1. Overview
2. Daily proxies vs. SMARD observations
3. Wind–solar complementarity and low-output diagnostics
4. Area vs. installed-capacity weighting
5. November 2018 hourly analysis

The app reads only the compact assets stored in `app_assets/2018/`. It does not download data or decode GRIB files at runtime.

### Run locally

```bash
pip install -r app/requirements.txt
streamlit run app/streamlit_app.py
```

## Reproduce the analysis

The main analysis can be rebuilt using the conda environment or the provided requirements.

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

The executed notebooks can also be opened directly without rebuilding the analysis:

```bash
jupyter lab notebooks/00_story_germany_2018.ipynb
```

## Reproducible notebooks

The executed notebooks are generated from Python scripts so that the notebook code and supporting text can be kept consistent.

Relevant generation scripts include:

```bash
python scripts/create_story_notebook.py
python scripts/create_hourly_case_study_notebook.py
python -c "from scripts.create_executed_notebooks import domain_notebook; domain_notebook()"
python scripts/create_germany_2018_notebooks.py
python scripts/create_capacity_weighting_notebook.py
```

## Method limitations

The capacity-factor calculations are simplified resource proxies and do not represent individual power plants.

### Wind

* 10 m wind is extrapolated to 100 m using a generic power-law profile.
* A generic turbine power curve is used.
* Actual plant hub heights, turbine specifications, wakes, availability, and curtailment are not modelled.

### Solar

* Solar conversion uses a simplified horizontal-panel, temperature-aware model.
* Tilt, orientation, snow, inverter losses, and clipping are not modelled.

### Validation and scope

* Validation is against national generation rather than individual plants.
* The daily analysis covers 2018.
* The hourly analysis is a November 2018 case study, not a full-year hourly validation.
* The 50/50 hybrid is a hypothetical equal-rated-capacity portfolio, not Germany's actual historical generation mix.
* Low-output events are resource-screening diagnostics. Demand, storage, imports, transmission constraints, dispatchable generation, and electricity-system adequacy are not modelled.
* One year of analysis should not be interpreted as a long-term climate climatology.

## License and data attribution

Code is released under the **MIT License**.

* **SMARD:** Bundesnetzagentur | SMARD.de, CC BY 4.0.
* **COSMO-REA6:** © Deutscher Wetterdienst (DWD), according to the applicable open-data terms.
* **OPSD:** Open Data Commons Open Database License (ODbL).

## Roadmap

* Extend the hourly analysis to additional years if a specific research or energy-system use case justifies it.

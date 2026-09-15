# Streamlit app documentation

## What this app is

This is a **portfolio-ready Streamlit demo** for the *Germany 2018 wind–solar
complementarity* analysis. It turns a one-year weather-to-generation research
project into a five-page, interviewer-facing web app that runs without any
raw weather downloads, GRIB decoding, or re-execution of the research pipeline.

The app is intentionally a **thin display layer**:

- It reads only small, committed, pre-computed files under `app_assets/2018/`.
- It performs **no** external downloads.
- It performs **no** GRIB decoding.
- It does **not** run `src/cosmo_rea6_pipeline.py` or the notebooks.
- It has only five lightweight dependencies: `streamlit`, `pandas`, `numpy`,
  `pyarrow`, `plotly`.
- All numbers are either read directly from QC JSON / Parquet assets or are
  simple display-level summaries (correlations, event catalogues, mix sweeps)
  computed from those assets.

## Research question

> Can wind and solar complement each other across Germany, and does the answer
> change when renewable resources are weighted according to the location of the
> existing renewable fleet?

The app answers this using:

- one year of **daily** model/observed capacity-factor data (2018);
- one month of **hourly** model/observed data (November 2018);
- three representations of the national signal: **area-weighted**,
  **capacity-weighted** and **SMARD-observed**.

## Run the app locally

Install the app-only dependencies and start the server:

```bash
pip install -r app/requirements.txt
streamlit run app/streamlit_app.py
```

The default URL is `http://localhost:8501`.

### Local run note (this machine)

On the Windows machine where the project was developed, calling the bare
`streamlit.exe` shim without an activated conda environment exits right after the
startup banner. The reliable launches are:

```bat
conda activate renewable-diagnostics
streamlit run app\streamlit_app.py
```

or directly through the environment interpreter:

```bat
<env-python> -m streamlit run app\streamlit_app.py
```

Keep the terminal open — the server lives inside it.

## Deploy on Streamlit Community Cloud

1. Push the repository to GitHub.
2. Create a new app on [Streamlit Community Cloud](https://streamlit.io/cloud).
3. Set the **main file** to `app/streamlit_app.py`.
4. Set the **requirements file** to `app/requirements.txt`.
5. No `secrets.toml` or environment variables are needed.

The app cannot run if only `streamlit_app.py` is uploaded; it also needs the
`app_assets/2018/` data files at the exact relative paths shown below.

## Files the app reads at runtime (`app_assets/2018/`)

| File | What the app uses it for |
|---|---|
| `cosmo_rea6_dach_2018_daily_country_timeseries.parquet` | Daily, area-weighted CF time series for Germany (wind, solar, hybrid) |
| `cosmo_rea6_de_2018_daily_capacity_weighted.parquet` | Daily, capacity-weighted CF time series for Germany |
| `smard_de_2018_daily.csv` | Observed daily CF from SMARD (national generation ÷ end-2018 capacity) |
| `fig_201811_field_means.png` | Static 2×2 domain map on the Overview (matplotlib/cartopy, 50 m borders + coastlines) |
| `cosmo_rea6_dach_201811_november_field_means.parquet` | Source table for the field-means map; the app only reads the `in_germany` cell count |
| `fig_capacity_maps.png` | Static installed-capacity map on the Overview and Capacity page (log(1 + MW) per cell, `plasma`, fixed 0–5 MW colourbar) |
| `cosmo_rea6_de_2018_capacity_maps.parquet` | Source table for the capacity map (lon/lat + MW per cell for wind and PV) |
| `cosmo_rea6_de_2018_capacity_weighting_qc.json` | OPSD coverage numbers (records, mapped MW, coverage fractions, occupied cells) |
| `cosmo_smard_de_201811_hourly_validation.parquet` | Hourly, aligned model/observed CF time series for November 2018 |
| `cosmo_smard_de_201811_hourly_validation_qc.json` | Hourly validation metrics, ramp metrics, low-output classification, and 1–9 Nov event window info |
| `cosmo_rea6_de_201811_hourly_qc.json` | Hourly-versus-daily processing comparison |

The two PNG maps are rendered offline by `scripts/create_app_map_assets.py` using
matplotlib/cartopy, so the deployed app needs **no geospatial stack**. It only
displays them with `st.image`.

## App architecture

The app is a single file, `app/streamlit_app.py`, structured in four parts:

1. **Imports, constants and shared styling**
   - `ROOT` and `ASSETS` are resolved from `__file__` using `pathlib`. This keeps
     all paths relative and platform-independent.
   - `COLOR`, `LINE`, `LABEL`, `PLOT_LAYOUT` enforce one visual identity across
     every page.
   - A small CSS block in `st.markdown(unsafe_allow_html=True)` controls fonts,
     spacing, KPI cards and section dividers.

2. **Data loading utilities** (`@st.cache_data`)
   - `load_daily()` merges the three daily layers (area, capacity, observed) into
     one wide DataFrame with columns `wind_area`, `wind_capacity`, `wind_observed`,
     etc.
   - `load_hourly()` loads the aligned November 2018 hourly table.
   - `load_json()` reads the QC JSON files.
   - `load_n_germany_cells()` returns the Germany cell count from the
     `november_field_means` asset.

3. **Display helpers**
   - `metrics(observed, model)` computes observed mean, model mean, correlation,
     bias, MAE and RMSE.
   - `find_events(dates, values, threshold, min_days)` identifies runs of
     consecutive days below a hybrid-CF threshold.
   - `line_chart(...)` returns a consistently styled Plotly time-series figure.
   - `metric_table(...)`, `kpi_row(...)`, `section(...)` and `footer()` keep the
     five pages visually coherent.

4. **Five page functions** plus the sidebar router.

## Colour and terminology contract

The same colour language is used on every page:

- Wind = **blue** (`#1f77b4`)
- Solar = **orange** (`#f28e2b`)
- 50/50 hybrid = **green** (`#2ca02c`)
- SMARD observed = **dark grey** (`#222222`)
- Area-weighted = **blue** (conceptually neutral meteorological average)
- Capacity-weighted = **red** (`#e15759`)

The same terminology is enforced everywhere:

- **COSMO model** = weather-derived capacity-factor proxy from DWD COSMO-REA6,
  converted with generic turbine/PV curves. It is **never fitted** to SMARD.
- **Area-weighted** = grid cells weighted by `cos(latitude)`.
- **Capacity-weighted** = grid cells weighted by mapped installed MW from the
  OPSD 2018 plant register.
- **SMARD observed** = measured national grid-fed generation ÷ fixed end-2018
  capacity (52,565 MW onshore wind, 45,277 MW PV).
- **50/50 hybrid** = a hypothetical equal-rated-capacity wind/solar portfolio.
- **Low-output events** = resource-screening diagnostics for that hypothetical
  portfolio, **not** electricity shortages, loss-of-load, or adequacy results.
- **CF** = dimensionless, 0–1.
- **Ramps** = CF change per hour.
- **Severity** = CF-days.

## Pages

### 1. Overview

The Overview is designed as a **2–3 minute, self-contained scientific narrative**
that does not require opening the notebooks.

**Structure (top to bottom):**

1. Title, subtitle and research-question callout.
2. A one-line outline: *From weather to CF proxies → Three views → Capacity
   weighting → Validation → Findings → Complementarity → Low-output → Hourly
   → Takeaway*.
3. A workflow chip strip showing the analysis chain.
4. A `COSMO model = weather-derived CF proxy` definition and the static
   `fig_201811_field_means.png` map (with an expander for map details).
5. Three view cards: area-weighted, capacity-weighted, SMARD observed.
6. Why capacity weighting matters — the static capacity map and a prompt:
   *Does this change the agreement with observed generation?*
7. SMARD validation — side-by-side wind and solar time series, KPI cards and a
   full metrics table inside an expander.
8. Four key-finding cards.
9. A wind-vs-solar seasonal scatter and a brief complementarity pointer.
10. Low-output-period summary and the *not a shortage* warning.
11. Hourly case-study KPIs and the *November only* limitation.
12. A highlighted takeaway box.
13. A limitations caption and the data-source footer.

The page never duplicates the full analyses from the other pages; it only
highlights results and points the reader to the relevant page.

### 2. Daily CF: model vs observed

**Question:** *How well do the weather-derived renewable-resource proxies
reproduce observed generation?*

**What the user can do:**

- Select a technology: wind, solar or 50/50 hybrid (radio buttons).
- Select which layers to plot: area-weighted, capacity-weighted, SMARD observed
  (multiselect, defaults to all three).

**What the app shows:**

- A single large Plotly time series of the selected technology.
- Validation metrics for whichever layers are selected:
  - observed mean CF (KPI)
  - correlation, bias, MAE, RMSE (table)
  - a result callout derived from the metrics
- A monthly-mean table.
- A limitation caption about national means, fixed capacities and missing
  grid/system effects.

The observed mean is shown once as a KPI and as a dashed horizontal line on the
plot. If `hybrid` is selected, the app explains that the observed hybrid is a
normalised reference portfolio, not actual combined generation.

### 3. Complementarity and low-output events

**Question:** *How do wind and solar vary together, and when are both resources
simultaneously low?*

**Structure:**

1. A `NOT_SHORTAGE` warning at the top.
2. Weighting selector (area vs. capacity).
3. **Relationship and seasonal behaviour** — wind-vs-solar scatter by season,
   full-year correlation KPI and monthly-anomaly correlation KPI.
4. **Portfolio mix sensitivity** — a mix sweep: wind share 0–1 in steps of 0.1,
   showing mean, std and 5th-percentile of the resulting portfolio.
5. **Low-output events** — two sliders:
   - Hybrid CF threshold (0.05–0.20, default 0.10)
   - Minimum consecutive days (1–5, default 2)

   A time series of the 50/50 hybrid is shown with a threshold line and shaded
   events. The detected events are listed in a table. KPIs report the event
   count, the longest event and the total severity in CF-days.

6. Result and limitation callouts repeated.

All event detection is done live from the daily DataFrame using the same
`find_events()` routine used in the notebooks.

### 4. Area vs capacity weighting

**Question:** *How does the spatial weighting assumption change the
renewable-resource signal?*

**What the app shows:**

1. A short analysis callout explaining that both layers start from the same
   COSMO CF fields and only the cell weights differ.
2. The static capacity map (`fig_capacity_maps.png`) and a caption explaining the
   log(1 + MW) colour scale.
3. The OPSD coverage table (records, mapped MW, published MW, coverage fraction,
   occupied cells).
4. Wind and solar validation side-by-side (two columns of time series) with
   KPI cards for bias, correlation and RMSE.
5. A full metrics table.
6. A result callout.
7. A low-output comparison table (area vs. capacity: event count, longest event,
   total severity).
8. A limitation callout about the static 2018 weights and ~80% wind coverage.

### 5. November 2018 hourly case study

**Question:** *What does hourly resolution reveal that daily aggregation can
hide?*

**Limitation made prominent at the top:** this is a one-month case study
(November 2018), not a full-year hourly validation.

**What the app shows:**

1. Headline KPI cards for solar/wind hourly and ramp correlations.
2. Technology selector (wind / solar / hybrid) and a window selector
   ("Full month" or "1–9 November" fixed event window).
3. Hourly model-vs-observed time series (interactive Plotly).
4. **Level and timing** validation table.
5. **Ramps** validation table (719 consecutive-hour pairs).
6. **Hourly vs. daily processing routes** comparison table (from
   `cosmo_rea6_de_201811_hourly_qc.json`).
7. **Low-output classification** table for hybrid CF < 0.10, with precision,
   recall, agreement and longest observed/model runs.

The hourly validation uses exactly the same SMARD and COSMO data as the
notebooks; the app only reads the pre-computed hourly validation asset.

## Design notes

- **Template:** `plotly_white` for every chart.
- **Fonts and spacing:** a small CSS block enforces a shared typography hierarchy.
- **Legends:** horizontal, placed below the x-axis.
- **Hover:** `x unified` on time series; rich, explicit hover labels.
- **Y-axes:** `rangemode="tozero"` for capacity factors; tick format `0.2f`.
- **X-axes:** month labels (`%b`) on daily charts; no vertical grid.
- **KPI cards:** styled `st.metric` boxes for key numbers.
- **Static maps:** displayed with `st.image(..., width="stretch")` so no
  `cartopy`, `shapely`, `geopandas` or `matplotlib` is needed at runtime.
- **Footer:** on every page, with data sources and the *no GRIB decoding* note.

## Scientific numbers shown (source of truth)

All numbers are computed from the committed `app_assets/2018/` files. During the
final QA pass, the values were recomputed and matched the app display:

- **Daily wind validation** — area: `r = 0.945`, `bias = -0.039`, `RMSE = 0.064`;
  capacity: `r = 0.941`, `bias = +0.011`, `RMSE = 0.067`.
- **Daily solar validation** — `r ≈ 0.980`, `bias ≈ +0.024`, `RMSE ≈ 0.034`.
- **Complementarity** — full-year wind/solar `r = -0.37`; monthly-anomaly
  `r = -0.18`; 50/50 hybrid std area `0.072`, capacity `0.084`.
- **Low-output events** (threshold 0.10, ≥2 days) — area: 22 events, longest 10
  days, severity 3.05 CF-days; capacity: 16 events, longest 9 days, severity
  1.95 CF-days.
- **Hourly November validation** — solar correlation `0.968`, ramp correlation
  `0.964`; wind correlation area/capacity `0.877 / 0.886`, ramp correlation
  `0.18 / 0.24`.
- **OPSD coverage** — wind ~80%, solar ~97% of published end-2018 capacity.

These numbers are not hard-coded; they are recomputed by the app from the assets
on every run.

## What is *not* in the app

The Streamlit app deliberately does **not**:

- Download or decode GRIB data.
- Run the COSMO reanalysis pipeline.
- Import `src/cosmo_rea6_pipeline.py` or the research `requirements.txt`.
- Use `cartopy`, `shapely`, `xarray`, `netCDF4`, `cfgrib`, etc.
- Claim grid adequacy, electricity shortages, security of supply or an optimised
  national energy mix.

For the full research code, reproducible environment and raw-data pipeline, see
the notebooks (`notebooks/00`–`notebooks/06`), `src/cosmo_rea6_pipeline.py`,
`scripts/`, `environment.yml` and the top-level `requirements.txt`.

## Recommended repository layout for publication

```
germany-wind-solar-complementarity-2018-publish/
├── app/
│   ├── streamlit_app.py      # this app
│   └── requirements.txt      # lightweight runtime deps
├── app_assets/2018/          # compact pre-computed assets (see table above)
├── notebooks/                # 00–06, executed
├── scripts/                  # reproducible pipeline and downloaders
├── src/                      # shared pipeline module
├── docs/
│   ├── STREAMLIT_APP.md      # this file
│   └── NOTEBOOKS_CONCATENATED.md  # notebook-to-streamlit sync document
├── README.md
├── DATA_NOTES_COSMO_REA6.md
├── DATA_STORAGE_AND_DEPLOYMENT.md
├── LICENSE
├── .gitignore
├── environment.yml           # full research env
└── requirements.txt          # full research deps
```

---

**Version notes:**

- Capacity map uses `plasma` and a fixed `0–5 MW` colourbar for `log(1 + MW)`.
- Field-means map is a 2×2 panel with fixed ranges:
  - t2m: −4 to 12 °C
  - wind speed: 0 to 10 m s⁻¹
  - ASWDIR_S: 5 to 45 W m⁻²
  - ASWDIFD_S: 5 to 45 W m⁻²
- Both maps are generated by `scripts/create_app_map_assets.py` from pipeline
  outputs and are committed as PNGs.

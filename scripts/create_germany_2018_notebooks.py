"""Create and execute the Germany-focused 2018 analysis notebooks."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

ROOT = Path(__file__).parent.parent
NOTEBOOK_DIR = ROOT / "notebooks"
NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)


def md(text: str):
    return nbformat.v4.new_markdown_cell(dedent(text).strip())


def code(text: str):
    return nbformat.v4.new_code_cell(dedent(text).strip())


def notebook(cells):
    nb = nbformat.v4.new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {
        "display_name": "renewable-diagnostics",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {"name": "python", "version": "3.12"}
    return nb


def execute_and_save(nb, name: str, timeout: int = 1200):
    path = NOTEBOOK_DIR / name
    ExecutePreprocessor(timeout=timeout, kernel_name="python3").preprocess(
        nb,
        {"metadata": {"path": str(ROOT)}},
    )
    with path.open("w", encoding="utf-8") as handle:
        nbformat.write(nb, handle)
    print(f"Executed notebook -> {path}")


def eda_notebook():
    cells = [
        md("""
        # 02 — Germany 2018 renewable-resource exploratory analysis

        **Question.** What are the characteristic wind and solar patterns in Germany during 2018,
        and what does the raw daily variability look like before we ask any portfolio questions?

        This notebook uses the pre-computed 2018 country time series and the spatial annual / seasonal
        mean maps. It uses the **area-weighted resource proxy**; notebook 04 validates it against
        SMARD, and notebook 05 adds the capacity-weighted modelled fleet proxy. It is Germany-only
        because the simple power-law wind extrapolation is not reliable in the Alps and the literature
        on Dunkelflaute is dominated by the German case.
        """),
        md("""
        ## 1. Concepts used in this notebook

        **Capacity factor (CF)** is actual energy produced divided by the energy that would be
        produced if the same capacity ran at rated power for the whole period:

        $$CF = \\frac{E}{P_{rated}\\,\\Delta t}.$$

        A daily CF of 0.25 means that the fleet produced one quarter of its maximum possible daily
        energy. CF is dimensionless and lies between 0 and 1. Here, `wind_cf` and `solar_cf` are
        **meteorology-derived resource proxies**, not measured fleet CFs: every German grid-cell
        centre is area-weighted equally after a latitude correction, irrespective of where plants
        are actually installed.

        The **50/50 hybrid CF** is

        $$CF_{hybrid}=0.5CF_{wind}+0.5CF_{solar}.$$

        The weights describe shares of total rated portfolio capacity, not shares of annual energy.
        Consequently, hybrid CF is a normalized portfolio output—not an additional physical plant.
        """),
        md("## 2. Load the data"),
        code("""
        %matplotlib inline
        from pathlib import Path
        import sys

        import numpy as np
        import pandas as pd
        import xarray as xr
        import matplotlib.pyplot as plt
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature

        ROOT = Path.cwd()
        sys.path.insert(0, str(ROOT / "src"))
        from cosmo_rea6_pipeline import build_country_masks

        ASSETS = ROOT / "app_assets" / "2018"
        daily_all = pd.read_parquet(ASSETS / "cosmo_rea6_dach_2018_daily_country_timeseries.parquet")
        germany = daily_all[daily_all["country"] == "de"].copy()
        germany = germany.sort_values("time").reset_index(drop=True)

        annual = xr.open_dataset(ASSETS / "cosmo_rea6_dach_2018_annual_means.nc")
        seasonal = xr.open_dataset(ASSETS / "cosmo_rea6_dach_2018_seasonal_means.nc")
        masks = build_country_masks(annual)

        de_cells = int(masks["mask_de"].sum())
        qc = pd.Series({
            "unique dates": germany["time"].nunique(),
            "missing analysis values": int(germany[["wind_cf", "solar_cf", "combined_cf", "ghi"]].isna().sum().sum()),
            "minimum CF": germany[["wind_cf", "solar_cf", "combined_cf"]].min().min(),
            "maximum CF": germany[["wind_cf", "solar_cf", "combined_cf"]].max().max(),
            "German model grid cells": de_cells,
        }, name="pre-analysis check")
        display(qc)
        print(germany[["time", "wind_cf", "solar_cf", "combined_cf", "ghi", "wspd100"]].head())
        """),
        md("""
        **Before interpreting figures.** The 365 unique dates confirm complete non-leap-year coverage;
        zero missing values and CF values inside 0–1 are basic integrity checks. The cell count refers
        to reanalysis grid centres inside Germany, not observations or plants.

        **GHI** (global horizontal irradiance) is direct horizontal plus diffuse horizontal shortwave
        irradiance, in W m⁻². It describes solar energy arriving per second on a horizontal square metre.
        The solar CF converts daily-mean GHI and temperature with a deliberately simple proxy. It omits
        panel tilt, orientation, shading, snow, inverter clipping, and fleet location, so it is not
        measured PV output. Example: a proxy CF of 0.20 means 20% of rated output over that day; it does
        not mean GHI itself was 20%.

        ## 3. Daily distributions and CDFs

        These are the empirical distributions of capacity factors over Germany in 2018. The wind
        distribution is right-skewed, the solar distribution is bounded at zero with a long upper tail,
        and the combined series is less extreme than either on its own.
        """),
        code("""
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        cols = ["wind_cf", "solar_cf", "combined_cf"]
        titles = ["Wind CF", "Solar CF", "Combined 50/50 CF"]

        for ax, col, title in zip(axes, cols, titles):
            ax.hist(germany[col], bins=40, density=True, alpha=0.7, color="steelblue", edgecolor="white")
            ax.set_xlabel("capacity factor")
            ax.set_ylabel("density")
            ax.set_title(f"Germany 2018 — {title}")
            ax.set_xlim(0, 1)
            ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plt.show()
        """),
        md("""
        ## 4. Cumulative distribution functions

        CDFs make the tails visible. The 5th and 10th percentiles describe the low-output tail used
        later for Dunkelflaute screening and portfolio sensitivity.
        """),
        code("""
        fig, ax = plt.subplots(figsize=(8, 5))
        for col, label in zip(["wind_cf", "solar_cf", "combined_cf"], ["wind", "solar", "combined 50/50"]):
            sorted_data = np.sort(germany[col].values)
            cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
            ax.plot(sorted_data, cdf, label=label, linewidth=2)

        ax.set_xlabel("capacity factor")
        ax.set_ylabel("cumulative probability")
        ax.set_title("Germany 2018 — capacity factor CDFs")
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_xlim(0, 0.7)
        plt.tight_layout()
        plt.show()

        q = germany[["wind_cf", "solar_cf", "combined_cf"]].quantile([0.05, 0.1, 0.5, 0.9]).round(3)
        q
        """),
        md("""
        ## 5. Annual cycle with interquartile bands

        Monthly means alone hide the spread. The shaded band shows the middle 50% of daily values.
        """),
        code("""
        germany["month"] = germany["time"].dt.month
        monthly = germany.groupby("month").agg(
            wind_mean=("wind_cf", "mean"),
            wind_q25=("wind_cf", lambda x: x.quantile(0.25)),
            wind_q75=("wind_cf", lambda x: x.quantile(0.75)),
            solar_mean=("solar_cf", "mean"),
            solar_q25=("solar_cf", lambda x: x.quantile(0.25)),
            solar_q75=("solar_cf", lambda x: x.quantile(0.75)),
        ).reset_index()

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(monthly["month"], monthly["wind_mean"], "o-", color="#1f77b4", label="wind mean")
        ax.fill_between(monthly["month"], monthly["wind_q25"], monthly["wind_q75"], color="#1f77b4", alpha=0.2)
        ax.plot(monthly["month"], monthly["solar_mean"], "o-", color="#ff7f0e", label="solar mean")
        ax.fill_between(monthly["month"], monthly["solar_q25"], monthly["solar_q75"], color="#ff7f0e", alpha=0.2)
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
        ax.set_ylabel("capacity factor (dimensionless)")
        ax.set_title("Germany 2018 — monthly mean and interquartile band")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
        """),
        md("""
        **How to read this figure.** Compare each line's seasonal level and the width of its shaded
        band. The line is a monthly mean; the band is day-to-day spread, not uncertainty in the mean.
        Opposite seasonal movement suggests complementarity, but does not prove that both cannot be
        low together.

        ## 6. Persistence: autocorrelation and low-resource run lengths

        Day-to-day persistence is expected in mid-latitude weather, while daily solar means can remain
        correlated across multi-day cloudy periods. Atmospheric blocking can provide physical context
        for some prolonged low-wind episodes, but this notebook does not diagnose circulation regimes
        or attribute individual events to blocking.
        """),
        code("""
        max_lag = 14
        lags = range(1, max_lag + 1)
        wind_ac = [germany["wind_cf"].autocorr(lag=lag) for lag in lags]
        solar_ac = [germany["solar_cf"].autocorr(lag=lag) for lag in lags]

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(lags, wind_ac, "o-", label="wind CF")
        ax.plot(lags, solar_ac, "o-", label="solar CF")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xlabel("lag (days)")
        ax.set_ylabel("autocorrelation")
        ax.set_title("Germany 2018 — day-to-day autocorrelation")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
        """),
        md("""
        ## 7. Low-wind and low-solar spell-length distributions

        A "spell" is a sequence of consecutive days below the 10th percentile of the respective
        variable. The distribution describes persistence in this one-year resource proxy; implications
        for storage cannot be inferred without demand, system, and storage modelling.
        """),
        code("""
        def run_lengths(series, threshold):
            flags = (series < threshold).astype(int)
            runs = []
            current = 0
            for value in flags:
                if value:
                    current += 1
                else:
                    if current > 0:
                        runs.append(current)
                    current = 0
            if current > 0:
                runs.append(current)
            return pd.Series(runs) if runs else pd.Series(dtype=int)

        wind_thr = germany["wind_cf"].quantile(0.1)
        solar_thr = germany["solar_cf"].quantile(0.1)
        wind_runs = run_lengths(germany["wind_cf"], wind_thr)
        solar_runs = run_lengths(germany["solar_cf"], solar_thr)

        fig, ax = plt.subplots(figsize=(8, 4))
        max_len = max(wind_runs.max() if len(wind_runs) else 0, solar_runs.max() if len(solar_runs) else 0, 1)
        bins = range(1, max_len + 2)
        ax.hist(wind_runs, bins=bins, alpha=0.6, label=f"low wind (<{wind_thr:.3f})")
        ax.hist(solar_runs, bins=bins, alpha=0.6, label=f"low solar (<{solar_thr:.3f})")
        ax.set_xlabel("spell length (days)")
        ax.set_ylabel("frequency")
        ax.set_title("Germany 2018 — low-resource spell lengths")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.show()

        spell_stats = pd.DataFrame({
            "low-wind spells": [len(wind_runs), int(wind_runs.max() if len(wind_runs) else 0), float(wind_runs.mean() if len(wind_runs) else 0)],
            "low-solar spells": [len(solar_runs), int(solar_runs.max() if len(solar_runs) else 0), float(solar_runs.mean() if len(solar_runs) else 0)],
        }, index=["count", "max length", "mean length"])
        spell_stats
        """),
        md("""
        ## 8. North vs. south Germany spatial split

        Germany's wind resource is concentrated in the north, the solar resource in the south. We
        split the country at 51°N and report area-weighted annual and seasonal means for both halves.
        """),
        code("""
        def area_mean(da, mask):
            lat = da.latitude
            weights = np.cos(np.deg2rad(lat))
            weights = weights.where(mask)
            weights = weights / weights.sum()
            return float((da * weights).sum().values)

        de_mask = masks["mask_de"]
        north_mask = de_mask & (annual.latitude > 51)
        south_mask = de_mask & (annual.latitude <= 51)

        rows = []
        for ds, label in [(annual, "annual"), (seasonal, "seasonal")]:
            if label == "annual":
                rows.append({
                    "period": "annual",
                    "north_wind_cf": area_mean(ds["wind_cf"], north_mask),
                    "north_solar_cf": area_mean(ds["solar_cf"], north_mask),
                    "south_wind_cf": area_mean(ds["wind_cf"], south_mask),
                    "south_solar_cf": area_mean(ds["solar_cf"], south_mask),
                })
            else:
                for season in ds["season"].values:
                    s = ds.sel(season=season)
                    rows.append({
                        "period": season,
                        "north_wind_cf": area_mean(s["wind_cf"], north_mask),
                        "north_solar_cf": area_mean(s["solar_cf"], north_mask),
                        "south_wind_cf": area_mean(s["wind_cf"], south_mask),
                        "south_solar_cf": area_mean(s["solar_cf"], south_mask),
                    })

        ns_table = pd.DataFrame(rows).round(3).set_index("period")
        ns_table
        """),
        md("""
        ## 9. Seasonal spatial maps of wind and solar

        The maps emphasize the north–south contrast. Only grid-cell centres inside the German
        political boundary are shown.
        """),
        code("""
        def plot_germany_field(ds, var, cmap, title, vmin, vmax):
            fig, ax = plt.subplots(figsize=(8, 7), subplot_kw={"projection": ccrs.PlateCarree()})
            im = ax.pcolormesh(
                ds.longitude, ds.latitude, ds[var],
                shading="auto", cmap=cmap, transform=ccrs.PlateCarree(),
                vmin=vmin, vmax=vmax,
            )
            ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.8, edgecolor="black")
            ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.6, edgecolor="black")
            ax.set_extent([5.5, 17.25, 45.5, 55.25], crs=ccrs.PlateCarree())
            ax.set_title(title)
            unit_label = "capacity factor (dimensionless)" if var.endswith("_cf") else "GHI (W m⁻²)"
            plt.colorbar(im, ax=ax, shrink=0.7, label=unit_label)
            plt.tight_layout()
            plt.show()

        for season in ["DJF", "JJA"]:
            s = seasonal.sel(season=season)
            plot_germany_field(s, "wind_cf", "YlGnBu", f"Germany {season} 2018 — mean wind CF", 0, 0.45)
            plot_germany_field(s, "solar_cf", "YlOrRd", f"Germany {season} 2018 — mean solar CF", 0, 0.4)
        """),
        md("""
        ## 10. Wind vs. solar joint distributions by season

        This is the most direct visualisation of complementarity: do high-wind days coincide with
        low-solar days, and vice versa?
        """),
        code("""
        month_to_season = {
            12: "DJF", 1: "DJF", 2: "DJF",
            3: "MAM", 4: "MAM", 5: "MAM",
            6: "JJA", 7: "JJA", 8: "JJA",
            9: "SON", 10: "SON", 11: "SON",
        }
        germany["season"] = germany["time"].dt.month.map(month_to_season)

        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        for ax, season in zip(axes.flat, ["DJF", "MAM", "JJA", "SON"]):
            sub = germany[germany["season"] == season]
            ax.scatter(sub["wind_cf"], sub["solar_cf"], alpha=0.5, s=25, edgecolor="none")
            ax.set_xlabel("wind CF")
            ax.set_ylabel("solar CF")
            ax.set_title(f"{season} (n={len(sub)})")
            ax.grid(alpha=0.3)

        fig.suptitle("Germany 2018 — daily wind vs solar capacity factor by season", y=1.01)
        plt.tight_layout()
        plt.show()
        """),
        md("""
        ## Key EDA take-aways

        - Wind and solar have opposite seasonal cycles: wind peaks in winter, solar in summer.
        - Daily distributions are non-Gaussian; wind has a long upper tail and both have many near-zero days.
        - Autocorrelation dies out after 3–5 days, but individual low-resource spells can last much longer.
        - The north–south split confirms that Germany's resource is spatially heterogeneous; a single national mean hides strong gradients.
        - The wind–solar scatter shows negative association in all seasons, which is the basis for the complementarity analysis in the next notebook.
        """),
    ]
    execute_and_save(notebook(cells), "02_germany_2018_eda.ipynb", timeout=1200)


def complementarity_notebook():
    cells = [
        md("""
        # 03 — Germany 2018 wind–solar complementarity and Dunkelflaute

        **Question.** How does combining wind and solar change the variability of a hypothetical German
        equal-rated-capacity portfolio, and what do low-output screening events look like?

        We use the area-weighted Germany resource-proxy series to quantify complementarity through
        correlation, variance decomposition, mix sweeps, and an absolute-threshold Dunkelflaute
        catalogue. Notebook 04 validates this baseline; notebook 05 shows how capacity weighting
        changes the fleet proxy and event catalogue.
        """),
        md("""
        ## 1. From technology CF to a hybrid portfolio

        Capacity factor expresses output per unit of installed capacity. For a portfolio with wind
        capacity share $w$ and solar share $1-w$, its normalized output is

        $$CF_{hybrid}(t)=wCF_{wind}(t)+(1-w)CF_{solar}(t).$$

        Thus, a 50/50 portfolio means equal **rated capacities**, not equal annual generation or
        investment. Because wind and solar CF have different means, changing $w$ changes both average
        output and variability. Complementarity is valuable when one resource tends to be available
        while the other is weak: negative covariance then lowers portfolio variance. It cannot,
        however, eliminate days on which both resources are weak.
        """),
        md("## 2. Load data"),
        code("""
        %matplotlib inline
        from pathlib import Path
        import sys

        import numpy as np
        import pandas as pd
        import xarray as xr
        import matplotlib.pyplot as plt
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature

        ROOT = Path.cwd()
        ASSETS = ROOT / "app_assets" / "2018"
        sys.path.insert(0, str(ROOT / "src"))
        from cosmo_rea6_pipeline import build_country_masks

        daily_all = pd.read_parquet(ASSETS / "cosmo_rea6_dach_2018_daily_country_timeseries.parquet")
        g = daily_all[daily_all["country"] == "de"].copy().sort_values("time").reset_index(drop=True)

        seasonal = xr.open_dataset(ASSETS / "cosmo_rea6_dach_2018_seasonal_means.nc")
        annual = xr.open_dataset(ASSETS / "cosmo_rea6_dach_2018_annual_means.nc")

        month_to_season = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
                           6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}
        g["season"] = g["time"].dt.month.map(month_to_season)

        qc = pd.Series({
            "unique dates": g["time"].nunique(),
            "missing CF values": int(g[["wind_cf", "solar_cf", "combined_cf"]].isna().sum().sum()),
            "minimum CF": g[["wind_cf", "solar_cf", "combined_cf"]].min().min(),
            "maximum CF": g[["wind_cf", "solar_cf", "combined_cf"]].max().max(),
            "German model grid cells": int(build_country_masks(annual)["mask_de"].sum()),
        }, name="pre-analysis check")
        display(qc)
        g[["time", "wind_cf", "solar_cf"]].head()
        """),
        md("""
        **Checks and conversion reminder.** We expect 365 unique dates, no missing CF values, and
        CFs in 0–1. Grid-cell count describes the reanalysis mask. GHI is direct horizontal plus
        diffuse horizontal shortwave irradiance (W m⁻²). Its conversion to solar CF is a simplified
        weather-to-power proxy, not a plant model.

        **Area weighting in plain language.** Each cell contributes in proportion to its approximate
        surface area (using cosine latitude), not installed MW. If two equal-area cells have CF 0.10
        and 0.30, their simple area mean is 0.20 even if nearly all plants are in one cell.

        ## 3. Raw and anomaly correlation

        The overall wind–solar correlation is mostly driven by the seasonal cycle. The anomaly
        correlation (daily value minus its monthly mean) isolates weather-scale complementarity.
        """),
        code("""
        g["month"] = g["time"].dt.month
        monthly_mean = g.groupby("month")[["wind_cf", "solar_cf"]].transform("mean")
        g["wind_anom"] = g["wind_cf"] - monthly_mean["wind_cf"]
        g["solar_anom"] = g["solar_cf"] - monthly_mean["solar_cf"]

        corr_total = g["wind_cf"].corr(g["solar_cf"])
        corr_anom = g["wind_anom"].corr(g["solar_anom"])

        season_corr = g.groupby("season").apply(
            lambda x: x["wind_cf"].corr(x["solar_cf"])
        ).round(3)

        corr_table = pd.DataFrame({
            "wind–solar correlation": [np.round(corr_total, 3), np.round(corr_anom, 3), *season_corr.values],
        }, index=["full year", "monthly anomaly", *season_corr.index])
        corr_table
        """),
        md("""
        ## 3. Mix-sweep: what happens as the wind/solar share changes?

        For each wind share from 0 to 1 in 0.1 steps, we compute the resulting hybrid capacity
        factor statistics. The key outputs are the standard deviation and the 5th percentile: a mix
        that lowers both is genuinely complementary.
        """),
        code("""
        wind_shares = np.arange(0.0, 1.01, 0.1)
        mix_rows = []
        for w in wind_shares:
            s = 1.0 - w
            mix = w * g["wind_cf"] + s * g["solar_cf"]
            mix_rows.append({
                "wind_share": w,
                "mean": float(mix.mean()),
                "std": float(mix.std()),
                "q05": float(mix.quantile(0.05)),
                "q10": float(mix.quantile(0.10)),
            })
        mix = pd.DataFrame(mix_rows)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        axes[0].plot(mix["wind_share"], mix["mean"], "o-")
        axes[0].set_ylabel("mean CF")
        axes[0].set_title("mean capacity factor")
        axes[1].plot(mix["wind_share"], mix["std"], "o-", color="crimson")
        axes[1].set_ylabel("standard deviation")
        axes[1].set_title("portfolio variability")
        axes[2].plot(mix["wind_share"], mix["q05"], "o-", color="darkgreen")
        axes[2].set_ylabel("5th-percentile CF")
        axes[2].set_title("worst 5% of days")
        for ax in axes:
            ax.set_xlabel("wind share")
            ax.set_xlim(0, 1)
            ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

        mix
        """),
        md("""
        ## 4. Variance decomposition for the 50/50 mix

        This decomposes the variance of the 50/50 portfolio into the wind variance, solar variance,
        and the covariance term. A negative covariance means the two resources offset each other,
        reducing total variance.
        """),
        code("""
        w, s = 0.5, 0.5
        var_w = g["wind_cf"].var()
        var_s = g["solar_cf"].var()
        cov = g["wind_cf"].cov(g["solar_cf"])
        var_mix = (w**2 * var_w) + (s**2 * var_s) + (2 * w * s * cov)

        decomp = pd.DataFrame({
            "term": ["w²·Var(W)", "s²·Var(S)", "2ws·Cov(W,S)", "Var(50/50 mix)"],
            "value": [w**2 * var_w, s**2 * var_s, 2 * w * s * cov, var_mix],
        })
        decomp["value"] = decomp["value"].round(6)
        decomp
        """),
        md("""
        ## 5. Low-output CF exceedance metric

        These empirical 2018 percentiles show the CF exceeded on 90% and 95% of days. Higher values
        indicate a less extreme low-output tail in this resource proxy; they are not firm capacity,
        capacity credit, or an operational reliability metric.
        """),
        code("""
        firm_rows = []
        for w in wind_shares:
            s = 1.0 - w
            mix = w * g["wind_cf"] + s * g["solar_cf"]
            firm_rows.append({
                "wind_share": w,
                "cf_exceeded_90_pct": float(mix.quantile(0.10)),
                "cf_exceeded_95_pct": float(mix.quantile(0.05)),
            })
        firm = pd.DataFrame(firm_rows)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(firm["wind_share"], firm["cf_exceeded_90_pct"], "o-", label="CF exceeded 90% of days")
        ax.plot(firm["wind_share"], firm["cf_exceeded_95_pct"], "o-", label="CF exceeded 95% of days")
        ax.set_xlabel("wind share")
        ax.set_ylabel("capacity factor (dimensionless)")
        ax.set_title("Germany 2018 — low-output CF exceedance by mix")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

        firm
        """),
        md("""
        ## 6. Dunkelflaute: threshold and duration sensitivity

        We define a Dunkelflaute as consecutive days on which the combined capacity factor is below
        an absolute threshold. The heatmap shows how the number of events and the maximum duration
        change with threshold and minimum duration.
        """),
        code("""
        def find_events(series, threshold, min_duration):
            runs = []
            in_event = False
            start = None
            for date, val in series.items():
                if val < threshold and not in_event:
                    start = date
                    in_event = True
                elif val >= threshold and in_event:
                    end = date - pd.Timedelta(days=1)
                    length = (end - start).days + 1
                    if length >= min_duration:
                        runs.append(length)
                    in_event = False
            if in_event:
                end = series.index[-1]
                length = (end - start).days + 1
                if length >= min_duration:
                    runs.append(length)
            return pd.Series(runs, dtype=int)

        thresholds = [0.05, 0.10, 0.15, 0.20]
        min_durations = [1, 2, 3]
        g_indexed = g.set_index("time")["combined_cf"]

        count_rows = []
        max_rows = []
        for thr in thresholds:
            count_row = {"threshold": thr}
            max_row = {"threshold": thr}
            for dur in min_durations:
                ev = find_events(g_indexed, thr, dur)
                count_row[dur] = len(ev)
                max_row[dur] = int(ev.max()) if len(ev) else 0
            count_rows.append(count_row)
            max_rows.append(max_row)

        count_df = pd.DataFrame(count_rows).set_index("threshold")
        max_df = pd.DataFrame(max_rows).set_index("threshold")

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        im1 = axes[0].imshow(count_df.values, aspect="auto", cmap="YlOrRd")
        axes[0].set_xticks(range(len(min_durations)))
        axes[0].set_xticklabels(min_durations)
        axes[0].set_yticks(range(len(thresholds)))
        axes[0].set_yticklabels(thresholds)
        axes[0].set_xlabel("min duration (days)")
        axes[0].set_ylabel("combined CF threshold")
        axes[0].set_title("number of Dunkelflaute events")
        plt.colorbar(im1, ax=axes[0])

        im2 = axes[1].imshow(max_df.values, aspect="auto", cmap="YlOrRd")
        axes[1].set_xticks(range(len(min_durations)))
        axes[1].set_xticklabels(min_durations)
        axes[1].set_yticks(range(len(thresholds)))
        axes[1].set_yticklabels(thresholds)
        axes[1].set_xlabel("min duration (days)")
        axes[1].set_ylabel("combined CF threshold")
        axes[1].set_title("max event duration (days)")
        plt.colorbar(im2, ax=axes[1])
        plt.tight_layout()
        plt.show()

        count_df
        """),
        md("""
        ## 7. Dunkelflaute catalogue for the 10% / ≥2-day definition

        This is the same definition used in the 2018 metrics. We show the longest events and the full
        table is available as a DataFrame display.
        """),
        code("""
        def make_event_table(series, threshold, min_duration):
            records = []
            in_event = False
            start = None
            for date, val in series.items():
                if val < threshold and not in_event:
                    start = date
                    in_event = True
                elif val >= threshold and in_event:
                    end = date - pd.Timedelta(days=1)
                    length = (end - start).days + 1
                    if length >= min_duration:
                        records.append({
                            "start": start.strftime("%Y-%m-%d"),
                            "end": end.strftime("%Y-%m-%d"),
                            "duration_days": length,
                            "threshold": threshold,
                        })
                    in_event = False
            if in_event:
                end = series.index[-1]
                length = (end - start).days + 1
                if length >= min_duration:
                    records.append({
                        "start": start.strftime("%Y-%m-%d"),
                        "end": end.strftime("%Y-%m-%d"),
                        "duration_days": length,
                        "threshold": threshold,
                    })
            return pd.DataFrame(records)

        events_10_2 = make_event_table(g_indexed, 0.10, 2)
        events_10_2.sort_values("duration_days", ascending=False).head(10)
        """),
        md("""
        ## 8. Zoom into the longest 2018 Dunkelflaute event

        The plot shows wind, solar, and hypothetical 50/50 hybrid CF around the longest screening
        event. The red band marks the event window. No conclusion about electricity shortage or the
        need for storage, imports, or backup can be drawn without a power-system model.
        """),
        code("""
        longest = events_10_2.sort_values("duration_days", ascending=False).iloc[0]
        start, end = pd.Timestamp(longest["start"]), pd.Timestamp(longest["end"])
        zoom = g_indexed.loc[start - pd.Timedelta(days=5) : end + pd.Timedelta(days=5)]

        fig, ax = plt.subplots(figsize=(11, 4))
        ax.plot(zoom.index, g.set_index("time").loc[zoom.index, "wind_cf"], "o-", label="wind CF", markersize=4)
        ax.plot(zoom.index, g.set_index("time").loc[zoom.index, "solar_cf"], "o-", label="solar CF", markersize=4)
        ax.plot(zoom.index, zoom, "o-", color="green", label="combined 50/50 CF", markersize=4)
        ax.axvline(start, color="red", linestyle="--")
        ax.axvline(end, color="red", linestyle="--")
        ax.fill_between(zoom.index, 0, 1, where=(zoom.index >= start) & (zoom.index <= end), alpha=0.1, color="red")
        ax.set_ylabel("capacity factor (dimensionless)")
        ax.set_title(f"Germany 2018 — longest Dunkelflaute: {start.date()} to {end.date()} ({longest['duration_days']} days)")
        ax.legend(ncol=3)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
        """),
        md("""
        ## 9. Spatial context: the season of the longest event

        The longest event occurred in winter, so the winter-mean wind and GHI maps show the
        background conditions. Red colours highlight the German land area within the DACH window.
        """),
        code("""
        def plot_germany_field(ds, var, cmap, title, vmin, vmax):
            fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={"projection": ccrs.PlateCarree()})
            im = ax.pcolormesh(
                ds.longitude, ds.latitude, ds[var],
                shading="auto", cmap=cmap, transform=ccrs.PlateCarree(),
                vmin=vmin, vmax=vmax,
            )
            ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.8, edgecolor="black")
            ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.6, edgecolor="black")
            ax.set_extent([5.5, 17.25, 45.5, 55.25], crs=ccrs.PlateCarree())
            ax.set_title(title)
            unit_label = "capacity factor (dimensionless)" if var.endswith("_cf") else "GHI (W m⁻²)"
            plt.colorbar(im, ax=ax, shrink=0.7, label=unit_label)
            plt.tight_layout()
            plt.show()

        winter = seasonal.sel(season="DJF")
        plot_germany_field(winter, "wind_cf", "YlGnBu", "Germany DJF 2018 — mean wind CF", 0, 0.5)
        plot_germany_field(winter, "ghi", "plasma", "Germany DJF 2018 — mean GHI", 0, 120)
        """),
        md("""
        ## Key complementarity take-aways

        - The full-year wind–solar correlation is negative, but much of it is the seasonal cycle. The
          weather-scale (monthly-anomaly) correlation is weaker, meaning day-to-day complementarity is
          more modest than the annual cycle suggests.
        - The mix sweep is a **statistical sensitivity analysis for 2018**, not an economic or operational
          optimization. Different wind shares change the mean, variance, and low-output percentiles.
        - Complementarity reduces variability under some hypothetical mixes, but it does not guarantee
          high output when both resources are weak.
        - Dunkelflaute screening events are sensitive to threshold and minimum duration. Under the
          CF < 0.10 / ≥2-day definition, the longest area-weighted 2018 event lasts 10 days. One year
          cannot establish event frequency, rarity, or a climatology.
        """),
    ]
    execute_and_save(notebook(cells), "03_germany_2018_complementarity.ipynb", timeout=1200)


def validation_notebook():
    cells = [
        md("""
        # 04 — Validation and limitations of the Germany 2018 analysis

        **Question.** Are the 2018 Germany results physically plausible, and what is out of scope?

        This notebook validates the area-weighted resource proxy against Bundesnetzagentur SMARD
        generation data. It distinguishes structural agreement (timing and variability) from level bias,
        then demonstrates a simple, explicitly non-operational bias correction. Its area-weighting
        limitation is addressed—but not fully solved—by the OPSD capacity-weighted fleet proxy in
        notebook 05. No data is downloaded during notebook execution.
        """),
        md("""
        ## 1. What is being validated?

        **Capacity factor (CF)** normalizes energy by installed capacity and elapsed time:

        $$CF = \\frac{E}{P_{installed}\\,\\Delta t}.$$

        For example, 1,200 MWh generated by a 200 MW fleet in one day gives
        $1,200/(200\\times24)=0.25$. This permits technologies and fleets of different size to be
        compared on a common 0–1 scale.

        The COSMO-REA6 series is an **area-weighted resource proxy** produced from weather fields and
        simplified conversion curves. SMARD reports actual electricity fed into the German grid.
        They are related but not identical quantities. Validation therefore asks two questions:

        1. Does the proxy reproduce the timing and co-variability of high and low generation (correlation)?
        2. Does it reproduce the level and error magnitude (mean bias, MAE, and RMSE)?

        A discrepancy is not necessarily a reanalysis error: installed-capacity geography, turbine
        design, losses, curtailment, and the use of year-end capacity in the observed denominator all
        contribute.
        """),
        md("## 2. Load data"),
        code("""
        %matplotlib inline
        from pathlib import Path
        import json

        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt

        ROOT = Path.cwd()
        ASSETS = ROOT / "app_assets" / "2018"

        with open(ASSETS / "cosmo_rea6_dach_2018_metrics.json", "r", encoding="utf-8") as f:
            metrics = json.load(f)

        daily_all = pd.read_parquet(ASSETS / "cosmo_rea6_dach_2018_daily_country_timeseries.parquet")
        g = daily_all[daily_all["country"] == "de"].copy().sort_values("time").reset_index(drop=True)

        print(f"Germany records: {len(g)}")
        """),
        md("""
        ## 3. Internal consistency checks

        - Exactly 365 days in 2018 (not a leap year).
        - All capacity factors in [0, 1].
        - No missing Germany values.
        """),
        code("""
        n_days = g.shape[0]
        cf_min = g[["wind_cf", "solar_cf", "combined_cf"]].min().min()
        cf_max = g[["wind_cf", "solar_cf", "combined_cf"]].max().max()
        missing = g[["wind_cf", "solar_cf", "combined_cf"]].isna().sum().sum()

        internal = pd.Series({
            "days": n_days,
            "capacity factor range": f"[{cf_min:.4f}, {cf_max:.4f}]",
            "missing values": missing,
            "status": "PASS" if n_days == 365 and cf_min >= -0.01 and cf_max <= 1.01 and missing == 0 else "FAIL",
        }, name="internal check")
        internal
        """),
        md("""
        ## 3. Plausibility against literature and observations

        The table compares our 2018 Germany area-weighted proxies with independent reference ranges.
        Differences are expected because the proxy is area-weighted, does not include losses/wake,
        and uses daily means.
        """),
        code("""
        de = metrics["countries"]["de"]
        plausibility = pd.DataFrame([
            {
                "metric": "Annual wind CF",
                "our value": f"{de['mean_wind_cf']:.3f}",
                "reference": "SMARD-derived 2018 fleet CF calculated below",
                "verdict": "requires quantitative comparison",
            },
            {
                "metric": "Annual solar CF",
                "our value": f"{de['mean_solar_cf']:.3f}",
                "reference": "SMARD-derived 2018 fleet CF calculated below",
                "verdict": "requires quantitative comparison",
            },
            {
                "metric": "Wind–solar correlation",
                "our value": f"{de['wind_solar_correlation']:.3f}",
                "reference": "Negative expected for seasonal cycle",
                "verdict": "physically consistent",
            },
            {
                "metric": "Max Dunkelflaute duration",
                "our value": f"{metrics['dunkelflaute_events']['by_country']['de']['max_duration']} days",
                "reference": "Published values depend strongly on definition and data period",
                "verdict": "Germany result; test threshold and bias sensitivity",
            },
            {
                "metric": "100-m wind",
                "our value": f"{de['mean_wspd100_m_s']:.2f} m/s",
                "reference": "German onshore ~5–6 m/s at 100 m",
                "verdict": "low, expected for area-weighted incl. weak-wind regions",
            },
        ])
        plausibility
        """),
        md("""
        ## 4. External reference: SMARD actual generation

        **SMARD** is the Bundesnetzagentur (Germany's federal network regulator) national
        electricity-market data portal. It publishes transparent market and power-system records.
        The compact CSV used here was obtained from its day-resolution API with
        `scripts/smard_download.py`; this notebook makes no download. Filters **4067** (onshore wind) and **4068** (photovoltaic) select German
        generation. Attribution/licensing is **Bundesnetzagentur SMARD, CC BY 4.0**.

        The file has `date`, daily onshore-wind and PV generation in MWh, the capacity denominators in
        MW, and the resulting `wind_onshore_cf` and `solar_cf`. Daily MWh is divided by 24 h and by
        published end-of-2018 capacity: 52,565 MW onshore wind and 45,277 MW PV. Thus the aggregation
        is national and daily. There are **365 national daily records per technology**, not grid cells,
        plant records, or plant locations.

        Using year-end capacity slightly understates observed CF because capacity increased during
        2018—especially PV. The comparison is therefore an informative first validation, not a final
        fleet reconstruction. Wind is compared with **onshore wind only**, matching the land-based
        COSMO proxy.
        """),
        code("""
        ref = pd.read_csv(ASSETS / "smard_de_2018_daily.csv", parse_dates=["date"])
        model = g[["time", "wind_cf", "solar_cf"]].rename(columns={"time": "date"})
        model["date"] = pd.to_datetime(model["date"]).dt.tz_localize(None).dt.normalize()
        ref["date"] = pd.to_datetime(ref["date"]).dt.tz_localize(None).dt.normalize()
        comparison = model.merge(ref[["date", "wind_onshore_cf", "solar_cf"]], on="date", suffixes=("_model", "_observed"))

        coverage = pd.Series({
            "model days": len(model),
            "SMARD rows": len(ref),
            "SMARD unique dates": ref["date"].nunique(),
            "SMARD date minimum": ref["date"].min(),
            "SMARD date maximum": ref["date"].max(),
            "duplicate SMARD dates": int(ref["date"].duplicated().sum()),
            "matched days": len(comparison),
            "missing comparison values": int(comparison.isna().sum().sum()),
        }, name="coverage")
        coverage
        """),
        md("""
        ## 5. Daily validation metrics

        - **Correlation ($r$)** measures timing and co-variability. It is insensitive to a constant
          multiplicative bias; high $r$ means high and low days tend to occur together.
        - **Bias** is mean(model − observed). Positive values indicate overestimation.
        - **MAE** is the mean absolute daily error and retains the intuitive CF scale.
        - **RMSE** penalizes occasional large errors more strongly than MAE.

        A model can therefore correlate well but remain biased. Both dimensions must be reported.
        """),
        code("""
        def validation_metrics(observed, modelled):
            error = modelled - observed
            return {
                "observed mean CF": observed.mean(),
                "model mean CF": modelled.mean(),
                "correlation": observed.corr(modelled),
                "bias": error.mean(),
                "MAE": error.abs().mean(),
                "RMSE": np.sqrt(np.mean(error**2)),
            }

        validation = pd.DataFrame({
            "wind": validation_metrics(comparison["wind_onshore_cf"], comparison["wind_cf"]),
            "solar": validation_metrics(comparison["solar_cf_observed"], comparison["solar_cf_model"]),
        }).T.round(3)
        validation
        """),
        md("""
        **How to read the 2018 result.** Correlations above 0.9 indicate that the weather-derived
        proxies capture the day-to-day ordering of German generation unusually well for such a simple
        conversion model. Absolute levels tell a different story: area-weighted wind is low relative
        to the installed fleet, while simplified PV is high relative to the SMARD series normalized
        by year-end capacity. This combination is physically understandable—German wind capacity is
        concentrated in windier regions, while PV conversion and the capacity denominator introduce
        different biases—but it confirms that raw proxy CF should not be presented as actual output.
        """),
        code("""
        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        pairs = [
            ("wind_onshore_cf", "wind_cf", "Wind"),
            ("solar_cf_observed", "solar_cf_model", "Solar"),
        ]
        for row, (obs, mod, title) in enumerate(pairs):
            axes[row, 0].plot(comparison["date"], comparison[obs], label="SMARD observed", alpha=0.8)
            axes[row, 0].plot(comparison["date"], comparison[mod], label="COSMO proxy", alpha=0.75)
            axes[row, 0].set_ylabel("daily CF")
            axes[row, 0].set_title(f"{title}: daily time series")
            axes[row, 0].legend()
            axes[row, 0].grid(alpha=0.3)

            axes[row, 1].scatter(comparison[obs], comparison[mod], alpha=0.45, s=18)
            limit = max(comparison[obs].max(), comparison[mod].max())
            axes[row, 1].plot([0, limit], [0, limit], "k--", linewidth=1, label="1:1")
            axes[row, 1].set_xlabel("SMARD observed CF")
            axes[row, 1].set_ylabel("COSMO proxy CF")
            axes[row, 1].set_title(f"{title}: agreement and bias")
            axes[row, 1].legend()
            axes[row, 1].grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
        """),
        md("""
        **How to read the daily panels.** In each left panel, peaks and troughs at the same dates mean
        the proxy gets timing right; a persistent vertical gap means amplitude bias. Wind varies
        through the year, while solar also has a strong seasonal envelope, so compare like seasons.
        In each right panel, every dot is one national day. Dots on the dashed 1:1 line agree exactly;
        dots above it are overestimated and dots below it are underestimated. A narrow diagonal cloud
        indicates good timing, but an offset cloud can still have high correlation. Outliers are single
        days and should not be assigned to one cause without operational and weather investigation.

        ## 6. Monthly error structure

        Monthly aggregation suppresses day-to-day noise and reveals seasonal bias. Persistent monthly
        differences point toward conversion or spatial-weighting limitations; isolated daily errors
        may instead reflect weather timing or operational effects.
        """),
        code("""
        comparison["month"] = comparison["date"].dt.month
        monthly = comparison.groupby("month").agg(
            wind_observed=("wind_onshore_cf", "mean"),
            wind_model=("wind_cf", "mean"),
            solar_observed=("solar_cf_observed", "mean"),
            solar_model=("solar_cf_model", "mean"),
        )
        monthly["wind_bias"] = monthly["wind_model"] - monthly["wind_observed"]
        monthly["solar_bias"] = monthly["solar_model"] - monthly["solar_observed"]
        monthly.round(3)
        """),
        code("""
        fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
        for column, tech in enumerate(["wind", "solar"]):
            axes[0, column].plot(monthly.index, monthly[f"{tech}_observed"], "o-", label="SMARD observed")
            axes[0, column].plot(monthly.index, monthly[f"{tech}_model"], "o-", label="COSMO proxy")
            axes[0, column].set_ylabel("mean daily CF")
            axes[0, column].set_title(f"{tech.title()}: monthly validation")
            axes[0, column].grid(alpha=0.3)
            axes[0, column].legend()
            axes[1, column].bar(monthly.index, monthly[f"{tech}_bias"], color="#6b8eaa")
            axes[1, column].axhline(0, color="black", linewidth=1)
            axes[1, column].set_xticks(range(1, 13))
            axes[1, column].set_xlabel("month")
            axes[1, column].set_ylabel("bias (model − observed CF)")
            axes[1, column].set_title(f"{tech.title()}: monthly bias")
            axes[1, column].grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.show()
        """),
        md("""
        **How to read monthly error.** A positive bar means the proxy is too high; a negative bar means
        it is too low. The upper panels show whether seasonal timing agrees, while the bars expose
        amplitude error. The same sign across many months suggests a persistent denominator,
        conversion-curve, or spatial-weighting issue. A sign that changes by season suggests a
        season-dependent limitation. One isolated month may reflect unusual weather timing or
        operations. These are plausible categories, not proof of a particular cause.

        ## 7. Mean-ratio correction: a diagnostic experiment

        We scale each proxy by observed mean CF divided by model mean CF, clipping to [0, 1]. This
        removes annual level bias but does **not** repair timing, power-curve shape, or spatial
        weighting. Because the correction is fitted and evaluated on the same year, it is an
        **diagnostic sensitivity experiment—same-year calibration/evaluation; not independent validation**.

        The corrected 50/50 hybrid CF gives equal weight to onshore-wind and PV rated capacities:
        $CF_{hybrid}=0.5CF_{wind}+0.5CF_{solar}$. It is not Germany's historical capacity mix and
        should not be interpreted as actual national combined generation.
        """),
        code("""
        wind_scale = comparison["wind_onshore_cf"].mean() / comparison["wind_cf"].mean()
        solar_scale = comparison["solar_cf_observed"].mean() / comparison["solar_cf_model"].mean()
        comparison["wind_cf_corrected"] = (comparison["wind_cf"] * wind_scale).clip(0, 1)
        comparison["solar_cf_corrected"] = (comparison["solar_cf_model"] * solar_scale).clip(0, 1)
        comparison["hybrid_cf_raw"] = 0.5 * comparison["wind_cf"] + 0.5 * comparison["solar_cf_model"]
        comparison["hybrid_cf_corrected"] = 0.5 * comparison["wind_cf_corrected"] + 0.5 * comparison["solar_cf_corrected"]

        correction_summary = pd.Series({
            "wind scale factor": wind_scale,
            "solar scale factor": solar_scale,
            "raw hybrid mean": comparison["hybrid_cf_raw"].mean(),
            "corrected hybrid mean": comparison["hybrid_cf_corrected"].mean(),
        }).round(3)
        correction_summary
        """),
        md("""
        ## 8. Consequence for Dunkelflaute detection

        Here an event is a consecutive run of at least two days with 50/50 hybrid CF below 0.10.
        Comparing raw and corrected proxies shows how sensitive the catalogue is to systematic level
        bias. This threshold is a screening definition; it is not equivalent to unmet demand because
        load, storage, imports, transmission, and dispatchable generation are absent.
        """),
        code("""
        def event_table(dates, values, threshold=0.10, min_duration=2):
            low = pd.Series(values < threshold).reset_index(drop=True)
            dates = pd.Series(pd.to_datetime(dates)).reset_index(drop=True)
            groups = low.ne(low.shift()).cumsum()
            records = []
            for _, idx in low[low].groupby(groups[low]).groups.items():
                idx = list(idx)
                if len(idx) >= min_duration:
                    segment = values.iloc[idx]
                    records.append({
                        "start": dates.iloc[idx[0]].date(),
                        "end": dates.iloc[idx[-1]].date(),
                        "duration_days": len(idx),
                        "mean_cf": segment.mean(),
                        "severity": (threshold - segment).clip(lower=0).sum(),
                    })
            return pd.DataFrame(records)

        raw_events = event_table(comparison["date"], comparison["hybrid_cf_raw"])
        corrected_events = event_table(comparison["date"], comparison["hybrid_cf_corrected"])
        event_comparison = pd.DataFrame({
            "raw proxy": [len(raw_events), raw_events["duration_days"].max(), raw_events["severity"].sum()],
            "mean-corrected proxy": [len(corrected_events), corrected_events["duration_days"].max(), corrected_events["severity"].sum()],
        }, index=["event count", "longest event (days)", "total severity (CF-days)"])
        event_comparison.round(3)
        """),
        code("""
        corrected_events.sort_values("duration_days", ascending=False).head(10)
        """),
        md("""
        **2018 interpretation.** Mean correction does not automatically reduce event frequency. In
        this case it raises wind CF but lowers solar CF because the raw proxy underestimates observed
        wind and overestimates observed PV. More autumn/winter days cross the 10% hybrid threshold,
        so event count can increase even while the longest event shortens. This is precisely why a
        Dunkelflaute catalogue must not be "fixed" by assuming that removal of mean bias will move
        every risk statistic in the same direction. Distribution shape, season, technology mix, and
        threshold all matter.
        """),
        md("""
        ## 9. Limitations and scope

        These are not bugs; they are the boundary of what this prototype can claim. They should be
        stated explicitly in any presentation.

        | Limitation | Consequence |
        |---|---|
        | Daily means in the annual workflow | Notebook 06 supplies a one-month hourly resolution and SMARD validation case study, but not full-year or multi-year hourly coverage. |
        | 10 m → 100 m power law (α = 0.143) | Valid for flat, neutral conditions; unreliable in complex terrain and stable nocturnal layers. |
        | Generic wind power curve | No site-specific turbine, wake, availability, or icing losses. |
        | Simplified solar model | No tilt, azimuth, soiling, snow, or inverter clipping. |
        | Area weighting in this notebook | Notebook 05 subsequently adds OPSD capacity weighting, but incomplete coordinates, static weights, and generic conversion curves remain. |
        | One year (2018) | No interannual variability, no return periods, no trend. |
        | COSMO-REA6 period | Ends in August 2019; cannot assess future climate change. |
        | Absolute threshold 10% | Transparent but arbitrary; a real risk study would test a range of thresholds. |
        """),
        md("""
        ## 10. Conclusion

        Internal checks establish that the pipeline is complete and numerically consistent; the SMARD
        comparison determines whether the resource proxies reproduce actual German fleet behaviour.
        Correlation should be interpreted as temporal skill, while bias and MAE quantify errors in
        level and magnitude. Any reduction in low-output events after mean correction demonstrates
        threshold sensitivity rather than proving that the corrected catalogue is climatologically
        accurate.

        The appropriate scientific label remains **validated methods prototype**: stronger than a
        plausibility-only exercise, but not a production forecast, adequacy assessment, or bankable
        resource study. Notebook 05 supplies capacity-weighted aggregation and shows that it improves
        wind mean bias but not every daily metric. Time-varying installed capacity, improved conversion,
        and independent-year evaluation remain outside this study. Notebook 06 validates one November at
        hourly resolution; it does not provide full-year or multi-year hourly coverage.
        """),
    ]
    execute_and_save(notebook(cells), "04_germany_2018_validation.ipynb", timeout=600)


def main():
    eda_notebook()
    complementarity_notebook()
    validation_notebook()
    print("\nGermany 2018 notebooks generated and executed.")


if __name__ == "__main__":
    main()

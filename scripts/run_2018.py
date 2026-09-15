"""Stream and process the full year 2018 of COSMO-REA6 month-by-month.

This script is designed to keep only one month of raw GRIB data on disk at a
time and to be resumable if a download is interrupted. It produces:
- compressed monthly-mean spatial fields (data/processed/2018/monthly/)
- a daily country-level (DE, AT, CH, DACH) time-series table (Parquet)
- annual and monthly climatological mean maps
- a Dunkelflaute event catalogue using an absolute combined-CF threshold
- a compact set of Streamlit-ready assets under app_assets/2018/
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cosmo_rea6_pipeline import (
    load_monthly_dataset,
    wind_speed,
    extrapolate_10m_to_100m,
    capacity_factor_wind,
    capacity_factor_solar,
    country_spatial_mean,
    save_compressed,
    dunkelflaute_events,
)

YEAR = "2018"
VARIABLES = ["T_2M", "ASWDIR_S", "ASWDIFD_S", "U_10M", "V_10M"]
MONTHS = [f"{YEAR}{m:02d}" for m in range(1, 13)]

WIND_SHARE = 0.5
SOLAR_SHARE = 0.5

ROOT = Path(__file__).parent.parent
PILOT_DIR = ROOT / "data" / "pilot" / YEAR
WORK_DIR = ROOT / "data" / "processed" / YEAR
ASSETS_DIR = ROOT / "app_assets" / YEAR

DUNKELF_THRESHOLD = 0.10
MIN_EVENT_DURATION = 2


def process_month(
    yyyymm: str,
    work_dir: Path,
    pilot_dir: Path = PILOT_DIR,
    evict_raw: bool = True,
) -> dict:
    """Download, process and save a single month. Resumable if outputs exist."""
    monthly_dir = work_dir / "monthly"
    monthly_dir.mkdir(parents=True, exist_ok=True)
    pilot_dir.mkdir(parents=True, exist_ok=True)

    means_path = monthly_dir / f"cosmo_rea6_dach_{yyyymm}_monthly_means.nc"
    ts_path = monthly_dir / f"cosmo_rea6_dach_{yyyymm}_country_timeseries.parquet"
    dkf_path = monthly_dir / f"cosmo_rea6_dach_{yyyymm}_dunkelflaute_freq.nc"

    if means_path.exists() and ts_path.exists() and dkf_path.exists():
        print(f"{yyyymm}: using existing outputs")
        monthly_means = xr.open_dataset(means_path)
        ts_df = pd.read_parquet(ts_path)
        dkf_freq = xr.open_dataset(dkf_path)
        return {
            "yyyymm": yyyymm,
            "country_timeseries": ts_df,
            "monthly_means": monthly_means,
            "dunkelflaute_freq": dkf_freq,
        }

    print(f"\n=== {yyyymm} ===")
    ds = load_monthly_dataset(yyyymm, pilot_dir, variables=VARIABLES)

    # --- capacity factors ---
    wspd10 = wind_speed(ds["u10"], ds["v10"])
    wspd100 = extrapolate_10m_to_100m(wspd10)
    wind_cf = capacity_factor_wind(wspd100)
    ghi = ds["aswdir_s"] + ds["aswdifd_s"]
    ghi = ghi.assign_attrs({"long_name": "surface downward shortwave radiation", "units": "W m-2"})
    solar_cf = capacity_factor_solar(ghi, ds["t2m"])
    combined_cf = WIND_SHARE * wind_cf + SOLAR_SHARE * solar_cf

    mask_ds = ds[["mask_de", "mask_at", "mask_ch", "mask_dach"]]

    # --- country-level daily time series ---
    ts_vars = {}
    for name, da in [
        ("wind_cf", wind_cf),
        ("solar_cf", solar_cf),
        ("combined_cf", combined_cf),
        ("ghi", ghi),
        ("wspd100", wspd100),
    ]:
        ts = country_spatial_mean(da, mask_ds)
        ts = ts.rename(name)
        ts_vars[name] = ts

    ts = xr.merge(ts_vars.values(), compat="override")
    ts_df = ts.to_dataframe().reset_index()
    ts_df["time"] = pd.to_datetime(ts_df["time"])
    ts_df.to_parquet(ts_path, index=False)

    # --- monthly mean maps ---
    monthly_means = xr.Dataset(
        {
            "wind_cf": wind_cf.mean("time"),
            "solar_cf": solar_cf.mean("time"),
            "combined_cf": combined_cf.mean("time"),
            "ghi": ghi.mean("time"),
        }
    )
    save_compressed(monthly_means, means_path)

    # --- spatial Dunkelflaute frequency (monthly quantile definition) ---
    dkf = dunkelflaute_events(wind_cf, solar_cf)
    dkf_spatial_frac = (dkf.sum("time") / dkf.sizes["time"]).rename("dunkelflaute_freq")
    save_compressed(dkf_spatial_frac.to_dataset(), dkf_path)

    if evict_raw:
        for var in VARIABLES:
            grb = pilot_dir / f"{var}.2D.{yyyymm}.DayMean.grb"
            idx = pilot_dir / f"{var}.2D.{yyyymm}.DayMean.grb.idx"
            if grb.exists():
                grb.unlink()
            if idx.exists():
                idx.unlink()
        print(f"{yyyymm}: evicted raw GRIBs")

    return {
        "yyyymm": yyyymm,
        "country_timeseries": ts_df,
        "monthly_means": monthly_means,
        "dunkelflaute_freq": dkf_spatial_frac,
    }


def detect_dunkelflaute_events(df: pd.DataFrame, threshold: float, min_duration: int) -> pd.DataFrame:
    """Find consecutive-day events where country-level combined CF is below threshold."""
    records = []
    for country in sorted(df["country"].unique()):
        sub = df[df["country"] == country].sort_values("time").set_index("time")
        flag = sub["combined_cf"] < threshold

        in_event = False
        start = None
        for date, val in flag.items():
            if val and not in_event:
                start = date
                in_event = True
            elif not val and in_event:
                end = date - pd.Timedelta(days=1)
                duration = (end - start).days + 1
                if duration >= min_duration:
                    records.append({
                        "country": country,
                        "start": start.strftime("%Y-%m-%d"),
                        "end": end.strftime("%Y-%m-%d"),
                        "duration_days": duration,
                        "threshold_combined_cf": threshold,
                    })
                in_event = False

        if in_event:
            end = flag.index[-1]
            duration = (end - start).days + 1
            if duration >= min_duration:
                records.append({
                    "country": country,
                    "start": start.strftime("%Y-%m-%d"),
                    "end": end.strftime("%Y-%m-%d"),
                    "duration_days": duration,
                    "threshold_combined_cf": threshold,
                })

    return pd.DataFrame(records)


def main() -> None:
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    monthly_results = []
    for yyyymm in MONTHS:
        monthly_results.append(process_month(yyyymm, WORK_DIR, PILOT_DIR, evict_raw=True))

    # --- concatenate daily time series ---
    full_ts = pd.concat([r["country_timeseries"] for r in monthly_results], ignore_index=True)
    full_ts = full_ts.sort_values(["time", "country"]).reset_index(drop=True)
    ts_path = WORK_DIR / f"cosmo_rea6_dach_{YEAR}_daily_country_timeseries.parquet"
    full_ts.to_parquet(ts_path, index=False)
    print(f"\nSaved daily country time series -> {ts_path}")

    # --- monthly and annual mean maps ---
    monthly_means = xr.concat(
        [r["monthly_means"].assign_coords(month=r["yyyymm"]).expand_dims("month") for r in monthly_results],
        dim="month",
    )
    monthly_path = WORK_DIR / f"cosmo_rea6_dach_{YEAR}_monthly_means.nc"
    save_compressed(monthly_means, monthly_path)
    print(f"Saved monthly means -> {monthly_path}")

    annual_means = monthly_means.mean("month")
    annual_path = WORK_DIR / f"cosmo_rea6_dach_{YEAR}_annual_means.nc"
    save_compressed(annual_means, annual_path)
    print(f"Saved annual means -> {annual_path}")

    # --- seasonal means ---
    season_map = {
        "DJF": ["201812", "201801", "201802"],
        "MAM": ["201803", "201804", "201805"],
        "JJA": ["201806", "201807", "201808"],
        "SON": ["201809", "201810", "201811"],
    }
    seasonal_list = []
    for season, months in season_map.items():
        try:
            sub = monthly_means.sel(month=months)
        except KeyError:
            continue
        if sub.sizes["month"]:
            seasonal_list.append(
                sub.mean("month").assign_coords(season=season).expand_dims("season")
            )
    seasonal_ds = xr.concat(seasonal_list, dim="season") if seasonal_list else None
    if seasonal_ds is not None:
        seasonal_path = WORK_DIR / f"cosmo_rea6_dach_{YEAR}_seasonal_means.nc"
        save_compressed(seasonal_ds, seasonal_path)
        print(f"Saved seasonal means -> {seasonal_path}")

    # --- metrics per country ---
    metrics = {
        "year": YEAR,
        "wind_share": WIND_SHARE,
        "solar_share": SOLAR_SHARE,
        "dunkelflaute_threshold_combined_cf": DUNKELF_THRESHOLD,
        "dunkelflaute_min_duration_days": MIN_EVENT_DURATION,
    }
    country_metrics = {}
    for country in ["de", "at", "ch", "dach"]:
        sub = full_ts[full_ts["country"] == country]
        country_metrics[country] = {
            "n_days": int(sub.shape[0]),
            "mean_wind_cf": float(np.round(sub["wind_cf"].mean(), 6)),
            "mean_solar_cf": float(np.round(sub["solar_cf"].mean(), 6)),
            "mean_combined_cf": float(np.round(sub["combined_cf"].mean(), 6)),
            "mean_ghi_w_m2": float(np.round(sub["ghi"].mean(), 2)),
            "mean_wspd100_m_s": float(np.round(sub["wspd100"].mean(), 3)),
            "wind_solar_correlation": float(np.round(sub["wind_cf"].corr(sub["solar_cf"]), 4)),
            "max_wind_cf": float(np.round(sub["wind_cf"].max(), 4)),
            "max_solar_cf": float(np.round(sub["solar_cf"].max(), 4)),
            "min_combined_cf": float(np.round(sub["combined_cf"].min(), 4)),
        }
    metrics["countries"] = country_metrics

    # --- Dunkelflaute events ---
    events = detect_dunkelflaute_events(full_ts, DUNKELF_THRESHOLD, MIN_EVENT_DURATION)
    events_path = WORK_DIR / f"cosmo_rea6_dach_{YEAR}_dunkelflaute_events.parquet"
    events.to_parquet(events_path, index=False)
    print(f"Saved Dunkelflaute events -> {events_path}")

    by_country = events.groupby("country").agg(count=("duration_days", "size"), max_duration=("duration_days", "max"))
    metrics["dunkelflaute_events"] = {
        "total_events": int(len(events)),
        "max_duration_days": int(events["duration_days"].max()) if len(events) else 0,
        "by_country": by_country.to_dict(orient="index") if len(events) else {},
    }

    metrics_path = WORK_DIR / f"cosmo_rea6_dach_{YEAR}_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics -> {metrics_path}")

    # --- copy compact assets for the portfolio app ---
    full_ts.to_parquet(ASSETS_DIR / f"cosmo_rea6_dach_{YEAR}_daily_country_timeseries.parquet", index=False)
    events.to_parquet(ASSETS_DIR / f"cosmo_rea6_dach_{YEAR}_dunkelflaute_events.parquet", index=False)
    save_compressed(annual_means, ASSETS_DIR / f"cosmo_rea6_dach_{YEAR}_annual_means.nc")
    if "seasonal_ds" in dir():
        save_compressed(seasonal_ds, ASSETS_DIR / f"cosmo_rea6_dach_{YEAR}_seasonal_means.nc")
    with open(ASSETS_DIR / f"cosmo_rea6_dach_{YEAR}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Copied assets -> {ASSETS_DIR}")

    print("\n2018 full-year processing complete.")


if __name__ == "__main__":
    main()

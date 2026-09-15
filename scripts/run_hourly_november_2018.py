"""Build compact hourly COSMO-REA6 Germany diagnostics for November 2018 (v1.3)."""
from __future__ import annotations

import bz2
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
if "--self-test" not in sys.argv:
    from cosmo_rea6_pipeline import (  # noqa: E402
        build_country_masks,
        capacity_factor_solar,
        capacity_factor_wind,
        extrapolate_10m_to_100m,
        standardize_names,
        subset_dach,
        wind_speed,
    )

YYYYMM = "201811"
EXPECTED_HOURS = 720
VARS = ("T_2M", "ASWDIR_S", "ASWDIFD_S", "U_10M", "V_10M")
CANONICAL = {"T_2M": "t2m", "ASWDIR_S": "aswdir_s", "ASWDIFD_S": "aswdifd_s", "U_10M": "u10", "V_10M": "v10"}
BASE_URL = "https://opendata.dwd.de/climate_environment/REA/COSMO_REA6/hourly/2D"
URLS = {v: f"{BASE_URL}/{v}/{v}.2D.{YYYYMM}.grb.bz2" for v in VARS}
CACHE = ROOT / "data" / "hourly_cache" / YYYYMM
SUBSETS = CACHE / "subsets"
WEIGHTS = ROOT / "data" / "capacity_weighting" / "cosmo_rea6_de_capacity_weights_2018.nc"
DETAIL = ROOT / "data" / "processed" / "2018"
ASSETS = ROOT / "app_assets" / "2018"
OUTPUT_NAME = "cosmo_rea6_de_201811_hourly.parquet"
QC_NAME = "cosmo_rea6_de_201811_hourly_qc.json"
TIME_START, TIME_END = pd.Timestamp("2018-11-01 01:00:00"), pd.Timestamp("2018-12-01 00:00:00")
RANGES = {"t2m": (180.0, 330.0), "aswdir_s": (-10.0, 1500.0), "aswdifd_s": (-10.0, 1000.0), "u10": (-100.0, 100.0), "v10": (-100.0, 100.0)}


def download_atomic(url: str, target: Path) -> bool:
    """Download resumably to .part and atomically rename; return whether this invocation created it."""
    if target.exists():
        return False
    part = target.with_suffix(target.suffix + ".part")
    target.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "curl.exe", "-L", "--fail", "--retry", "20", "--retry-all-errors",
        "--continue-at", "-", "--speed-limit", "1024", "--speed-time", "30",
        "--output", str(part), url,
    ]
    subprocess.run(command, check=True)
    if part.stat().st_size == 0:
        raise IOError("empty response")
    part.replace(target)
    return True


def decompress_atomic(source: Path, target: Path) -> bool:
    if target.exists():
        return False
    part = target.with_suffix(target.suffix + ".part")
    try:
        with bz2.open(source, "rb") as compressed, part.open("wb") as raw:
            shutil.copyfileobj(compressed, raw, length=8 * 1024 * 1024)
        if part.stat().st_size == 0:
            raise IOError(f"decompression produced empty file: {source}")
        part.replace(target)
        return True
    except Exception:
        part.unlink(missing_ok=True)
        raise


def normalize_time(ds: xr.Dataset) -> xr.Dataset:
    if "valid_time" in ds.coords and ds["valid_time"].ndim == 1:
        dim = ds["valid_time"].dims[0]
        ds = ds.assign_coords({dim: pd.DatetimeIndex(ds["valid_time"].values)}).rename({dim: "time"}) if dim != "time" else ds.assign_coords(time=pd.DatetimeIndex(ds["valid_time"].values))
    if "time" not in ds.coords:
        raise ValueError("cfgrib dataset has no time/valid_time coordinate")
    index = pd.DatetimeIndex(ds.time.values)
    if index.tz is not None:
        index = index.tz_convert("UTC").tz_localize(None)
    return ds.assign_coords(time=index).sel(time=slice(TIME_START, TIME_END))


def validate_time(ds: xr.Dataset, label: str) -> dict:
    index = pd.DatetimeIndex(ds.time.values)
    duplicates = int(index.duplicated().sum())
    missing = len(pd.date_range(TIME_START, TIME_END, freq="h").difference(index))
    if len(index) != EXPECTED_HOURS or duplicates or missing or not index.is_monotonic_increasing:
        raise ValueError(f"{label}: invalid hourly axis records={len(index)}, duplicates={duplicates}, missing={missing}")
    return {"records": len(index), "start_utc": str(index[0]), "end_utc": str(index[-1]), "duplicates": duplicates, "missing_timestamps": missing, "monotonic": True}


def variable_metadata(da: xr.DataArray) -> dict:
    wanted = ("GRIB_shortName", "GRIB_name", "GRIB_stepType", "GRIB_stepUnits", "GRIB_typeOfLevel", "units", "long_name", "standard_name")
    return {key: str(da.attrs[key]) for key in wanted if key in da.attrs}


def extract_one(variable: str) -> tuple[Path, dict]:
    subset_path = SUBSETS / f"{variable}.2D.{YYYYMM}.dach.nc"
    if subset_path.exists():
        with xr.open_dataset(subset_path) as cached:
            qc = validate_time(cached, variable)
            da = cached[CANONICAL[variable]]
            qc.update({"metadata": variable_metadata(da), "resumed_subset": True, "range": [float(da.min()), float(da.max())]})
        return subset_path, qc
    archive = CACHE / f"{variable}.2D.{YYYYMM}.grb.bz2"
    grib = CACHE / f"{variable}.2D.{YYYYMM}.grb"
    created_archive = download_atomic(URLS[variable], archive)
    created_grib = decompress_atomic(archive, grib)
    success = False
    try:
        opened = xr.open_dataset(grib, engine="cfgrib", decode_timedelta=False, backend_kwargs={"indexpath": str(grib) + ".idx"})
        ds = subset_dach(standardize_names(normalize_time(opened)))
        name = CANONICAL[variable]
        if name not in ds:
            raise KeyError(f"{variable}: standardize_names did not produce {name}; got {list(ds.data_vars)}")
        masks = build_country_masks(ds)
        da = ds[name]
        qc = validate_time(ds, variable)
        german = da.where(masks.mask_de)
        missing_de = int(german.isnull().where(masks.mask_de, False).sum().item())
        lo, hi = map(float, RANGES[name])
        actual = (float(da.min()), float(da.max()))
        if missing_de:
            raise ValueError(f"{variable}: {missing_de} missing values in German mask")
        if actual[0] < lo or actual[1] > hi:
            raise ValueError(f"{variable}: implausible range {actual}, expected within {(lo, hi)}")
        compact = xr.Dataset({name: da.astype("float32"), "mask_de": masks.mask_de})
        compact[name].attrs.update(da.attrs)
        subset_path.parent.mkdir(parents=True, exist_ok=True)
        part = subset_path.with_suffix(".nc.part")
        compact.to_netcdf(part, engine="netcdf4", encoding={name: {"zlib": True, "complevel": 4, "dtype": "float32"}})
        with xr.open_dataset(part) as check:
            validate_time(check, variable)
            if int(check[name].isnull().where(check.mask_de, False).sum()) != 0:
                raise ValueError(f"{variable}: persisted subset failed German-mask validation")
        part.replace(subset_path)
        qc.update({"metadata": variable_metadata(da), "resumed_subset": False, "range": list(actual), "missing_german_mask_values": missing_de})
        success = True
        opened.close()
        return subset_path, qc
    finally:
        if success:
            if created_grib:
                grib.unlink(missing_ok=True)
                Path(str(grib) + ".idx").unlink(missing_ok=True)
            if created_archive:
                archive.unlink(missing_ok=True)


def area_weighted_mean(field: xr.DataArray, mask: xr.DataArray) -> xr.DataArray:
    selected = field.where(mask)
    weights = np.cos(np.deg2rad(field.latitude)).where(np.isfinite(selected))
    return (selected * weights).sum(("y", "x")) / weights.sum(("y", "x"))


def capacity_weighted_mean(field: xr.DataArray, capacity: xr.DataArray) -> tuple[xr.DataArray, dict]:
    field, capacity = xr.align(field, capacity, join="inner")
    positive = capacity.where(capacity > 0)
    valid = positive.where(np.isfinite(field))
    denominator = valid.sum(("y", "x"))
    if bool((denominator <= 0).any()):
        raise ValueError("no finite positive capacity weights for one or more hours")
    result = (field * valid).sum(("y", "x")) / denominator
    total = float(positive.sum())
    coverage = denominator / total
    return result, {"mapped_capacity_mw": total, "finite_weight_coverage_min": float(coverage.min()), "finite_weight_coverage_mean": float(coverage.mean()), "occupied_cells": int((positive > 0).sum())}


def run_records(times, values, threshold: float = 0.10) -> list[dict]:
    frame = pd.DataFrame({"time": pd.to_datetime(times), "value": np.asarray(values)})
    low = frame.value < threshold
    groups = low.ne(low.shift(fill_value=False)).cumsum()
    records = []
    for _, segment in frame[low].groupby(groups[low]):
        records.append({"start_utc": str(segment.time.iloc[0]), "end_utc": str(segment.time.iloc[-1]), "duration_hours": int(len(segment)), "mean_cf": float(segment.value.mean()), "severity_cf_hours": float((threshold - segment.value).sum())})
    return records


def comparison_metrics(model: pd.Series, reference: pd.Series) -> dict:
    pair = pd.concat([model, reference], axis=1, sort=False).dropna()
    error = pair.iloc[:, 0] - pair.iloc[:, 1]
    return {"n_days": len(pair), "correlation": float(pair.iloc[:, 0].corr(pair.iloc[:, 1])), "bias": float(error.mean()), "mae": float(error.abs().mean()), "rmse": float(np.sqrt(np.mean(error**2)))}


def compare_daily(hourly: pd.DataFrame) -> dict:
    interval_dates = (hourly["time"] - pd.Timedelta(nanoseconds=1)).dt.normalize()
    derived = hourly.drop(columns="time").groupby(interval_dates).mean(numeric_only=True)
    area = pd.read_parquet(ASSETS / "cosmo_rea6_dach_2018_daily_country_timeseries.parquet")
    area = area[area.country.eq("de")].assign(time=lambda x: pd.to_datetime(x.time).dt.normalize()).set_index("time")
    capacity = pd.read_parquet(ASSETS / "cosmo_rea6_de_2018_daily_capacity_weighted.parquet").assign(time=lambda x: pd.to_datetime(x.time).dt.normalize()).set_index("time")
    result = {}
    for tech in ("wind", "solar", "hybrid"):
        old_area = {"wind": "wind_cf", "solar": "solar_cf", "hybrid": "combined_cf"}[tech]
        old_capacity = {"wind": "wind_cf_capacity_weighted", "solar": "solar_cf_capacity_weighted", "hybrid": "combined_cf_capacity_weighted"}[tech]
        result[f"{tech}_area"] = comparison_metrics(derived[f"{tech}_cf_area"], area[old_area])
        result[f"{tech}_capacity"] = comparison_metrics(derived[f"{tech}_cf_capacity"], capacity[old_capacity])
    return {"metrics": result, "explanation": "Hourly-derived daily CF means need not match v1.2: conversion before temporal averaging is nonlinear, hourly and daily products may have different averaging/time conventions, and grid processing can differ."}


def ramp_metrics(series: pd.Series) -> dict:
    ramps = series.diff().dropna()
    return {"mean_absolute_hourly_ramp": float(ramps.abs().mean()), "max_up_ramp": float(ramps.max()), "max_down_ramp": float(ramps.min()), "p95_absolute_ramp": float(ramps.abs().quantile(.95))}


def evict_full_cache() -> None:
    for variable in VARS:
        stem = CACHE / f"{variable}.2D.{YYYYMM}.grb"
        for path in (stem, Path(str(stem) + ".idx"), Path(str(stem) + ".part"), Path(str(stem) + ".bz2"), Path(str(stem) + ".bz2.part")):
            path.unlink(missing_ok=True)


def _synthetic_tests() -> None:
    times = pd.date_range("2018-11-01", periods=7, freq="h")
    runs = run_records(times, [0.2, .05, .04, .2, .01, .02, .03])
    assert [x["duration_hours"] for x in runs] == [2, 3]
    field = xr.DataArray([[[1., np.nan], [3., 5.]], [[2., 4.], [6., 8.]]], dims=("time", "y", "x"))
    weights = xr.DataArray([[1., 100.], [3., 0.]], dims=("y", "x"))
    got, qc = capacity_weighted_mean(field, weights)
    np.testing.assert_allclose(got, [2.5, 420 / 104])
    assert qc["finite_weight_coverage_min"] < qc["finite_weight_coverage_mean"]


def main() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    ASSETS.mkdir(parents=True, exist_ok=True)
    DETAIL.mkdir(parents=True, exist_ok=True)
    source_qc, paths = {}, []
    for variable in VARS:
        print(f"Extracting {variable} (only this full archive is retained during extraction)")
        path, source_qc[variable] = extract_one(variable)
        paths.append(path)
    opened = [xr.open_dataset(path) for path in paths]
    try:
        ds = xr.merge(opened, join="exact", compat="override")
        validate_time(ds, "merged")
        masks = xr.Dataset({"mask_de": ds.mask_de})
        weights = xr.open_dataset(WEIGHTS)
        wspd100 = extrapolate_10m_to_100m(wind_speed(ds.u10, ds.v10))
        wind_cf = capacity_factor_wind(wspd100)
        ghi = ds.aswdir_s + ds.aswdifd_s
        solar_cf = capacity_factor_solar(ghi, ds.t2m)
        area = {name: area_weighted_mean(field, masks.mask_de) for name, field in {"wind_cf": wind_cf, "solar_cf": solar_cf, "ghi": ghi, "wspd100": wspd100}.items()}
        wind_capacity, wind_coverage = capacity_weighted_mean(wind_cf, weights.wind_capacity_mw)
        solar_capacity, solar_coverage = capacity_weighted_mean(solar_cf, weights.solar_capacity_mw)
        ghi_wind, _ = capacity_weighted_mean(ghi, weights.wind_capacity_mw)
        ghi_solar, _ = capacity_weighted_mean(ghi, weights.solar_capacity_mw)
        wspd_capacity, _ = capacity_weighted_mean(wspd100, weights.wind_capacity_mw)
        result = pd.DataFrame({"time": pd.DatetimeIndex(ds.time.values), "wind_cf_area": area["wind_cf"].values, "solar_cf_area": area["solar_cf"].values, "wind_cf_capacity": wind_capacity.values, "solar_cf_capacity": solar_capacity.values, "ghi_area_wm2": area["ghi"].values, "ghi_capacity_solar_wm2": ghi_solar.values, "ghi_capacity_wind_wm2": ghi_wind.values, "wspd100_area_ms": area["wspd100"].values, "wspd100_capacity_ms": wspd_capacity.values})
        result["hybrid_cf_area"] = .5 * (result.wind_cf_area + result.solar_cf_area)
        result["hybrid_cf_capacity"] = .5 * (result.wind_cf_capacity + result.solar_cf_capacity)
        validate_time(xr.Dataset(coords={"time": result.time}), "output")
        if result.drop(columns="time").isna().any().any():
            raise ValueError("final compact series contains missing values")
        for column in result.filter(regex="_cf_"):
            if not result[column].between(0, 1).all():
                raise ValueError(f"{column} outside [0, 1]")
        for target in (DETAIL / OUTPUT_NAME, ASSETS / OUTPUT_NAME):
            result.to_parquet(target, index=False)
        low = {}
        window = result.time.between("2018-11-01", "2018-11-09 23:00")
        for weighting in ("area", "capacity"):
            col = f"hybrid_cf_{weighting}"
            runs = run_records(result.time, result[col])
            window_runs = run_records(result.loc[window, "time"], result.loc[window, col])
            low[weighting] = {"threshold": 0.10, "low_hours": int((result[col] < .10).sum()), "runs": runs, "longest_run_hours": max((r["duration_hours"] for r in runs), default=0), "event_window_2018_11_01_through_09": {"low_hours": int((result.loc[window, col] < .10).sum()), "runs": window_runs}}
        cf_cols = list(result.filter(regex="_cf_").columns)
        evict_full_cache()
        qc = {"version": "1.3", "product": "DWD COSMO-REA6 hourly 2D monthly archives", "source_urls": URLS, "archive_fields": source_qc, "time_semantics": {"timezone": "UTC stored as timezone-naive valid-time timestamps", "archive_axis": "720 valid times from 2018-11-01 01:00 through 2018-12-01 00:00; for daily comparison, each timestamp is assigned to the one-hour interval ending at that time", "interpretation": "Each hourly CF is treated as representative of the hour ending at its valid-time label.", "radiation_caution": "Archive fields are hourly values. Instantaneous versus interval-average/accumulation semantics follow captured GRIB stepType metadata and are not assumed in advance."}, "records": len(result), "date_range_utc": [str(result.time.min()), str(result.time.max())], "cadence": "1 hour", "missing_values": int(result.isna().sum().sum()), "duplicate_timestamps": int(result.time.duplicated().sum()), "cf_ranges": {c: [float(result[c].min()), float(result[c].max())] for c in cf_cols}, "mask_coverage": {"german_grid_cells": int(ds.mask_de.sum())}, "weight_coverage": {"wind": wind_coverage, "solar": solar_coverage}, "daily_vs_existing_daily": compare_daily(result), "ramp_metrics": {c: ramp_metrics(result[c]) for c in cf_cols}, "low_output_screening": low, "event_window_note": "The original daily low-output event began 2018-10-31, outside this selected hourly archive. Hourly runs are screening intervals, not electricity shortages and not directly comparable with daily event counts.", "storage_cleanup": {"policy": "After all compact outputs and resumable DACH subsets validate, full compressed/decompressed hourly archives and index/partial files for the five known variables are evicted.", "remaining_full_archives": [str(p) for p in CACHE.glob("*.grb*") if not str(p).endswith(".idx")], "temporary_subsets_retained_for_resume": [str(p) for p in paths]}}
        for target in (DETAIL / QC_NAME, ASSETS / QC_NAME):
            target.write_text(json.dumps(qc, indent=2, allow_nan=False), encoding="utf-8")
        print(f"Saved {ASSETS / OUTPUT_NAME}\nSaved {ASSETS / QC_NAME}")
    finally:
        for item in opened:
            item.close()


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        _synthetic_tests()
        print("Synthetic run-detection and finite-weight renormalization tests passed.")
    else:
        main()

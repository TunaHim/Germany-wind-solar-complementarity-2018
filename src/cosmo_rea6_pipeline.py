"""COSMO-REA6 wind + solar + complementarity pipeline.

This module provides a lightweight, reproducible way to turn COSMO-REA6
reanalysis GRIB files into renewable-energy diagnostics over the DACH region.
It is intentionally simple (daily means, 10 m wind extrapolated to 100 m,
simple parametric capacity factors) so it can be validated quickly and
replaced with more sophisticated models later.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
import xarray as xr
from cartopy.io import shapereader
from shapely import contains_xy
from shapely.ops import unary_union

BASE_URL = "https://opendata.dwd.de/climate_environment/REA/COSMO_REA6"

# Daily-mean 2D variables used by the pipeline.
VARIABLES = {
    "T_2M": "daily/2D/T_2M/T_2M.2D.{yyyymm}.DayMean.grb",
    "ASWDIR_S": "daily/2D/ASWDIR_S/ASWDIR_S.2D.{yyyymm}.DayMean.grb",
    "ASWDIFD_S": "daily/2D/ASWDIFD_S/ASWDIFD_S.2D.{yyyymm}.DayMean.grb",
    "U_10M": "daily/2D/U_10M/U_10M.2D.{yyyymm}.DayMean.grb",
    "V_10M": "daily/2D/V_10M/V_10M.2D.{yyyymm}.DayMean.grb",
}

# Pre-crop bounds enclosing Germany, Austria, and Switzerland. Austria reaches
# approximately 17.15°E, so a 17°E boundary would still clip its eastern edge.
LON_BOUNDS = (5.5, 17.25)
LAT_BOUNDS = (45.5, 55.25)
DACH_COUNTRIES = ("Germany", "Austria", "Switzerland")


def download_var(
    var_name: str,
    yyyymm: str,
    pilot_dir: Path,
    base_url: str = BASE_URL,
) -> Path:
    """Download a single COSMO-REA6 daily-mean GRIB file if not cached."""
    remote_path = VARIABLES[var_name].format(yyyymm=yyyymm)
    local_path = pilot_dir / Path(remote_path).name
    url = f"{base_url}/{remote_path}"
    if not local_path.exists():
        print(f"Downloading {url}")
        urlretrieve(url, local_path)
    return local_path


def open_cosmo_rea6(path: Path) -> xr.Dataset:
    """Open a COSMO-REA6 GRIB file with cfgrib, filtering to a single variable."""
    return xr.open_dataset(path, engine="cfgrib", decode_timedelta=False)


def subset_dach(ds: xr.Dataset) -> xr.Dataset:
    """Pre-crop a native y/x dataset to bounds enclosing all three DACH countries."""
    lat = ds.latitude
    lon = ds.longitude
    bounds_mask = (
        (lon >= LON_BOUNDS[0])
        & (lon <= LON_BOUNDS[1])
        & (lat >= LAT_BOUNDS[0])
        & (lat <= LAT_BOUNDS[1])
    )
    return ds.where(bounds_mask, drop=True)


def load_dach_geometries() -> dict[str, object]:
    """Load Natural Earth 1:50m country geometries for Germany, Austria, Switzerland."""
    path = shapereader.natural_earth(
        resolution="50m",
        category="cultural",
        name="admin_0_countries",
    )
    records = shapereader.Reader(path).records()
    geometries = {
        record.attributes["ADMIN"]: record.geometry
        for record in records
        if record.attributes["ADMIN"] in DACH_COUNTRIES
    }
    missing = set(DACH_COUNTRIES) - set(geometries)
    if missing:
        raise ValueError(f"Natural Earth is missing DACH geometries: {sorted(missing)}")
    return geometries


def build_dach_mask(ds: xr.Dataset) -> xr.DataArray:
    """Return a Boolean mask selecting grid-cell centres inside DACH borders."""
    geometry = unary_union(list(load_dach_geometries().values()))
    values = contains_xy(geometry, ds.longitude.values, ds.latitude.values)
    return xr.DataArray(
        values,
        coords={"y": ds.y, "x": ds.x},
        dims=("y", "x"),
        name="dach_mask",
        attrs={
            "long_name": "Germany, Austria, and Switzerland political-boundary mask",
            "source": "Natural Earth admin_0_countries, 1:50m",
        },
    )


def build_country_masks(ds: xr.Dataset) -> xr.Dataset:
    """Return a Dataset with one Boolean mask per DACH country and the union."""
    geometries = load_dach_geometries()
    masks = {}
    for name in DACH_COUNTRIES:
        values = contains_xy(geometries[name], ds.longitude.values, ds.latitude.values)
        short = {
            "Germany": "de",
            "Austria": "at",
            "Switzerland": "ch",
        }[name]
        masks[f"mask_{short}"] = xr.DataArray(
            values,
            coords={"y": ds.y, "x": ds.x},
            dims=("y", "x"),
            name=f"mask_{short}",
            attrs={
                "long_name": f"{name} political-boundary mask",
                "source": "Natural Earth admin_0_countries, 1:50m",
            },
        )
    dach = masks["mask_de"] | masks["mask_at"] | masks["mask_ch"]
    dach = dach.rename("mask_dach").assign_attrs(
        {
            "long_name": "Germany, Austria, and Switzerland political-boundary mask",
            "source": "Natural Earth admin_0_countries, 1:50m",
        }
    )
    masks["mask_dach"] = dach
    return xr.Dataset(masks)


def standardize_names(ds: xr.Dataset) -> xr.Dataset:
    """Map GRIB short names to pipeline names."""
    rename_map = {}
    for v in list(ds.data_vars):
        short = getattr(ds[v], "shortName", None)
        if short is None:
            short = ds[v].attrs.get("GRIB_shortName", v)
        mapping = {
            "2t": "t2m",
            "t2m": "t2m",
            "T_2M": "t2m",
            "T_2M_CL": "t2m",
            "aswdir": "aswdir_s",
            "ASWDIR_S": "aswdir_s",
            "aswdif": "aswdifd_s",
            "aswdifd": "aswdifd_s",
            "ASWDIFD_S": "aswdifd_s",
            "10u": "u10",
            "u": "u10",
            "U_10M": "u10",
            "10v": "v10",
            "v": "v10",
            "V_10M": "v10",
        }
        if short in mapping:
            rename_map[v] = mapping[short]
    return ds.rename(rename_map)


def load_monthly_dataset(
    yyyymm: str,
    pilot_dir: Path,
    variables: Iterable[str] | None = None,
) -> xr.Dataset:
    """Download (if needed), open, subset and merge COSMO-REA6 variables."""
    if variables is None:
        variables = list(VARIABLES.keys())

    datasets = []
    for var in variables:
        local_path = download_var(var, yyyymm, pilot_dir)
        ds = open_cosmo_rea6(local_path)
        ds = standardize_names(ds)
        ds = subset_dach(ds)
        datasets.append(ds)

    merged = xr.merge(datasets, join="inner", compat="override")
    dach_mask = build_dach_mask(merged)
    country_masks = build_country_masks(merged)
    masked = merged.where(dach_mask)
    return masked.assign({"dach_mask": dach_mask, **country_masks})


# ---------------------------------------------------------------------------
# Renewable-energy calculations
# ---------------------------------------------------------------------------


def wind_speed(u: xr.DataArray, v: xr.DataArray) -> xr.DataArray:
    """Compute horizontal wind speed from U and V components."""
    return np.sqrt(u**2 + v**2)


def extrapolate_10m_to_100m(wspd10: xr.DataArray, alpha: float = 0.143) -> xr.DataArray:
    """Extrapolate 10 m wind speed to 100 m hub height using the power law."""
    return wspd10 * ((100.0 / 10.0) ** alpha)


def capacity_factor_wind(
    wspd: xr.DataArray,
    cut_in: float = 3.0,
    rated: float = 12.0,
    cut_out: float = 25.0,
) -> xr.DataArray:
    """Piecewise-linear capacity factor from wind speed (m/s).

    NaNs in the input are preserved as NaNs in the output.
    """
    cf = xr.where(wspd <= cut_in, 0.0, np.nan)
    cf = xr.where(wspd >= cut_out, 0.0, cf)
    region = (wspd > cut_in) & (wspd < rated)
    cf = xr.where(region, (wspd - cut_in) / (rated - cut_in), cf)
    cf = xr.where((wspd >= rated) & (wspd < cut_out), 1.0, cf)
    return cf


def capacity_factor_solar(
    ghi: xr.DataArray,
    t2m: xr.DataArray,
    ghi_ref: float = 1000.0,
    beta: float = 0.004,
    t_ref: float = 25.0,
) -> xr.DataArray:
    """Simple PV capacity factor from daily-mean GHI (W/m2) and 2 m temperature.

    Temperature is expected in K and is converted to Celsius internally.
    The capacity factor is clipped to [0, 1].
    """
    t_c = t2m - 273.15
    efficiency = 1.0 - beta * (t_c - t_ref)
    efficiency = xr.where(efficiency < 0.0, 0.0, efficiency)
    cf = (ghi / ghi_ref) * efficiency
    return xr.where(cf > 1.0, 1.0, xr.where(cf < 0.0, 0.0, cf))


# ---------------------------------------------------------------------------
# Complementarity / reliability metrics
# ---------------------------------------------------------------------------


def spatial_mean(da: xr.DataArray, dims: Iterable[str] = ("y", "x")) -> xr.DataArray:
    """Area-weighted spatial mean using cosine-of-latitude weights, ignoring NaNs."""
    lat = da.latitude
    weights = np.cos(np.deg2rad(lat))
    weights = weights.where(np.isfinite(da))
    weights = weights / weights.sum(dim=dims)
    return (da * weights).sum(dim=dims)


def country_spatial_mean(
    da: xr.DataArray,
    mask_ds: xr.Dataset,
    countries: tuple[str, ...] = ("de", "at", "ch"),
) -> xr.DataArray:
    """Area-weighted mean for each DACH country and the full union."""
    all_countries = (*countries, "dach")
    means = {}
    for c in all_countries:
        mask = mask_ds[f"mask_{c}"]
        sub = da.where(mask)
        lat = da.latitude
        weights = np.cos(np.deg2rad(lat))
        weights = weights.where(np.isfinite(sub))
        weights = weights / weights.sum(dim=("y", "x"))
        means[c] = (sub * weights).sum(dim=("y", "x"))
    arr = xr.concat([means[c] for c in all_countries], dim="country")
    arr = arr.assign_coords(country=list(all_countries))
    return arr.rename("mean")


def dunkelflaute_events(
    wind_cf: xr.DataArray,
    solar_cf: xr.DataArray,
    wind_quantile: float = 0.1,
    solar_quantile: float = 0.1,
    dims: Iterable[str] = ("y", "x"),
) -> xr.DataArray:
    """Flag grid points/times where both wind and solar are in their lowest decile."""
    w_thresh = wind_cf.quantile(wind_quantile, dim=dims)
    s_thresh = solar_cf.quantile(solar_quantile, dim=dims)
    return (wind_cf <= w_thresh) & (solar_cf <= s_thresh)


def save_compressed(
    ds: xr.Dataset,
    path: Path,
    complevel: int = 4,
) -> None:
    """Write a NetCDF with float32 + zlib encoding for all floating-point variables."""
    encoding = {}
    for v in ds.data_vars:
        dtype = ds[v].dtype
        if np.issubdtype(dtype, np.floating):
            encoding[v] = {"dtype": "float32", "zlib": True, "complevel": complevel}
        elif np.issubdtype(dtype, np.integer) and v != "time":
            encoding[v] = {"zlib": True, "complevel": complevel}
    ds.to_netcdf(path, encoding=encoding)


def ramp_rate(da: xr.DataArray, dim: str = "time") -> xr.DataArray:
    """Absolute day-to-day change."""
    return da.diff(dim).assign_attrs({"units": "d-1"})


def run_pipeline(
    yyyymm: str,
    pilot_dir: Path,
    output_dir: Path,
    wind_share: float = 0.5,
    solar_share: float = 0.5,
) -> dict:
    """Run the full COSMO-REA6 renewable pipeline for a given month."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading COSMO-REA6 month {yyyymm} ...")
    ds = load_monthly_dataset(yyyymm, pilot_dir)

    # --- wind ---
    wspd10 = wind_speed(ds["u10"], ds["v10"])
    wspd100 = extrapolate_10m_to_100m(wspd10)
    wind_cf = capacity_factor_wind(wspd100)

    # --- solar ---
    ghi = ds["aswdir_s"] + ds["aswdifd_s"]
    ghi = ghi.assign_attrs({"long_name": "surface downward shortwave radiation", "units": "W m-2"})
    solar_cf = capacity_factor_solar(ghi, ds["t2m"])

    # --- combined ---
    combined_cf = wind_share * wind_cf + solar_share * solar_cf

    # Build diagnostic dataset
    diag = xr.Dataset(
        {
            "dach_mask": ds["dach_mask"],
            "wspd10": wspd10,
            "wspd100": wspd100,
            "ghi": ghi,
            "wind_cf": wind_cf,
            "solar_cf": solar_cf,
            "combined_cf": combined_cf,
        },
        attrs={
            "region": "Germany, Austria, and Switzerland",
            "country_mask_source": "Natural Earth admin_0_countries, 1:50m",
        },
    )

    # Save full spatial diagnostics
    nc_path = output_dir / f"cosmo_rea6_dach_{yyyymm}_diagnostics.nc"
    diag.to_netcdf(nc_path)
    print(f"Saved spatial diagnostics -> {nc_path}")

    # --- spatial-mean time series ---
    ts_wind = spatial_mean(wind_cf).rename("wind_cf_mean")
    ts_solar = spatial_mean(solar_cf).rename("solar_cf_mean")
    ts_combined = spatial_mean(combined_cf).rename("combined_cf_mean")
    ts_ghi = spatial_mean(ghi).rename("ghi_mean")
    ts_wspd100 = spatial_mean(wspd100).rename("wspd100_mean")

    ts = xr.merge([ts_wind, ts_solar, ts_combined, ts_ghi, ts_wspd100], compat="override")
    ts_df = ts.to_dataframe()
    csv_path = output_dir / f"cosmo_rea6_dach_{yyyymm}_timeseries.csv"
    ts_df.to_csv(csv_path)
    print(f"Saved time series -> {csv_path}")

    # --- complementarity metrics ---
    corr = xr.corr(ts_wind, ts_solar, dim="time").values.item()
    dkf = dunkelflaute_events(wind_cf, solar_cf)
    dkf_spatial_frac = (dkf.sum("time") / dkf.sizes["time"]).rename("dunkelflaute_freq")
    dkf_mean_freq = float(spatial_mean(dkf_spatial_frac).values)
    ramp = ramp_rate(ts_combined)
    ramp_mean = float(np.abs(ramp).mean().values)
    ramp_max = float(np.abs(ramp).max().values)

    metrics = {
        "yyyymm": yyyymm,
        "wind_cf_mean": float(ts_wind.mean().values),
        "solar_cf_mean": float(ts_solar.mean().values),
        "combined_cf_mean": float(ts_combined.mean().values),
        "wind_solar_correlation": corr,
        "dunkelflaute_mean_frequency": dkf_mean_freq,
        "combined_ramp_mean_abs": ramp_mean,
        "combined_ramp_max_abs": ramp_max,
    }

    # Save Dunkelflaute frequency map
    dkf_path = output_dir / f"cosmo_rea6_dach_{yyyymm}_dunkelflaute_freq.nc"
    dkf_spatial_frac.to_netcdf(dkf_path)
    print(f"Saved Dunkelflaute frequency map -> {dkf_path}")

    print("\nMetrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    return {
        "dataset": diag,
        "timeseries": ts,
        "metrics": metrics,
        "paths": {
            "diagnostics": nc_path,
            "timeseries_csv": csv_path,
            "dunkelflaute": dkf_path,
        },
    }

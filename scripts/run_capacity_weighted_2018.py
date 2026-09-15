"""Create Germany 2018 capacity-weighted wind and solar resource-proxy time series."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cosmo_rea6_pipeline import (
    capacity_factor_solar,
    capacity_factor_wind,
    extrapolate_10m_to_100m,
    load_monthly_dataset,
    wind_speed,
)

YEAR = 2018
PUBLISHED_CAPACITY_MW = {"wind": 52_565, "solar": 45_277}
MONTHS = [f"{YEAR}{month:02d}" for month in range(1, 13)]
VARIABLES = ["T_2M", "ASWDIR_S", "ASWDIFD_S", "U_10M", "V_10M"]
OPSD_URL = "https://data.open-power-system-data.org/renewable_power_plants/2020-08-25/renewable_power_plants_DE.csv"
DATA_DIR = ROOT / "data" / "capacity_weighting"
SOURCE_PATH = DATA_DIR / "renewable_power_plants_DE_2020-08-25.csv"
PLANTS_PATH = DATA_DIR / "opsd_de_wind_solar_operating_2018.parquet"
WEIGHTS_PATH = DATA_DIR / "cosmo_rea6_de_capacity_weights_2018.nc"
OUTPUT_PATH = ROOT / "data" / "processed" / "2018" / "cosmo_rea6_de_2018_daily_capacity_weighted.parquet"
ASSET_PATH = ROOT / "app_assets" / "2018" / OUTPUT_PATH.name
QC_PATH = ROOT / "app_assets" / "2018" / "cosmo_rea6_de_2018_capacity_weighting_qc.json"
PILOT_DIR = ROOT / "data" / "pilot" / "2018_capacity_weighted"


def prepare_plants() -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if PLANTS_PATH.exists():
        return pd.read_parquet(PLANTS_PATH)
    if not SOURCE_PATH.exists():
        print(f"Downloading OPSD plant register from {OPSD_URL}")
        urlretrieve(OPSD_URL, SOURCE_PATH)

    selected = []
    usecols = [
        "electrical_capacity",
        "energy_source_level_2",
        "technology",
        "lon",
        "lat",
        "commissioning_date",
        "decommissioning_date",
        "federal_state",
    ]
    for chunk in pd.read_csv(SOURCE_PATH, usecols=usecols, chunksize=200_000, low_memory=False):
        source = chunk["energy_source_level_2"].astype(str).str.lower()
        technology = chunk["technology"].astype(str).str.lower()
        keep_technology = source.isin(["wind", "solar"]) & (
            technology.str.contains("onshore|photovoltaic", regex=True, na=False)
            | source.eq("solar")
        )
        chunk = chunk.loc[keep_technology].copy()
        chunk["commissioning_date"] = pd.to_datetime(chunk["commissioning_date"], errors="coerce")
        chunk["decommissioning_date"] = pd.to_datetime(chunk["decommissioning_date"], errors="coerce")
        operating = (
            chunk["commissioning_date"].notna()
            & (chunk["commissioning_date"] <= f"{YEAR}-12-31")
            & (
                chunk["decommissioning_date"].isna()
                | (chunk["decommissioning_date"] >= f"{YEAR}-01-01")
            )
        )
        valid = (
            operating
            & chunk["electrical_capacity"].gt(0)
            & chunk["lon"].between(5.5, 15.5)
            & chunk["lat"].between(47.0, 55.5)
        )
        chunk = chunk.loc[valid]
        chunk["technology_group"] = np.where(source.loc[chunk.index].eq("wind"), "wind", "solar")
        selected.append(chunk)

    plants = pd.concat(selected, ignore_index=True)
    plants.to_parquet(PLANTS_PATH, index=False)
    print(f"Saved {len(plants):,} filtered plant records to {PLANTS_PATH}")
    return plants


def spherical_xyz(lon, lat):
    lon_rad = np.deg2rad(np.asarray(lon))
    lat_rad = np.deg2rad(np.asarray(lat))
    return np.column_stack(
        [
            np.cos(lat_rad) * np.cos(lon_rad),
            np.cos(lat_rad) * np.sin(lon_rad),
            np.sin(lat_rad),
        ]
    )


def build_weights(ds: xr.Dataset, plants: pd.DataFrame) -> xr.Dataset:
    mask = ds["mask_de"].values.astype(bool)
    flat_indices = np.flatnonzero(mask.ravel())
    grid_lon = ds["longitude"].values.ravel()[flat_indices]
    grid_lat = ds["latitude"].values.ravel()[flat_indices]
    tree = cKDTree(spherical_xyz(grid_lon, grid_lat))

    _, nearest = tree.query(spherical_xyz(plants["lon"], plants["lat"]), k=1)
    plants = plants.copy()
    plants["flat_grid_index"] = flat_indices[nearest]

    arrays = {}
    for technology in ["wind", "solar"]:
        sub = plants[plants["technology_group"] == technology]
        capacities = sub.groupby("flat_grid_index")["electrical_capacity"].sum()
        values = np.zeros(mask.size, dtype="float64")
        values[capacities.index.to_numpy(dtype=int)] = capacities.to_numpy()
        arrays[f"{technology}_capacity_mw"] = xr.DataArray(
            values.reshape(mask.shape), coords={"y": ds.y, "x": ds.x}, dims=("y", "x")
        )

    weights = xr.Dataset(arrays).assign_coords(
        latitude=ds["latitude"], longitude=ds["longitude"]
    )
    weights.to_netcdf(WEIGHTS_PATH)
    return weights


def capacity_weighted_mean(field: xr.DataArray, capacity: xr.DataArray) -> xr.DataArray:
    valid_capacity = capacity.where(np.isfinite(field))
    return (field * valid_capacity).sum(("y", "x")) / valid_capacity.sum(("y", "x"))


def evict_raw(yyyymm: str):
    for variable in VARIABLES:
        path = PILOT_DIR / f"{variable}.2D.{yyyymm}.DayMean.grb"
        if path.exists():
            path.unlink()


def main():
    plants = prepare_plants()
    PILOT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ASSET_PATH.parent.mkdir(parents=True, exist_ok=True)

    daily = []
    weights = None
    for yyyymm in MONTHS:
        print(f"Processing {yyyymm}")
        ds = load_monthly_dataset(yyyymm, PILOT_DIR, variables=VARIABLES)
        if weights is None:
            weights = build_weights(ds, plants)

        wspd100 = extrapolate_10m_to_100m(wind_speed(ds["u10"], ds["v10"]))
        wind_cf = capacity_factor_wind(wspd100)
        ghi = ds["aswdir_s"] + ds["aswdifd_s"]
        solar_cf = capacity_factor_solar(ghi, ds["t2m"])
        wind_weighted = capacity_weighted_mean(wind_cf, weights["wind_capacity_mw"])
        solar_weighted = capacity_weighted_mean(solar_cf, weights["solar_capacity_mw"])
        frame = xr.Dataset(
            {
                "wind_cf_capacity_weighted": wind_weighted,
                "solar_cf_capacity_weighted": solar_weighted,
            }
        ).to_dataframe().reset_index()
        frame["combined_cf_capacity_weighted"] = (
            0.5 * frame["wind_cf_capacity_weighted"]
            + 0.5 * frame["solar_cf_capacity_weighted"]
        )
        daily.append(frame)
        evict_raw(yyyymm)

    result = pd.concat(daily, ignore_index=True).sort_values("time")
    result.to_parquet(OUTPUT_PATH, index=False)
    result.to_parquet(ASSET_PATH, index=False)

    totals = plants.groupby("technology_group")["electrical_capacity"].agg(["count", "sum"])
    qc = {
        "year": YEAR,
        "plant_snapshot": "operating at any time in 2018; static year snapshot",
        "source": OPSD_URL,
        "plant_records": {technology: int(totals.loc[technology, "count"]) for technology in totals.index},
        "mapped_capacity_mw": {technology: round(float(totals.loc[technology, "sum"]), 3) for technology in totals.index},
        "published_end_2018_capacity_mw": PUBLISHED_CAPACITY_MW,
        "mapped_capacity_coverage_fraction": {
            technology: round(float(totals.loc[technology, "sum"]) / PUBLISHED_CAPACITY_MW[technology], 4)
            for technology in totals.index
        },
        "occupied_grid_cells": {
            technology: int((weights[f"{technology}_capacity_mw"] > 0).sum().item())
            for technology in ["wind", "solar"]
        },
        "daily_records": int(len(result)),
        "mean_capacity_factor": {
            column: round(float(result[column].mean()), 6)
            for column in result.columns
            if column.endswith("weighted")
        },
    }
    QC_PATH.write_text(json.dumps(qc, indent=2), encoding="utf-8")
    print(json.dumps(qc, indent=2))
    print(f"Saved daily series to {OUTPUT_PATH} and {ASSET_PATH}")


if __name__ == "__main__":
    main()

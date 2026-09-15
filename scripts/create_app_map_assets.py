"""Build the compact map assets used by app/streamlit_app.py.

Inputs (produced by the hourly November pipeline, see scripts/run_hourly_november_2018.py
and scripts/run_capacity_weighted_2018.py):
    data/hourly_cache/201811/subsets/{T_2M,U_10M,V_10M,ASWDIR_S,ASWDIFD_S}.2D.201811.dach.nc
    data/capacity_weighting/cosmo_rea6_de_capacity_weights_2018.nc

Outputs (committed, read by the app):
    app_assets/2018/cosmo_rea6_dach_201811_november_field_means.parquet
    app_assets/2018/cosmo_rea6_de_2018_capacity_maps.parquet
    app_assets/2018/fig_201811_field_means.png   (static 2x2 domain map)
    app_assets/2018/fig_capacity_maps.png      (static 1x2 mapped-capacity map)

Run from the repository root:  python scripts/create_app_map_assets.py
Requires the research environment (xarray, matplotlib, cartopy); the app
itself only reads the committed outputs.
"""
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parent.parent
# Pipeline outputs live under data/ in the working repository; pass
# --data-root <path> when regenerating assets without rerunning the pipeline here.
import argparse

_parser = argparse.ArgumentParser()
_parser.add_argument("--data-root", default=str(ROOT / "data"))
DATA = Path(_parser.parse_args().data_root)
SUB = DATA / "hourly_cache" / "201811" / "subsets"
OUT = ROOT / "app_assets" / "2018"


def style_map(ax, extent):
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.8, edgecolor="black")
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.6, edgecolor="black")
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    x0, x1, y0, y1 = extent
    xt = np.arange(np.ceil(x0), x1, 4.0)
    yt = np.arange(np.ceil(y0), y1, 4.0)
    ax.set_xticks(xt, crs=ccrs.PlateCarree())
    ax.set_yticks(yt, crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.tick_params(labelsize=8)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")


# --- November 2018 monthly-mean input fields ---
means = {}
coords = {}
for name, var in [("T_2M", "t2m"), ("U_10M", "u10"), ("ASWDIR_S", "aswdir_s"),
                  ("ASWDIFD_S", "aswdifd_s")]:
    ds = xr.open_dataset(SUB / f"{name}.2D.201811.dach.nc")
    if var == "u10":
        # mean wind speed = time-mean of the speed, not the speed of the mean vector
        v10 = xr.open_dataset(SUB / "V_10M.2D.201811.dach.nc")["v10"]
        means["wspd10"] = np.sqrt(ds["u10"] ** 2 + v10 ** 2).mean("time").values.astype(np.float32)
    else:
        means[var] = ds[var].mean("time").values.astype(np.float32)
    if not coords:
        coords["latitude"] = ds["latitude"].values.astype(np.float32)
        coords["longitude"] = ds["longitude"].values.astype(np.float32)
        coords["mask_de"] = ds["mask_de"].values.astype(np.float32)
    ds.close()

lon, lat = coords["longitude"], coords["latitude"]
ny, nx = lat.shape
iy, ix = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
fields = pd.DataFrame({
    "iy": iy.ravel().astype(np.int16),
    "ix": ix.ravel().astype(np.int16),
    "latitude": lat.ravel(),
    "longitude": lon.ravel(),
    "in_germany": coords["mask_de"].ravel().astype(bool),
    "t2m_degC": (means["t2m"].ravel() - 273.15).astype(np.float32),
    "wspd10_ms": means["wspd10"].ravel(),
    "aswdir_wm2": means["aswdir_s"].ravel(),
    "aswdifd_wm2": means["aswdifd_s"].ravel(),
})
fields.to_parquet(OUT / "cosmo_rea6_dach_201811_november_field_means.parquet", index=False)
print(f"field means -> {fields.shape[0]} cells ({int(fields['in_germany'].sum())} in Germany)")

# --- static 2x2 domain map (full DACH extent) ---
dach_extent = [2.0, 18.0, 44.5, 56.3]
panels = [
    (means["t2m"] - 273.15, "RdYlBu_r", "2 m temperature", "°C", -4.0, 12.0),
    (means["wspd10"], "viridis", "10 m wind speed", "m s$^{-1}$", 0.0, 10.0),
    (means["aswdir_s"], "YlOrRd", "Direct shortwave ASWDIR_S", "W m$^{-2}$", 5.0, 45.0),
    (means["aswdifd_s"], "YlOrRd", "Diffuse shortwave ASWDIFD_S", "W m$^{-2}$", 5.0, 45.0),
]
fig, axes = plt.subplots(2, 2, figsize=(11, 9), subplot_kw={"projection": ccrs.PlateCarree()})
for ax, (arr, cmap, title, unit, vmin, vmax) in zip(axes.flat, panels):
    im = ax.pcolormesh(lon, lat, arr, shading="auto", cmap=cmap, transform=ccrs.PlateCarree(),
                       vmin=vmin, vmax=vmax)
    style_map(ax, dach_extent)
    ax.set_title(title)
    plt.colorbar(im, ax=ax, shrink=0.75, label=unit)
fig.suptitle("COSMO-REA6 input fields — November 2018 monthly mean (DACH domain)", y=0.98)
fig.tight_layout()
fig.savefig(OUT / "fig_201811_field_means.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("fig_201811_field_means.png written")

# --- Mapped installed capacity per grid cell ---
ds = xr.open_dataset(DATA / "capacity_weighting" / "cosmo_rea6_de_capacity_weights_2018.nc")
cap = pd.DataFrame({
    "latitude": ds["latitude"].values.astype(np.float32).ravel(),
    "longitude": ds["longitude"].values.astype(np.float32).ravel(),
    "wind_mw": ds["wind_capacity_mw"].values.astype(np.float32).ravel(),
    "solar_mw": ds["solar_capacity_mw"].values.astype(np.float32).ravel(),
})
cap = cap[(cap["wind_mw"] > 0) | (cap["solar_mw"] > 0)].reset_index(drop=True)
cap.to_parquet(OUT / "cosmo_rea6_de_2018_capacity_maps.parquet", index=False)
print(f"capacity maps -> {cap.shape[0]} occupied cells")

# --- static 1x2 mapped-capacity map (Germany extent) ---
# Same rendering as notebook 05: colour = log(1 + MW), which keeps zero defined
# and prevents a few very large cells from hiding the small ones.
de_extent = [5.5, 15.5, 47.0, 55.5]
cap_panels = [
    (ds["wind_capacity_mw"].values, "Mapped onshore-wind capacity"),
    (ds["solar_capacity_mw"].values, "Mapped PV capacity"),
]
fig, axes = plt.subplots(1, 2, figsize=(14, 6), subplot_kw={"projection": ccrs.PlateCarree()})
for ax, (arr, title) in zip(axes.flat, cap_panels):
    im = ax.pcolormesh(lon, lat, np.log1p(arr), shading="auto", cmap="plasma",
                       transform=ccrs.PlateCarree(), vmin=0.0, vmax=5.0)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.7)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.7)
    ax.set_extent(de_extent, crs=ccrs.PlateCarree())
    xt = np.arange(np.ceil(de_extent[0]), de_extent[1], 4.0)
    yt = np.arange(np.ceil(de_extent[2]), de_extent[3], 4.0)
    ax.set_xticks(xt, crs=ccrs.PlateCarree())
    ax.set_yticks(yt, crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.tick_params(labelsize=8)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"{title}\ncolour = log(1 + MW)")
    fig.colorbar(im, ax=ax, shrink=0.7, label="log(1 + MW)", extend="max")
fig.suptitle("OPSD installed capacity mapped to COSMO-REA6 grid cells (2018 snapshot)", y=1.0)
fig.tight_layout()
fig.savefig(OUT / "fig_capacity_maps.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("fig_capacity_maps.png written")
ds.close()

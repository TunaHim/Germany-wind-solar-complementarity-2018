"""Measure prototype storage and estimate COSMO-REA6 scaling requirements."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
GROUPS = {
    "raw_and_legacy_pilot_data": ROOT / "data" / "pilot",
    "processed_outputs": ROOT / "data" / "processed",
    "notebooks_including_checkpoints": ROOT / "notebooks",
    "figures": ROOT / "reports" / "figures",
    "source": ROOT / "src",
    "scripts": ROOT / "scripts",
}
RAW_PATTERN = "*.201907.DayMean.grb"
DAYS_IN_PILOT = 31
MONTHS_IN_COSMO_REA6 = 296


def file_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def mib(size: int) -> float:
    return size / 1024**2


def gib(size: int) -> float:
    return size / 1024**3


def main() -> None:
    group_sizes = {name: file_bytes(path) for name, path in GROUPS.items()}
    raw_files = sorted((ROOT / "data" / "pilot").glob(RAW_PATTERN))
    raw_month_bytes = sum(path.stat().st_size for path in raw_files)
    processed_month_bytes = group_sizes["processed_outputs"]

    report = {
        "prototype": {
            "raw_cosmo_rea6_files": len(raw_files),
            "raw_cosmo_rea6_month_bytes": raw_month_bytes,
            "raw_cosmo_rea6_month_mib": round(mib(raw_month_bytes), 3),
            "group_sizes_mib": {name: round(mib(size), 3) for name, size in group_sizes.items()},
        },
        "linear_projection": {
            "raw_one_year_gib": round(gib(raw_month_bytes * 12), 3),
            "raw_1995_to_2019_gib": round(gib(raw_month_bytes * MONTHS_IN_COSMO_REA6), 3),
            "current_processed_one_year_gib": round(gib(processed_month_bytes * 12), 3),
            "current_processed_1995_to_2019_gib": round(
                gib(processed_month_bytes * MONTHS_IN_COSMO_REA6), 3
            ),
        },
        "notes": [
            "Raw projection assumes five full-domain monthly GRIB files of the current size.",
            "Processed projection assumes the current uncompressed monthly NetCDF design.",
            "Legacy EERIE pilot NetCDF files are included in the pilot directory group, but not in raw COSMO-REA6 projections.",
        ],
    }

    output = ROOT / "data" / "processed" / "storage_analysis.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()

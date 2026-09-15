"""Download and QC SMARD hourly German wind/PV generation for November 2018."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
BASE = "https://www.smard.de/app/chart_data"
FILTERS = {"wind": 4067, "solar": 4068}
CAPACITY_MW = {"wind": 52_565.0, "solar": 45_277.0}
START = pd.Timestamp("2018-11-01 00:00:00", tz="UTC")
END = pd.Timestamp("2018-11-30 23:00:00", tz="UTC")
EXPECTED = pd.date_range(START, END, freq="h")
REFERENCE = ROOT / "data/references/smard_de_201811_hourly.parquet"
ASSET = ROOT / "app_assets/2018/smard_de_201811_hourly.parquet"
QC_PATH = ROOT / "app_assets/2018/smard_de_201811_hourly_qc.json"


def get_json(url: str) -> dict:
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return response.json()


def fetch_technology(name: str, filter_id: int) -> tuple[pd.DataFrame, dict]:
    index_url = f"{BASE}/{filter_id}/DE/index_hour.json"
    stamps = sorted(set(get_json(index_url)["timestamps"]))
    # Index entries identify weekly windows. Fetch only windows whose interval
    # overlaps the exact requested UTC timestamps, then filter returned series.
    start_ms = int(START.timestamp() * 1000)
    end_ms = int(END.timestamp() * 1000)
    selected = [
        stamp for position, stamp in enumerate(stamps)
        if stamp <= end_ms and (stamps[position + 1] if position + 1 < len(stamps) else float("inf")) > start_ms
    ]
    if not selected:
        raise ValueError(f"{name}: index supplied no overlapping chunks")
    rows, urls = [], []
    for stamp in selected:
        url = f"{BASE}/{filter_id}/DE/{filter_id}_DE_hour_{stamp}.json"
        payload = get_json(url)
        urls.append(url)
        rows.extend(payload.get("series", []))
    raw = pd.DataFrame(rows, columns=["timestamp_ms", "generation_mwh"])
    null_values = int(raw["generation_mwh"].isna().sum())
    raw = raw.dropna(subset=["timestamp_ms", "generation_mwh"])
    raw["interval_start_utc"] = pd.to_datetime(raw.timestamp_ms, unit="ms", utc=True)
    raw = raw[raw.interval_start_utc.between(START, END)]
    duplicate_count = int(raw.interval_start_utc.duplicated().sum())
    raw = raw.sort_values("interval_start_utc").drop_duplicates("interval_start_utc", keep="last")
    missing = EXPECTED.difference(pd.DatetimeIndex(raw.interval_start_utc))
    if len(raw) != 720 or len(missing) or not raw.interval_start_utc.is_monotonic_increasing:
        raise ValueError(f"{name}: records={len(raw)}, missing={len(missing)}, duplicates={duplicate_count}")
    frame = raw[["interval_start_utc", "generation_mwh"]].rename(columns={"generation_mwh": f"{name}_generation_mwh"})
    return frame, {"filter": filter_id, "index_url": index_url, "chunk_urls": urls,
                   "candidate_chunks": selected, "raw_records": len(rows), "null_values": null_values,
                   "duplicate_timestamps_before_deduplication": duplicate_count,
                   "records_after_exact_utc_filter": len(frame), "missing_expected_timestamps": len(missing)}


def main() -> None:
    frames, downloads = [], {}
    for name, filter_id in FILTERS.items():
        frame, info = fetch_technology(name, filter_id)
        frames.append(frame); downloads[name] = info
    data = frames[0].merge(frames[1], on="interval_start_utc", how="inner", validate="one_to_one")
    if len(data) != 720:
        raise ValueError(f"merged coverage is {len(data)}, expected 720")
    # Preserve UTC semantics in a serialization-portable naive timestamp; local is an ISO label.
    local = data.interval_start_utc.dt.tz_convert("Europe/Berlin")
    data["interval_start_local"] = local.map(lambda x: x.isoformat())
    data["interval_start_utc"] = data.interval_start_utc.dt.tz_localize(None)
    for tech in FILTERS:
        data[f"{tech}_capacity_mw"] = CAPACITY_MW[tech]
        data[f"observed_{tech}_cf"] = data[f"{tech}_generation_mwh"] / CAPACITY_MW[tech]
    data["observed_hybrid_cf"] = .5 * data.observed_wind_cf + .5 * data.observed_solar_cf
    final_index = pd.DatetimeIndex(data.interval_start_utc)
    expected_naive = pd.date_range(START.tz_localize(None), END.tz_localize(None), freq="h")
    if data.isna().any().any() or not final_index.equals(expected_naive):
        raise ValueError("final SMARD table failed exact coverage/null validation")
    REFERENCE.parent.mkdir(parents=True, exist_ok=True); ASSET.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(REFERENCE, index=False, compression="zstd"); shutil.copy2(REFERENCE, ASSET)
    qc = {"records": len(data), "expected_records": 720, "missing_timestamps": 0, "duplicate_timestamps": 0,
          "start_interval_start_utc": str(data.interval_start_utc.min()), "end_interval_start_utc": str(data.interval_start_utc.max()),
          "cadence": "1 hour", "downloads": downloads,
          "units": {"generation": "MWh per one-hour interval", "capacity": "MW", "capacity_factor": "dimensionless"},
          "time_semantics": "interval_start_utc is timezone-naive storage representing UTC interval starts; exact range 2018-11-01 00:00 through 2018-11-30 23:00. interval_start_local is an ISO-8601 Europe/Berlin display label.",
          "capacity_assumptions": {"wind_onshore_end_2018_mw": 52565, "solar_end_2018_mw": 45277,
            "caveat": "Static year-end denominators slightly understate CF where installed capacity was lower; correlation and ramp timing are less affected than level."},
          "hybrid_definition": "0.5 observed wind CF + 0.5 observed solar CF; hypothetical equal-rated-capacity portfolio, not actual combined national generation.",
          "source": "Bundesnetzagentur | SMARD.de; underlying data received from ENTSO-E", "licence": "CC BY 4.0"}
    QC_PATH.write_text(json.dumps(qc, indent=2), encoding="utf-8")
    print(f"Wrote {len(data)} exact hours to {REFERENCE} and {ASSET}")

if __name__ == "__main__":
    main()

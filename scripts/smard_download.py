import requests
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

CAPACITY_MW = {"wind_onshore": 52_565, "solar": 45_277}
BASE_URL = "https://www.smard.de/app/chart_data"
FILTERS = {"wind_onshore": 4067, "solar": 4068}


def fetch_json(url):
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return response.json()


def find_year_timestamp(filter_id, year):
    index_url = f"{BASE_URL}/{filter_id}/DE/index_day.json"
    timestamps = fetch_json(index_url)["timestamps"]
    target = int(datetime(year, 1, 1, tzinfo=timezone.utc).timestamp() * 1000) - 3_600_000
    return min(timestamps, key=lambda timestamp: abs(timestamp - target))


def fetch_series(filter_id, year):
    timestamp = find_year_timestamp(filter_id, year)
    data_url = f"{BASE_URL}/{filter_id}/DE/{filter_id}_DE_day_{timestamp}.json"
    series = fetch_json(data_url)["series"]
    return pd.DataFrame(series, columns=["timestamp_ms", "generation_mwh"])


def main():
    year = 2018
    frames = []

    for technology, filter_id in FILTERS.items():
        print(f"Fetching {technology} from SMARD...")
        frame = fetch_series(filter_id, year).rename(
            columns={"generation_mwh": f"{technology}_mwh"}
        )
        frames.append(frame)
        print(f"  Received {len(frame)} daily records")

    data = frames[0].merge(frames[1], on="timestamp_ms", how="outer")
    data["time_utc"] = pd.to_datetime(data["timestamp_ms"], unit="ms", utc=True)
    data["date"] = data["time_utc"].dt.tz_convert("Europe/Berlin").dt.date
    data = data.sort_values("timestamp_ms").drop_duplicates("date").copy()
    data = data[pd.to_datetime(data["date"]).dt.year == year].copy()

    data["wind_onshore_cf"] = (
        data["wind_onshore_mwh"] / (CAPACITY_MW["wind_onshore"] * 24)
    )
    data["solar_cf"] = data["solar_mwh"] / (CAPACITY_MW["solar"] * 24)

    output = Path(__file__).resolve().parent.parent / "app_assets" / "2018" / "smard_de_2018_daily.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "date",
        "wind_onshore_mwh",
        "solar_mwh",
        "wind_onshore_cf",
        "solar_cf",
    ]
    data[columns].to_csv(output, index=False)

    print(f"Saved {len(data)} rows to {output}")
    print(data[columns].head().to_string(index=False))
    print(f"Annual mean wind CF:  {data['wind_onshore_cf'].mean():.4f}")
    print(f"Annual mean solar CF: {data['solar_cf'].mean():.4f}")


if __name__ == "__main__":
    main()

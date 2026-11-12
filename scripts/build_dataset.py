"""Export the training dataset (committed in data/ so the model can be retrained without the warehouse).

    python scripts/build_dataset.py --dw postgresql://aeria:aeria@127.0.0.1:5440/aeria \
        --start 2025-11-01 --end 2026-11-12
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import pandas as pd

from aeria_forecast import data, weather

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--dw", default=os.getenv("DW_URL", "postgresql://aeria:aeria@127.0.0.1:5440/aeria"))
    p.add_argument("--start", default="2025-11-01")
    p.add_argument("--end", default="2026-11-12", help="exclusive")
    p.add_argument("--weather-cache", type=Path, default=None)
    args = p.parse_args()

    hourly = data.load_hourly(args.dw, args.start, args.end)
    daily = data.daily_table(hourly)
    data.save(daily, ROOT / "data" / "daily.csv.gz")

    stations = data.load_stations(args.dw)
    stations["point"] = [weather.point_of(c, z) for c, z in zip(stations.station_code, stations.zone)]
    stations.to_csv(ROOT / "data" / "stations.csv", index=False)

    end_incl = (pd.Timestamp(args.end) - pd.Timedelta(days=1)).date().isoformat()
    w = weather.fetch_previous_day1(args.start, end_incl, args.weather_cache)
    wd = weather.daily(w)
    data.save(wd, ROOT / "data" / "weather_daily.csv.gz")
    print(f"daily rows: {len(daily)}, stations: {len(stations)}, weather days: {len(wd)}")


if __name__ == "__main__":
    main()

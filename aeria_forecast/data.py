"""Daily station table from the Aeria warehouse (module 2).

One row per (station, local date, pollutant) with the ATMO daily aggregate and two helpers used as
features: the daily mean and the evening level (mean of the last 6 local hours of the day), which is
the most recent information available when the forecast for the next day is issued.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from aeria_forecast.atmo import DAILY_AGG

log = logging.getLogger(__name__)

TZ = "Europe/Paris"
MIN_HOURS = 18

HOURLY_SQL = """
    SELECT ts_utc, station_code, pollutant, value
    FROM dw.v_measurement
    WHERE is_reference AND usable AND pollutant = ANY(%(pollutants)s)
      AND ts_utc >= %(start)s AND ts_utc < %(end)s
"""
STATION_SQL = """
    SELECT station_code, station_name, zone, area_type, influence
    FROM dw.dim_station WHERE is_reference ORDER BY station_code
"""


def load_hourly(dw_url: str, start: str = "2000-01-01", end: str = "2100-01-01") -> pd.DataFrame:
    import psycopg

    with psycopg.connect(dw_url) as conn:
        cur = conn.execute(HOURLY_SQL, {"pollutants": list(DAILY_AGG), "start": start, "end": end})
        df = pd.DataFrame(cur.fetchall(), columns=[c.name for c in cur.description])
    df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)
    df["value"] = df["value"].astype(float)
    log.info("warehouse: %d hourly values, %d stations", len(df), df["station_code"].nunique())
    return df


def load_stations(dw_url: str) -> pd.DataFrame:
    import psycopg

    with psycopg.connect(dw_url) as conn:
        cur = conn.execute(STATION_SQL)
        return pd.DataFrame(cur.fetchall(), columns=[c.name for c in cur.description])


def daily_table(hourly: pd.DataFrame) -> pd.DataFrame:
    """Hourly usable values -> (station_code, date, pollutant, agg, mean, evening, n_hours)."""
    df = hourly[["ts_utc", "station_code", "pollutant", "value"]].copy()
    local = df["ts_utc"].dt.tz_convert(TZ)
    df["date"] = local.dt.tz_localize(None).dt.normalize()
    df["evening"] = local.dt.hour >= 18
    g = df.groupby(["station_code", "date", "pollutant"])
    out = g["value"].agg(mean="mean", max="max", n_hours="count").reset_index()
    eve = df[df["evening"]].groupby(["station_code", "date", "pollutant"])["value"].mean().rename("evening")
    out = out.merge(eve.reset_index(), how="left", on=["station_code", "date", "pollutant"])
    is_mean = out["pollutant"].map(DAILY_AGG).eq("mean")
    out["agg"] = out["mean"].where(is_mean, out["max"])
    out.loc[out["n_hours"] < MIN_HOURS, "agg"] = float("nan")
    return out.drop(columns="max")


def save(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, compression="gzip", float_format="%.3f")


def load_csv(path: Path, date_cols=("date",)) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"station_code": str})
    for c in date_cols:
        if c in df:
            df[c] = pd.to_datetime(df[c])
    return df

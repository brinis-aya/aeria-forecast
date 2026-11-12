"""Weather forecasts from Open-Meteo (free, no key, CC BY 4.0).

The model must only see what is known when the forecast is issued. For training we therefore use the
**Previous Runs API**: ``*_previous_day1`` variables are the values forecast by the run of the day
before, i.e. genuine 24-47 h forecasts, not observations. At inference time the regular forecast API
returns the same kind of forecast for tomorrow.

Normandy is covered by 8 points; each station is mapped to the nearest one.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

log = logging.getLogger(__name__)

PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
VARS = ("temperature_2m", "relative_humidity_2m", "wind_speed_10m", "wind_direction_10m",
        "precipitation", "surface_pressure", "shortwave_radiation", "cloud_cover")
TZ = "Europe/Paris"

POINTS = {  # approximate town centres
    "le_havre": (49.49, 0.11), "rouen": (49.44, 1.10), "caen": (49.18, -0.37),
    "cherbourg": (49.64, -1.62), "alencon": (48.43, 0.09), "evreux": (49.02, 1.15),
    "dieppe": (49.92, 1.08), "saint_lo": (49.12, -1.09),
}
ZONE_POINT = {"ZAR LE-HAVRE": "le_havre", "ZAG ROUEN": "rouen", "ZAR CAEN": "caen"}
STATION_POINT = {  # stations of the regional zone ("ZR NORMANDIE")
    "FR05040": "le_havre", "FR05082": "le_havre", "FR05087": "rouen", "FR21031": "alencon",
    "FR21038": "cherbourg", "FR21039": "cherbourg", "FR21040": "saint_lo", "FR21050": "alencon",
    "FR25039": "evreux", "FR25049": "rouen", "FR25050": "dieppe", "FR25051": "dieppe", "FR25055": "dieppe",
}


def point_of(station_code: str, zone: str | None) -> str:
    return STATION_POINT.get(station_code) or ZONE_POINT.get(zone or "", "rouen")


def _get(url: str, params: dict, retries: int = 4) -> list[dict]:
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, params=params, timeout=120)
            r.raise_for_status()
            j = r.json()
            return j if isinstance(j, list) else [j]
        except requests.RequestException as exc:
            log.warning("Open-Meteo request failed (%d/%d): %s", attempt, retries, exc)
            time.sleep(3 * attempt)
    raise RuntimeError("Open-Meteo unreachable")


def _frame(payload: list[dict], names: list[str], suffix: str) -> pd.DataFrame:
    frames = []
    for name, loc in zip(names, payload):
        h = loc["hourly"]
        df = pd.DataFrame({"time": pd.to_datetime(h["time"], utc=True)})
        for v in VARS:
            df[v] = pd.to_numeric(pd.Series(h.get(v + suffix)), errors="coerce")
        df.insert(0, "point", name)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _params(hourly_vars: list[str], **extra) -> dict:
    names = list(POINTS)
    return {
        "latitude": ",".join(str(POINTS[n][0]) for n in names),
        "longitude": ",".join(str(POINTS[n][1]) for n in names),
        "hourly": ",".join(hourly_vars), "timezone": "UTC", **extra,
    }


def fetch_previous_day1(start: str, end: str, cache: Path | None = None) -> pd.DataFrame:
    """Hourly D+1 forecasts (issued the day before) for all points between two dates."""
    if cache and cache.exists():
        df = pd.read_csv(cache)
        df["time"] = pd.to_datetime(df["time"], utc=True)
        return df
    payload = _get(PREVIOUS_RUNS_URL, _params([v + "_previous_day1" for v in VARS], start_date=start, end_date=end))
    df = _frame(payload, list(POINTS), "_previous_day1")
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache, index=False, compression="gzip")
    return df


def fetch_forecast(forecast_days: int = 3) -> pd.DataFrame:
    """Latest forecast for today and the next days (used by the API)."""
    return _frame(_get(FORECAST_URL, _params(list(VARS), forecast_days=forecast_days)), list(POINTS), "")


def daily(hourly: pd.DataFrame) -> pd.DataFrame:
    """Hourly weather -> daily features per point and local date."""
    df = hourly.copy()
    df["date"] = df["time"].dt.tz_convert(TZ).dt.tz_localize(None).dt.normalize()
    rad = np.deg2rad(df["wind_direction_10m"])
    df["u"] = -df["wind_speed_10m"] * np.sin(rad)  # west -> east component
    df["v"] = -df["wind_speed_10m"] * np.cos(rad)  # south -> north component
    g = df.groupby(["point", "date"])
    out = pd.DataFrame({
        "t_mean": g["temperature_2m"].mean(), "t_max": g["temperature_2m"].max(),
        "t_min": g["temperature_2m"].min(),
        "rh_mean": g["relative_humidity_2m"].mean(),
        "wind_mean": g["wind_speed_10m"].mean(), "wind_min": g["wind_speed_10m"].min(),
        "u_mean": g["u"].mean(), "v_mean": g["v"].mean(),
        "precip_sum": g["precipitation"].sum(min_count=1),
        "pressure_mean": g["surface_pressure"].mean(),
        "radiation_sum": g["shortwave_radiation"].sum(min_count=1) / 1000,  # kWh/m²
        "cloud_mean": g["cloud_cover"].mean(),
        "n": g["temperature_2m"].count(),
    }).reset_index()
    return out[out["n"] >= 18].drop(columns="n")

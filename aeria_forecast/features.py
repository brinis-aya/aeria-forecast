"""Feature table: one row per (station, issue date D), describing what is known on the evening of D,
plus the weather forecast for D+1. Targets are the ATMO daily aggregates of D+1.

Feature groups
--------------
* persistence: daily aggregate of D, D-1, D-2 and the 7-day mean, for every pollutant of the station
* evening level of D (mean 18-24 h local): the freshest signal at issue time
* regional background: mean daily aggregate of D over all Normandy stations, per pollutant
* weather forecast for D+1 (issued on D): temperature, wind (speed + u/v), rain, humidity, pressure,
  radiation, cloud; plus the pressure and temperature change from D to D+1
* calendar of D+1: day of week, public holiday, day of year (sin/cos)
* station type: influence (background / traffic / industrial) and area type
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from aeria_forecast.atmo import POLLUTANTS

WEATHER_VARS = ("t_mean", "t_max", "t_min", "rh_mean", "wind_mean", "wind_min", "u_mean", "v_mean",
                "precip_sum", "pressure_mean", "radiation_sum", "cloud_mean")
INFLUENCES = ["Fond", "Trafic", "Industrielle"]
AREAS = ["Urbaine", "Périurbaine", "Rurale près des villes", "Rurale régionale", "Rurale nationale"]

# French public holidays (fixed + Easter-based) for the years covered by the data and the API.
HOLIDAYS = pd.to_datetime([
    "2025-01-01", "2025-04-21", "2025-05-01", "2025-05-08", "2025-05-29", "2025-06-09", "2025-07-14",
    "2025-08-15", "2025-11-01", "2025-11-11", "2025-12-25",
    "2026-01-01", "2026-04-06", "2026-05-01", "2026-05-08", "2026-05-14", "2026-05-25", "2026-07-14",
    "2026-08-15", "2026-11-01", "2026-11-11", "2026-12-25",
    "2027-01-01", "2027-03-29", "2027-05-01", "2027-05-06", "2027-05-08", "2027-05-17", "2027-07-14",
    "2027-08-15", "2027-11-01", "2027-11-11", "2027-12-25",
])


def _wide(daily: pd.DataFrame, col: str, prefix: str) -> pd.DataFrame:
    w = daily.pivot_table(index=["station_code", "date"], columns="pollutant", values=col, aggfunc="first")
    return w.reindex(columns=list(POLLUTANTS)).add_prefix(prefix)


def build(daily: pd.DataFrame, weather_daily: pd.DataFrame, stations: pd.DataFrame,
          issue_dates: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    """Return features (and ``y_<pollutant>`` targets when D+1 is known), indexed by (station_code, date=D)."""
    daily = daily.copy()
    daily["date"] = pd.to_datetime(daily["date"])
    start, end = daily["date"].min(), daily["date"].max()
    if issue_dates is not None:
        end = max(end, issue_dates.max())
    grid = pd.MultiIndex.from_product([sorted(daily["station_code"].unique()), pd.date_range(start, end)],
                                      names=["station_code", "date"])
    agg = _wide(daily, "agg", "").reindex(grid)
    eve = _wide(daily, "evening", "eve_").reindex(grid)

    g = agg.groupby(level="station_code")
    parts = [agg.add_prefix("lag0_"), g.shift(1).add_prefix("lag1_"), g.shift(2).add_prefix("lag2_"),
             g.transform(lambda s: s.rolling(7, min_periods=4).mean()).add_prefix("mean7_"), eve]
    target = g.shift(-1).add_prefix("y_")

    regional = agg.groupby(level="date").mean().add_prefix("reg0_")
    X = pd.concat(parts + [target], axis=1).reset_index()
    X = X.merge(regional.reset_index(), on="date", how="left")

    # station attributes and weather point
    st = stations.set_index("station_code")
    X["point"] = X["station_code"].map(st["point"])
    X["influence"] = pd.Categorical(X["station_code"].map(st["influence"]), categories=INFLUENCES)
    X["area_type"] = pd.Categorical(X["station_code"].map(st["area_type"]), categories=AREAS)

    # weather: forecast for D+1 and for D (the latter for day-to-day changes)
    w = weather_daily.copy()
    w["date"] = pd.to_datetime(w["date"])
    w = w.set_index(["point", "date"])[list(WEATHER_VARS)]
    tomorrow = w.add_prefix("w1_")
    today = w[["t_mean", "pressure_mean", "wind_mean"]].add_prefix("w0_")
    X["target_date"] = X["date"] + pd.Timedelta(days=1)
    X = X.merge(tomorrow, left_on=["point", "target_date"], right_index=True, how="left")
    X = X.merge(today, left_on=["point", "date"], right_index=True, how="left")
    X["d_pressure"] = X["w1_pressure_mean"] - X["w0_pressure_mean"]
    X["d_temp"] = X["w1_t_mean"] - X["w0_t_mean"]
    X["d_wind"] = X["w1_wind_mean"] - X["w0_wind_mean"]
    X = X.drop(columns=["w0_pressure_mean", "w0_wind_mean"])

    # calendar of D+1
    t = X["target_date"]
    X["dow"] = t.dt.day_of_week
    X["is_off"] = ((t.dt.day_of_week >= 5) | t.isin(HOLIDAYS)).astype(int)
    doy = t.dt.day_of_year
    X["doy_sin"], X["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)

    if issue_dates is not None:
        X = X[X["date"].isin(issue_dates)]
    return X.drop(columns=["point"]).set_index(["station_code", "date"]).sort_index()


def feature_columns(X: pd.DataFrame) -> list[str]:
    return [c for c in X.columns if not c.startswith("y_") and c != "target_date"]

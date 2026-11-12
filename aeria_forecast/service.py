"""Inference: build the features of an issue date from live data and run the saved model.

Data sources
------------
* ``WarehouseSource``: last days of measurements from the Aeria warehouse (module 2) - production.
* ``CsvSource``: the committed dataset in ``data/`` - offline demo and tests.

Weather: for an issue date of yesterday or today the regular Open-Meteo forecast is used; for older
dates the forecast that was issued at the time (Previous Runs API), so a replayed forecast is
exactly what the service would have answered that day.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from aeria_forecast import data, features, weather
from aeria_forecast.atmo import ALERT_LEVEL, LEVELS, POLLUTANTS, sub_index
from aeria_forecast.model import with_index

log = logging.getLogger(__name__)
HISTORY_DAYS = 9  # 7-day mean + 2 lags
CACHE_SECONDS = 3600


class CsvSource:
    def __init__(self, root: Path):
        self.daily = data.load_csv(root / "data" / "daily.csv.gz")
        self.weather = data.load_csv(root / "data" / "weather_daily.csv.gz")

    def daily_table(self, start: date, end: date) -> pd.DataFrame:
        d = self.daily
        return d[(d["date"] >= pd.Timestamp(start)) & (d["date"] <= pd.Timestamp(end))]

    def latest_complete_day(self) -> date:
        counts = self.daily.dropna(subset=["agg"]).groupby("date")["station_code"].nunique()
        return counts[counts >= 0.5 * counts.max()].index.max().date()

    def weather_daily(self, issue: date) -> pd.DataFrame:
        w = self.weather
        days = [pd.Timestamp(issue), pd.Timestamp(issue + timedelta(days=1))]
        out = w[w["date"].isin(days)]
        if out["date"].nunique() < 2:  # beyond the dataset: ask Open-Meteo
            return live_weather(issue)
        return out


class WarehouseSource(CsvSource):
    def __init__(self, dw_url: str):
        self.dw_url = dw_url

    def daily_table(self, start: date, end: date) -> pd.DataFrame:
        # one extra day on each side: local days straddle two UTC dates
        hourly = data.load_hourly(self.dw_url, (start - timedelta(days=1)).isoformat(),
                                  (end + timedelta(days=2)).isoformat())
        d = data.daily_table(hourly)
        return d[(d["date"] >= pd.Timestamp(start)) & (d["date"] <= pd.Timestamp(end))]

    def latest_complete_day(self) -> date:
        today = date.today()
        d = self.daily_table(today - timedelta(days=5), today)
        counts = d.dropna(subset=["agg"]).groupby("date")["station_code"].nunique()
        if counts.empty:
            raise LookupError("no recent data in the warehouse")
        return counts[counts >= 0.5 * counts.max()].index.max().date()

    def weather_daily(self, issue: date) -> pd.DataFrame:
        return live_weather(issue)


def live_weather(issue: date) -> pd.DataFrame:
    if issue >= date.today() - timedelta(days=1):
        hourly = weather.fetch_forecast(forecast_days=3)
        past = weather._get(weather.FORECAST_URL, weather._params(list(weather.VARS), past_days=2, forecast_days=1))
        hourly = pd.concat([weather._frame(past, list(weather.POINTS), ""), hourly]).drop_duplicates(["point", "time"])
    else:
        hourly = weather.fetch_previous_day1(issue.isoformat(), (issue + timedelta(days=1)).isoformat())
    return weather.daily(hourly)


@dataclass
class Forecast:
    station_code: str
    station_name: str | None
    issue_date: date
    target_date: date
    atmo_index: int | None
    atmo_label: str | None
    responsible: str | None
    alert: bool
    concentrations: dict
    sub_indices: dict


class Forecaster:
    def __init__(self, bundle: dict, source):
        self.bundle = bundle
        self.model = bundle["model"]
        self.stations = bundle["stations"]
        self.source = source
        self._cache: dict = {}
        self._latest_day: tuple | None = None

    @classmethod
    def load(cls, model_path: Path, source) -> "Forecaster":
        return cls(joblib.load(model_path), source)

    def station_list(self) -> list[dict]:
        sp = self.model.station_pollutants
        rows = self.stations.to_dict(orient="records")
        return [{**{k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in r.items()},
                 "index_pollutants": sp.get(r["station_code"], [])} for r in rows]

    def forecast(self, issue: date | None = None, station: str | None = None,
                 alert_margin: float = 1.0) -> list[Forecast]:
        """Forecasts of every station (or one) for issue date + 1. Results are cached for an hour."""
        issue = issue or self._latest()
        key = (issue, round(alert_margin, 3))
        hit = self._cache.get(key)
        if hit is None or time.monotonic() - hit[0] > CACHE_SECONDS:
            hit = (time.monotonic(), self._compute(issue, alert_margin))
            self._cache[key] = hit
        if station is None:
            return hit[1]
        found = [f for f in hit[1] if f.station_code == station]
        if not found:
            raise KeyError(station)
        return found

    def _latest(self) -> date:
        if self._latest_day is None or time.monotonic() - self._latest_day[0] > CACHE_SECONDS:
            self._latest_day = (time.monotonic(), self.source.latest_complete_day())
        return self._latest_day[1]

    def _compute(self, issue: date, alert_margin: float) -> list[Forecast]:
        hist = self.source.daily_table(issue - timedelta(days=HISTORY_DAYS), issue)
        if hist.empty or pd.Timestamp(issue) not in set(hist["date"]):
            raise LookupError(f"no measurements for {issue}")
        wx = self.source.weather_daily(issue)
        X = features.build(hist, wx, self.stations, issue_dates=pd.DatetimeIndex([pd.Timestamp(issue)]))
        has_index = [bool(self.model.station_pollutants.get(s)) for s in X.index.get_level_values("station_code")]
        X = X[np.array(has_index, dtype=bool)]
        conc = self.model.predict_concentrations(X)
        res = with_index(conc, self.model.station_pollutants)
        # alert with a safety margin: alert when a forecast reaches margin x the "Dégradé" threshold
        alert = with_index(conc / alert_margin, self.model.station_pollutants)["atmo_index"] >= ALERT_LEVEL
        names = self.stations.set_index("station_code")["station_name"]
        out = []
        for (code, _), row in res.iterrows():
            pols = self.model.station_pollutants.get(code, [])
            idx = None if pd.isna(row["atmo_index"]) else int(row["atmo_index"])
            out.append(Forecast(
                station_code=code, station_name=names.get(code), issue_date=issue,
                target_date=issue + timedelta(days=1), atmo_index=idx, atmo_label=LEVELS.get(idx),
                responsible=row["responsible"] if isinstance(row["responsible"], str) else None,
                alert=bool(alert.loc[(code, pd.Timestamp(issue))]),
                concentrations={p: round(float(row[p]), 1) for p in pols if p in row},
                sub_indices={p: int(sub_index(p, row[p])) for p in pols if p in row},
            ))
        return out


def default_source(root: Path, dw_url: str | None):
    if dw_url:
        log.info("data source: warehouse")
        return WarehouseSource(dw_url)
    log.info("data source: committed CSV dataset")
    return CsvSource(root)


__all__ = ["Forecaster", "Forecast", "CsvSource", "WarehouseSource", "default_source", "POLLUTANTS"]

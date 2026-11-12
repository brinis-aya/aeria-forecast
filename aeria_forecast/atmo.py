"""French ATMO index thresholds (arrêté du 10 juillet 2020).

Same rules as the data pipeline (module 2): PM daily mean, NO2/O3/SO2 daily max of the hourly means,
concentrations rounded to the unit, level = first threshold not exceeded.
"""
from __future__ import annotations

import numpy as np

POLLUTANTS = ("pm25", "pm10", "no2", "o3", "so2")
DAILY_AGG = {"pm25": "mean", "pm10": "mean", "no2": "max", "o3": "max", "so2": "max"}
LEVELS = {1: "Bon", 2: "Moyen", 3: "Dégradé", 4: "Mauvais", 5: "Très mauvais", 6: "Extrêmement mauvais"}
THRESHOLDS = {
    "pm25": (10, 20, 25, 50, 75),
    "pm10": (20, 40, 50, 100, 150),
    "no2": (40, 90, 120, 230, 340),
    "o3": (50, 100, 130, 240, 380),
    "so2": (100, 200, 350, 500, 750),
}
ALERT_LEVEL = 3  # "Dégradé" or worse: the level from which the public is informed


def sub_index(pollutant: str, value) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    levels = np.searchsorted(np.asarray(THRESHOLDS[pollutant], dtype=float), np.round(arr), side="left") + 1.0
    return np.where(np.isnan(arr), np.nan, levels)

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aeria_forecast import weather  # noqa: E402

STATIONS = pd.DataFrame({
    "station_code": ["A", "B", "C"],
    "station_name": ["Station A", "Station B", "Station C"],
    "zone": ["ZAR LE-HAVRE", "ZAG ROUEN", "ZAR CAEN"],
    "area_type": ["Urbaine", "Urbaine", "Périurbaine"],
    "influence": ["Fond", "Trafic", "Fond"],
})
STATIONS["point"] = [weather.point_of(c, z) for c, z in zip(STATIONS.station_code, STATIONS.zone)]


@pytest.fixture(scope="session")
def synthetic():
    """300 days, 3 stations. O3 follows tomorrow's radiation; PM2.5 is persistent (AR(1))."""
    rng = np.random.default_rng(1)
    dates = pd.date_range("2025-11-01", periods=300)
    wx_rows, daily_rows = [], []
    for point in weather.POINTS:
        rad = 3.5 + 2.5 * np.sin(2 * np.pi * (dates.day_of_year - 80) / 365) + rng.normal(0, 1.0, len(dates))
        for d, r in zip(dates, rad):
            wx_rows.append({"point": point, "date": d, "t_mean": 12 + 2 * r, "t_max": 16 + 2 * r, "t_min": 8 + r,
                            "rh_mean": 80 - 4 * r, "wind_mean": 12, "wind_min": 5, "u_mean": 2, "v_mean": 1,
                            "precip_sum": 1, "pressure_mean": 1012, "radiation_sum": r, "cloud_mean": 60})
    wx = pd.DataFrame(wx_rows)
    rad_by = wx.set_index(["point", "date"])["radiation_sum"]
    for _, st in STATIONS.iterrows():
        pm = 10.0
        for d in dates:
            pm = 0.8 * pm + 2 + rng.normal(0, 2)
            o3 = 25 * rad_by[(st.point, d)] + rng.normal(0, 5)
            for p, v in (("pm25", max(pm, 0)), ("o3", max(o3, 0))):
                if st.station_code == "C" and p == "pm25":
                    continue  # C measures ozone only
                daily_rows.append({"station_code": st.station_code, "date": d, "pollutant": p, "agg": v,
                                   "mean": v, "evening": v + rng.normal(0, 1), "n_hours": 24})
    return pd.DataFrame(daily_rows), wx, STATIONS.copy()

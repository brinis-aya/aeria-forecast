import numpy as np
import pandas as pd
import pytest

from aeria_forecast import atmo, data, weather


@pytest.mark.parametrize("pollutant,value,level", [
    ("pm25", 10, 1), ("pm25", 10.6, 2), ("pm10", 41, 3), ("no2", 90, 2), ("o3", 131, 4), ("so2", 751, 6),
])
def test_sub_index(pollutant, value, level):
    assert atmo.sub_index(pollutant, value) == level


def hourly(pollutant, values, day="2026-07-01"):
    ts = pd.date_range(f"{day} 00:00", periods=len(values), freq="h", tz="Europe/Paris").tz_convert("UTC")
    return pd.DataFrame({"ts_utc": ts, "station_code": "A", "pollutant": pollutant, "value": values})


def test_daily_table_mean_max_evening_and_local_day():
    o3 = hourly("o3", [50.0] * 18 + [100.0] * 6)
    pm = hourly("pm25", [10.0] * 24)
    d = data.daily_table(pd.concat([o3, pm])).set_index("pollutant")
    assert d.loc["o3", "agg"] == 100.0          # gas: daily max
    assert d.loc["o3", "evening"] == 100.0      # 18-24 h local
    assert d.loc["pm25", "agg"] == 10.0         # particles: daily mean
    assert str(d.loc["o3", "date"].date()) == "2026-07-01"


def test_daily_table_needs_18_hours():
    d = data.daily_table(hourly("no2", [30.0] * 17))
    assert np.isnan(d["agg"].iloc[0]) and d["mean"].iloc[0] == 30.0


def test_weather_daily_wind_components_and_units():
    t = pd.date_range("2026-07-01", periods=24, freq="h", tz="Europe/Paris").tz_convert("UTC")
    h = pd.DataFrame({"point": "le_havre", "time": t, "temperature_2m": 20.0, "relative_humidity_2m": 70,
                      "wind_speed_10m": 10.0, "wind_direction_10m": 270.0,  # wind FROM the west
                      "precipitation": 0.5, "surface_pressure": 1015, "shortwave_radiation": 250.0,
                      "cloud_cover": 40})
    d = weather.daily(h).iloc[0]
    assert d.u_mean == pytest.approx(10.0)    # blows towards the east
    assert d.v_mean == pytest.approx(0.0, abs=1e-9)
    assert d.precip_sum == pytest.approx(12.0)
    assert d.radiation_sum == pytest.approx(6.0)  # 24 x 250 W/m² = 6 kWh/m²


def test_station_mapping():
    assert weather.point_of("FR05089", "ZAR LE-HAVRE") == "le_havre"
    assert weather.point_of("FR21038", "ZR NORMANDIE") == "cherbourg"
    assert weather.point_of("UNKNOWN", "ZAG ROUEN") == "rouen"

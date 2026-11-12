import numpy as np
import pandas as pd

from aeria_forecast import evaluate, features
from aeria_forecast.model import ForecastModel, observed_index, persistence, with_index

FAST = dict(max_iter=80, learning_rate=0.1, max_leaf_nodes=15, min_samples_leaf=10,
            l2_regularization=1.0, early_stopping=False, random_state=0)


def fast_model():
    return ForecastModel(params=dict(FAST))


def test_target_is_next_day_and_features_do_not_leak(synthetic):
    daily, wx, st = synthetic
    X = features.build(daily, wx, st)
    d0 = pd.Timestamp("2026-03-01")
    a = X.loc[("A", d0)]
    nxt = daily[(daily.station_code == "A") & (daily.date == d0 + pd.Timedelta(days=1)) & (daily.pollutant == "o3")]
    assert a["y_o3"] == nxt["agg"].iloc[0]

    # changing everything that happens after D must not change the features of D (except D+1 weather)
    future = daily.copy()
    future.loc[future.date > d0, "agg"] += 1000
    future.loc[future.date > d0, "evening"] += 1000
    X2 = features.build(future, wx, st)
    cols = [c for c in features.feature_columns(X) if not c.startswith("y_")]
    pd.testing.assert_series_equal(X.loc[("A", d0), cols], X2.loc[("A", d0), cols], check_names=False)


def test_weather_is_for_the_target_day(synthetic):
    daily, wx, st = synthetic
    X = features.build(daily, wx, st)
    d0 = pd.Timestamp("2026-02-10")
    expected = wx[(wx.point == "le_havre") & (wx.date == d0 + pd.Timedelta(days=1))]["radiation_sum"].iloc[0]
    assert X.loc[("A", d0), "w1_radiation_sum"] == expected


def test_holiday_flag(synthetic):
    daily, wx, st = synthetic
    X = features.build(daily, wx, st)
    # issue 31 Dec 2025 -> target 1 Jan 2026 (holiday, Thursday)
    assert X.loc[("A", pd.Timestamp("2025-12-31")), "is_off"] == 1
    assert X.loc[("A", pd.Timestamp("2026-01-06")), "is_off"] == 0  # target Wednesday 7 Jan


def test_model_learns_weather_signal_and_beats_persistence(synthetic):
    daily, wx, st = synthetic
    X = features.build(daily, wx, st)
    dates = X.index.get_level_values("date")
    train, test = X[dates < "2026-06-01"], X[dates >= "2026-06-01"]
    m = fast_model().fit(train)
    pred = m.predict_concentrations(test)["o3"]
    mae = (pred - test["y_o3"]).abs().mean()
    mae_pers = (test["lag0_o3"] - test["y_o3"]).abs().mean()
    assert mae < 0.6 * mae_pers
    assert m.station_pollutants["C"] == ["o3"]
    assert set(m.station_pollutants["A"]) == {"pm25", "o3"}


def test_index_uses_only_station_pollutants():
    idx = pd.MultiIndex.from_tuples([("A", pd.Timestamp("2026-07-01")), ("C", pd.Timestamp("2026-07-01"))],
                                    names=["station_code", "date"])
    conc = pd.DataFrame({"pm25": [60.0, 60.0], "o3": [40.0, 40.0]}, index=idx)
    out = with_index(conc, {"A": ["pm25", "o3"], "C": ["o3"]})
    assert out["atmo_index"].tolist() == [5, 1]           # PM2.5 60 -> Très mauvais, ignored at C
    assert out["responsible"].tolist() == ["pm25", "o3"]
    assert out["alert"].tolist() == [True, False]


def test_observed_index_requires_all_station_pollutants(synthetic):
    daily, wx, st = synthetic
    X = features.build(daily, wx, st)
    row = X.loc[[("A", pd.Timestamp("2026-03-01"))]].copy()
    sp = {"A": ["pm25", "o3"]}
    assert not np.isnan(observed_index(row, sp).iloc[0])
    row["y_pm25"] = np.nan
    assert np.isnan(observed_index(row, sp).iloc[0])


def test_rolling_origin_never_trains_on_the_future(synthetic):
    daily, wx, st = synthetic
    X = features.build(daily, wx, st)
    seen = []

    class Spy(ForecastModel):
        def fit(self, train):
            seen.append(train.index.get_level_values("date").max())
            return super().fit(train)

    pred = evaluate.rolling_origin(X, "2026-05", "2026-07", model_factory=lambda: Spy(params=dict(FAST)),
                                   min_train_rows=300)
    months = sorted(pred["month"].unique())
    assert months == ["2026-05", "2026-06", "2026-07"]
    for last_train, m in zip(seen, months):
        # the target (D+1) of the last training row is before the first test day
        assert last_train + pd.Timedelta(days=1) < pd.Period(m).start_time
    metrics = evaluate.index_metrics(pred).set_index("method")
    assert set(metrics.index) == {"model", "persistence", "climatology"}
    assert metrics.loc["model", "accuracy"] >= metrics.loc["climatology", "accuracy"]


def test_persistence_baseline(synthetic):
    daily, wx, st = synthetic
    X = features.build(daily, wx, st).dropna(subset=["lag0_o3"]).head(5)
    p = persistence(X, {"A": ["pm25", "o3"], "B": ["pm25", "o3"], "C": ["o3"]})
    assert (p["o3"] == X["lag0_o3"]).all()

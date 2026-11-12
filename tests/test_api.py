from datetime import date

import joblib
import pytest
import sklearn
from fastapi.testclient import TestClient

from aeria_forecast import api
from aeria_forecast.service import CsvSource, Forecaster
from aeria_forecast.train import load_dataset

from .conftest import ROOT

ISSUE = "2026-08-12"  # inside the committed dataset: no network call


@pytest.fixture(scope="module")
def client():
    bundle = joblib.load(ROOT / "models" / "aeria-forecast.joblib")
    api.state["forecaster"] = Forecaster(bundle, CsvSource(ROOT))
    with TestClient(api.app) as c:
        yield c
    api.state.clear()


def test_committed_model_matches_pinned_sklearn():
    bundle = joblib.load(ROOT / "models" / "aeria-forecast.joblib")
    assert bundle["sklearn"] == sklearn.__version__
    assert set(bundle["model"].models) == {"pm25", "pm10", "no2", "o3", "so2"}


def test_health_and_model_card(client):
    assert client.get("/health").json()["status"] == "ok"
    card = client.get("/model").json()
    assert card["n_features"] > 40
    assert any(r["method"] == "model" for r in card["backtest"]["index"])


def test_stations(client):
    st = {s["station_code"]: s for s in client.get("/stations").json()}
    assert len(st) == 36
    assert set(st["FR05090"]["index_pollutants"]) == {"pm25", "pm10", "no2", "o3", "so2"}


def test_forecast_one_station(client):
    r = client.get("/forecast/FR05090", params={"issue_date": ISSUE})
    assert r.status_code == 200
    f = r.json()
    assert f["target_date"] == "2026-08-13"
    assert 1 <= f["atmo_index"] <= 6
    assert f["responsible"] in f["concentrations"]
    assert f["sub_indices"][f["responsible"]] == f["atmo_index"]


def test_forecast_all_and_alert_margin(client):
    strict = client.get("/forecast", params={"issue_date": ISSUE, "alert_margin": 1.0}).json()
    loose = client.get("/forecast", params={"issue_date": ISSUE, "alert_margin": 0.7}).json()
    assert len(strict) == len(loose) >= 30
    assert sum(f["alert"] for f in loose) >= sum(f["alert"] for f in strict)


def test_errors(client):
    assert client.get("/forecast/NOPE", params={"issue_date": ISSUE}).status_code == 404
    assert client.get("/forecast", params={"issue_date": "2019-01-01"}).status_code == 404
    assert client.get("/forecast", params={"issue_date": ISSUE, "alert_margin": 2}).status_code == 422


def test_replayed_forecast_matches_offline_prediction():
    """The service and the training code build exactly the same features."""
    bundle = joblib.load(ROOT / "models" / "aeria-forecast.joblib")
    f = Forecaster(bundle, CsvSource(ROOT)).forecast(date.fromisoformat(ISSUE), "FR05089")[0]
    X, _ = load_dataset()
    offline = bundle["model"].predict_concentrations(X.loc[[("FR05089", __import__("pandas").Timestamp(ISSUE))]])
    assert f.concentrations["no2"] == pytest.approx(round(float(offline["no2"].iloc[0]), 1))

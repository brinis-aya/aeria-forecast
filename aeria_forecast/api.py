"""REST API of the 24 h ATMO forecast.

    uvicorn aeria_forecast.api:app --port 8083
    docs: http://127.0.0.1:8083/docs
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import date
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from pydantic import BaseModel, Field

from aeria_forecast import __version__
from aeria_forecast.service import Forecaster, default_source

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT / "models" / "aeria-forecast.joblib"))
DEFAULT_MARGIN = 0.95  # recall/precision balance chosen on the backtest (docs/rapport.md §6)


class ForecastOut(BaseModel):
    station_code: str
    station_name: str | None
    issue_date: date = Field(description="last day of measurements used")
    target_date: date = Field(description="forecast day (issue date + 1)")
    atmo_index: int | None = Field(description="1 Bon ... 6 Extrêmement mauvais")
    atmo_label: str | None
    responsible: str | None = Field(description="pollutant that sets the index")
    alert: bool = Field(description="'Dégradé' or worse expected (with the alert margin)")
    concentrations: dict[str, float] = Field(description="forecast ATMO aggregates, µg/m³")
    sub_indices: dict[str, int]


state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if "forecaster" not in state:
        state["forecaster"] = Forecaster.load(MODEL_PATH, default_source(ROOT, os.getenv("DW_URL")))
    yield


app = FastAPI(title="Aeria · 24 h air-quality forecast", version=__version__, lifespan=lifespan,
              description="Forecast of tomorrow's French ATMO index for the Normandy reference stations.")


# Security headers (audit finding F5 of the security module). The API only serves public data, but
# browsers must not sniff, frame or leak it; the interactive /docs page needs inline scripts from its CDN.
SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Permissions-Policy": "geolocation=(), camera=(), microphone=()",
    "Cache-Control": "no-store",
}
API_CSP = "default-src 'none'; frame-ancestors 'none'"


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers.update(SECURITY_HEADERS)
    if not request.url.path.startswith(("/docs", "/redoc", "/openapi.json")):
        response.headers["Content-Security-Policy"] = API_CSP
    return response


def forecaster() -> Forecaster:
    return state["forecaster"]


@app.get("/health")
def health():
    b = forecaster().bundle
    return {"status": "ok", "model_version": b["version"], "trained_until": b["trained_until"]}


@app.get("/model")
def model_card():
    b = forecaster().bundle
    return {"version": b["version"], "trained_at": b["trained_at"], "trained_until": b["trained_until"],
            "n_train_rows": b["n_train_rows"], "sklearn": b["sklearn"],
            "n_features": len(b["model"].features), "pollutants": list(b["model"].models),
            "backtest": b["metrics"]}


@app.get("/stations")
def stations():
    return forecaster().station_list()


def _run(issue: date | None, station: str | None, margin: float) -> list[ForecastOut]:
    try:
        return [ForecastOut(**asdict(f)) for f in forecaster().forecast(issue, station, margin)]
    except KeyError:
        raise HTTPException(404, f"unknown station or no index for station {station}")
    except LookupError as exc:
        raise HTTPException(404, str(exc))
    except RuntimeError as exc:  # weather provider unreachable
        raise HTTPException(503, str(exc))


@app.get("/forecast", response_model=list[ForecastOut])
def forecast_all(issue_date: date | None = Query(None, description="default: latest complete day"),
                 alert_margin: float = Query(DEFAULT_MARGIN, gt=0.5, le=1.0)):
    return _run(issue_date, None, alert_margin)


@app.get("/forecast/{station_code}", response_model=ForecastOut)
def forecast_station(station_code: str, issue_date: date | None = None,
                     alert_margin: float = Query(DEFAULT_MARGIN, gt=0.5, le=1.0)):
    return _run(issue_date, station_code, alert_margin)[0]

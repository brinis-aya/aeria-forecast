# Aeria · 24 h air-quality forecast

[![CI](https://github.com/brinis-aya/aeria-forecast/actions/workflows/ci.yml/badge.svg)](https://github.com/brinis-aya/aeria-forecast/actions/workflows/ci.yml)

Module 3 of **Aeria**, an open air-quality monitoring platform. A **scikit-learn** model forecasts
**tomorrow's French ATMO index** (Bon → Extrêmement mauvais) for the 36 regulatory stations of Normandy,
and a **FastAPI** service serves the forecasts.

> Other modules: [IoT sensors](https://github.com/brinis-aya/aeria-iot-sensors) ·
> [data pipeline](https://github.com/brinis-aya/aeria-data-pipeline) ·
> [back-office](https://github.com/brinis-aya/aeria-backoffice) ·
> [security](https://github.com/brinis-aya/aeria-security) ·
> [dashboard](https://github.com/brinis-aya/aeria-dashboard) ·
> [overview](https://github.com/brinis-aya/aeria)

```
 warehouse (module 2)              Open-Meteo forecast
 last 9 days of measurements       for tomorrow (issued today)
            │                                │
            └────────────┬───────────────────┘
                         ▼
            52 features per station ──▶ 5 gradient-boosting regressors ──▶ ATMO sub-indices ──▶ index J+1
                                         (PM2.5, PM10, NO₂, O₃, SO₂)        worst one wins      + alert
```

## Results

Rolling-origin backtest, **March → November 2026**: each month is forecast by a model trained only on
the data before it. 8,180 station-days with an observed index. Real data only (LCSQA / Geod'air).

| ATMO index, D+1 | Model | Persistence (D+1 = D) | Climatology |
|---|---|---|---|
| Exact index | **83.1 %** | 79.6 % | 75.2 % |
| Within one level | 98.4 % | 98.7 % | 95.2 % |
| "Dégradé or worse" alerts, F1 | **0.71** | 0.71 | 0 |
| Alert precision / recall | 81 % / 64 % | 71 % / 70 % | – |
| False alarms | **203** | 389 | 0 |

| Mean absolute error (µg/m³) | PM2.5 | PM10 | NO₂ | O₃ | SO₂ |
|---|---|---|---|---|---|
| Model | 2.80 | 4.16 | **7.01** | **10.18** | **6.40** |
| Persistence | 2.72 | 4.31 | 9.48 | 12.67 | 8.62 |
| Gain vs persistence | −3 % | +4 % | **+26 %** | **+20 %** | **+26 %** |

<p align="center"><img src="docs/figures/01_precision_mensuelle.png" width="720" alt="Monthly index accuracy"></p>

**What the numbers say, honestly:**

- The model beats persistence **every month**, most clearly in summer (+7 to +8 points) when ozone,
  driven by sunshine and wind, sets the index. Weather forecasts are where the skill comes from.
- Particles are **not** better forecast than "same as today": their day-to-day inertia is strong and
  the evening level of the day before is by far the most useful variable.
- The model is cautious: fewer false alarms but some episodes missed. An adjustable **alert margin**
  trades one for the other; at 0.95 (API default) recall and precision are both ≈ 72 %, slightly above
  persistence on both. At the **Le Havre** stations alone (790 days), the index is better than
  persistence (76.6 % vs 74.3 %), but episode detection is not (F1 0.55 vs 0.68): 35 % of Le Havre
  episodes are caused by PM2.5 (13 % region-wide), the pollutant where the model adds nothing.

Full analysis in French: [`docs/rapport.md`](docs/rapport.md) · raw numbers: [`docs/metrics.json`](docs/metrics.json).

## No data leakage

- Features only use what is known on the evening of day D (measurements up to midnight).
- Weather comes from the Open-Meteo **Previous Runs API**: `*_previous_day1` values are forecasts issued
  the day before, not observations. The backtest uses exactly the weather the service would have had.
- Tests check that changing anything after D leaves the features of D unchanged, and that no backtest
  model is trained on a date whose target falls in its test month.

## API

```bash
pip install -r requirements.txt
uvicorn aeria_forecast.api:app --port 8083        # docs: http://127.0.0.1:8083/docs
```

| Endpoint | |
|---|---|
| `GET /forecast?issue_date=&alert_margin=` | tomorrow's index for every station |
| `GET /forecast/{station_code}` | one station, e.g. `FR05090` (Le Havre ville-haute) |
| `GET /stations` | stations and the pollutants in their index |
| `GET /model` | model card: version, training period, backtest metrics |
| `GET /health` | liveness |

```json
GET /forecast/FR05090?issue_date=2026-08-12
{"station_code": "FR05090", "station_name": "Le Havre ville-haute", "issue_date": "2026-08-12",
 "target_date": "2026-08-13", "atmo_index": 4, "atmo_label": "Mauvais", "responsible": "o3", "alert": true,
 "concentrations": {"pm25": 8.4, "pm10": 26.5, "no2": 49.4, "o3": 165.7, "so2": 65.1},
 "sub_indices": {"pm25": 1, "pm10": 2, "no2": 2, "o3": 4, "so2": 1}}
```

Without `DW_URL` the service uses the dataset committed in `data/` (offline demo); with
`DW_URL=postgresql://…` it reads the latest measurements from the module 2 warehouse and fetches the
current weather forecast. Replaying a past date uses the final model, which has seen that date in
training — the backtest above is the honest measure of quality.

```bash
docker build -t aeria-forecast . && docker run -p 8083:8083 aeria-forecast
```

## Retrain

```bash
pip install -r requirements-dev.txt
python scripts/build_dataset.py --dw postgresql://aeria:aeria@127.0.0.1:5440/aeria   # refresh data/
python -m aeria_forecast.train         # backtest (≈ 2 min) + final model -> models/, docs/metrics.json
python scripts/make_figures.py         # docs/figures/
pytest                                 # 25 tests
```

## Layout

```
aeria_forecast/
  data.py       daily ATMO aggregates from the warehouse        weather.py   Open-Meteo, 8 points
  features.py   lags, evening level, regional mean, weather J+1, calendar, station type
  model.py      5 HistGradientBoosting regressors -> ATMO index   evaluate.py  backtest, baselines, metrics
  train.py      training CLI                                      service.py   inference + data sources
  api.py        FastAPI
data/           committed dataset (daily measurements, weather, stations)
models/         trained model (joblib, pinned scikit-learn 1.9.1)
docs/           rapport.md (FR), metrics.json, backtest.csv.gz, figures/
```

## License

MIT. Data: LCSQA / Geod'air (Licence Ouverte 2.0); weather: [Open-Meteo](https://open-meteo.com) (CC BY 4.0).

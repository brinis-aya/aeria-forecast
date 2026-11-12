"""Time-aware evaluation.

Rolling origin (expanding window): for every test month M, a model is trained on all issue dates
before M and predicts every day of M. Nothing from the future is ever seen, and the scores cover
spring, summer and autumn instead of a single test period.

Baselines
---------
* persistence: tomorrow's concentration = today's (hard to beat at 24 h for slow pollutants)
* climatology: mean of the same station, pollutant and calendar month in the training data
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error, r2_score

from aeria_forecast.atmo import ALERT_LEVEL, POLLUTANTS
from aeria_forecast.model import ForecastModel, observed_index, persistence, with_index

log = logging.getLogger(__name__)


def climatology(train: pd.DataFrame, test: pd.DataFrame, station_pollutants: dict) -> pd.DataFrame:
    month_tr = train.index.get_level_values("date").month
    month_te = test.index.get_level_values("date").month
    conc = pd.DataFrame(index=test.index)
    for p in POLLUTANTS:
        clim = train[f"y_{p}"].groupby([train.index.get_level_values("station_code"), month_tr]).mean()
        overall = train[f"y_{p}"].groupby(level="station_code").mean()
        keys = list(zip(test.index.get_level_values("station_code"), month_te))
        vals = clim.reindex(keys).to_numpy()
        fallback = overall.reindex(test.index.get_level_values("station_code")).to_numpy()
        conc[p] = np.where(np.isnan(vals), fallback, vals)
    return with_index(conc, station_pollutants)


def rolling_origin(X: pd.DataFrame, first_test_month: str = "2026-03", last_test_month: str | None = None,
                   model_factory=ForecastModel, min_train_rows: int = 1000) -> pd.DataFrame:
    """Predictions of the model and the baselines for every test month (long format)."""
    dates = X.index.get_level_values("date")
    months = pd.period_range(first_test_month, last_test_month or dates.max().to_period("M"), freq="M")
    frames = []
    for m in months:
        start = m.start_time
        train = X[dates < start - pd.Timedelta(days=1)]  # the D+1 target of the last train day is < start
        test = X[(dates >= start) & (dates <= m.end_time)]
        if test.empty or len(train) < min_train_rows:
            continue
        model = model_factory().fit(train)
        sp = model.station_pollutants
        preds = {"model": model.predict(test), "persistence": persistence(test, sp),
                 "climatology": climatology(train, test, sp)}
        obs = observed_index(test, sp)
        for name, pr in preds.items():
            f = pr.reindex(columns=list(POLLUTANTS) + ["atmo_index"]).add_prefix("pred_")
            f["method"] = name
            f["month"] = str(m)
            f["obs_index"] = obs
            for p in POLLUTANTS:
                f[f"obs_{p}"] = test[f"y_{p}"]
            frames.append(f)
        log.info("%s: train %d rows, test %d rows", m, len(train), len(test))
    return pd.concat(frames).reset_index()


def regression_metrics(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (method, p), _ in pd.MultiIndex.from_product([pred["method"].unique(), POLLUTANTS]).to_frame().iterrows():
        d = pred[pred["method"] == method][[f"pred_{p}", f"obs_{p}"]].dropna()
        if len(d) < 30:
            continue
        rows.append({"method": method, "pollutant": p, "n": len(d),
                     "MAE": mean_absolute_error(d[f"obs_{p}"], d[f"pred_{p}"]),
                     "RMSE": mean_squared_error(d[f"obs_{p}"], d[f"pred_{p}"]) ** 0.5,
                     "R2": r2_score(d[f"obs_{p}"], d[f"pred_{p}"]),
                     "bias": (d[f"pred_{p}"] - d[f"obs_{p}"]).mean()})
    out = pd.DataFrame(rows)
    pers = out[out.method == "persistence"].set_index("pollutant")["MAE"]
    out["skill_vs_persistence"] = 1 - out["MAE"] / out["pollutant"].map(pers)
    return out


def index_metrics(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, d in pred.groupby("method"):
        d = d.dropna(subset=["obs_index", "pred_atmo_index"])
        obs, est = d["obs_index"].astype(int), d["pred_atmo_index"].astype(int)
        alert_o, alert_p = obs >= ALERT_LEVEL, est >= ALERT_LEVEL
        tp = int((alert_o & alert_p).sum())
        rows.append({
            "method": method, "n": len(d),
            "accuracy": (obs == est).mean(),
            "within_1": ((obs - est).abs() <= 1).mean(),
            "macro_f1": f1_score(obs, est, average="macro"),
            "alert_days": int(alert_o.sum()),
            "alert_recall": tp / max(int(alert_o.sum()), 1),
            "alert_precision": tp / max(int(alert_p.sum()), 1),
            "alert_f1": f1_score(alert_o, alert_p),
            "false_alarms": int((~alert_o & alert_p).sum()),
        })
    return pd.DataFrame(rows)


def confusion(pred: pd.DataFrame, method: str = "model") -> pd.DataFrame:
    d = pred[pred["method"] == method].dropna(subset=["obs_index", "pred_atmo_index"])
    return pd.crosstab(d["obs_index"].astype(int), d["pred_atmo_index"].astype(int),
                       rownames=["observed"], colnames=["predicted"])

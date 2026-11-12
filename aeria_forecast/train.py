"""Train the final model, run the backtest and write the metrics.

    python -m aeria_forecast.train                 # backtest + final model
    python -m aeria_forecast.train --skip-backtest # final model only
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import sklearn

from aeria_forecast import __version__, data, evaluate, features
from aeria_forecast.model import ForecastModel

ROOT = Path(__file__).resolve().parents[1]
LE_HAVRE = ("FR05074", "FR05083", "FR05089", "FR05090")
log = logging.getLogger("aeria_forecast")


def load_dataset(root: Path = ROOT) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily = data.load_csv(root / "data" / "daily.csv.gz")
    weather = data.load_csv(root / "data" / "weather_daily.csv.gz")
    stations = pd.read_csv(root / "data" / "stations.csv", dtype={"station_code": str})
    return features.build(daily, weather, stations), stations


def summarize(pred: pd.DataFrame) -> dict:
    reg = evaluate.regression_metrics(pred)
    idx = evaluate.index_metrics(pred)
    lh = pred[pred["station_code"].isin(LE_HAVRE)]
    by_month = (pred.dropna(subset=["obs_index", "pred_atmo_index"])
                .assign(ok=lambda d: d["obs_index"] == d["pred_atmo_index"])
                .groupby(["month", "method"])["ok"].mean().unstack())
    return {
        "regression": reg.round(4).to_dict(orient="records"),
        "index": idx.round(4).to_dict(orient="records"),
        "index_le_havre": evaluate.index_metrics(lh).round(4).to_dict(orient="records"),
        "regression_le_havre": evaluate.regression_metrics(lh).round(4).to_dict(orient="records"),
        "accuracy_by_month": by_month.round(4).reset_index().to_dict(orient="records"),
        "confusion": {str(k): {str(c): int(v) for c, v in row.items()}
                      for k, row in evaluate.confusion(pred).iterrows()},
        "test_period": [pred["date"].min().date().isoformat(), pred["date"].max().date().isoformat()],
    }


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--skip-backtest", action="store_true")
    p.add_argument("--out", type=Path, default=ROOT / "models" / "aeria-forecast.joblib")
    args = p.parse_args(argv)

    X, stations = load_dataset()
    log.info("feature table: %d rows x %d columns", *X.shape)

    metrics = {}
    if not args.skip_backtest:
        pred = evaluate.rolling_origin(X)
        pred.to_csv(ROOT / "docs" / "backtest.csv.gz", index=False, compression="gzip", float_format="%.2f")
        metrics = summarize(pred)
        (ROOT / "docs" / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    model = ForecastModel().fit(X)
    trained_until = X.index.get_level_values("date").max()
    bundle = {
        "model": model,
        "stations": stations,
        "version": __version__,
        "sklearn": sklearn.__version__,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "trained_until": trained_until.date().isoformat(),
        "n_train_rows": int(len(X)),
        "metrics": {k: metrics[k] for k in ("index", "regression", "test_period")} if metrics else {},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.out, compress=3)
    log.info("model saved to %s (%.1f MB)", args.out, args.out.stat().st_size / 1e6)


if __name__ == "__main__":
    main()

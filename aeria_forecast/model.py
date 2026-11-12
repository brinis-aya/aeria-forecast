"""24 h (D+1) ATMO forecast model.

One gradient-boosting regressor per pollutant predicts tomorrow's ATMO aggregate (PM daily mean,
gas daily maximum). The ATMO index is then derived exactly like the official one: sub-index per
pollutant, worst sub-index wins. Predicting concentrations rather than the class directly keeps the
model interpretable (µg/m³ errors) and lets one model serve every station whatever its pollutants.

NO2 and SO2 regressors are trained on log1p(concentration): they are very right-skewed (SO2 is near
zero most days with industrial peaks) and the log target reduces the weight of a few extreme days.
For O3 and particles the log target biases forecasts downwards and misses pollution episodes, so they
are trained on raw concentrations (choice made on the rolling-origin backtest, see docs/rapport.md).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from aeria_forecast.atmo import ALERT_LEVEL, LEVELS, POLLUTANTS, sub_index
from aeria_forecast.features import feature_columns

HGB_PARAMS = dict(max_iter=500, learning_rate=0.04, max_leaf_nodes=31, min_samples_leaf=30,
                  l2_regularization=1.0, early_stopping=False, random_state=42)


@dataclass
class ForecastModel:
    features: list[str] = field(default_factory=list)
    models: dict = field(default_factory=dict)
    station_pollutants: dict = field(default_factory=dict)
    log_target: tuple[str, ...] = ("no2", "so2")
    params: dict = field(default_factory=lambda: dict(HGB_PARAMS))

    def fit(self, X: pd.DataFrame) -> "ForecastModel":
        # features never observed in the training period carry no information (and cannot be binned)
        self.features = [c for c in feature_columns(X) if X[c].notna().any()]
        for p in POLLUTANTS:
            rows = X[f"y_{p}"].notna()
            if rows.sum() < 50:
                continue
            y = X.loc[rows, f"y_{p}"].clip(lower=0)
            reg = HistGradientBoostingRegressor(categorical_features="from_dtype", **self.params)
            reg.fit(X.loc[rows, self.features], np.log1p(y) if p in self.log_target else y)
            self.models[p] = reg
        # a pollutant belongs to a station's index if the station measured it on >= 50 % of the days
        cover = X[[f"lag0_{p}" for p in POLLUTANTS]].notna().groupby(level="station_code").mean()
        self.station_pollutants = {s: [p for p in POLLUTANTS if cover.loc[s, f"lag0_{p}"] >= 0.5]
                                   for s in cover.index}
        return self

    def predict_concentrations(self, X: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=X.index)
        for p, reg in self.models.items():
            pred = reg.predict(X[self.features])
            out[p] = np.clip(np.expm1(pred) if p in self.log_target else pred, 0, None)
        return out

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        conc = self.predict_concentrations(X)
        return with_index(conc, self.station_pollutants)


def with_index(conc: pd.DataFrame, station_pollutants: dict) -> pd.DataFrame:
    """Add sub-indices, the ATMO index, its label and the responsible pollutant.

    Only the pollutants measured at the station enter its index (as in module 2), so a forecast and
    the observed index are always computed on the same pollutants.
    """
    out = conc.copy()
    stations = out.index.get_level_values("station_code")
    subs = pd.DataFrame(index=out.index)
    for p in POLLUTANTS:
        if p not in out:
            continue
        idx = sub_index(p, out[p])
        mask = np.array([p in station_pollutants.get(s, ()) for s in stations])
        subs[p] = np.where(mask, idx, np.nan)
    out["atmo_index"] = subs.max(axis=1)
    responsible = subs.fillna(-1).idxmax(axis=1) if len(subs.columns) else pd.Series(np.nan, index=out.index)
    out["responsible"] = responsible.where(out["atmo_index"].notna())
    out["atmo_label"] = out["atmo_index"].map(LEVELS)
    out["alert"] = out["atmo_index"] >= ALERT_LEVEL
    return out


def observed_index(X: pd.DataFrame, station_pollutants: dict) -> pd.Series:
    """Observed D+1 index on the station's pollutants; NaN unless all of them were measured."""
    y = pd.DataFrame({p: X[f"y_{p}"] for p in POLLUTANTS}, index=X.index)
    complete = pd.Series(
        [all(pd.notna(y.at[i, p]) for p in station_pollutants.get(i[0], ())) and bool(station_pollutants.get(i[0]))
         for i in y.index], index=y.index)
    idx = with_index(y, station_pollutants)["atmo_index"]
    return idx.where(complete)


def persistence(X: pd.DataFrame, station_pollutants: dict) -> pd.DataFrame:
    """Baseline: tomorrow = today."""
    conc = pd.DataFrame({p: X[f"lag0_{p}"] for p in POLLUTANTS}, index=X.index)
    return with_index(conc, station_pollutants)


def importances(model: ForecastModel, X: pd.DataFrame, pollutant: str, n_repeats: int = 5,
                seed: int = 0) -> pd.Series:
    """Permutation importance: increase of the MAE (µg/m³) when one feature is shuffled."""
    rows = X[f"y_{pollutant}"].notna()
    Xp, y = X.loc[rows, model.features].copy(), X.loc[rows, f"y_{pollutant}"].to_numpy()
    rng = np.random.default_rng(seed)

    def mae(frame):
        return np.abs(model.predict_concentrations(frame)[pollutant].to_numpy() - y).mean()

    base = mae(Xp)
    scores = {}
    for col in model.features:
        original = Xp[col]
        losses = []
        for _ in range(n_repeats):
            Xp[col] = original.iloc[rng.permutation(len(original))].to_numpy()
            losses.append(mae(Xp) - base)
        Xp[col] = original
        scores[col] = float(np.mean(losses))
    return pd.Series(scores).sort_values(ascending=False)

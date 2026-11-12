"""Report figures from docs/backtest.csv.gz (written by ``python -m aeria_forecast.train``).

    python scripts/make_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from aeria_forecast import evaluate  # noqa: E402
from aeria_forecast.atmo import ALERT_LEVEL, POLLUTANTS, THRESHOLDS  # noqa: E402
from aeria_forecast.model import ForecastModel, importances  # noqa: E402
from aeria_forecast.train import load_dataset  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COLORS = {"model": "#2a78d6", "persistence": "#eb6834", "climatology": "#1baf7a"}
LABELS = {"model": "Modèle (gradient boosting)", "persistence": "Persistance (J+1 = J)",
          "climatology": "Climatologie (moyenne du mois)"}
POLL_LABEL = {"pm25": "PM2.5", "pm10": "PM10", "no2": "NO₂", "o3": "O₃", "so2": "SO₂"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "lines.linewidth": 2, "legend.frameon": False,
})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=150)
    plt.close(fig)
    print("wrote", name)


def fig_monthly(pred):
    d = pred.dropna(subset=["obs_index", "pred_atmo_index"])
    acc = (d.assign(ok=d.obs_index == d.pred_atmo_index).groupby(["month", "method"])["ok"].mean() * 100).unstack()
    fig, ax = plt.subplots(figsize=(9, 3.8))
    for m in ("model", "persistence", "climatology"):
        ax.plot(acc.index, acc[m], marker="o", markersize=6, color=COLORS[m], label=LABELS[m])
    ax.set_ylabel("% de jours avec le bon indice")
    ax.set_title("Indice ATMO J+1 - précision par mois de test (backtest glissant)")
    ax.legend(loc="lower left")
    save(fig, "01_precision_mensuelle.png")


def fig_mae(pred):
    reg = evaluate.regression_metrics(pred).set_index(["pollutant", "method"])["MAE"].unstack()
    reg = reg.reindex(list(POLLUTANTS))
    fig, ax = plt.subplots(figsize=(9, 3.8))
    x = np.arange(len(reg))
    for i, m in enumerate(("model", "persistence", "climatology")):
        ax.bar(x + (i - 1) * 0.27, reg[m], width=0.25, color=COLORS[m], label=LABELS[m],
               edgecolor=SURFACE, linewidth=2)
    for xi, (p, row) in zip(x, reg.iterrows()):
        skill = 1 - row["model"] / row["persistence"]
        ax.text(xi - 0.27, row["model"] + 0.4, f"{skill:+.0%}", ha="center", fontsize=8, color=INK2)
    ax.set_xticks(x, [POLL_LABEL[p] for p in reg.index])
    ax.set_ylabel("Erreur absolue moyenne (µg/m³)")
    ax.set_title("Erreur de prévision J+1 par polluant (étiquette : gain du modèle vs persistance)")
    ax.legend(loc="upper left")
    save(fig, "02_erreur_par_polluant.png")


def fig_confusion(pred):
    cm = evaluate.confusion(pred).reindex(index=range(1, 6), columns=range(1, 6), fill_value=0)
    share = cm.div(cm.sum(axis=1).replace(0, np.nan), axis=0)
    labels = ["Bon", "Moyen", "Dégradé", "Mauvais", "Très m."]
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    ax.imshow(share.fillna(0).to_numpy(), cmap="Blues", vmin=0, vmax=1)
    for i in range(5):
        for j in range(5):
            n = cm.iat[i, j]
            if n:
                ax.text(j, i, f"{n}", ha="center", va="center", fontsize=9,
                        color="#ffffff" if share.iat[i, j] > 0.55 else INK)
    ax.set_xticks(range(5), labels, rotation=30)
    ax.set_yticks(range(5), labels)
    ax.set_xlabel("Prévu")
    ax.set_ylabel("Observé")
    ax.grid(False)
    ax.set_title("Matrice de confusion (jours)")
    save(fig, "03_confusion.png")


def fig_series(pred, station="FR05090", pollutant="o3"):
    d = pred[(pred.station_code == station) & (pred.date >= "2026-06-01")]
    piv = d.pivot_table(index="date", columns="method", values=f"pred_{pollutant}")
    obs = d[d.method == "model"].set_index("date")[f"obs_{pollutant}"]
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.plot(obs.index, obs, color=INK2, linewidth=1.4, label="Observé")
    ax.plot(piv.index, piv["model"], color=COLORS["model"], label=LABELS["model"])
    ax.plot(piv.index, piv["persistence"], color=COLORS["persistence"], linewidth=1.2, alpha=0.8,
            label=LABELS["persistence"])
    for t, lab in zip(THRESHOLDS[pollutant][:3], ("Bon", "Moyen", "Dégradé")):
        ax.axhline(t, color=GRID, linewidth=1, linestyle="--")
        ax.text(piv.index.max(), t, f" {lab} ≤ {t}", va="bottom", ha="right", fontsize=8, color=INK2)
    ax.set_title(f"{POLL_LABEL[pollutant]} max journalier - Le Havre ville-haute, prévision J+1 (µg/m³)")
    ax.legend(loc="upper right", fontsize=9)
    save(fig, "04_serie_o3_le_havre.png")


def fig_alert_tradeoff(pred):
    """Raise the alert when a forecast reaches `margin` x the 'Dégradé' threshold."""
    d = pred[pred.method == "model"].dropna(subset=["obs_index"])
    obs = d["obs_index"] >= ALERT_LEVEL
    pts = []
    from aeria_forecast.model import with_index
    for margin in np.arange(0.70, 1.01, 0.05):
        conc = d.set_index(["station_code", "date"])[[f"pred_{p}" for p in POLLUTANTS]] / margin
        conc.columns = list(POLLUTANTS)
        sp = {s: [p for p in POLLUTANTS if d.loc[d.station_code == s, f"obs_{p}"].notna().mean() >= 0.5]
              for s in d.station_code.unique()}
        alert = (with_index(conc, sp)["atmo_index"] >= ALERT_LEVEL).to_numpy()
        tp = (alert & obs.to_numpy()).sum()
        pts.append((margin, tp / obs.sum(), tp / max(alert.sum(), 1)))
    pts = pd.DataFrame(pts, columns=["margin", "recall", "precision"])
    pers = evaluate.index_metrics(pred).set_index("method").loc["persistence"]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(pts.recall * 100, pts.precision * 100, color=COLORS["model"], marker="o", markersize=6,
            label="Modèle, marge d'alerte variable")
    for _, r in pts.iloc[[0, 2, 4, 6]].iterrows():
        ax.annotate(f"×{r.margin:.2f}", (r.recall * 100, r.precision * 100), xytext=(6, 4),
                    textcoords="offset points", fontsize=8, color=INK2)
    ax.scatter([pers.alert_recall * 100], [pers.alert_precision * 100], color=COLORS["persistence"],
               s=60, zorder=3, label=LABELS["persistence"])
    ax.set_xlabel("Rappel : épisodes « Dégradé » ou pire détectés (%)")
    ax.set_ylabel("Précision des alertes (%)")
    ax.set_title("Alertes : compromis rappel / précision")
    ax.legend(loc="lower left", fontsize=9)
    save(fig, "05_compromis_alertes.png")
    return pts


WEATHER_FR = {"t_mean": "température moy.", "t_max": "température max", "t_min": "température min",
              "rh_mean": "humidité", "wind_mean": "vent moyen", "wind_min": "vent min",
              "u_mean": "vent d'ouest (composante u)", "v_mean": "vent du sud (composante v)",
              "precip_sum": "pluie", "pressure_mean": "pression", "radiation_sum": "rayonnement solaire",
              "cloud_mean": "nébulosité"}


def pretty(name: str) -> str:
    head, _, tail = name.partition("_")
    if head == "w1":
        return f"Météo J+1 : {WEATHER_FR.get(tail, tail)}"
    if head == "w0":
        return f"Météo J : {WEATHER_FR.get(tail, tail)}"
    if head == "d":
        return {"pressure": "Variation de pression J→J+1", "temp": "Variation de température J→J+1",
                "wind": "Variation du vent J→J+1"}.get(tail, name)
    p = POLL_LABEL.get(tail, tail)
    return {"lag0": f"{p} jour J", "lag1": f"{p} jour J-1", "lag2": f"{p} jour J-2", "mean7": f"{p} moyenne 7 j",
            "eve": f"{p} soirée de J (18-24 h)", "reg0": f"{p} moyenne régionale J"}.get(head, {
            "doy": "Saison (jour de l'année)", "dow": "Jour de la semaine", "is": "Week-end / férié",
            "influence": "Type de station", "area": "Type de zone"}.get(head, name))


def fig_importance():
    X, _ = load_dataset()
    dates = X.index.get_level_values("date")
    train, test = X[dates < "2026-08-31"], X[dates >= "2026-09-01"]
    model = ForecastModel().fit(train)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    tops = {}
    for ax, p in zip(axes, ("o3", "pm25")):
        imp = importances(model, test, p).head(12)[::-1]
        tops[p] = imp[::-1].round(3).to_dict()
        ax.barh([pretty(n) for n in imp.index], imp.values, color=COLORS["model"], height=0.6)
        ax.set_title(f"{POLL_LABEL[p]}")
        ax.set_xlabel("Hausse de l'erreur quand la variable est mélangée (µg/m³)", fontsize=9)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", labelsize=8)
    fig.suptitle("Variables les plus utiles au modèle (test sept.-nov. 2026)", x=0.01, ha="left",
                 fontweight="bold", fontsize=12)
    save(fig, "06_importance_variables.png")
    return tops


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pred = pd.read_csv(ROOT / "docs" / "backtest.csv.gz", parse_dates=["date"], dtype={"station_code": str})
    fig_monthly(pred)
    fig_mae(pred)
    fig_confusion(pred)
    fig_series(pred)
    pts = fig_alert_tradeoff(pred)
    tops = fig_importance()
    (ROOT / "docs" / "analysis.json").write_text(json.dumps(
        {"alert_tradeoff": pts.round(4).to_dict(orient="records"), "importance": tops}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

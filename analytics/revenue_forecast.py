"""
Weekly revenue forecasting by protein group, with honest backtesting.

Same discipline as the demand-forecast module in the supply-chain repo:
rolling-origin evaluation decides which model ships — a single lucky
train/test split decides nothing.

Models: seasonal naive (same week last cycle), 8-week moving average,
Holt-Winters exponential smoothing (additive trend).

Outputs: output/forecast_backtest.csv, output/revenue_forecast_8w.csv
Visual:  docs/revenue_forecast.png

Usage:
    python analytics/revenue_forecast.py
"""

import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "engine"))
from recommend import load_sales

OUT = ROOT / "output"
DOCS = ROOT / "docs"
OUT.mkdir(exist_ok=True)
DOCS.mkdir(exist_ok=True)

HORIZON = 8    # forecast 8 weeks ahead
N_FOLDS = 3
NAVY, TEAL, ORANGE = "#12436D", "#28A197", "#F46A25"


def weekly_revenue(sales: pd.DataFrame) -> pd.DataFrame:
    weekly = (sales.set_index("order_date")
              .groupby("protein")
              .resample("W")["revenue"].sum()
              .rename("revenue").reset_index())
    return weekly


def seasonal_naive(train, horizon):
    return np.resize(train[-4:], horizon)


def moving_average(train, horizon):
    return np.full(horizon, train[-8:].mean())


def holt_winters(train, horizon):
    fit = ExponentialSmoothing(train, trend="add",
                               initialization_method="estimated").fit(optimized=True)
    return np.clip(fit.forecast(horizon), 0, None)


MODELS = {"seasonal_naive": seasonal_naive,
          "moving_avg_8w": moving_average,
          "holt_winters": holt_winters}


def wape(actual, forecast):
    return float(np.abs(actual - forecast).sum() / np.abs(actual).sum())


def backtest(weekly: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for protein, g in weekly.groupby("protein"):
        series = g.sort_values("order_date")["revenue"].values
        for fold in range(N_FOLDS):
            cutoff = len(series) - HORIZON * (N_FOLDS - fold)
            if cutoff < 16:
                continue
            train, actual = series[:cutoff], series[cutoff:cutoff + HORIZON]
            for name, fn in MODELS.items():
                rows.append({"protein": protein, "fold": fold + 1, "model": name,
                             "wape": round(wape(actual, fn(train, HORIZON)), 4)})
    return pd.DataFrame(rows)


def main():
    sales = load_sales()
    weekly = weekly_revenue(sales)
    results = backtest(weekly)
    results.to_csv(OUT / "forecast_backtest.csv", index=False)

    summary = results.groupby("model")["wape"].mean().sort_values()
    best = summary.index[0]
    print("Rolling-origin backtest (avg WAPE, weekly revenue by protein):")
    for model, w in summary.items():
        print(f"  {model:<16} {w:.1%}" + ("  <-- shipped" if model == best else ""))

    # 8-week forward forecast with the winner
    fwd = []
    for protein, g in weekly.groupby("protein"):
        g = g.sort_values("order_date")
        fc = MODELS[best](g["revenue"].values, HORIZON)
        dates = pd.date_range(g["order_date"].max() + pd.Timedelta(weeks=1),
                              periods=HORIZON, freq="W")
        fwd.append(pd.DataFrame({"protein": protein, "week": dates.date,
                                 "forecast_revenue": np.round(fc, 0),
                                 "model": best}))
    pd.concat(fwd, ignore_index=True).to_csv(OUT / "revenue_forecast_8w.csv", index=False)

    # chart: biggest protein, final fold, all models
    biggest = weekly.groupby("protein")["revenue"].sum().idxmax()
    g = weekly[weekly["protein"] == biggest].sort_values("order_date")
    series, dates = g["revenue"].values, g["order_date"].values
    cutoff = len(series) - HORIZON
    fig, ax = plt.subplots(figsize=(10.5, 4.4))
    ax.plot(dates[max(0, cutoff - 20):cutoff], series[max(0, cutoff - 20):cutoff],
            color="#5A6570", lw=1.3, label="history")
    ax.plot(dates[cutoff:], series[cutoff:], color=NAVY, lw=2.2, label="actual")
    palette = {"seasonal_naive": "#A285D1", "moving_avg_8w": ORANGE, "holt_winters": TEAL}
    for name, fn in MODELS.items():
        fc = fn(series[:cutoff], HORIZON)
        ax.plot(dates[cutoff:], fc, ls="--", lw=1.6, color=palette[name],
                label=f"{name} (WAPE {wape(series[cutoff:], fc):.0%})")
    ax.set_title(f"Weekly revenue forecast backtest — {biggest} (final {HORIZON}-week fold)",
                 fontsize=11, fontweight="bold", color=NAVY, loc="left")
    ax.set_ylabel("weekly revenue ($)")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(DOCS / "revenue_forecast.png", dpi=130)
    print(f"\nwrote forecast_backtest.csv, revenue_forecast_8w.csv ({best}), "
          f"revenue_forecast.png")


if __name__ == "__main__":
    main()

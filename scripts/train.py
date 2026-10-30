"""Prepare daily forecasts offline; never run Stan during a page request."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from prophet import Prophet, __version__ as prophet_version
from prophet.serialize import model_to_json
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from energy_forecast.core import daily_observations, metrics, weekly_naive

ROOT = Path(__file__).resolve().parents[1]


def predict(training, dates, model):
    """Fit a fixed candidate using past observations only."""
    if model == "weekly_naive":
        values = weekly_naive(training["y"], len(dates))
        return pd.DataFrame({"ds": dates, "yhat": values}), None
    if model == "exponential_smoothing":
        fitted = ExponentialSmoothing(
            training["y"].to_numpy(),
            trend="add",
            seasonal="add",
            seasonal_periods=7,
            initialization_method="estimated",
        ).fit()
        values = fitted.forecast(len(dates))
        return pd.DataFrame({"ds": dates, "yhat": values}), None
    if model != "prophet":
        raise ValueError("Unknown forecasting candidate")
    fitted = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        interval_width=0.8,
        uncertainty_samples=300,
    )
    fitted.fit(training[["ds", "y"]])
    np.random.seed(42)
    return (
        fitted.predict(pd.DataFrame({"ds": dates}))[
            ["ds", "yhat", "yhat_lower", "yhat_upper"]
        ],
        fitted,
    )


def train():
    """Select on chronological validation MAE, test once, then forecast from full
    history."""
    source = ROOT / "train_IxoE5JN.csv"
    raw = pd.read_csv(source)
    daily = daily_observations(raw)
    first, second = int(len(daily) * 0.6), int(len(daily) * 0.8)
    training = daily.iloc[:first]
    validation = daily.iloc[first:second]
    test = daily.iloc[second:]
    candidates = {}
    for model in ["weekly_naive", "exponential_smoothing", "prophet"]:
        print("Fitting", model, flush=True)
        prediction, _ = predict(training, validation["ds"], model)
        candidates[model] = metrics(validation["y"], prediction["yhat"])
    winner = min(candidates, key=lambda name: candidates[name]["mae"])
    test_forecast, _ = predict(daily.iloc[:second], test["ds"], winner)
    dates = pd.date_range(
        daily.iloc[-1]["ds"] + pd.Timedelta(days=1), periods=365, freq="D"
    )
    future, fitted = predict(daily, dates, winner)
    # Keep an explicit Prophet comparison even if a simpler model wins.
    prophet_forecast, prophet_model = (
        predict(daily, dates, "prophet") if winner != "prophet" else (future, fitted)
    )
    destination = ROOT / "reports"
    destination.mkdir(exist_ok=True)
    daily.to_csv(destination / "daily-history.csv", index=False)
    result = test[["ds", "y"]].merge(test_forecast, on="ds", validate="one_to_one")
    result.to_csv(destination / "test-forecast.csv", index=False)
    future.to_csv(destination / "future-forecast.csv", index=False)
    prophet_forecast.to_csv(destination / "prophet-forecast.csv", index=False)
    (destination / "prophet-model.json").write_text(model_to_json(prophet_model))
    report = {
        "version": "1.0.0",
        "prophet_version": prophet_version,
        "selected_model": winner,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "hourly_rows": len(raw),
        "missing_hourly_targets": int(raw["energy"].isna().sum()),
        "daily_rows": len(daily),
        "partial_days": int((daily["observed_hours"] < 24).sum()),
        "minimum_observed_hours": int(daily["observed_hours"].min()),
        "aggregation": (
            "daily mean of observed source values; no target "
            "imputation; units undocumented"
        ),
        "first_observation": str(daily.iloc[0]["ds"].date()),
        "last_observation": str(daily.iloc[-1]["ds"].date()),
        "split": {
            "train": first,
            "validation": second - first,
            "test": len(daily) - second,
        },
        "validation_candidates": candidates,
        "test": metrics(test["y"], test_forecast["yhat"]),
        "prophet_interval_width": 0.8,
        "prophet_uncertainty_samples": 300,
        "serving": (
            "precomputed historical forecast; no automatic retraining " "or live feed"
        ),
    }
    if "yhat_lower" in result:
        report["test"]["interval_coverage"] = float(
            (
                (result["y"] >= result["yhat_lower"])
                & (result["y"] <= result["yhat_upper"])
            ).mean()
        )
    (destination / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    train()

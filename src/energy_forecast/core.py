"""Data aggregation and metrics shared by training, serving and tests."""

import numpy as np
import pandas as pd


def daily_observations(frame: pd.DataFrame) -> pd.DataFrame:
    """Average observed hourly values, retaining coverage rather than filling
    targets."""
    if not {"datetime", "energy"} <= set(frame):
        raise ValueError("Source requires datetime and energy columns.")
    data = frame[["datetime", "energy"]].copy()
    data["datetime"] = pd.to_datetime(data["datetime"], errors="raise")
    data["energy"] = pd.to_numeric(data["energy"], errors="raise")
    if data["datetime"].isna().any() or data["datetime"].duplicated().any():
        raise ValueError("Hourly timestamps must be nonmissing and unique.")
    present = data["energy"].dropna()
    if not np.isfinite(present).all() or (present < 0).any():
        raise ValueError("Observed energy must be finite and nonnegative.")
    data = data.set_index("datetime").sort_index()
    daily = data["energy"].resample("D").agg(["mean", "count"]).reset_index()
    daily.columns = ["ds", "y", "observed_hours"]
    if daily["y"].isna().any():
        raise ValueError("A day has no observed target; review before modeling.")
    return daily


def metrics(actual: object, predicted: object) -> dict[str, float]:
    """Scale-dependent errors without a misleading percentage denominator."""
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(
        predicted, dtype=float
    )
    if (
        actual.shape != predicted.shape
        or len(actual) == 0
        or not np.isfinite(actual).all()
        or not np.isfinite(predicted).all()
    ):
        raise ValueError(
            "Require matching, finite, nonempty actual and forecast arrays."
        )
    return {
        "mae": float(np.abs(actual - predicted).mean()),
        "rmse": float(np.sqrt(np.mean((actual - predicted) ** 2))),
    }


def weekly_naive(values: object, horizon: int) -> np.ndarray:
    """Repeat the last observed seven-day cycle over a complete daily calendar."""
    values = np.asarray(values, dtype=float)
    if (
        len(values) < 7
        or not np.isfinite(values).all()
        or not isinstance(horizon, int)
        or horizon < 1
    ):
        raise ValueError(
            "Need seven finite observations and a positive integer horizon."
        )
    return np.resize(values[-7:], horizon)

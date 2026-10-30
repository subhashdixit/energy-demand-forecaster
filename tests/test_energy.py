"""Aggregation, error metrics, weekly baseline and rendered app checks."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest
from energy_forecast.core import daily_observations, metrics, weekly_naive

ROOT = Path(__file__).resolve().parents[1]


def test_daily_observed_mean():
    """Missing hourly values are excluded, not treated as zero or filled targets."""
    data = pd.DataFrame(
        {
            "datetime": pd.date_range("2020-01-01", periods=3, freq="h"),
            "energy": [1, np.nan, 3],
        }
    )
    daily = daily_observations(data)
    assert daily.iloc[0]["y"] == 2
    assert daily.iloc[0]["observed_hours"] == 2


@pytest.mark.parametrize(
    "case", ["missing", "duplicate", "negative", "infinite", "all_missing"]
)
def test_invalid_hourly_source(case):
    """Reject invalid measurement contracts before aggregation."""
    frame = pd.DataFrame(
        {
            "datetime": pd.date_range("2020-01-01", periods=3, freq="h"),
            "energy": [1, 2, 3],
        }
    )
    if case == "missing":
        frame = frame.drop(columns="energy")
    elif case == "duplicate":
        frame.loc[1, "datetime"] = frame.loc[0, "datetime"]
    elif case == "negative":
        frame.loc[0, "energy"] = -1
    elif case == "infinite":
        frame["energy"] = frame["energy"].astype(float)
        frame.loc[0, "energy"] = np.inf
    else:
        frame["energy"] = np.nan
    with pytest.raises(ValueError):
        daily_observations(frame)


def test_weekly_naive():
    """Repeat a seven-day seasonal cycle."""
    assert weekly_naive(range(14), 9).tolist() == [7, 8, 9, 10, 11, 12, 13, 7, 8]
    with pytest.raises(ValueError):
        weekly_naive([1, 2], 3)


def test_metrics():
    """Compute errors in the source units without percentage denominators."""
    assert metrics([1, 3], [2, 1])["mae"] == 1.5
    with pytest.raises(ValueError):
        metrics([1], [1, 2])
    with pytest.raises(ValueError):
        metrics([np.nan], [1])


def test_artifact_totals():
    """Published history, test and forecasts agree with the recorded report."""
    import json

    report = json.loads((ROOT / "reports/metrics.json").read_text())
    assert sum(report["split"].values()) == report["daily_rows"]
    assert len(pd.read_csv(ROOT / "reports/future-forecast.csv")) == 365
    assert (
        len(pd.read_csv(ROOT / "reports/test-forecast.csv")) == report["split"]["test"]
    )
    assert report["missing_hourly_targets"] == 1900


def test_app_and_prophet_comparison():
    """App serves artifacts without loading Prophet or running training."""
    app = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=20).run()
    assert not app.exception
    assert app.title[0].value == "Energy Forecast Explorer"
    app.sidebar.selectbox[0].select("Prophet comparison").run()
    assert not app.exception

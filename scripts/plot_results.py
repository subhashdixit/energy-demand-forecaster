"""Plot measured errors and Prophet comparison components offline."""

import json
from pathlib import Path

from matplotlib.figure import Figure
import pandas as pd
from prophet.serialize import model_from_json

ROOT = Path(__file__).resolve().parents[1]


def plot_results() -> None:
    """Keep selected-model evaluation distinct from Prophet explanation."""
    destination = ROOT / "docs/images"
    destination.mkdir(parents=True, exist_ok=True)
    report = json.loads((ROOT / "reports/metrics.json").read_text())
    test = pd.read_csv(ROOT / "reports/test-forecast.csv", parse_dates=["ds"])
    figure = Figure(figsize=(10, 6), layout="constrained", facecolor="#f4f6f5")
    top, bottom = figure.subplots(2, 1)
    top.plot(test["ds"], test["y"], label="Observed daily mean", color="#263b56")
    top.plot(test["ds"], test["yhat"], label=report["selected_model"], color="#26704e")
    top.legend()
    top.set(
        title="Energy: selected model on later held-out days",
        ylabel="Source daily-average units",
    )
    candidates = report["validation_candidates"]
    bottom.barh(
        list(candidates),
        [value["mae"] for value in candidates.values()],
        color="#3266aa",
    )
    bottom.set(xlabel="Validation MAE (lower is better)")
    figure.savefig(destination / "evaluation.png", dpi=150)
    fitted = model_from_json((ROOT / "reports/prophet-model.json").read_text())
    dates = pd.read_csv(ROOT / "reports/prophet-forecast.csv", parse_dates=["ds"])[
        ["ds"]
    ]
    components = fitted.predict(dates)
    figure = Figure(figsize=(10, 8), layout="constrained", facecolor="#f4f6f5")
    trend, weekly, yearly = figure.subplots(3, 1)
    trend.plot(components["ds"], components["trend"], color="#3266aa")
    trend.set(
        title="Prophet comparison components (selected model is weekly naive)",
        ylabel="Trend",
    )
    weekly_values = components.groupby(components["ds"].dt.dayofweek)["weekly"].mean()
    weekly.plot(range(7), weekly_values, color="#26704e", marker="o")
    weekly.set(
        xticks=range(7),
        xticklabels=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
        ylabel="Weekly component",
    )
    yearly.plot(components["ds"], components["yearly"], color="#996538")
    yearly.set(
        ylabel="Yearly component",
        xlabel="Historical future calendar, not current forecasts",
    )
    figure.savefig(destination / "prophet-components.png", dpi=150)


if __name__ == "__main__":
    plot_results()

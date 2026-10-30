"""Fast historical energy dashboard served from reproducible offline artifacts."""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent


def main() -> None:
    """Show forecast horizons, model comparisons, uncertainty and measured errors."""
    st.set_page_config(
        page_title="Energy Forecast Explorer", page_icon="⚡", layout="wide"
    )
    st.caption("FORECASTING · SUBHASH DIXIT")
    st.title("Energy Forecast Explorer")
    report = json.loads((ROOT / "reports/metrics.json").read_text())
    st.info(
        (
            f"Historical learning demonstration: source observations "
            f"end {report['last_observation']}. These forecasts start "
            f"after that date and are not today's energy forecast."
        )
    )
    history = pd.read_csv(ROOT / "reports/daily-history.csv", parse_dates=["ds"])
    test = pd.read_csv(ROOT / "reports/test-forecast.csv", parse_dates=["ds"])
    forecast = pd.read_csv(ROOT / "reports/future-forecast.csv", parse_dates=["ds"])
    prophet = pd.read_csv(ROOT / "reports/prophet-forecast.csv", parse_dates=["ds"])
    horizon = st.sidebar.slider(
        "Forecast horizon (days after source history)", 7, 365, 90
    )
    view = st.sidebar.selectbox(
        "Forecast view", ["Validation-selected model", "Prophet comparison"]
    )
    tabs = st.tabs(["History and forecast", "Evaluation", "Learn the method"])
    with tabs[0]:
        st.subheader("Daily mean of observed hourly energy")
        start, end = st.sidebar.slider(
            "History years",
            int(history.ds.dt.year.min()),
            int(history.ds.dt.year.max()),
            (int(history.ds.dt.year.min()), int(history.ds.dt.year.max())),
        )
        filtered = history[history.ds.dt.year.between(start, end)]
        st.line_chart(filtered.set_index("ds")["y"], use_container_width=True)
        st.caption(
            (
                "Source units are undocumented. This is a daily average, "
                "not daily total energy; partial days average only their "
                "observed hours."
            )
        )
        chosen = forecast if view == "Validation-selected model" else prophet
        selected = chosen.head(horizon)
        columns = [
            name for name in ["yhat", "yhat_lower", "yhat_upper"] if name in selected
        ]
        st.subheader("Forecast beyond the historical cutoff")
        st.line_chart(selected.set_index("ds")[columns], use_container_width=True)
        st.write(
            (
                f"Validation-selected algorithm: "
                f"**{report['selected_model'].replace('_',' ')}**. Prophet "
                f"is shown separately when it does not win."
            )
        )
        if "yhat_lower" in selected:
            st.caption(
                (
                    "Prophet's nominal 80% model-based interval is not a "
                    "guarantee of 80% future coverage. Lines show lower/upper "
                    "bounds; uncertainty assumptions can fail."
                )
            )
        st.download_button(
            "Download visible forecast",
            selected.to_csv(index=False),
            "energy-forecast.csv",
            "text/csv",
        )
        st.dataframe(selected, use_container_width=True)
    with tabs[1]:
        left, right = st.columns(2)
        left.metric("Held-out MAE", f"{report['test']['mae']:.2f}")
        right.metric("Held-out RMSE", f"{report['test']['rmse']:.2f}")
        st.dataframe(
            pd.DataFrame(report["validation_candidates"]).T, use_container_width=True
        )
        st.line_chart(test.set_index("ds")[["y", "yhat"]], use_container_width=True)
        st.json(report["split"])
        st.warning(
            (
                f"{report['missing_hourly_targets']:,} hourly targets are "
                f"missing; {report['partial_days']:,} daily means use fewer "
                f"than 24 observations. No missing hourly targets were "
                f"imputed."
            )
        )
        st.caption(
            (
                "Validation selects the algorithm by MAE. The selected "
                "model is refitted on train + validation before testing, "
                "and on all historical observations only for the final "
                "future forecast."
            )
        )
    with tabs[2]:
        st.markdown(
            """### From measurements to a forecast
1. Parse and validate unique hourly timestamps.
2. Aggregate observed values to a daily mean and report coverage.
3. Keep time order: training, validation, then test.
4. Compare weekly naive, exponential smoothing and Prophet.
5. Freeze the selected method, evaluate later observations, then refit on
full history.

**Weekly naive** repeats last week's values. **Exponential smoothing**
updates level, trend and weekly seasonality. **Prophet** combines a
piecewise trend, Fourier seasonal patterns and an error model. It is useful
for flexible calendar patterns, but changes in demand, measurement coverage
and rare events can break its assumptions.

MAE measures average absolute error in source units. RMSE emphasizes larger
errors. These are historical holdout measurements, not a service-level
promise. The interface serves precomputed outputs; Stan fitting runs
offline, so page loads are quick and reproducible."""
        )
        st.json(report)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from .data_loader import load_sales, save_analytics_result, write_json


def _pycaret_available() -> bool:
    try:
        import pycaret  # noqa: F401
        from pycaret.time_series import TSForecastingExperiment  # noqa: F401
        return True
    except Exception:
        return False


def _forecast_series(series: pd.Series, horizon: int) -> list[dict]:
    daily = series.asfreq("D").fillna(0)
    if len(daily) < 14 or daily.sum() <= 0:
        baseline = float(daily.tail(7).mean() if len(daily) else 0)
        forecast = np.repeat(baseline, horizon)
    else:
        forecast = None
        if _pycaret_available() and len(daily) >= 30:
            try:
                from pycaret.time_series import TSForecastingExperiment
                exp = TSForecastingExperiment()
                df = pd.DataFrame({"sales": daily})
                df.index = pd.to_datetime(df.index)
                exp.setup(data=df, fh=horizon, session_id=42, verbose=False)
                best_model = exp.compare_models(sort="mase", n_select=1, turbo=True, verbose=False)
                preds = exp.predict_model(best_model)
                if preds is not None and not preds.empty:
                    forecast = np.maximum(preds["y_pred"].values, 0)
            except Exception as e:
                print(f"PyCaret forecasting failed: {e}. Falling back to Statsmodels.")
        
        if forecast is None:
            seasonal_periods = 7 if len(daily) >= 28 else None
            model = ExponentialSmoothing(
                daily,
                trend="add",
                seasonal="add" if seasonal_periods else None,
                seasonal_periods=seasonal_periods,
                initialization_method="estimated",
            )
            fitted = model.fit(optimized=True)
            forecast = np.maximum(fitted.forecast(horizon).to_numpy(), 0)

    start = (daily.index.max() if len(daily) else pd.Timestamp.utcnow()).date()
    return [
        {"date": str(start + timedelta(days=i + 1)), "predicted_sales": round(float(value), 2)}
        for i, value in enumerate(forecast)
    ]


def run(days: int, horizon: int, shop_id: str | None) -> dict:
    sales = load_sales()
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    if sales.empty or "date" not in sales:
        return {"status": "empty", "message": "No sales rows available.", "forecasts": []}

    recent = sales[sales["date"] >= cutoff].copy()
    if shop_id:
        recent = recent[recent.get("shop_id") == shop_id]

    amount_col = "total_with_tax" if "total_with_tax" in recent else "amount"
    recent["day"] = recent["date"].dt.floor("D")
    grouped = recent.groupby(["shop_id", "day"], dropna=False)[amount_col].sum().reset_index()

    forecasts = []
    for sid, frame in grouped.groupby("shop_id", dropna=False):
        series = frame.set_index("day")[amount_col].sort_index()
        forecasts.append({
            "shop_id": sid or "unknown",
            "history_days": int(series.shape[0]),
            "history_total": round(float(series.sum()), 2),
            "forecast": _forecast_series(series, horizon),
        })

    return {
        "status": "success",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "window_days": days,
        "horizon_days": horizon,
        "forecasts": forecasts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Forecast Nirvana shop demand from sales history.")
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--horizon", type=int, default=14)
    parser.add_argument("--shop-id")
    parser.add_argument("--output")
    parser.add_argument("--save-db", action="store_true", help="Save this snapshot to analytics_results.")
    args = parser.parse_args()
    payload = run(args.days, args.horizon, args.shop_id)
    if args.save_db:
        forecast_count = len(payload.get("forecasts", []))
        save_analytics_result("demand_forecast", payload, f"{forecast_count} shop forecasts generated")
    write_json(payload, args.output)


if __name__ == "__main__":
    main()

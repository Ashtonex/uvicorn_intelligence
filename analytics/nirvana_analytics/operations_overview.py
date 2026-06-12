from __future__ import annotations

import argparse
import base64
from datetime import datetime, timedelta, timezone
from io import BytesIO

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize
from sklearn.ensemble import IsolationForest

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
except Exception:  # pragma: no cover - optional runtime guard
    ExponentialSmoothing = None

from .data_loader import load_operations, load_sales, load_table, save_analytics_result, write_json


ACCOUNT_ORDER = ["savings", "overhead", "invest", "tshirts", "stockvel", "round", "vault", "other"]
OVERHEAD_WORDS = ("rent", "salary", "salaries", "wages", "payroll", "utilities", "utility", "wifi", "zesa", "electric", "water")


def _norm(value: object) -> str:
    return str(value or "").lower().replace("_", " ").replace("-", " ").strip()


def _account(row: pd.Series) -> str:
    kind = _norm(row.get("kind"))
    text = f"{kind} {_norm(row.get('title'))} {_norm(row.get('notes'))}"
    if kind.startswith("stockvel") or text.startswith("stockvel") or " stockvel " in f" {text} ":
        return "stockvel"
    if kind.startswith("round") or text.startswith("round") or " round " in f" {text} ":
        return "round"
    if any(word in text for word in ("savings", "saving", "blackbox", "black box")):
        return "savings"
    if kind.startswith("invest") or " invest " in f" {text} ":
        return "invest"
    if kind.startswith("overhead") or any(word in text for word in OVERHEAD_WORDS):
        return "overhead"
    if kind in ("eod deposit", "drawer post", "capital injection", "loan received", "other income"):
        return "vault"
    return "other"


def _chart_base64(account_totals: pd.DataFrame, daily: pd.DataFrame) -> str | None:
    if account_totals.empty:
        return None
    sns.set_theme(style="darkgrid")
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), dpi=150)

    plot_accounts = account_totals.sort_values("balance", ascending=False).head(8)
    sns.barplot(data=plot_accounts, x="account", y="balance", ax=axes[0], palette="viridis", hue="account", legend=False)
    axes[0].set_title("Account Balances")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("USD")
    axes[0].tick_params(axis="x", rotation=35)

    if daily.empty:
        axes[1].text(0.5, 0.5, "No daily flow data", ha="center", va="center")
        axes[1].set_axis_off()
    else:
        sns.lineplot(data=daily, x="date", y="net_flow", ax=axes[1], color="#10b981")
        axes[1].set_title("Daily Net Operations Flow")
        axes[1].set_xlabel("")
        axes[1].set_ylabel("USD")
        axes[1].tick_params(axis="x", rotation=35)

    fig.tight_layout()
    buffer = BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight", facecolor="#020617")
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _safe_records(df: pd.DataFrame, limit: int = 50) -> list[dict]:
    if df.empty:
        return []
    out = df.head(limit).replace({np.nan: None}).to_dict(orient="records")
    return out


def _forecast_overhead(monthly_overhead: pd.DataFrame, horizon: int) -> list[dict]:
    if monthly_overhead.empty:
        return []
    series = monthly_overhead.set_index("month")["amount"].asfreq("MS").fillna(0)
    if len(series) < 3:
        baseline = float(series.tail(1).iloc[0]) if len(series) else 0.0
        start = (series.index.max() if len(series) else pd.Timestamp.utcnow()).to_period("M").to_timestamp()
        dates = pd.date_range(start + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
        return [{"month": d.strftime("%Y-%m"), "predicted_overhead": round(baseline, 2), "model": "naive"} for d in dates]
    try:
        if ExponentialSmoothing is None:
            raise RuntimeError("statsmodels unavailable")
        model = ExponentialSmoothing(series, trend="add", seasonal=None, initialization_method="estimated")
        fit = model.fit(optimized=True)
        forecast = fit.forecast(horizon)
        return [
            {"month": idx.strftime("%Y-%m"), "predicted_overhead": round(max(0.0, float(value)), 2), "model": "statsmodels_holt"}
            for idx, value in forecast.items()
        ]
    except Exception:
        x = np.arange(len(series))
        slope, intercept, *_ = stats.linregress(x, series.to_numpy())
        future = []
        for i in range(1, horizon + 1):
            month = (series.index.max() + pd.offsets.MonthBegin(i)).strftime("%Y-%m")
            future.append({"month": month, "predicted_overhead": round(max(0.0, intercept + slope * (len(series) + i - 1)), 2), "model": "scipy_linregress"})
        return future


def _optimize_account_targets(account_totals: pd.DataFrame) -> dict:
    balances = account_totals.set_index("account")["balance"].reindex(["savings", "overhead", "invest", "stockvel", "round"]).fillna(0)
    total = float(balances.sum())
    if total <= 0:
        return {"backend": "scipy_slsqp", "total": 0, "targets": [], "recommendations": []}

    accounts = balances.index.tolist()
    min_bounds = np.array([0.18, 0.22, 0.08, 0.05, 0.03])
    max_bounds = np.array([0.42, 0.45, 0.25, 0.18, 0.15])
    expected_resilience = np.array([0.16, 0.20, 0.10, 0.12, 0.08])
    risk = np.array([0.10, 0.08, 0.24, 0.18, 0.14])
    current = (balances / total).to_numpy()

    def objective(weights: np.ndarray) -> float:
        drift_penalty = np.sum((weights - current) ** 2) * 0.35
        return float(-(weights @ expected_resilience - weights @ risk * 0.45) + drift_penalty)

    result = minimize(
        objective,
        x0=np.clip(current, min_bounds, max_bounds),
        method="SLSQP",
        bounds=list(zip(min_bounds, max_bounds)),
        constraints=[{"type": "eq", "fun": lambda w: float(np.sum(w) - 1)}],
        options={"maxiter": 500},
    )
    weights = result.x if result.success else np.array([0.30, 0.35, 0.14, 0.12, 0.09])
    weights = weights / weights.sum()

    targets = []
    recommendations = []
    for account, current_amount, weight in zip(accounts, balances.to_numpy(), weights):
        target_amount = float(total * weight)
        delta = target_amount - float(current_amount)
        targets.append({
            "account": account,
            "current": round(float(current_amount), 2),
            "target_weight": round(float(weight), 4),
            "target_amount": round(target_amount, 2),
            "delta": round(delta, 2),
        })
        if abs(delta) >= max(25, total * 0.04):
            recommendations.append({
                "account": account,
                "action": "increase" if delta > 0 else "reduce",
                "amount": round(abs(delta), 2),
                "reason": f"Target {weight * 100:.1f}% of operating pool",
            })

    return {"backend": "scipy_slsqp", "total": round(total, 2), "targets": targets, "recommendations": recommendations}


def run(days: int = 180, horizon: int = 3, limit: int = 12) -> dict:
    generated_at = datetime.now(timezone.utc).isoformat()
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    operations = load_operations(limit=50000)
    sales = load_sales(limit=50000)
    invest = load_table("invest_deposits", "*", limit=50000)

    if operations.empty and sales.empty and invest.empty:
        return {"status": "empty", "generated_at": generated_at, "message": "No operations data available."}

    ops = operations.copy() if not operations.empty else pd.DataFrame()
    if not ops.empty:
        ops["created_at"] = pd.to_datetime(ops.get("created_at"), errors="coerce", utc=True)
        ops["effective_date"] = pd.to_datetime(ops.get("effective_date"), errors="coerce", utc=True).fillna(ops["created_at"])
        ops["amount"] = pd.to_numeric(ops.get("amount"), errors="coerce").fillna(0)
        ops = ops[ops["created_at"] >= cutoff]
        ops["account"] = ops.apply(_account, axis=1)
        ops["month"] = ops["effective_date"].dt.to_period("M").dt.to_timestamp()
        ops["date"] = ops["effective_date"].dt.date.astype(str)
    else:
        ops = pd.DataFrame(columns=["amount", "account", "shop_id", "kind", "title", "notes", "created_at", "effective_date", "month", "date"])

    invest_rows = pd.DataFrame()
    if not invest.empty:
        invest_rows = invest.copy()
        invest_rows["amount"] = pd.to_numeric(invest_rows.get("amount"), errors="coerce").fillna(0) - pd.to_numeric(invest_rows.get("withdrawn_amount"), errors="coerce").fillna(0)

    tee_total = 0.0
    if not sales.empty:
        sales["date"] = pd.to_datetime(sales.get("date"), errors="coerce", utc=True)
        tee_sales = sales[(sales.get("shop_id").fillna("") == "tshirts")]
        tee_total = float(pd.to_numeric(tee_sales.get("total_with_tax"), errors="coerce").fillna(0).sum())

    account_totals = ops.groupby("account", as_index=False)["amount"].sum().rename(columns={"amount": "balance"})
    if not invest_rows.empty:
        invest_balance = float(invest_rows["amount"].sum()) + float(account_totals.loc[account_totals["account"] == "invest", "balance"].sum())
        account_totals = account_totals[account_totals["account"] != "invest"]
        account_totals = pd.concat([account_totals, pd.DataFrame([{"account": "invest", "balance": invest_balance}])], ignore_index=True)
    account_totals = account_totals[account_totals["account"].isin(ACCOUNT_ORDER)]
    account_totals = pd.concat([account_totals, pd.DataFrame([{"account": "tshirts", "balance": tee_total}])], ignore_index=True)
    account_totals = account_totals.groupby("account", as_index=False)["balance"].sum()
    account_totals["balance"] = account_totals["balance"].round(2)
    account_totals["account"] = pd.Categorical(account_totals["account"], ACCOUNT_ORDER, ordered=True)
    account_totals = account_totals.sort_values("account")

    current_month = pd.Timestamp.utcnow().to_period("M").to_timestamp().tz_localize(None)
    monthly_overhead_rows = ops[(ops["account"] == "overhead") & (ops["amount"] > 0)].copy()
    monthly_overhead = monthly_overhead_rows.groupby("month", as_index=False)["amount"].sum()
    current_month_overhead = float(monthly_overhead[monthly_overhead["month"] == current_month]["amount"].sum()) if not monthly_overhead.empty else 0.0

    shop_matrix = pd.DataFrame()
    if not ops.empty:
        shop_matrix = ops.pivot_table(index="shop_id", columns="account", values="amount", aggfunc="sum", fill_value=0).reset_index()
        for account in ACCOUNT_ORDER:
            if account not in shop_matrix.columns:
                shop_matrix[account] = 0.0
        shop_matrix = shop_matrix[["shop_id", *ACCOUNT_ORDER]].round(2)

    daily = pd.DataFrame()
    if not ops.empty:
        daily = ops.groupby("date", as_index=False)["amount"].sum().rename(columns={"amount": "net_flow"})
        daily["net_flow"] = daily["net_flow"].round(2)

    anomaly_rows = ops.copy()
    if not anomaly_rows.empty:
        anomaly_rows["abs_amount"] = anomaly_rows["amount"].abs()
        anomaly_rows["zscore"] = np.abs(stats.zscore(anomaly_rows["abs_amount"], nan_policy="omit"))
        anomaly_rows["zscore"] = anomaly_rows["zscore"].replace([np.inf, -np.inf], 0).fillna(0)
        if len(anomaly_rows) >= 10:
            model = IsolationForest(contamination=min(0.2, max(0.05, 6 / len(anomaly_rows))), random_state=42)
            anomaly_rows["model_flag"] = model.fit_predict(anomaly_rows[["abs_amount"]])
        else:
            anomaly_rows["model_flag"] = 1
        anomaly_rows = anomaly_rows[(anomaly_rows["zscore"] >= 2.25) | (anomaly_rows["model_flag"] == -1)]
        if anomaly_rows.empty:
            anomaly_rows = ops.reindex(ops["amount"].abs().sort_values(ascending=False).index).head(min(limit, 5)).copy()
            anomaly_rows["reason"] = "largest movement in period"
        else:
            anomaly_rows["reason"] = np.where(anomaly_rows["zscore"] >= 2.25, "statistical outlier", "isolation forest anomaly")

    account_trends = []
    if not ops.empty:
        month_account = ops.groupby(["month", "account"], as_index=False)["amount"].sum()
        for account, subset in month_account.groupby("account"):
            if len(subset) < 2:
                continue
            ordered = subset.sort_values("month")
            slope, _, r_value, _, _ = stats.linregress(np.arange(len(ordered)), ordered["amount"].to_numpy())
            account_trends.append({
                "account": account,
                "monthly_slope": round(float(slope), 2),
                "r2": round(float(r_value * r_value), 3),
                "direction": "rising" if slope > 0 else "falling" if slope < 0 else "flat",
            })

    return {
        "status": "success",
        "generated_at": generated_at,
        "window_days": days,
        "stack": {
            "pandas_numpy": "ledger normalization, pivots, account/shop/month tables",
            "scipy_statsmodels": "trend slopes and overhead forecast",
            "matplotlib_seaborn": "embedded report chart artifact",
            "sklearn_pycaret_path": "IsolationForest anomaly detection; PyCaret can replace this backend later",
            "risk_optimization": "SciPy SLSQP constrained operating-pool allocation",
        },
        "summary": {
            "total_operations_rows": int(len(ops)),
            "current_month_overhead": round(current_month_overhead, 2),
            "tee_sales_total": round(tee_total, 2),
            "accounts_tracked": int(account_totals["account"].nunique()),
            "anomalies_flagged": int(len(anomaly_rows)),
        },
        "account_totals": _safe_records(account_totals, 20),
        "shop_matrix": _safe_records(shop_matrix, 25),
        "daily_flow": _safe_records(daily.tail(60), 60),
        "overhead_forecast": _forecast_overhead(monthly_overhead, horizon),
        "anomalies": _safe_records(anomaly_rows[["id", "shop_id", "account", "kind", "title", "amount", "created_at", "reason", "zscore"]], limit),
        "account_trends": account_trends,
        "allocation": _optimize_account_targets(account_totals),
        "chart_png_base64": _chart_base64(account_totals, daily.tail(45)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Operations overview intelligence engine.")
    parser.add_argument("--days", type=int, default=180)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--output")
    parser.add_argument("--save-db", action="store_true")
    args = parser.parse_args()
    payload = run(args.days, args.horizon, args.limit)
    if args.save_db:
        summary = f"{payload.get('summary', {}).get('accounts_tracked', 0)} accounts analyzed; {payload.get('summary', {}).get('anomalies_flagged', 0)} flags"
        save_analytics_result("operations_overview", payload, summary)
    write_json(payload, args.output)


if __name__ == "__main__":
    main()

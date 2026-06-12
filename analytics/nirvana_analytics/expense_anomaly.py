from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest

from .data_loader import load_ledger, load_operations, save_analytics_result, write_json

def _pycaret_available() -> bool:
    try:
        import pycaret  # noqa: F401
        from pycaret.anomaly import AnomalyExperiment  # noqa: F401
        return True
    except Exception:
        return False


NON_CASH_CATEGORIES = {
    "Cash Drawer Opening",
    "Cash Drawer Adjustment",
    "Stock Adjustment",
    "Operations Transfer",
    "Inventory Acquisition",
    "Shipping & Logistics",
    "Lay-by Completed",
    "Lay-by Pending",
    "Lay-by Payment",
    "Return",
    "Refund",
}


def _expense_frame(days: int) -> pd.DataFrame:
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    ledger = load_ledger()
    operations = load_operations()

    frames = []
    if not ledger.empty:
        ledger = ledger.copy()
        ledger["source"] = "ledger_entries"
        ledger["when"] = ledger.get("date")
        ledger["label"] = ledger.get("description", "").fillna(ledger.get("category", ""))
        ledger = ledger[
            (ledger["when"] >= cutoff)
            & (ledger["amount"] > 0)
            & (~ledger["category"].fillna("").isin(NON_CASH_CATEGORIES))
        ]
        frames.append(ledger[["source", "id", "shop_id", "when", "amount", "category", "label"]])

    if not operations.empty:
        operations = operations.copy()
        operations["source"] = "operations_ledger"
        operations["when"] = operations.get("created_at")
        operations["category"] = operations.get("kind", "")
        operations["label"] = operations.get("title", "").fillna(operations.get("notes", ""))
        operations = operations[(operations["when"] >= cutoff) & (operations["amount"] < 0)]
        operations["amount"] = operations["amount"].abs()
        frames.append(operations[["source", "id", "shop_id", "when", "amount", "category", "label"]])

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values("amount", ascending=False)


def run(days: int, limit: int) -> dict:
    df = _expense_frame(days)
    if df.empty:
        return {"status": "empty", "message": "No expense rows available.", "anomalies": []}

    df["amount_zscore"] = np.abs(stats.zscore(df["amount"], nan_policy="omit"))
    df["amount_zscore"] = df["amount_zscore"].replace([np.inf, -np.inf], 0).fillna(0)

    if len(df) >= 10:
        model_run = False
        if _pycaret_available():
            try:
                from pycaret.anomaly import AnomalyExperiment
                exp = AnomalyExperiment()
                exp.setup(df[["amount"]], session_id=42, verbose=False)
                iforest = exp.create_model('iforest', fraction=min(0.2, max(0.05, 5 / len(df))))
                res = exp.assign_model(iforest)
                df["model_score"] = np.where(res["Anomaly"] == 1, -1, 1)
                model_run = True
            except Exception as e:
                print(f"PyCaret anomaly detection failed: {e}. Falling back to scikit-learn.")
        
        if not model_run:
            model = IsolationForest(contamination=min(0.2, max(0.05, 5 / len(df))), random_state=42)
            df["model_score"] = model.fit_predict(df[["amount"]])
    else:
        df["model_score"] = 1

    suspicious = df[(df["amount_zscore"] >= 2.5) | (df["model_score"] == -1)].copy()
    if suspicious.empty:
        suspicious = df.head(min(limit, 5)).copy()
        suspicious["reason"] = "largest expense in period"
    else:
        suspicious["reason"] = np.where(
            suspicious["amount_zscore"] >= 2.5,
            "amount is statistically unusual",
            "model flagged unusual amount pattern",
        )

    anomalies = []
    for row in suspicious.head(limit).to_dict(orient="records"):
        anomalies.append({
            "source": row.get("source"),
            "id": row.get("id"),
            "shop_id": None if pd.isna(row.get("shop_id")) else row.get("shop_id"),
            "date": row.get("when"),
            "amount": round(float(row.get("amount") or 0), 2),
            "category": row.get("category"),
            "label": row.get("label"),
            "reason": row.get("reason"),
            "zscore": round(float(row.get("amount_zscore") or 0), 2),
        })

    return {
        "status": "success",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "window_days": days,
        "rows_scanned": int(len(df)),
        "anomalies": anomalies,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Flag unusual Nirvana expenses and operations outflows.")
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--limit", type=int, default=15)
    parser.add_argument("--output")
    parser.add_argument("--save-db", action="store_true", help="Save this snapshot to analytics_results.")
    args = parser.parse_args()
    payload = run(args.days, args.limit)
    if args.save_db:
        count = len(payload.get("anomalies", []))
        save_analytics_result("expense_anomaly", payload, f"{count} expense anomalies flagged")
    write_json(payload, args.output)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from .data_loader import load_allocations, load_inventory, load_sales, load_table, save_analytics_result, write_json


def _historical_model_comparison(recent_sales: pd.DataFrame, item_key: str, days: int) -> dict:
    if recent_sales.empty or item_key not in recent_sales or "date" not in recent_sales:
        return {"status": "empty", "models": [], "winner": None, "items_tested": 0}

    frame = recent_sales.copy()
    frame["day"] = frame["date"].dt.floor("D")
    models = {
        "naive_7d": {"absolute_error": 0.0, "checks": 0},
        "rolling_30d": {"absolute_error": 0.0, "checks": 0},
        "trend_linear": {"absolute_error": 0.0, "checks": 0},
    }
    items_tested = 0
    horizon = min(14, max(7, days // 5))

    for _, item_sales in frame.groupby(item_key):
        daily = item_sales.groupby("day")["quantity"].sum().sort_index().asfreq("D").fillna(0)
        if len(daily) < 21 or daily.sum() <= 0:
            continue

        train = daily.iloc[:-horizon]
        test = daily.iloc[-horizon:]
        if train.empty or test.empty:
            continue

        items_tested += 1
        actual = float(test.sum())
        forecasts = {
            "naive_7d": float(train.tail(7).sum() / 7 * horizon),
            "rolling_30d": float(train.tail(min(30, len(train))).sum() / min(30, len(train)) * horizon),
        }

        x = np.arange(len(train), dtype=float)
        y = train.to_numpy(dtype=float)
        if len(train) >= 7 and y.sum() > 0:
            slope, intercept = np.polyfit(x, y, 1)
            future_x = np.arange(len(train), len(train) + horizon, dtype=float)
            forecasts["trend_linear"] = float(np.maximum(intercept + slope * future_x, 0).sum())
        else:
            forecasts["trend_linear"] = forecasts["rolling_30d"]

        for name, predicted in forecasts.items():
            models[name]["absolute_error"] += abs(predicted - actual)
            models[name]["checks"] += 1

    model_rows = []
    for name, stats in models.items():
        checks = max(stats["checks"], 1)
        model_rows.append({
            "model": name,
            "mae_units": round(float(stats["absolute_error"] / checks), 2),
            "checks": int(stats["checks"]),
        })

    model_rows.sort(key=lambda row: row["mae_units"])
    return {
        "status": "success" if items_tested else "insufficient_history",
        "window_days": days,
        "test_horizon_days": horizon,
        "items_tested": items_tested,
        "winner": model_rows[0]["model"] if items_tested else None,
        "models": model_rows,
    }


def run(days: int, dead_stock_days: int, limit: int, lead_time_days: int = 7, service_level_z: float = 1.65) -> dict:
    sales = load_sales()
    inventory = load_inventory()
    allocations = load_allocations()
    shops = load_table("shops", "id,name", limit=1000)
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    if inventory.empty:
        return {"status": "empty", "message": "No inventory rows available.", "items": []}

    recent_sales = sales[sales["date"] >= cutoff].copy() if not sales.empty and "date" in sales else pd.DataFrame()
    item_key = "item_id" if "item_id" in recent_sales else "inventory_item_id"
    if not recent_sales.empty and item_key in recent_sales:
        velocity = recent_sales.groupby(item_key).agg(
            sold_units=("quantity", "sum"),
            sales_value=("total_with_tax", "sum"),
            last_sale=("date", "max"),
        ).reset_index().rename(columns={item_key: "id"})
    else:
        velocity = pd.DataFrame(columns=["id", "sold_units", "sales_value", "last_sale"])

    if not allocations.empty and "item_id" in allocations:
        stock = allocations.groupby("item_id")["quantity"].sum().reset_index().rename(columns={"item_id": "id", "quantity": "allocated_stock"})
    else:
        stock_col = "quantity" if "quantity" in inventory else None
        stock = inventory[["id", stock_col]].rename(columns={stock_col: "allocated_stock"}) if stock_col else pd.DataFrame(columns=["id", "allocated_stock"])

    merged = inventory.merge(stock, on="id", how="left").merge(velocity, on="id", how="left")
    merged["allocated_stock"] = merged["allocated_stock"].fillna(0)
    merged["sold_units"] = merged["sold_units"].fillna(0)
    merged["sales_value"] = merged["sales_value"].fillna(0)
    merged["daily_velocity"] = merged["sold_units"] / max(days, 1)
    merged["days_to_zero"] = np.where(merged["daily_velocity"] > 0, merged["allocated_stock"] / merged["daily_velocity"], np.inf)
    merged["safety_stock"] = np.ceil(merged["daily_velocity"] * lead_time_days * 0.5)
    merged["reorder_point"] = np.ceil((merged["daily_velocity"] * lead_time_days) + merged["safety_stock"])
    merged["suggested_order_qty"] = np.maximum(0, np.ceil((merged["daily_velocity"] * 30) - merged["allocated_stock"]))
    merged["confidence"] = np.where(
        merged["sold_units"] >= 20,
        0.9,
        np.where(merged["sold_units"] >= 5, 0.65, 0.35),
    )

    now = pd.Timestamp.now(tz="UTC")
    date_added = merged["date_added"] if "date_added" in merged else pd.NaT
    merged["days_in_stock"] = (now - date_added).dt.days if hasattr(date_added, "dt") else np.nan
    merged["dead_stock_age"] = np.where(merged["sold_units"] <= 0, merged["days_in_stock"], 0)
    cost_col = "landed_cost" if "landed_cost" in merged else "cost" if "cost" in merged else None
    merged["capital_tied"] = merged["allocated_stock"] * (merged[cost_col].fillna(0) if cost_col else 0)

    shop_names = dict(zip(shops.get("id", []), shops.get("name", []))) if not shops.empty else {}
    shop_needs_by_item: dict[str, list[dict]] = {}
    if not allocations.empty and not recent_sales.empty and "shop_id" in recent_sales and item_key in recent_sales:
        shop_sales = recent_sales.groupby([item_key, "shop_id"]).agg(sold_units=("quantity", "sum")).reset_index()
        shop_allocs = allocations.groupby(["item_id", "shop_id"])["quantity"].sum().reset_index()
        shop_frame = shop_sales.merge(
            shop_allocs,
            left_on=[item_key, "shop_id"],
            right_on=["item_id", "shop_id"],
            how="left",
        )
        shop_frame["quantity"] = shop_frame["quantity"].fillna(0)
        shop_frame["daily_velocity"] = shop_frame["sold_units"] / max(days, 1)
        shop_frame["days_to_zero"] = np.where(shop_frame["daily_velocity"] > 0, shop_frame["quantity"] / shop_frame["daily_velocity"], np.inf)
        shop_frame["suggested_transfer_qty"] = np.maximum(0, np.ceil((shop_frame["daily_velocity"] * lead_time_days) - shop_frame["quantity"]))
        needs = shop_frame[(shop_frame["days_to_zero"] <= lead_time_days) | (shop_frame["quantity"] <= 0)]
        for item_id, frame in needs.groupby(item_key):
            rows = []
            for row in frame.sort_values("days_to_zero").head(5).to_dict(orient="records"):
                days_left = row.get("days_to_zero")
                rows.append({
                    "shop_id": row.get("shop_id"),
                    "shop_name": shop_names.get(row.get("shop_id"), row.get("shop_id")),
                    "stock": round(float(row.get("quantity") or 0), 2),
                    "daily_velocity": round(float(row.get("daily_velocity") or 0), 3),
                    "days_to_zero": None if np.isinf(days_left) else round(float(days_left), 1),
                    "suggested_transfer_qty": int(row.get("suggested_transfer_qty") or 0),
                })
            shop_needs_by_item[str(item_id)] = rows

    def status(row: pd.Series) -> str:
        if row["allocated_stock"] <= 0:
            return "out_of_stock"
        if row["sold_units"] <= 0 and (row.get("days_in_stock") or 0) >= dead_stock_days:
            return "dead_stock"
        if row["allocated_stock"] <= row["reorder_point"] and row["daily_velocity"] > 0:
            return "reorder_risk"
        if row["days_to_zero"] <= 14:
            return "reorder_risk"
        return "healthy"

    merged["status"] = merged.apply(status, axis=1)
    priority = merged[merged["status"].isin(["dead_stock", "reorder_risk"])].copy()
    priority = priority.sort_values(["status", "capital_tied"], ascending=[True, False]).head(limit)

    name_col = "name" if "name" in priority else "item_name" if "item_name" in priority else "id"
    items = []
    for row in priority.to_dict(orient="records"):
        days_to_zero = row.get("days_to_zero")
        items.append({
            "item_id": row.get("id"),
            "item_name": row.get(name_col),
            "status": row.get("status"),
            "stock": round(float(row.get("allocated_stock") or 0), 2),
            "sold_units": round(float(row.get("sold_units") or 0), 2),
            "daily_velocity": round(float(row.get("daily_velocity") or 0), 3),
            "days_to_zero": None if np.isinf(days_to_zero) else round(float(days_to_zero), 1),
            "capital_tied": round(float(row.get("capital_tied") or 0), 2),
            "safety_stock": int(row.get("safety_stock") or 0),
            "reorder_point": int(row.get("reorder_point") or 0),
            "suggested_order_qty": int(row.get("suggested_order_qty") or 0),
            "dead_stock_age": None if pd.isna(row.get("dead_stock_age")) else int(row.get("dead_stock_age") or 0),
            "confidence": round(float(row.get("confidence") or 0), 2),
            "shop_allocation_needs": shop_needs_by_item.get(str(row.get("id")), []),
        })

    return {
        "status": "success",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "window_days": days,
        "lead_time_days": lead_time_days,
        "items_scanned": int(len(merged)),
        "dead_stock_value": round(float(merged.loc[merged["status"] == "dead_stock", "capital_tied"].sum()), 2),
        "stockout_risk_14d": int((merged["days_to_zero"] <= 14).sum()),
        "capital_tied": round(float(merged["capital_tied"].sum()), 2),
        "model_comparison": _historical_model_comparison(recent_sales, item_key, days),
        "priority_items": items,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Calculate inventory velocity, reorder risk, and dead stock.")
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--dead-stock-days", type=int, default=60)
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--lead-time-days", type=int, default=7)
    parser.add_argument("--output")
    parser.add_argument("--save-db", action="store_true", help="Save this snapshot to analytics_results.")
    args = parser.parse_args()
    payload = run(args.days, args.dead_stock_days, args.limit, args.lead_time_days)
    if args.save_db:
        count = len(payload.get("priority_items", []))
        save_analytics_result("inventory_velocity", payload, f"{count} priority inventory items identified")
    write_json(payload, args.output)


if __name__ == "__main__":
    main()

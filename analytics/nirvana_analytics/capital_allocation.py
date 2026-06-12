from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .data_loader import load_allocations, load_inventory, load_operations, load_sales, load_table, save_analytics_result, write_json


BUCKETS = ["inventory", "invest", "blackbox", "reserves", "stockvel"]
BOUNDS = {
    "inventory": (0.25, 0.55),
    "invest": (0.05, 0.30),
    "blackbox": (0.05, 0.25),
    "reserves": (0.15, 0.40),
    "stockvel": (0.05, 0.25),
}


def _riskfolio_available() -> bool:
    try:
        import riskfolio  # noqa: F401
        return True
    except Exception:
        return False


def _inventory_capital() -> float:
    inventory = load_inventory(limit=10000)
    allocations = load_allocations(limit=10000)
    if inventory.empty:
        return 0.0

    if not allocations.empty and "item_id" in allocations:
        stock = allocations.groupby("item_id")["quantity"].sum().reset_index().rename(columns={"item_id": "id", "quantity": "allocated_stock"})
        inventory = inventory.merge(stock, on="id", how="left")
    else:
        inventory["allocated_stock"] = inventory.get("quantity", 0)

    cost_col = "landed_cost" if "landed_cost" in inventory else "cost" if "cost" in inventory else None
    if not cost_col:
        return 0.0
    return float((inventory["allocated_stock"].fillna(0) * inventory[cost_col].fillna(0)).sum())


def _capital_buckets() -> dict[str, float]:
    operations = load_operations(limit=10000)
    invest = load_table("invest_deposits", "*", limit=10000)

    invest_available = 0.0
    if not invest.empty:
        invest_available = float((pd.to_numeric(invest.get("amount", 0), errors="coerce").fillna(0) - pd.to_numeric(invest.get("withdrawn_amount", 0), errors="coerce").fillna(0)).sum())

    blackbox = 0.0
    reserves = 0.0
    stockvel = 0.0
    if not operations.empty:
        ops = operations.copy()
        ops["kind_norm"] = ops["kind"].fillna("").astype(str).str.lower()
        positive = ops[pd.to_numeric(ops["amount"], errors="coerce").fillna(0) > 0]
        blackbox = float(positive[positive["kind_norm"].isin(["blackbox", "black_box", "black-box"])]["amount"].sum())
        reserves = float(positive[positive["kind_norm"].isin(["savings_deposit", "savings_contribution", "savings"])]["amount"].sum())
        stockvel = float(ops[ops["kind_norm"].isin(["stockvel_loan", "stockvel_repayment"])]["amount"].sum())

    return {
        "inventory": max(0.0, _inventory_capital()),
        "invest": max(0.0, invest_available),
        "blackbox": max(0.0, blackbox),
        "reserves": max(0.0, reserves),
        "stockvel": max(0.0, stockvel),
    }


def _monthly_returns(days: int) -> pd.DataFrame:
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    sales = load_sales(limit=10000)
    operations = load_operations(limit=10000)
    invest = load_table("invest_deposits", "*", limit=10000)

    index = pd.date_range(cutoff, datetime.now(timezone.utc), freq="D").tz_convert("UTC").floor("D")
    flows = pd.DataFrame(index=index, columns=BUCKETS, data=0.0)

    if not sales.empty and "date" in sales:
        s = sales[sales["date"] >= cutoff].copy()
        s["day"] = s["date"].dt.floor("D")
        amount_col = "total_with_tax" if "total_with_tax" in s else "amount"
        flows.loc[s.groupby("day")[amount_col].sum().index, "inventory"] = s.groupby("day")[amount_col].sum().values

    if not operations.empty:
        ops = operations[operations["created_at"] >= cutoff].copy()
        ops["day"] = ops["created_at"].dt.floor("D")
        ops["kind_norm"] = ops["kind"].fillna("").astype(str).str.lower()
        for kind, bucket in [
            ("blackbox", "blackbox"),
            ("black_box", "blackbox"),
            ("black-box", "blackbox"),
            ("savings_deposit", "reserves"),
            ("savings_contribution", "reserves"),
            ("savings", "reserves"),
            ("stockvel_repayment", "stockvel"),
            ("stockvel_loan", "stockvel"),
        ]:
            subset = ops[ops["kind_norm"] == kind]
            if not subset.empty:
                grouped = subset.groupby("day")["amount"].sum()
                flows.loc[grouped.index, bucket] = flows.loc[grouped.index, bucket].add(grouped, fill_value=0)

    if not invest.empty and "deposited_at" in invest:
        inv = invest.copy()
        inv["deposited_at"] = pd.to_datetime(inv["deposited_at"], errors="coerce", utc=True)
        inv = inv[inv["deposited_at"] >= cutoff]
        inv["day"] = inv["deposited_at"].dt.floor("D")
        grouped = pd.to_numeric(inv.get("amount", 0), errors="coerce").fillna(0).groupby(inv["day"]).sum()
        flows.loc[grouped.index, "invest"] = grouped.values

    returns = flows.fillna(0).rolling(7, min_periods=1).sum().pct_change().replace([np.inf, -np.inf], np.nan).fillna(0)
    return returns


def _heuristic_expected_returns(returns: pd.DataFrame) -> np.ndarray:
    observed = returns.mean().reindex(BUCKETS).fillna(0).to_numpy()
    priors = np.array([0.11, 0.09, 0.025, 0.015, 0.08]) / 365
    return np.maximum(observed * 0.35 + priors * 0.65, 0.00001)


def _optimize(returns: pd.DataFrame, risk_aversion: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    mu = _heuristic_expected_returns(returns)
    cov = returns.reindex(columns=BUCKETS).cov().fillna(0).to_numpy()
    cov = cov + np.eye(len(BUCKETS)) * 1e-6
    bounds = [BOUNDS[bucket] for bucket in BUCKETS]

    if _riskfolio_available():
        try:
            import riskfolio as rp
            port = rp.Portfolio(returns=returns.reindex(columns=BUCKETS).fillna(0))
            port.assets_stats(method_mu='hist', method_cov='hist')
            # Inject our heuristic mu and covariance
            port.mu = pd.DataFrame(mu, index=BUCKETS).T
            port.cov = pd.DataFrame(cov, index=BUCKETS, columns=BUCKETS)
            
            # Enforce bounds
            port.lower_bound = [BOUNDS[b][0] for b in BUCKETS]
            port.upper_bound = [BOUNDS[b][1] for b in BUCKETS]
            
            w = port.optimization(model='Classic', rm='MV', obj='Sharpe', rf=0, l=0)
            if w is not None and not w.empty:
                weights = w['weights'].to_numpy().flatten()
                return weights, mu, cov, "riskfolio-classic-sharpe"
        except Exception as e:
            print(f"Riskfolio optimization failed: {e}. Falling back to SciPy SLSQP.")

    def objective(weights: np.ndarray) -> float:
        expected = float(weights @ mu)
        risk = float(np.sqrt(weights @ cov @ weights))
        return -(expected - risk_aversion * risk)

    result = minimize(
        objective,
        x0=np.repeat(1 / len(BUCKETS), len(BUCKETS)),
        method="SLSQP",
        bounds=bounds,
        constraints=[{"type": "eq", "fun": lambda w: float(np.sum(w) - 1)}],
        options={"maxiter": 1000, "ftol": 1e-9},
    )
    if not result.success:
        weights = np.array([0.4, 0.15, 0.1, 0.25, 0.1])
    else:
        weights = result.x
    return weights, mu, cov, "scipy-constrained-optimizer"


def run(days: int, risk_aversion: float) -> dict:
    buckets = _capital_buckets()
    total = float(sum(buckets.values()))
    returns = _monthly_returns(days)
    weights, mu, cov, backend_name = _optimize(returns, risk_aversion)

    current_weights = {bucket: (amount / total if total > 0 else 0) for bucket, amount in buckets.items()}
    target = {bucket: float(weights[i]) for i, bucket in enumerate(BUCKETS)}
    target_amounts = {bucket: round(total * target[bucket], 2) for bucket in BUCKETS}
    rebalance = {bucket: round(target_amounts[bucket] - buckets[bucket], 2) for bucket in BUCKETS}

    portfolio_return = float(weights @ mu) * 365
    portfolio_risk = float(np.sqrt(weights @ cov @ weights)) * np.sqrt(365)
    concentration = max(current_weights.values()) if current_weights else 0

    recommendations = []
    for bucket in BUCKETS:
        delta = rebalance[bucket]
        if abs(delta) < max(10, total * 0.02):
            continue
        recommendations.append({
            "bucket": bucket,
            "action": "add" if delta > 0 else "reduce",
            "amount": abs(delta),
            "reason": f"Target {target[bucket] * 100:.1f}% vs current {current_weights[bucket] * 100:.1f}%",
        })

    return {
        "status": "success",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "backend": backend_name,
        "riskfolio_available": _riskfolio_available(),
        "window_days": days,
        "risk_aversion": risk_aversion,
        "total_capital": round(total, 2),
        "current_amounts": {bucket: round(amount, 2) for bucket, amount in buckets.items()},
        "current_weights": {bucket: round(current_weights[bucket], 4) for bucket in BUCKETS},
        "target_weights": {bucket: round(target[bucket], 4) for bucket in BUCKETS},
        "target_amounts": target_amounts,
        "rebalance": rebalance,
        "portfolio": {
            "expected_return_annualized": round(portfolio_return, 4),
            "risk_annualized": round(portfolio_risk, 4),
            "concentration": round(concentration, 4),
        },
        "recommendations": recommendations,
        "notes": [
            "Advisory only: no money is moved by this job.",
            "Riskfolio-Lib is optional here; the production path uses SciPy constraints so Nirvana can enforce business bucket limits.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize Nirvana capital allocation across major capital buckets.")
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--risk-aversion", type=float, default=2.5)
    parser.add_argument("--output")
    parser.add_argument("--save-db", action="store_true", help="Save this snapshot to analytics_results.")
    args = parser.parse_args()
    payload = run(args.days, args.risk_aversion)
    if args.save_db:
        save_analytics_result("capital_allocation", payload, f"${payload['total_capital']:.2f} capital optimized")
    write_json(payload, args.output)


if __name__ == "__main__":
    main()

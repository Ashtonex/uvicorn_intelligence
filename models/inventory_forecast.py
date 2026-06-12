from __future__ import annotations
from analytics.nirvana_analytics import demand_forecast, inventory_velocity

def run_sales_forecast(days: int = 90, horizon: int = 14, shop_id: str | None = None) -> dict:
    """Thin wrapper around the centralized demand_forecast logic."""
    return demand_forecast.run(days=days, horizon=horizon, shop_id=shop_id)

def calculate_inventory_velocity() -> dict:
    """Thin wrapper around the centralized inventory_velocity logic."""
    return inventory_velocity.run()
